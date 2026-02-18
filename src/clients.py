"""
API clients for WikiBot (Twitter, OpenAI, OpenRouter, Wikipedia).
"""
import json
import time
import numpy as np
import tiktoken
import wikipediaapi
import tweepy
from requests_oauthlib import OAuth1
import requests
from openai import OpenAI
import os


class ContentFilterError(Exception):
    """Raised when an API refuses a request due to content policy."""
    pass


class CreditsExhaustedError(Exception):
    """Raised when OpenRouter credits are depleted."""
    pass

class APIClients:
    """
    Wrapper for all API clients used by WikiBot.
    """

    def __init__(self, project_path, openai_api_key, twitter_api_keys, openrouter_api_key=None):
        """
        Initialize API clients with their respective keys.

        Args:
            openai_api_key (str): OpenAI API key
            twitter_api_keys (list): Twitter API keys in order:
                [api_key, api_secret, bearer, access_token, access_token_secret]
            openrouter_api_key (str): OpenRouter API key (for Claude etc.)
        """
        self.openai_api_key = openai_api_key
        self.openrouter_api_key = openrouter_api_key
        self.twitter_api_keys = twitter_api_keys
        self.project_path = project_path

        # Initialize clients
        self.wiki = self._setup_wikipedia()
        self.twitter_auth = self._setup_twitter_auth()
        self.twitter_client_v2 = self._setup_twitter_v2()
        self.openai_client = self._setup_openai()
        self.openrouter_client = self._setup_openrouter()
        self.encoder = tiktoken.get_encoding("o200k_base")

        self.usage_path = os.path.join(self.project_path, 'data', 'usage.json')
        self.usage = {}  # model -> {input_tokens, output_tokens}
        self.img_count = 0
        self.load_usage()

    # Per-model pricing (USD per million tokens)
    MODEL_PRICING = {
        'gpt-4o-mini-2024-07-18': (0.15, 0.60),
        'anthropic/claude-haiku-4.5': (0.80, 4.00),
        'google/gemini-2.0-flash-001': (0.10, 0.40),
        'google/gemini-2.0-flash-lite-001': (0.075, 0.30),
        'gpt-image-1-mini': (0.0, 0.0),  # tracked via img_count
    }
    IMG_COST = 0.02  # per image

    def load_usage(self):
        if os.path.exists(self.usage_path):
            with open(self.usage_path, 'r') as f:
                data = json.load(f)
            self.usage = data.get('models', {})
            self.img_count = data.get('img_count', 0)

    def save_usage(self):
        data = {'models': self.usage, 'img_count': self.img_count}
        with open(self.usage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def track_tokens(self, model, input_tokens, output_tokens):
        if model not in self.usage:
            self.usage[model] = {'input_tokens': 0, 'output_tokens': 0}
        self.usage[model]['input_tokens'] += input_tokens
        self.usage[model]['output_tokens'] += output_tokens
        self.save_usage()

    def print_costs(self, n_runs=None):
        total_cost = self.img_count * self.IMG_COST
        for model, counts in self.usage.items():
            in_price, out_price = self.MODEL_PRICING.get(model, (0.50, 2.00))  # default conservative
            total_cost += counts['input_tokens'] / 1e6 * in_price + counts['output_tokens'] / 1e6 * out_price
        if n_runs and n_runs > 0:
            print(f'costs: ${total_cost:.3f} total, ${total_cost/n_runs*100:.2f} cts/run')
        else:
            print(f'costs: ${total_cost:.3f} total')

    def _setup_wikipedia(self):
        """Set up Wikipedia API client."""
        return wikipediaapi.Wikipedia(
            language='en',
            user_agent='WikiBot/1.0 (wiki_voyager)'
        )

    def _setup_twitter_auth(self):
        """Set up OAuth1 auth for v2 media upload."""
        return OAuth1(
            self.twitter_api_keys[0],
            self.twitter_api_keys[1],
            self.twitter_api_keys[3],
            self.twitter_api_keys[4]
        )

    def upload_media(self, file_path):
        """Upload media via Twitter API v2 (initialize → append → finalize)."""
        auth = self.twitter_auth
        base = "https://api.x.com/2/media/upload"

        file_size = os.path.getsize(file_path)
        mime = "image/png" if file_path.endswith(".png") else "image/jpeg"

        # INIT
        resp = requests.post(f"{base}/initialize", auth=auth, json={
            "media_type": mime,
            "total_bytes": file_size,
            "media_category": "tweet_image"
        })
        resp.raise_for_status()
        media_id = resp.json()["data"]["id"]

        # APPEND
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{base}/{media_id}/append",
                auth=auth,
                files={"media": f},
                data={"segment_index": 0}
            )
            resp.raise_for_status()

        # FINALIZE
        resp = requests.post(f"{base}/{media_id}/finalize", auth=auth)
        resp.raise_for_status()

        return media_id

    def _setup_twitter_v2(self):
        """Set up Twitter API v2 client (for tweeting)."""
        return tweepy.Client(
            consumer_key=self.twitter_api_keys[0],
            consumer_secret=self.twitter_api_keys[1],
            access_token=self.twitter_api_keys[3],
            access_token_secret=self.twitter_api_keys[4]
        )

    def _setup_openai(self):
        """Set up OpenAI client."""
        return OpenAI(api_key=self.openai_api_key, timeout=50.0)

    def _setup_openrouter(self):
        """Set up OpenRouter client (OpenAI-compatible)."""
        if not self.openrouter_api_key:
            return None
        return OpenAI(
            api_key=self.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0
        )

    def call_text_model(self, messages, model="gpt-4o-mini-2024-07-18", max_tokens=1000):
        """
        Call a text model with retry logic. Routes to OpenAI or OpenRouter based on model name.

        Args:
            messages (list): List of message dictionaries
            model (str): Model name to use
            max_tokens (int): Maximum tokens to generate

        Returns:
            The model's response
        """
        if model.startswith('gpt-') or model.startswith('o1'):
            return self._call_openai(messages, model, max_tokens)
        else:
            return self._call_openrouter(messages, model, max_tokens)

    def _call_openai(self, messages, model, max_tokens):
        """Call OpenAI API with retry logic."""
        i_attempt = 0
        response = None
        error = ""

        while i_attempt < 5:
            i_attempt += 1
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens
                )
                self.track_tokens(model, response.usage.prompt_tokens, response.usage.completion_tokens)
                output = response.choices[0].message.content
                return output
            except Exception as err:
                error = str(err)
                if 'content_policy' in error or 'content policy' in error.lower() or 'safety' in error.lower():
                    raise ContentFilterError(f"Content filtered: {error}")
                print(f'API error: {error}')
                time.sleep(np.random.randint(5, 60))

        if response is None:
            raise RuntimeError(f"Error in text model call: {error}")

    def _call_openrouter(self, messages, model, max_tokens):
        """Call OpenRouter API (OpenAI-compatible) with retry logic."""
        if not self.openrouter_client:
            raise ValueError("OpenRouter API key not set")

        i_attempt = 0
        response = None
        error = ""

        while i_attempt < 5:
            i_attempt += 1
            try:
                response = self.openrouter_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens
                )
                if response.usage:
                    self.track_tokens(model, response.usage.prompt_tokens, response.usage.completion_tokens)
                output = response.choices[0].message.content
                return output
            except Exception as err:
                error = str(err)
                if '402' in error or 'credit' in error.lower() or 'insufficient' in error.lower():
                    raise CreditsExhaustedError(f"OpenRouter credits exhausted: {error}")
                if 'content_policy' in error or 'content policy' in error.lower() or 'safety' in error.lower():
                    raise ContentFilterError(f"Content filtered: {error}")
                print(f'API error: {error}')
                time.sleep(np.random.randint(5, 60))

        if response is None:
            raise RuntimeError(f"Error in OpenRouter call: {error}")

    def call_image_model(self, prompt, model='gpt-image-1-mini', size="1024x1024", quality='low'):
        """
        Call OpenAI image model with retry logic.

        Args:
            prompt (str): Text prompt for image generation
            model (str): Model name to use
            size (str): Image size
            quality (str): Image quality ('low', 'medium', 'high')

        Returns:
            Base64-encoded image data (str)
        """
        i_attempt = 0
        response = None
        error = ""

        while i_attempt < 5:
            i_attempt += 1
            try:
                response = self.openai_client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=1,
                    size=size,
                    quality=quality
                )
                self.img_count += 1
                break
            except Exception as err:
                error = str(err)
                if 'content_policy' in error or 'content policy' in error.lower() or 'safety' in error.lower():
                    raise ContentFilterError(f"Content filtered: {error}")
                print(f'API error: {error}')
                time.sleep(np.random.randint(5, 60))

        if response is None:
            raise RuntimeError(f"Error in image model call: {error}")
        self.save_usage()
        return response.data[0].b64_json

    def get_page(self, title):
        i_attempts = 0
        error = ''
        while i_attempts < 5:
            try:
                page = self.wiki.page(title)
                return page
            except Exception as err:
                error = str(err)
                i_attempts += 1
                time.sleep(3)
        raise ValueError(f'Wikipedia API error: {error}')

    def page_exists(self, page):
         i_attempts = 0
         error = ''
         while i_attempts < 5:
             try:
                 page_exists = page.exists()
                 return page_exists
             except Exception as err:
                 error = str(err)
                 i_attempts += 1
                 time.sleep(3)
         raise ValueError(f'Wikipedia API error: {error}')


    def crop_text(self, text, n_tokens):
        """
        Crop text to a maximum number of tokens.

        Args:
            text (str): Text to crop
            n_tokens (int): Maximum number of tokens

        Returns:
            str: Cropped text
        """
        tokens = self.encoder.encode(text)
        cropped_text = self.encoder.decode(tokens[:n_tokens])
        if len(tokens) > n_tokens:
            cropped_text += ' [...]'
        return cropped_text