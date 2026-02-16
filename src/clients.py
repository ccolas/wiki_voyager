"""
API clients for WikiBot (Twitter, OpenAI, Wikipedia).
"""
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
    """Raised when OpenAI refuses a request due to content policy."""
    pass

class APIClients:
    """
    Wrapper for all API clients used by WikiBot.
    """

    def __init__(self, project_path, openai_api_key, twitter_api_keys):
        """
        Initialize API clients with their respective keys.

        Args:
            openai_api_key (str): OpenAI API key
            twitter_api_keys (list): Twitter API keys in order:
                [api_key, api_secret, bearer, access_token, access_token_secret]
        """
        self.openai_api_key = openai_api_key
        self.twitter_api_keys = twitter_api_keys
        self.project_path = project_path

        # Initialize clients
        self.wiki = self._setup_wikipedia()
        self.twitter_auth = self._setup_twitter_auth()
        self.twitter_client_v2 = self._setup_twitter_v2()
        self.openai_client = self._setup_openai()
        self.encoder = tiktoken.get_encoding("o200k_base")

        self.input_tokens = 0
        self.output_tokens = 0
        self.img_count = 0
        self.load_usage()

    def load_usage(self):
        if os.path.exists(self.project_path + 'data/usage.txt'):
            with open(self.project_path + 'data/usage.txt', 'r') as f:
                usage_txt = f.read()
            self.input_tokens, self.output_tokens, self.img_count = [int(val) for val in usage_txt.split('\n')]

    def save_usage(self):
        usage_txt = '\n'.join([str(val) for val in [self.input_tokens, self.output_tokens, self.img_count]])
        with open(self.project_path + 'data/usage.txt', 'w') as f:
            f.write(usage_txt)

    def print_costs(self):
        total_cost = self.img_count * 0.02 + self.input_tokens / 1e6 * 0.15 + self.output_tokens / 1e6 * 0.60
        cost_per_day = total_cost / self.img_count
        print(f'costs: USD {total_cost:.2f}, USD {cost_per_day*100:.2f} cts/day')

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

    def call_text_model(self, messages, model="gpt-4o-mini-2024-07-18", max_tokens=1000):
        """
        Call OpenAI text model with retry logic.

        Args:
            messages (list): List of message dictionaries
            model (str): Model name to use
            response_model: Pydantic model for response parsing
            max_tokens (int): Maximum tokens to generate

        Returns:
            The model's response
        """
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
                self.output_tokens += response.usage.completion_tokens
                self.input_tokens += response.usage.prompt_tokens
                output = response.choices[0].message.content
                self.save_usage()
                return output
            except Exception as err:
                error = str(err)
                if 'content_policy' in error or 'content policy' in error.lower() or 'safety' in error.lower():
                    raise ContentFilterError(f"Content filtered: {error}")
                print(f'API error: {error}')
                time.sleep(np.random.randint(5, 60))

        if response is None:
            raise RuntimeError(f"Error in text model call: {error}")

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