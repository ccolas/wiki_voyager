"""
Content generation module for WikiBot (tweets and images).
"""
import os
from PIL import Image
from pydantic import BaseModel
from typing import Optional

from utils import download_image, format_tweet
from tweet_splitter import split_tweets


class PromptOrSearch(BaseModel):
    """Pydantic model for the prompt or search response."""
    thinking: str
    search: bool
    prompt: Optional[str] = None


class ContentGenerator:
    """
    Content generator for WikiBot.
    Handles tweet and image generation.
    """

    def __init__(self, clients, params):
        """
        Initialize the content generator.

        Args:
            clients: APIClients instance
            params: paramsuration dictionary
        """
        self.clients = clients
        self.params = params

    def generate_image(self, title, page, step_id, new_tweets):
        """
        Generate an image for a Wikipedia page.

        Args:
            title (str): Page title
            page: Wikipedia page object
            step_id (str): Unique identifier for this exploration step

        Returns:
            dict: Image information dictionary or None if generation fails
        """
        # Skip image generation if configured to do so
        if 'img' in self.params['to_skip']:
            return None

        try:
            # Prepare page content
            summary = self.clients.crop_text(page.summary, n_tokens=2000)

            # Read prompt from file
            with open(self.params['project_path'] + "src/prompts/image_generation.txt", "r") as f:
                system_prompt = f.read().strip()

            # Create user prompt
            user_prompt = (f"The title is: {title}. The beginning of the page is:\n{summary}\n\n"
                           f"Here the tweet you want to illustrate:\n{format_tweet(new_tweets)}\n\n"
                           f"Please generate an image prompt for DALL-E that follows the guidelines.")
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            # Get image prompt from AI
            image_prompt = self.clients.call_text_model(messages, self.params['text_model'], max_tokens=200)

            # Generate image using OpenAI API
            image_url = self.clients.call_image_model(image_prompt, self.params['img_model'])

            # Download and save the image
            img_path = self.params['img_path'] + f'{step_id}.png'
            download_image(image_url, img_path)
            img_info = dict(prompt=image_prompt, url=image_url, path=img_path)
            return img_info
        except Exception as err:
            print(f"Error in image generation: {str(err)}")
            return None

    def generate_tweet(self, title, page, memories, tweet_limit=280):
        """
        Generate tweets for a Wikipedia page, including reflections on the exploration journey.

        Args:
            title (str): Page title
            page: Wikipedia page object
            recent_memories (list): Recent exploration history
            tweet_limit (int): Maximum tweet length

        Returns:
            list: List of tweet strings
        """
        if 'tweets' in self.params.get('to_skip', []):
            return None

        # Get page text and clean title
        text = self.clients.crop_text(page.text, n_tokens=2000)

        # Load tweet generation prompt from file
        with open(self.params['project_path'] + 'src/prompts/tweet_generation.txt', 'r') as f:
            system_prompt = f.read().strip()

        # Add exploration history context
        user_prompt = ""
        if memories and len(memories) > 0:
            user_prompt += (f"# Context\n\nToday's page is: {title}\n\n"
                            "## Recent exploration history (most recent first):\n")
            for i, memory in enumerate(reversed(memories[-self.params['context_length']:])):
                user_prompt += f"{i + 1}. {memory['title']}: {memory['summary']}\n"
            user_prompt += f"\nEach page led to the next, e.g. you arrived at '{title}' from page '{memories[-1]['title']}'.\n\n"

            user_prompt += f"# Yesterday's thread\n\n{format_tweet(memories[-1]['tweets'], remove_url=True)}\n\n"


        # Create user prompt with exploration context
        user_prompt += (f"# Task\n\n"
                        f"Today's title is: {title}. The text is:\n\n<WIKIPEDIA>\n\n{text}\n\n</WIKIPEDIA>\n\n"
                        f"Please write today's thread now!")

        # Call the model
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        text = self.clients.call_text_model(messages, self.params['text_model'], max_tokens=800)

        # Add Wikipedia URL to the end if not already included
        text = text + f'\nlearn more: {page.fullurl}'

        # Split into tweets
        tweets = split_tweets(text)

        # Validate tweet lengths
        for t in tweets:
            assert len(t) < tweet_limit, f"Tweet exceeds limit: {len(t)} > {tweet_limit}"

        return tweets

