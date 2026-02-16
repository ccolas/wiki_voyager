"""
Content generation module for WikiBot (tweets and images).
"""
import os
import base64
from PIL import Image
from pydantic import BaseModel
from typing import Optional

from utils import format_tweet
from tweet_splitter import split_tweets
from clients import ContentFilterError


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
        self.debug = params['debug']

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

        # Build image prompt directly — no LLM call needed
        summary_sentences = page.summary.split('. ')[:3]
        short_summary = '. '.join(summary_sentences).strip()
        if not short_summary.endswith('.'):
            short_summary += '.'

        style = (
            "Risograph print style, 2-4 muted colors on warm cream paper, "
            "subtle grain and halftone texture, clean minimal composition, "
            "soft desaturated palette. No text, no letters, no words, no labels, no captions."
        )
        image_prompt = f"An illustration of: {title}.\n\nContext: {short_summary}\n\nStyle: {style}"

        # if self.debug:
        #     print(f'Image prompt: {image_prompt}')

        # Generate image (ContentFilterError propagates up)
        image_b64 = self.clients.call_image_model(image_prompt, self.params['img_model'])

        # Save the image from base64
        img_path = self.params['img_path'] + f'{step_id}.png'
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        img_info = dict(prompt=image_prompt, path=img_path)
        return img_info

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
            start_date = memories[0]['date'].replace(':', ' ')
            day_number = len(memories) + 1
            ctx = self.params['context_length']
            user_prompt += (f"# Context\n\nToday's page is: {title}\n\n"
                            f"## Journey info\n\nYou started this exploration on {start_date} (day 1). Today is day {day_number}.\n\n")

            # Older memories (titles only)
            older = memories[:-ctx] if len(memories) > ctx else []
            older = older[-100:]
            if older:
                user_prompt += "## Earlier exploration (titles only, oldest first):\n"
                for i, memory in enumerate(older):
                    user_prompt += f"{i + 1}. {memory['title']}\n"
                user_prompt += "\n"

            # Recent memories (title + summary)
            recent = memories[-ctx:]
            user_prompt += "## Recent exploration history (most recent first):\n"
            for i, memory in enumerate(reversed(recent)):
                user_prompt += f"{i + 1}. {memory['title']}: {memory['summary']}\n"
            user_prompt += f"\nEach page led to the next, e.g. you arrived at '{title}' from page '{memories[-1]['title']}'.\n\n"

            user_prompt += f"# Yesterday's thread\n\n{format_tweet(memories[-1]['tweets'], remove_url=True)}\n\n"
        else:
            user_prompt += (f"# Context\n\nToday's page is: {title}\n\n"
                            "## Journey info\n\nThis is DAY 1 — the very first step of your Wikipedia exploration! "
                            "Comment on this new beginning: you're starting a journey through Wikipedia, hopping from page to page by following links, "
                            "and you have no idea where it will take you.\n\n")

        # Create user prompt with exploration context
        user_prompt += (f"# Task\n\n"
                        f"Today's page is: {title}. The text is:\n\n<WIKIPEDIA>\n\n{text}\n\n</WIKIPEDIA>\n\n"
                        f"Please write today's thread now!")

        # Call the model
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        # if self.debug:
        #     print('##########################\n'
        #           'PROMPT FOR SUMMARIZER')
        #     for m in messages:
        #         print(f'\n\n{m["role"]}\n{m["content"]}')
        text = self.clients.call_text_model(messages, self.params['text_model'], max_tokens=800)

        # Add Wikipedia URL to the end if not already included
        text = f"{text}\n\nlearn more: {page.fullurl}"

        # Split into tweets
        tweets = split_tweets(text)

        # Validate tweet lengths
        for t in tweets:
            assert len(t) <= tweet_limit, f"Tweet exceeds limit: {len(t)} > {tweet_limit}"

        return tweets

