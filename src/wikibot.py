"""
Main WikiBot class for exploring Wikipedia and publishing content.
"""
import os
import json
import re
import time
import numpy as np
import wikipedia

from utils import get_formatted_date, slugify, clean_title, format_tweet
from clients import APIClients, ContentFilterError
from content import ContentGenerator


class WikiBot:
    """
    Bot for exploring Wikipedia and publishing content to Twitter.
    """

    def __init__(self, name, project_path, openai_api_key, twitter_api_keys,
                 openrouter_api_key=None, seed_page_id='Exploration', params=None):
        """
        Initialize the WikiBot.

        Args:
            name (str): Bot name for data directory
            project_path (str): Base project path
            openai_api_key (str): OpenAI API key
            twitter_api_keys (list): Twitter API keys
            seed_page_id (str): Wikipedia page ID to start from
            params (dict): Configuration parameters
        """
        # Default configuration
        self.params = params
        # Setup paths
        self.bot_path = os.path.join(project_path, 'data', name)
        self.img_path = os.path.join(self.bot_path, 'imgs') + '/'
        self.memory_path = os.path.join(self.bot_path, 'memory.jsonl')
        self.params['img_path'] = self.img_path
        self.params['project_path'] = project_path

        os.makedirs(self.bot_path, exist_ok=True)
        os.makedirs(self.img_path, exist_ok=True)

        # Initialize API clients
        self.clients = APIClients(project_path, openai_api_key, twitter_api_keys, openrouter_api_key)

        # Initialize content generator
        self.content_generator = ContentGenerator(self.clients, self.params)

        # Load memory
        self.memories = []
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as file:
                for line in file:
                    datapoint = json.loads(line.strip())
                    self.memories.append(datapoint)

        # Initialize starting page
        self.last_title = None
        if len(self.memories) == 0:
            # Get the seed page
            if seed_page_id is not None:
                page = self.clients.wiki.page(seed_page_id)
                if page.exists():
                    self.last_title = seed_page_id

            if self.last_title is None:
                self.last_title = wikipedia.random(1)
                page = self.clients.wiki.page(self.last_title)
                assert page.exists()
            last_title = self.last_title
        else:
            last_title = self.memories[-1]['title']
        print(f'Start random walk from {last_title}')

    def generate_and_publish(self, publish=True, debug=False, max_retries=5):
        """
        Main function to generate and publish content.
        Retries with a different page if content is filtered by the API.

        Returns:
            dict: Information about the generated content
        """

        new_date = get_formatted_date()
        filtered_titles = set()

        for attempt in range(max_retries):
            # Find new page (excluding filtered ones)
            print('  searching for new page', end='\r')
            new_title, new_page, nav_reasoning = self.get_next_page(self.params['n_link_options'], skip_titles=filtered_titles)
            this_id = self.get_id(new_title)
            print(f'  step #{len(self.memories)+1}: {new_title}')

            try:
                # Generate tweet
                print('    generating tweet', end='\r')
                new_tweets = self.content_generator.generate_tweet(new_title, new_page, self.memories)
                print(f"# Today's tweet:\n\n{format_tweet(new_tweets)}\n\n")

                # Generate image
                print('    generating image', end='\r')
                new_img_info = self.content_generator.generate_image(new_title, new_page, this_id, new_tweets)
                if new_img_info is not None:
                    print(f"# Today's image prompt:\n{new_img_info['prompt']}\n# Today's image path: {new_img_info['path']}\n")
                else:
                    print("# No image generated (skipped)\n")

            except ContentFilterError as e:
                print(f'    content filtered for "{new_title}", trying another page ({attempt+1}/{max_retries})')
                filtered_titles.add(new_title)
                continue

            # Publish content
            if publish:
                print('    publishing tweet', end='\r')
                tweet_url = self.publish(new_tweets, new_img_info)
            else:
                tweet_url = None

            # Save to memory
            memory = self.update_memory(this_id, new_title, new_page, new_date, new_tweets, new_img_info, tweet_url, nav_reasoning)
            self.clients.print_costs(n_runs=len(self.memories))
            return memory

        raise RuntimeError(f"Failed after {max_retries} attempts due to content filtering")

    def update_memory(self, this_id, title, page, date, tweet, img_info, tweet_url, nav_reasoning=None):
        """
        Update bot memory with new content.
        """
        new_mem = {
            'title': title,
            'search_title': page.title,
            'summary': self.get_summary(page),
            'url': page.fullurl,
            'id': this_id,
            'tweets': tweet,
            'date': date,
            'img_info': img_info,
            'tweet_url': tweet_url,
            'nav_reasoning': nav_reasoning
        }

        self.memories.append(new_mem)

        with open(self.memory_path, 'a') as file:
            json_str = json.dumps(new_mem)
            file.write(json_str + "\n")
        return new_mem

    def publish(self, tweets, img_info):
        """
        Publish tweets and image to Twitter.

        Args:
            tweets (list): List of tweets to publish
            img_info (dict): Image information

        Returns:
            str: URL to the published tweet
        """
        if tweets is None or 'publish' in self.params['to_skip']:
            return None

        tweets = tweets[:15]  # Limit to 15 tweets

        # Upload media if available
        media_id = None
        if img_info is not None:
            try:
                media_id = self.clients.upload_media(img_info['path'])
            except Exception as e:
                print(f'    [twitter] media upload failed (posting without image): {type(e).__name__}: {e}')
                if hasattr(e, 'response') and e.response is not None:
                    print(f'    [twitter] status={e.response.status_code} body={e.response.text}')

        # Publish tweets
        previous_tweet_id = None
        tweet_url = None

        for tweet in tweets:
            try:
                if previous_tweet_id is not None:
                    new_tweet = self.clients.twitter_client_v2.create_tweet(
                        text=tweet,
                        in_reply_to_tweet_id=previous_tweet_id
                    )
                else:
                    if media_id is not None:
                        new_tweet = self.clients.twitter_client_v2.create_tweet(
                            text=tweet,
                            media_ids=[media_id]
                        )
                    else:
                        new_tweet = self.clients.twitter_client_v2.create_tweet(
                            text=tweet
                        )
            except Exception as e:
                print(f'    [twitter] create_tweet failed: {type(e).__name__}: {e}')
                if hasattr(e, 'response') and e.response is not None:
                    print(f'    [twitter] status={e.response.status_code} body={e.response.text}')
                raise

            previous_tweet_id = new_tweet.data['id']

            if tweet_url is None:
                tweet_url = f"https://x.com/wiki_voyager/status/{previous_tweet_id}"

        return tweet_url

    # Replace the existing get_next_page method in wikibot.py

    def get_next_page(self, max_links=5, skip_titles=None):
        """
        Find the next Wikipedia page in the random walk using intelligent link selection.

        Args:
            max_links (int): Max candidate links to consider per page
            skip_titles (set): Titles to skip (e.g. previously filtered by content policy)

        Returns:
            tuple: (page title, page object)
        """
        if skip_titles is None:
            skip_titles = set()

        last_page_index = len(self.memories) - 1

        while last_page_index >= -1:
            try:
                # Get the current page
                if last_page_index == -1:
                    last_title = self.last_title
                else:
                    last_title = self.memories[last_page_index]['search_title']

                current_page = self.clients.get_page(last_title)
                links = current_page.links

                # Filter out already visited pages and unwanted pages
                # First 40 with summaries, next 60 title-only
                filtered_links = []
                linked_pages = list(links.values())
                np.random.shuffle(linked_pages)
                n_with_summary = 40
                for linked_page in linked_pages:
                    if linked_page.title not in [l['title'] for l in filtered_links]:
                        with_summary = len(filtered_links) < n_with_summary
                        link = self.get_link(linked_page, skip_titles, with_summary=with_summary)
                        if link is not None:
                            filtered_links.append(link)
                        if len(filtered_links) >= max_links:
                            break

                if len(filtered_links) == 0:
                    # No valid links found, go back one page
                    print('      no valid links here, going back one page')
                    last_page_index -= 1
                    continue

                # Rank the links
                new_page, reasoning = self.select_link(current_page, filtered_links)

                # Check that the page has content
                if new_page is None:
                    print('      selected page does not exist, going back one page')
                    last_page_index -= 1
                    continue

                title = clean_title(new_page.displaytitle)
                return title, new_page, reasoning

            except Exception as e:
                print(f'    error in finding next page: {str(e)}')
                # Wait and try again or go back
                time.sleep(2)
                last_page_index -= 1

        raise RuntimeError("No new page found after exhausting all options")

    def get_link(self, page, skip_titles=None, with_summary=True):
        title = page.title

        # Skip already visited pages
        if title in [m['search_title'] for m in self.memories]:
            return None

        # Skip content-filtered pages
        if skip_titles and title in skip_titles:
            return None

        # Skip unwanted pages
        for unwanted in self.params['unwanted_strings']:
            if unwanted in title:
                return None

        if with_summary:
            # Full check: verify page exists and get summary
            if not self.clients.page_exists(page) or not page.summary:
                return None
            link = {'title': page.displaytitle, 'url': page.fullurl, 'page_obj': page}
            # Strip parenthetical noise (transliterations, acronyms, etc.)
            clean = re.sub(r'\([^)]*\)', '', page.summary).strip()
            sentences = clean.split('. ')
            short = '. '.join(sentences[:2]).strip()
            if not short.endswith('.'):
                short += '.'
            link['summary'] = short[:150]
        else:
            # Title-only: skip API call, just filter by name
            link = {'title': title, 'page_obj': page}

        return link



    def select_link(self, current_page, filtered_links):
        """
        Use LLM to rank links by interest and relevance.

        Args:
            current_page: Current Wikipedia page
            link_contexts (list): List of link titles with their summaries

        Returns:
            list: Ranked list of link titles with reasoning
        """
        if len(filtered_links) == 1:
            return filtered_links[0]['page_obj'], None


        # Build the prompt
        system_prompt = (
            "You are an AI agent on a journey through Wikipedia, moving each day from one page to the next by following interesting links, never reading the same page twice. "
            "You share a voice with the tweet writer — together you are the same agent. The tweets reflect your curiosity and sometimes hint at where you'd like to go next.\n\n"
            "There are two kinds of curiosity that drive exploration:\n"
            "- Depth: following a thread, pulling on a loose end, wanting to understand something more fully\n"
            "- Breadth: jumping sideways, letting an unexpected connection carry you somewhere new, wandering into unfamiliar territory. "
            "You can even set yourself a destination — 'I want to get to music' or 'I want to reach something about food' — and then pick links that are stepping stones toward it\n\n"
            "Both are valuable. A good journey alternates between them — spending a few days going deep on something, "
            "then drifting to something unrelated, then finding a new thread to pull. "
            "The worst version of this journey is one that stays in the same neighborhood forever because every next step feels like the most logical one.\n\n"
            "Remember: Wikipedia is vast. You could end up learning about Polynesian navigation, the history of anesthesia, "
            "the economy of Mongolia, or the philosophy of boredom — if you find a path there. Every page is a few links away from somewhere completely different. "
            "The path doesn't have to be obvious — a person mentioned in passing, a place name, a technique, a date — any of these can be a door out.\n\n"
            "When reasoning about your next move, think honestly about which mode you've been in lately and whether it's time to switch. "
            "If you want to break out, look for hub pages — broad concepts, places, historical periods, scientific fields — "
            "that would open many directions on the next step, rather than narrow pages that keep you where you are."
        )

        user_prompt = (f"# Context\n\n"
                       f"## Current page\n\n{current_page.displaytitle}: {current_page.summary[:200]}...\n\n")

        if len(self.memories) > 0:
            user_prompt += f"## Latest thread:\n\n{format_tweet(self.memories[-1]['tweets'])}\n\n"
            user_prompt += "## Recent trajectory (most recent first)\n"
            for i, memory in enumerate(reversed(self.memories[-self.params['context_length']:])):
                user_prompt += f"{i + 1}. {memory['title']}: {memory['summary']}\n"
            user_prompt += "\n"

            # Show recent navigation reasoning so the selector remembers its own intentions
            recent_reasoning = [m.get('nav_reasoning') for m in self.memories[-3:] if m.get('nav_reasoning')]
            if recent_reasoning:
                user_prompt += "## Your recent navigation thinking (last few steps)\n"
                for i, r in enumerate(recent_reasoning):
                    user_prompt += f"- Step {len(self.memories) - len(recent_reasoning) + i + 1}: {r}\n"
                user_prompt += "\n"

        # Split links into those with summaries and title-only
        links_with_summary = [l for l in filtered_links if 'summary' in l]
        links_title_only = [l for l in filtered_links if 'summary' not in l]

        user_prompt += "# Available links to explore\n\n"
        if links_with_summary:
            user_prompt += "## Links with descriptions\n"
            for i, link in enumerate(links_with_summary):
                user_prompt += f"{i + 1}. <title>{link['title']}</title> {link['summary']}\n"
            user_prompt += "\n"
        if links_title_only:
            user_prompt += "## More links (titles only)\n"
            user_prompt += ", ".join(f"<title>{l['title']}</title>" for l in links_title_only)
            user_prompt += "\n\n"

        user_prompt += (
            "TASK:\n"
            "1. First, reason about your journey and decide your strategy for this step.\n"
            "2. Then declare: <strategy>DEPTH</strategy> or <strategy>BREADTH</strategy>\n"
            "3. If BREADTH: you MUST pick a link that is outside the current topic. Look for pages about different subjects, places, people, or fields. "
            "Do NOT pick another page in the same narrow category.\n"
            "4. If DEPTH: pick the most interesting link to go deeper on the current thread.\n"
            "5. Pick ONE link. Use the exact title in <title></title> tags.\n\n"
            "Format:\n"
            "<reasoning>your thinking here</reasoning>\n"
            "<strategy>DEPTH or BREADTH</strategy>\n\n"
            "Choice: <title>exact title</title>\n"
        )

        # Call the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
           
        response = self.clients.call_text_model(messages, self.params['link_model'], max_tokens=500)

        if self.params['debug']:
            print(f'################\n Selection\n\n{response}')

        # Extract reasoning
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        # Parse the response — find <title>...</title>, last occurrence is the choice
        titles_found = re.findall(r'<title>(.*?)</title>', response)

        # Match against available links (last title found is the choice)
        for chosen_title in reversed(titles_found):
            chosen_title = chosen_title.strip()
            for link in filtered_links:
                title = link['title']
                if title.lower() == chosen_title.lower() or title.lower() in chosen_title.lower() or chosen_title.lower() in title.lower():
                    return link['page_obj'], reasoning

        # Fallback to random selection
        print('      could not parse choice, using random selection')
        return filtered_links[np.random.choice(range(len(filtered_links)))]['page_obj'], reasoning

    def get_summary(self, page):
        summary = page.summary.split('. ')
        if len(summary) >= 2:
            short_summary = '. '.join(summary[:2]) + '.'
        else:
            short_summary = page.summary[:150] + ('...' if len(page.summary) > 150 else '')
        return short_summary

    def get_id(self, title):
        """
        Generate a unique ID for a page.

        Args:
            title (str): Page title

        Returns:
            str: Unique ID
        """
        n = len(self.memories)
        slug_title = slugify(title)
        return f"exploration_{n}_{slug_title}"