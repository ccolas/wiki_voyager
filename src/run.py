"""
Run script for WikiBot - executes the bot on a daily schedule.
"""
from datetime import datetime, timedelta
import os
import json
import time
import numpy as np
import zoneinfo

from wikibot import WikiBot

DEBUG = False
PUBLISH = True
POST_HOUR = 19  # 7pm Paris time
TIMEZONE = zoneinfo.ZoneInfo("Europe/Paris")

def read_api_keys(project_path):
    """Read API keys from files."""
    with open(os.path.join(project_path, '.api_openai'), 'r') as f:
        openai_api_key = f.read().strip()

    with open(os.path.join(project_path, '.api_twitter'), 'r') as f:
        twitter_api_keys = f.read().strip().split('\n')

    with open(os.path.join(project_path, 'src', '.api_openrouter'), 'r') as f:
        openrouter_api_key = f.read().strip()

    return openai_api_key, twitter_api_keys, openrouter_api_key

PARAMS = {
    'unwanted_strings': ['wiki', 'Wiki', 'Category', 'List', 'Template',
                         'Help', 'ISO', 'User', 'Talk', 'Portal', '501'],
    'tweet_model': "anthropic/claude-haiku-4.5",
    'link_model': "google/gemini-2.0-flash-001",
    'img_model': "gpt-image-1-mini",
    'to_skip': [],
    'context_length': 30,
    'n_link_options': 100,
    'debug': DEBUG
}

if __name__ == '__main__':
    # Set up paths
    script_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(script_path) + '/'

    # Read API keys
    openai_api_key, twitter_api_keys, openrouter_api_key = read_api_keys(project_path)

    # Bot configuration
    bot_name = 'twitter_trail'
    bot_path = os.path.join(project_path, 'data', bot_name)
    last_bot_path = os.path.join(bot_path, 'last_run.json')

    # Ensure the bot directory exists
    os.makedirs(bot_path, exist_ok=True)

    # Create the WikiBot instance
    wikibot = WikiBot(
        name=bot_name,
        project_path=project_path,
        openai_api_key=openai_api_key,
        twitter_api_keys=twitter_api_keys,
        openrouter_api_key=openrouter_api_key,
        params=PARAMS
    )

    def seconds_until_post():
        """Compute seconds until next POST_HOUR in Paris time."""
        now = datetime.now(TIMEZONE)
        target = now.replace(hour=POST_HOUR, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    def already_posted_today():
        """Check if the bot already posted today (Paris time)."""
        if not os.path.exists(last_bot_path):
            return False
        with open(last_bot_path, 'r') as f:
            last_run = json.load(f)
        today = datetime.now(TIMEZONE).date()
        last_date = datetime(year=last_run['year'], month=last_run['month'], day=last_run['day']).date()
        return last_date >= today

    if DEBUG:
        # Debug mode: run immediately, up to 10 steps
        for i_step in range(15):
            print(f"[debug] Running step {i_step + 1}")
            try:
                wikibot.generate_and_publish(publish=PUBLISH, debug=DEBUG)
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Production: post immediately on first run, then daily at POST_HOUR
        first_run = False
        while True:
            if not first_run and already_posted_today():
                wait = seconds_until_post()
                print(f"Already posted today. Sleeping {wait/3600:.1f}h until {POST_HOUR}:00 Paris time.")
                time.sleep(wait)
                continue

            # Add a small random delay (0-30 min) so it doesn't post at exactly :00 every day
            if not first_run:
                jitter = np.random.randint(0, 1800)
            else:
                jitter = 0
            print(f"Posting in {jitter//60} minutes...")
            time.sleep(jitter)
            first_run = False

            try:
                print(f"Running WikiBot at {datetime.now(TIMEZONE)}")
                wikibot.generate_and_publish(publish=PUBLISH, debug=DEBUG)

                # Save today's date
                today = datetime.now(TIMEZONE)
                last_run = {'day': today.day, 'month': today.month, 'year': today.year}
                with open(last_bot_path, 'w') as f:
                    json.dump(last_run, f)

                print(f"Done. Next post tomorrow at ~{POST_HOUR}:00 Paris time.")
            except Exception as e:
                print(f"Error: {e}. Retrying in 15 minutes.")
                time.sleep(900)
                continue

            # Sleep until tomorrow's post time
            time.sleep(seconds_until_post())