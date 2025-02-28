"""
Run script for WikiBot - executes the bot on a daily schedule.
"""
from datetime import datetime
import os
import json
import time
import numpy as np

from wikibot import WikiBot


def read_api_keys(project_path):
    """Read API keys from files."""
    with open(os.path.join(project_path, '.api_openai'), 'r') as f:
        openai_api_key = f.read().strip()

    with open(os.path.join(project_path, '.api_twitter'), 'r') as f:
        twitter_api_keys = f.read().strip().split('\n')

    return openai_api_key, twitter_api_keys

PARAMS = {
    'unwanted_strings': ['wiki', 'Wiki', 'Category', 'List', 'Template',
                         'Help', 'ISO', 'User', 'Talk', 'Portal', '501'],
    'text_model': "gpt-4o-mini-2024-07-18",
    'img_model': "dall-e-3",
    'to_skip': ['publish'],
    'context_length': 10,
    'n_link_options': 19
}

if __name__ == '__main__':
    # Set up paths
    script_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(script_path) + '/'

    # Read API keys
    openai_api_key, twitter_api_keys = read_api_keys(project_path)

    # Bot configuration
    bot_name = 'trail_1'
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
        params=PARAMS
    )

    # Main loop
    i_attempt = 0
    while True:
        i_attempt += 1
        if i_attempt > 10:
            break
        # Check last run date
        if os.path.exists(last_bot_path):
            with open(last_bot_path, 'r') as f:
                last_run = json.load(f)
            last_date = datetime(year=last_run['year'], month=last_run['month'], day=last_run['day'])
        else:
            last_date = datetime(year=1975, month=1, day=1)

        now = datetime.now()

        # For testing purposes, remove the comment to run regardless of time
        # In production, uncomment the time check
        if True:  # (now - last_date).days > 1 and now.hour > 12:
            print(f"Running WikiBot at {now}")

            try:
                wikibot.generate_and_publish()

                # Update last run time
                last_run = {'day': now.day, 'month': now.month, 'year': now.year}

                with open(last_bot_path, 'w') as f:
                    json.dump(last_run, f)

                # print(f"WikiBot run completed successfully at {datetime.now()}")

                # Wait between 1-2.5 hours before checking again
                # time.sleep(np.random.randint(3600, 9000))

            except Exception as e:
                print(f"Error running WikiBot: {str(e)}")
                # Wait 15 minutes before trying again on error
                time.sleep(900)
        else:
            # Check again in an hour
            time.sleep(3600)