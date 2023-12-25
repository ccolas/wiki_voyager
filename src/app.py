from datetime import datetime
import os
import pickle
from wikibot import WikiBot
import time

project_path = "/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Perso/Scratch/wikibot/"



if __name__ == '__main__':
    bot_name = 'bot1'
    bot_path = project_path + 'data/' + bot_name
    last_bot_path = bot_path + 'last_run.pkl'
    wikibot = WikiBot(bot_name)
    while True:
        if os.path.exists(last_bot_path):
            with open(last_bot_path, 'rb') as f:
                last_run = pickle.load(f)

        else:
            last_run = dict(day=1, month=1, year=1975)
        last_date = datetime(**last_run)

        now = datetime.now()
        if (now - last_date).days > 1 and now.hour > 12:
            wikibot.generate_and_publish()
            last_run = dict(day=now.day, month=now.month, year=now.year)
            with open(last_bot_path, 'wb') as f:
                pickle.dump(last_run, f)
            time.sleep(3600)










