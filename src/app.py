from datetime import datetime
import os
import pickle
import time

import numpy as np

from wikibot import WikiBot


script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(script_path)

if __name__ == '__main__':
    bot_name = 'bot2'
    bot_path = project_path + '/data/' + bot_name + '/'
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
            time.sleep(np.random.randint(3600, 9000))










