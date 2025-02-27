import os
from datetime import datetime
import re
import time
import json
import requests

from openai import OpenAI
import wikipediaapi
import wikipedia
from PIL import Image
from io import BytesIO
import numpy as np
import tiktoken
import tweepy

from tweet_splitter import split_tweets

project_path = "/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Perso/Scratch/wikibot/"
params = dict(unwanted_strings=['wiki','Wiki','Category', 'List', 'Template', 'Help', 'ISO', 'User','Talk','Portal'], text_model="gpt-4o-mini-2024-07-18", img_model="dall-e-3")
handle = "@wiki_voyager"
SKIP = ['tweet', 'publish', 'img']

with open(project_path + '.api_openai', 'r') as f:
    openai_api_key = f.read()

with open(project_path + '.api_twitter', 'r') as f:
    twitter_api_keys = f.read().split('\n')
# order: api, api secret, bearer, access, access secret

class WikiBot:
    def __init__(self, name, seed_page_id='Wikipedia', params=params):
        self.params = params
        self.bot_path = project_path + 'data/' + name + '/'
        self.img_path = self.bot_path + 'imgs/'
        self.memory_path = self.bot_path + 'memory.jsonl'
        os.makedirs(self.bot_path, exist_ok=True)
        os.makedirs(self.img_path, exist_ok=True)

        self.page_titles = []
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as file:
                # Read each line and parse it as JSON
                for line in file:
                    datapoint = json.loads(line.strip())
                    self.page_titles.append(datapoint['title'])

        self.setup_apis()

        self.last_page_id = None
        if len(self.page_titles) == 0:
            # get the seed page
            if seed_page_id is not None:
                page = self.wiki.page(seed_page_id)
                if page.exists():
                    self.last_page_id = seed_page_id

            if self.last_page_id is None:
                self.last_page_id = wikipedia.random(1)
                page = self.wiki.page(self.last_page_id)
                assert page.exists()
            last_page_title = self.last_page_id
        else:
            last_page_title = self.page_titles[-1]
        print(f'start random walk from {last_page_title}')

    def update_memory(self, this_id, title, page, date, tweet, img_info, tweet_url):
        new_mem = dict(title=title, url=page.fullurl, id=this_id, tweets=tweet, date=date, img_info=img_info, tweet_url=tweet_url)
        self.page_titles.append(title)
        with open(self.memory_path, 'a') as file:
            json_str = json.dumps(new_mem)
            file.write(json_str + "\n")

    def generate_and_publish(self):
        new_date = self.get_date()
        print('  find wiki page', end='\r')
        new_title, new_page = self.get_next_page()
        this_id = self.get_id(new_title)
        print(f'  step #{len(self.page_titles)}: {new_title}')
        print('    generate image', end='\r')
        new_img_info = self.generate_img(new_title, new_page, this_id)
        print('    generate tweet', end='\r')
        new_tweets = self.format_tweet(new_title, new_page)
        print('    publish tweet', end='\r')
        tweet_url = self.publish(new_tweets, new_img_info)
        self.update_memory(this_id, new_title, new_page, new_date, new_tweets, new_img_info, tweet_url)
        stop = 1

    def generate_img(self, new_title, new_page, this_id):
        if 'img' in SKIP:
            return None
        try:
            summary = self.crop_prompt(new_page.summary, n_tokens=2000)
    
            new_title = clean_title(new_title)
    
            # get a prompt
            system_prompt = """You are a creative AI assistant who takes a wikipedia page title and the beginning of the page and returns a concise prompt for an AI image generator to illustrate the concept or topic.
            
    Guidelines:
    - The prompt should be short (two sentences max)
    - The resulting generated image should be creative and appealing while capturing essential components of the wikipedia concept such that we can recognize it from the picture
    - If the topic is sensitive, make sure to craft a prompt that will not be censored, maybe try to draw something related without using the bad keywords"""
            prompt = f"The title is: {new_title}. The beginning of the page is:\n{summary}\n\nPrompt for an AI image generator:"
            messages = [{"role": "system",
                         "content": system_prompt},
                        {"role": "user",
                         "content": prompt}]
            image_prompt = self.call_text_model(messages, self.params['text_model'], max_tokens=200)
            print('    img prompt generated')
            image_url = self.call_image_model(image_prompt, self.params['img_model'])
            print('    image generated')
            image = download_image(image_url)
            img_path = self.img_path + f'{this_id}.png'
            img = Image.fromarray(image)
            img.save(img_path)
            img_info = dict(prompt=image_prompt, path=img_path)
            return img_info
        except Exception as err:
            print(f'    error in img generation: {str(err)}')
            return None
            
    def format_tweet(self, new_title, new_page, tweet_limit=280):
        if 'tweet' in SKIP:
            return None
        text = self.crop_prompt(new_page.text, n_tokens=2000)

        new_title = clean_title(new_title)

        # get a prompt
        system_prompt = f"""You are a creative AI assistant who takes a wikipedia page title and the beginning of its text and writes a short thread of 3-5 engaging tweets on the topic to inform people in a fun way.

Guidelines:
- Tweets should be engaging, but in a cool way: don't use too much emoji, don't overdo the hashtags
- Be critical, be subjective, be edgy (but not rude)
- Highlight interesting facts instead of boilerplates, engage people!
- Start your tweet by mentioning the topic of the day.

Formatting Rules:
- Each tweet should be maximum 280 characters long
- Tweets must be separated by markdown separators \\n***\\n
- Do not indicate tweet numbers. 
- Write without capitalizing words, including at the beginning of sentences, but capitalize them when they are acronyms."""
        prompt = f"Today's title is: {new_title}. The text is:\n{text}"
        messages = [{"role": "system",
                     "content": system_prompt},
                    {"role": "user",
                     "content": prompt}]
        text = self.call_text_model(messages, self.params['text_model'], max_tokens=500)
        text = text + f'\nlearn more: {new_page.fullurl}'
        tweets = split_tweets(text)
        for t in tweets:
            assert len(t) < tweet_limit
        print(f'    generated {len(tweets)} tweets')
        return tweets

    def publish(self, tweets, new_img_info):
        if 'publish' in SKIP:
            return None
        tweets = tweets[:10]

        if new_img_info is not None:
            media = self.twitter_client_v1.media_upload(filename=new_img_info['path'])
            media_id = media.media_id
        else:
            media_id = None
        previous_tweet_id = None
        tweet_url = None
        for tweet in tweets:
            if previous_tweet_id is not None:
                new_tweet = self.twitter_client_v2.create_tweet(text=tweet, in_reply_to_tweet_id=previous_tweet_id)
            else:
                if media_id is not None:
                    new_tweet = self.twitter_client_v2.create_tweet(text=tweet, media_ids=[media_id])
                else:
                    new_tweet = self.twitter_client_v2.create_tweet(text=tweet)
            previous_tweet_id = new_tweet.data['id']
            if tweet_url is None:
                tweet_url = f"https://x.com/wiki_voyager/status/{previous_tweet_id}"
        return tweet_url

    def get_date(self):
        # Get the current date
        now = datetime.now()

        # Extract the day, month, and year
        day = now.day
        month = now.strftime("%B")
        year = now.year

        # Format the date with the ordinal suffix
        formatted_date = f"{month} {day}{get_ordinal_suffix(day)} {year}"
        return formatted_date

    def call_text_model(self, messages, model="gpt-4o-mini-2024-07-18", max_tokens=1000):
        i_attempt = 0
        response = None
        error = ""
        while i_attempt < 5:
            i_attempt += 1
            try:
                response = self.openai_client.chat.completions.create(model=model,
                                                                      messages=messages,
                                                                      max_tokens=max_tokens)
                break
            except Exception as err:
                error = str(err)
                print(f'API error: {str(err)}')
                time.sleep(np.random.randint(5, 60))
                
        if response is None:
            assert False, f"error in text model call: ERR: {error}"
        return response.choices[0].message.content

    def call_image_model(self, prompt, model='dall-e-3', size="1024x1024", quality='standard'):  # 4 cts
        i_attempt = 0
        response = None
        error = ""
        while i_attempt < 5:
            i_attempt += 1
            try:
                response = self.openai_client.images.generate(model=model,
                                                              prompt=prompt,
                                                              n=1,
                                                              size=size,
                                                              quality=quality)
                break
            except Exception as err:
                error = str(err)
                print(f'API error: {str(err)}')
                time.sleep(np.random.randint(5, 60))

        if response is None:
            assert False, f"error in img model call: ERR: {error}"
        return response.data[0].url

    def get_next_page(self):
        last_page_index = len(self.page_titles) - 1
        new_page = None
        while last_page_index >= -1:
            while True:
                try:
                    success, new_page = self.get_new_page(last_page_index)
                    break
                except:
                    time.sleep(2)#np.random.randint(10, 3*60))
                    print('    error in loading new page')
            if success:
                break
            else:
                # go back one page to find valid page ids
                print('      no new page here, going back one page')
                last_page_index -= 1
        assert new_page is not None, "no new page found (weird)"
        return new_page.displaytitle, new_page

    def get_new_page(self, last_page_index):
        if last_page_index == -1:
            last_page_id = self.last_page_id
        else:
            last_page_id = self.page_titles[last_page_index]
        if last_page_id is None:
            assert False, 'could not find new valid page ids'
        last_page = self.wiki.page(last_page_id)
        links = last_page.links
        list_keys = sorted(links.keys())
        np.random.shuffle(list_keys)
        n_keys = len(list_keys)
        success = False
        candidate_page = None
        for k in range(n_keys):
            # check for unwanted pages
            reject = False  # reject current keyword if True
            candidate_keyword = list_keys[k]
            candidate_page = links[list_keys[k]]
            if candidate_page.exists():
                # check that it has not been selected yet
                if candidate_keyword in self.page_titles:
                    continue

                # check that it does not contain unwanted substring
                for strg in self.params['unwanted_strings']:
                    if strg in candidate_keyword:
                        reject = True
                        break
                if reject:
                    continue

                # check whether next page contains links
                if len(sorted(candidate_page.links.keys())) < 1:
                    continue

                # page seems valid
                success = True
                break

        return success, candidate_page

    def crop_prompt(self, prompt, n_tokens):
        tokens = self.encoder.encode(prompt)
        cropped_text = self.encoder.decode(tokens[:n_tokens])
        if len(tokens) > n_tokens:
            cropped_text += ' [...]'
        return cropped_text
    
    def setup_apis(self):
        self.wiki = wikipediaapi.Wikipedia(language='en',
                                           user_agent='my_app/0.1 (my_email@example.com)'
                                           )

        # setup api v1 to upload media
        auth = tweepy.OAuth1UserHandler(twitter_api_keys[0], twitter_api_keys[1], twitter_api_keys[3], twitter_api_keys[4])
        self.twitter_client_v1 = tweepy.API(auth)

        # setup api v2 to tweet
        self.twitter_client_v2 = tweepy.Client(consumer_key=twitter_api_keys[0],
                                               consumer_secret=twitter_api_keys[1],
                                               access_token=twitter_api_keys[3],
                                               access_token_secret=twitter_api_keys[4])
        self.openai_client = OpenAI(api_key=openai_api_key, timeout=50.0)
        self.encoder = tiktoken.get_encoding("o200k_base")

    def get_id(self, title):
        n = len(self.page_titles)
        slug_title = slugify(title)
        return f"exploration_{n}_{slug_title}"


# utils
def clean_title(new_title):
    for c in ['<i>', '<b>', '</i>', '</b>']:
        new_title = new_title.replace(c, '')
    new_title = new_title.replace("&amp;", "&")
    return new_title

def slugify(text):
    # Convert to lowercase
    text = text.lower()
    # Replace non-word (word here means [a-zA-Z0-9_]) characters with a hyphen
    text = re.sub(r'\W+', '-', text)
    # Trim hyphens from the start and end
    text = text.strip('-')
    return text

def get_ordinal_suffix(day):
    if 4 <= day <= 20 or 24 <= day <= 30:
        return "th"
    else:
        return ["st", "nd", "rd"][day % 10 - 1]

def download_image(url):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image_array = np.array(image)
        print('    image downloaded')
        return image_array
    else:
        assert False, "Failed to download image. Status code: {response.status_code}"



