"""
Utility functions for the WikiBot project.
"""
import re
import time
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from datetime import datetime


def clean_title(title):
    """Remove HTML tags and fix HTML entities in titles."""
    for c in ['<i>', '<b>', '</i>', '</b>']:
        title = title.replace(c, '')
    title = title.replace("&amp;", "&")
    return title


def slugify(text):
    """Convert text to a URL-friendly slug."""
    # Convert to lowercase
    text = text.lower()
    # Replace non-word characters with a hyphen
    text = re.sub(r'\W+', '_', text)
    # Trim hyphens from the start and end
    text = text.strip('-').strip('_')
    return text


def get_ordinal_suffix(day):
    """Return the ordinal suffix for a day number (1st, 2nd, 3rd, etc.)."""
    if 4 <= day <= 20 or 24 <= day <= 30:
        return "th"
    else:
        return ["st", "nd", "rd"][day % 10 - 1]


def get_formatted_date():
    """Get the current date formatted with ordinal suffix."""
    now = datetime.now()
    day = now.day
    month = now.strftime("%B")
    year = now.year
    formatted_date = f"{year}:{month}:{day}"
    return formatted_date

def download_image(url, img_path):
    """Download an image from a URL and return it as a numpy array."""
    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    i_attempt = 0
    img = None
    while i_attempt < 5:
        try:
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image_array = np.array(image)
                img = Image.fromarray(image_array)
                img.save(img_path)
                break
            else:
                raise ValueError(f"Failed to download image. Status code: {response.status_code}")
        except:
            i_attempt += 1
            time.sleep(np.random.randint(3, 10))
            pass
    if img is None:
        raise ValueError(f"Failed to download image (5 attempts). Status code: {response.status_code}")

def format_tweet(tweets, remove_url=False):
    s = f""
    for t in tweets:
        if 'learn more: http' in t and remove_url:
            t = t.split('learn more: http')[0]
        s += f'{t}\n***\n'
    s = s[:-5]
    return s
