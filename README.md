# WikiBot

A Twitter bot that explores Wikipedia through random walks, generates interesting content about each page, and publishes it to Twitter with AI-generated images.

## Project Structure

```
wikibot/
├── data                     # historic of random walks saved as jsonl files
└── src                 
    ├── prompts/             # prompts for content generation    
    ├── wikibot.py           # Main WikiBot class
    ├── clients.py           # API client setup (Twitter, OpenAI, Wikipedia)
    ├── content.py           # Content generation (tweets and images)
    ├── utils.py             # Utility functions
    ├── tweet_splitter.py    # Tweet text splitting
    └── run.py               # Script to run the bot
```

## Features

- Random exploration of Wikipedia pages
- AI-generated tweets summarizing page content
- AI-generated images based on page content
- Automatic publishing to Twitter
- Memory of visited pages and published content

## Requirements

- Python 3.8+
- OpenAI API key
- Twitter API keys (v1 & v2)
- Required Python packages (see below)

## Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -t requirements.txt
   ```
3. Create API key files:
    - `.api_openai` with your OpenAI API key
    - `.api_twitter` with your Twitter API keys (5 lines: API key, API secret, bearer token, access token, access token secret)

## Usage

Run the bot:

```
python src/run.py
```

By default, the bot will:
1. Find a new Wikipedia page
2. Generate tweets about the page
3. Generate an image for the page
4. Publish the content to Twitter
5. Store the information in memory
6. Wait for the next day to run again

## Configuration

You can modify the bot's behavior by changing the parameters in `wikibot.py`.

## License

MIT License