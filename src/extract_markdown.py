#!/usr/bin/env python
"""
Script to convert WikiBot memory.jsonl to a markdown summary file.
"""
import os
import json
import argparse
import shutil
from datetime import datetime


def load_memories(dir):
    """Load memories from a JSONL file."""
    memories = []
    jsonl_path = dir + 'memory.jsonl'
    with open(jsonl_path, 'r') as file:
        for line in file:
            try:
                memory = json.loads(line.strip())
                memories.append(memory)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return memories


def format_tweets(tweets):
    """Format tweets for markdown display."""
    if not tweets:
        return "*No tweets available*"

    formatted = ""
    for i, tweet in enumerate(tweets):
        formatted += f"**Tweet {i + 1}:**\n"
        formatted += f"{tweet}\n\n"

    return formatted.strip()


def copy_image(src_path, dest_dir, filename):
    """Copy an image to the destination directory."""
    if not os.path.exists(src_path):
        return None

    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    try:
        shutil.copy2(src_path, dest_path)
        return filename
    except Exception as e:
        print(f"Error copying image: {e}")
        return None


def generate_markdown(dir):
    """Generate a markdown file from memories."""

    memories = load_memories(dir)

    if not memories:
        print("No memories found!")
        return

    # Create markdown file
    markdown_path = os.path.join(dir, "exploration_journey.md")

    with open(markdown_path, 'w') as md_file:
        # Write header
        md_file.write("# WikiBot Exploration Journey\n\n")
        md_file.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d')}*\n\n")
        md_file.write("---\n\n")

        # Write each memory
        for i, memory in enumerate(memories):
            title = memory.get('title', 'Unknown Page')
            date = memory.get('date', 'Unknown Date')
            url = memory.get('url', '#')
            tweets = memory.get('tweets', [])
            img_info = memory.get('img_info', {})
            tweet_url = memory.get('tweet_url', None)

            # Write memory header
            md_file.write(f"## {date}: [{title}]({url})\n\n")

            # Add tweet URL if available
            if tweet_url:
                md_file.write(f"[View on Twitter]({tweet_url})\n\n")

            # Format and write tweets
            md_file.write("### Tweets\n\n")
            md_file.write(f"{format_tweets(tweets)}\n\n")

            # Handle image
            if img_info and 'path' in img_info and 'prompt' in img_info:
                md_file.write("### Image\n\n")
                md_file.write(f"**Prompt:** {img_info['prompt']}\n\n")

                # Copy image and add to markdown
                img_src_path = img_info.get('path')
                if img_src_path and os.path.exists(img_src_path):
                    img_filename = img_src_path.split('/')[-1]
                    relative_img_path = os.path.join("imgs", img_filename).replace("\\", "/")
                    md_file.write(f"![{title}]({relative_img_path})\n\n")
                else:
                    md_file.write("*Image not available*\n\n")

            # Add separator between memories
            md_file.write("---\n\n")

    print(f"Markdown file generated: {markdown_path}")
    return markdown_path


def main():
    parser = argparse.ArgumentParser(description='Convert WikiBot memory.jsonl to markdown')
    parser.add_argument('--memory', required=True, help='Path to memory.jsonl file')
    parser.add_argument('--output', required=True, help='Output directory for markdown and images')

    args = parser.parse_args()




if __name__ == "__main__":
    dir = "/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Perso/Scratch/wikibot/data/trail_1/"
    # Generate markdown
    generate_markdown(dir)