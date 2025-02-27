def split_tweets(text, max_length=280):
    """
    Split a string of tweets (separated by '***') to ensure each is under max_length
    while maintaining sentence integrity when possible.
    """
    tweets = text.split('***')
    result = []

    for tweet in tweets:
        tweet = tweet.strip()
        if len(tweet) <= max_length:
            result.append(tweet)
        else:
            result.extend(split_long_tweet(tweet, max_length))

    return result


def split_long_tweet(tweet, max_length):
    """Split a tweet that exceeds max_length into multiple tweets."""
    if len(tweet) <= max_length:
        return [tweet]

    # Strict priority-based split approach
    # Try each level of priority before moving to the next

    # 1. First try paragraph breaks (\n\n)
    paragraph_splits = find_split_positions(tweet, '\n\n', True, max_length)
    if paragraph_splits:
        split_pos = choose_best_split(tweet, paragraph_splits, max_length)
        first_segment = tweet[:split_pos].strip()
        remaining = tweet[split_pos:].strip()
        return [first_segment] + split_long_tweet(remaining, max_length)

    # 2. Then try line breaks (\n)
    line_splits = find_split_positions(tweet, '\n', True, max_length)
    if line_splits:
        split_pos = choose_best_split(tweet, line_splits, max_length)
        first_segment = tweet[:split_pos].strip()
        remaining = tweet[split_pos:].strip()
        return [first_segment] + split_long_tweet(remaining, max_length)

    # 3. Then try sentence endings
    for marker in ['. ', '! ', '? ']:
        sentence_splits = find_split_positions(tweet, marker, True, max_length)
        if sentence_splits:
            split_pos = choose_best_split(tweet, sentence_splits, max_length)
            first_segment = tweet[:split_pos].strip()
            remaining = tweet[split_pos:].strip()
            return [first_segment] + split_long_tweet(remaining, max_length)

    # 4. Then try clause breaks
    for marker in [', ', '; ', ': ']:
        clause_splits = find_split_positions(tweet, marker, False, max_length)
        if clause_splits:
            split_pos = choose_best_split(tweet, clause_splits, max_length)
            first_segment = tweet[:split_pos].strip()
            remaining = tweet[split_pos:].strip()
            return [first_segment] + split_long_tweet(remaining, max_length)

    # 5. Then try spaces
    space_splits = find_split_positions(tweet, ' ', False, max_length)
    if space_splits:
        split_pos = choose_best_split(tweet, space_splits, max_length)
        first_segment = tweet[:split_pos].strip()
        remaining = tweet[split_pos:].strip()
        return [first_segment] + split_long_tweet(remaining, max_length)

    # Last resort: character-level split
    return character_level_split(tweet, max_length)


def find_split_positions(tweet, marker, is_sentence_boundary, max_length):
    """Find all valid split positions for the given marker."""
    positions = find_all(tweet, marker)
    valid_positions = []

    for pos in positions:
        split_pos = pos + len(marker)
        if split_pos <= max_length and is_valid_split(tweet, split_pos, is_sentence_boundary):
            valid_positions.append(split_pos)

    return valid_positions


def choose_best_split(tweet, positions, max_length):
    """Choose the best split position from the valid ones."""
    # Target length is 70% of max_length for better balance
    target_length = max_length * 0.7

    best_pos = positions[0]
    best_score = abs(len(tweet[:positions[0]]) - target_length)

    for pos in positions[1:]:
        score = abs(len(tweet[:pos]) - target_length)
        if score < best_score:
            best_score = score
            best_pos = pos

    return best_pos


def find_all(text, substring):
    """Find all occurrences of substring in text."""
    positions = []
    start = 0

    while True:
        start = text.find(substring, start)
        if start == -1:
            break
        positions.append(start)
        start += 1

    return positions


def is_valid_split(tweet, split_pos, is_sentence_boundary):
    """
    Check if splitting at position creates valid segments.
    If is_sentence_boundary is True, no further checks needed as we're splitting at a sentence boundary.
    Otherwise, ensure remaining text contains at least one sentence.
    """
    if is_sentence_boundary:
        return True

    remaining = tweet[split_pos:].strip()
    # Check if remaining contains a sentence boundary
    sentence_markers = ['. ', '! ', '? ', '\n\n', '\n']
    for marker in sentence_markers:
        if marker in remaining:
            return True

    return False


def character_level_split(tweet, max_length):
    """
    Last resort: Split at character level, prioritizing commas and then
    spaces to maintain readability.
    """
    result = []

    while tweet:
        if len(tweet) <= max_length:
            result.append(tweet)
            break

        # First try to split at a comma within reasonable range
        comma_pos = tweet[:max_length].rfind(', ')
        if comma_pos > max_length // 2:
            split_pos = comma_pos + 2  # Include the comma and space
        else:
            # Try to split at a space near the limit
            space_pos = tweet[:max_length].rfind(' ')
            if space_pos > max_length // 2:
                split_pos = space_pos + 1  # Include the space
            else:
                # Forced character split at maximum position
                split_pos = max_length

        result.append(tweet[:split_pos].strip())
        tweet = tweet[split_pos:].strip()

    return result


if __name__ == "__main__":
    # Example usage
    sample_text = """This is tweet one that's under the limit.
    ***
    This is a very long tweet that needs to be split because it exceeds the character limit. It has multiple sentences. It goes on for quite a while. It goes on for quite a while. It goes on for quite a while. It goes on for quite a while. It goes on for quite a while. We need to ensure it's split properly at sentence boundaries.
    ***
    Short tweet here.
    ***
    Another tweet with multiple paragraphs.

    This is the second paragraph.

    And this is a third one that makes the tweet too long for the 280 character limit. And this is a third one that makes the tweet too long for the 280 character limit. And this is a third one that makes the tweet too long for the 280 character limit.

    *** 
    now a tweet with a super long sentence that doesn't end, now a tweet with a super long sentence that doesn't end, now a tweet with a super long sentence that doesn't end, now a tweet with a super long sentence that doesn't end, now a tweet with a super long sentence that doesn't end, now a tweet with a super long sentence that doesn't end,"""

    split_results = split_tweets(sample_text)

    print("Split Tweets:")
    for i, tweet in enumerate(split_results):
        print(f"Tweet {i + 1} ({len(tweet)} chars):")
        print(tweet)
        print("-" * 40)
