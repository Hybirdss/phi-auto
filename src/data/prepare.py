"""
Data preparation for phi-auto.
Downloads TinyStories dataset and trains tokenizer.
"""

import os
import json
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.engine.tokenizer import ByteBPETokenizer

CACHE_DIR = os.path.expanduser("~/.cache/phi-auto")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_PATH = os.path.join(CACHE_DIR, "tokenizer.json")


def download_tinystories(max_stories=50000):
    """Download TinyStories from HuggingFace."""
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "train.jsonl")
    val_path = os.path.join(DATA_DIR, "val.jsonl")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Data already exists at {DATA_DIR}")
        return train_path, val_path

    print("Downloading TinyStories dataset...")
    try:
        import urllib.request
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
        txt_path = os.path.join(DATA_DIR, "raw.txt")

        if not os.path.exists(txt_path):
            print(f"  Fetching from {url}")
            urllib.request.urlretrieve(url, txt_path)
            print(f"  Downloaded to {txt_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Generating synthetic data instead...")
        return generate_synthetic_data(max_stories)

    # parse stories
    print("Parsing stories...")
    stories = []
    current = []
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip() == '<|endoftext|>':
                if current:
                    story = ' '.join(current).strip()
                    if 50 < len(story) < 2000:  # filter very short/long
                        stories.append(story)
                    current = []
                    if len(stories) >= max_stories:
                        break
            else:
                current.append(line.strip())

    if not stories:
        print("No stories found, generating synthetic data...")
        return generate_synthetic_data(max_stories)

    random.shuffle(stories)
    split = int(len(stories) * 0.95)
    train_stories = stories[:split]
    val_stories = stories[split:]

    # save as jsonl
    for path, data in [(train_path, train_stories), (val_path, val_stories)]:
        with open(path, 'w') as f:
            for story in data:
                f.write(json.dumps({"text": story}) + "\n")

    print(f"Data: {len(train_stories)} train, {len(val_stories)} val")
    return train_path, val_path


def generate_synthetic_data(n=10000):
    """Generate simple synthetic stories as fallback."""
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "train.jsonl")
    val_path = os.path.join(DATA_DIR, "val.jsonl")

    names = ["Lily", "Max", "Sam", "Emma", "Tom", "Mia", "Ben", "Zoe", "Jack", "Anna"]
    animals = ["dog", "cat", "bird", "fish", "bunny", "frog", "duck", "bear"]
    places = ["garden", "park", "forest", "beach", "house", "school", "lake", "hill"]
    actions = ["played", "ran", "jumped", "danced", "laughed", "smiled", "sang", "swam"]
    objects = ["ball", "flower", "toy", "book", "hat", "kite", "cake", "star"]
    feelings = ["happy", "excited", "brave", "kind", "gentle", "proud", "glad", "warm"]

    stories = []
    for _ in range(n):
        name = random.choice(names)
        animal = random.choice(animals)
        place = random.choice(places)
        action = random.choice(actions)
        obj = random.choice(objects)
        feeling = random.choice(feelings)

        templates = [
            f"Once upon a time, there was a little child named {name}. {name} had a {feeling} {animal}. They {action} in the {place} every day. One day, {name} found a {obj}. It made them very {feeling}. The end.",
            f"{name} loved to play with a {obj} in the {place}. One sunny day, {name} saw a {animal}. The {animal} {action} around {name}. They became best friends. {name} was so {feeling}.",
            f"There was a {feeling} {animal} who lived near the {place}. A child named {name} came to visit. {name} {action} with the {animal}. They shared a {obj}. It was a {feeling} day.",
            f"In a big {place}, {name} found a little {animal}. The {animal} was looking for a {obj}. {name} helped the {animal} find it. The {animal} was {feeling}. {name} {action} with joy.",
        ]
        stories.append(random.choice(templates))

    random.shuffle(stories)
    split = int(len(stories) * 0.95)

    for path, data in [(train_path, stories[:split]), (val_path, stories[split:])]:
        with open(path, 'w') as f:
            for story in data:
                f.write(json.dumps({"text": story}) + "\n")

    print(f"Synthetic data: {split} train, {len(stories) - split} val")
    return train_path, val_path


def train_tokenizer(train_path, vocab_size=1024, sample_size=5000):
    """Train BPE tokenizer on training data."""
    if os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer already exists at {TOKENIZER_PATH}")
        return ByteBPETokenizer.load(TOKENIZER_PATH)

    print(f"Training tokenizer (vocab_size={vocab_size})...")
    texts = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            texts.append(json.loads(line)["text"])

    tok = ByteBPETokenizer(vocab_size=vocab_size)
    tok.train(texts, verbose=True)
    tok.save(TOKENIZER_PATH)
    print(f"Tokenizer saved to {TOKENIZER_PATH}")
    return tok


def prepare_all(vocab_size=1024, max_stories=50000):
    """Full data preparation pipeline."""
    train_path, val_path = download_tinystories(max_stories)
    tokenizer = train_tokenizer(train_path, vocab_size)
    return tokenizer, train_path, val_path


if __name__ == "__main__":
    tokenizer, train_path, val_path = prepare_all()
    # quick test
    sample = json.loads(open(train_path).readline())["text"]
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    print(f"\nSample: {sample[:80]}...")
    print(f"Tokens: {len(ids)}")
    print(f"Decoded: {decoded[:80]}...")
