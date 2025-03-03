# tasks/save_articles.py
import json
import os

def save_articles(articles, output_file):
    """Save extracted articles to a JSON file."""
    existing_articles = []

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as file:
                existing_articles = json.load(file)
        except json.JSONDecodeError:
            print("Warning: JSON file is empty or corrupted.")
            return

    all_articles = existing_articles + articles
    unique_articles = { (article['title'], article['link']): article for article in all_articles }
    unique_articles = list(unique_articles.values())
    unique_articles.sort(key=lambda x: x['title'])

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(unique_articles, file, indent=4, ensure_ascii=False)
