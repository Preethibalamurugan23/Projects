import time
import logging
import datetime
import traceback
import sys

class TimestampedLogger:
    def __init__(self, stream):
        self.stream = stream

    def write(self, message):
        # Only add timestamp if the message is not empty (like newlines)
        if message.strip():
            self.stream.write(f"[{datetime.datetime.now()}] {message}")
        else:
            self.stream.write(message)

    def flush(self):
        self.stream.flush()

# Redirect standard output and error to a file with timestamps
detailed_log_file = open('detailed_log.log', 'a')
sys.stdout = TimestampedLogger(detailed_log_file)
sys.stderr = TimestampedLogger(detailed_log_file)

logging.basicConfig(
    filename='C:/Users/Preethi/Downloads/assgn 1/orchestration_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Assuming the imported functions from previous code
from google_news_scraper import get_google_news_homepage, parse_google_news, save_articles, download_thumbnail, get_actual_url
from pymongo import MongoClient
import json


# Function to log the start of each task
def log_start(task_name):
    logging.info(f"Starting task: {task_name}")


# Function to log the completion of each task
def log_end(task_name):
    logging.info(f"Completed task: {task_name}")


# Function to log any error encountered
def log_error(task_name, error):
    logging.error(f"Error in task {task_name}: {error}")
    logging.error(f"Traceback: {traceback.format_exc()}")


def load_config(config_file):
    """Load configuration from a JSON file."""
    try:
        with open(config_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        log_error('Load Config', e)
        raise


def get_and_parse_articles(config):
    try:
        log_start('Get Google News Homepage')
        html = get_google_news_homepage(config.get("google_news_url", "https://news.google.com/"))
        log_end('Get Google News Homepage')

        if not html:
            raise Exception("Failed to fetch Google News homepage.")

        log_start('Parse Google News Articles')
        articles = parse_google_news(html)
        log_end('Parse Google News Articles')

        return articles
    except Exception as e:
        log_error('Get and Parse Articles', e)
        return []


def save_article_data(articles, config):
    try:
        log_start('Save Articles')
        save_articles(articles, config.get("output_file", "news_articles.json"))
        log_end('Save Articles')
    except Exception as e:
        log_error('Save Articles', e)


def main():
    # Load config file
    try:
        config = load_config("config.json")
    except Exception as e:
        log_error('Main', e)
        return

    # Step 1: Get and parse Google News articles
    articles = get_and_parse_articles(config)
    if not articles:
        logging.warning("No articles to process.")
        return

    # Step 2: Save articles to file
    save_article_data(articles, config)

    # Additional logic could be added to process articles, store them in a database, etc.


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    logging.info(f"Orchestration script started at {start_time}")

    try:
        main()
    except Exception as e:
        log_error("Main Execution", e)

    end_time = datetime.datetime.now()
    logging.info(f"Orchestration script ended at {end_time}")
