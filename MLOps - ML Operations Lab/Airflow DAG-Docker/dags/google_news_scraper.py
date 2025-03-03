import requests
import json
import argparse
from bs4 import BeautifulSoup
import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
from urllib.parse import urljoin
import time
import os
from urllib.parse import urljoin
import datetime
from bson import ObjectId


def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_google_news_homepage(url):
    """Fetch the Google News homepage."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch the page. Status code: {response.status_code}")
        return None

def get_actual_url(google_news_url):
    print(google_news_url)
    """Follow Google News redirection using Selenium headless mode to get the actual news website URL."""
    try:
        # Set up headless Chrome options
        chrome_options = Options()
        chrome_options.add_argument("user-agent=Your_Custom_User_Agent")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        # Initialize Selenium WebDriver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        # Navigate to the URL
        driver.get(google_news_url)
        
        # Introduce random sleep to mimic human behavior
        time.sleep(random.uniform(2, 5))
        
        try:
            # Wait for the URL to change with a timeout of 15 seconds
            WebDriverWait(driver, 15).until(
                lambda d: d.current_url != google_news_url
            )
            # Get the redirected URL
            redirected_url = driver.current_url
        except TimeoutException:
            print("Redirection took too long. Returning original URL.")
            redirected_url = google_news_url
        
        # Quit the driver
        driver.quit()

        # Return the final redirected URL
        print(redirected_url)
        return redirected_url
        
    except Exception as e:
        print(f"Error with Selenium: {e}")
        return google_news_url  # Fallback to original URL if Selenium fails

def download_thumbnail(thumbnail_url, article_id):
    """Download the thumbnail image for a given article using the article_id for unique naming."""
    if not thumbnail_url:
        return None  # No thumbnail URL

    try:
        response = requests.get(thumbnail_url, stream=True)
        if response.status_code == 200:
            # Check if the response is an image
            content_type = response.headers.get('Content-Type')
            if content_type and content_type.startswith('image'):
                # Use jpg as default extension for simplicity
                image_filename = f"thumbnail_{article_id}.jpg"
                image_path = os.path.join("thumbnails", image_filename)

                # Create the folder if it doesn't exist
                if not os.path.exists("thumbnails"):
                    os.makedirs("thumbnails")

                with open(image_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)

                print(f"[INFO] Downloaded thumbnail: {image_filename}")
                return image_path
            else:
                print(f"[WARNING] URL did not return an image: {thumbnail_url}")
                return None
        else:
            print(f"[ERROR] Failed to download image: {thumbnail_url} (Status: {response.status_code})")
            return None
        
    except Exception as e:
        print(f"[ERROR] Exception occurred while downloading image: {e}")
        return None

# # MongoDB connection setup
# client = MongoClient('mongodb://localhost:27017/') ----------------------------
# db = client['google_news']
# collection = db['news_articles']

# def get_next_article_id():
#     """Fetch the next article_id by continuing from the last ID in the database."""
#     last_article = collection.find_one(sort=[("article_id", -1)])
#     return last_article["article_id"] + 1 if last_article else 1

import json
import os

def get_next_article_id(file_path='news_articles.json'):
    """Fetch the next article_id by continuing from the last ID in the JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            if data:  # Check if data is not empty
                last_article = max(data, key=lambda x: x['article_id'])
                return last_article["article_id"] + 1
    return 1  # Start from 1 if file doesn't exist or is empty


import json
import os

def get_next_article_id_from_json(file_path='news_articles.json'):
    """Fetch the next article_id by continuing from the last ID in the JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            if data:  # Check if data is not empty
                last_article = max(data, key=lambda x: x['article_id'])
                return last_article["article_id"] + 1
    return 1  # Start from 1 if file doesn't exist or is empty

# Function to check if the title is unique using TF-IDF similarity
def is_title_unique(new_title, threshold=0.7):
    # Get all existing titles from the database -------- collection as params
    # existing_titles = [article['title'] for article in collection.find({}, {'title': 1})]
    existing_titles = [article['title'] for article in json.load(open('news_articles.json'))] if os.path.exists('news_articles.json') else []

    # If no existing titles, it's unique
    if not existing_titles:
        return True

    # Append the new title to the list for comparison
    titles = existing_titles + [new_title]
    
    # Vectorize the titles using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(titles)
    
    # Calculate cosine similarity between the new title and all existing ones
    similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    
    # Check if the maximum similarity exceeds the threshold
    max_similarity = similarity_matrix.max()
    if max_similarity > threshold:
        print(f"Duplicate title detected with similarity: {max_similarity}")
        return False  # Duplicate found
    
    return True  # Unique title

def parse_google_news(html):
    """Parse Google News homepage to extract headlines, URLs, and metadata."""
    soup = BeautifulSoup(html, "html.parser")
    articles = []

    article_counter = get_next_article_id()  # Initialize counter from the last ID in the DB

    for item in soup.select("a.gPFEn"):  # Targeting article links
        title = item.get_text(strip=True)
        relative_link = item.get("href")

        if relative_link.startswith("./"):
            google_news_url = urljoin("https://news.google.com", relative_link)
            # actual_link = (google_news_url)
            actual_link = get_actual_url(google_news_url)


            # Extract metadata
            author = "Unknown"
            source_publisher = "Unknown"
            category = "Unknown"
            time_posted = "Unknown"

            # Extracting author
            author_tag = item.find_previous("div", class_="bInasb")
            if author_tag:
                author = author_tag.find("span").get_text(strip=True).replace("By ", "")

            # Extracting source/publisher
            publisher_tag = item.find_previous("div", class_="vr1PYe")
            if publisher_tag:
                source_publisher = publisher_tag.get_text(strip=True)

            # Extracting category
            category_tag = item.find_previous("span", class_="CUjhod")
            if category_tag:
                category = category_tag.get_text(strip=True)

            # Extracting time posted
            time_tag = item.find_previous("time")
            if time_tag and time_tag.has_attr("datetime"):
                time_posted = time_tag["datetime"]

            # Extracting the image URL for the article
            image_url = None
            image_tag = item.find_previous("img")
            if image_tag:
                image_url = image_tag.get("src")

            # Download the thumbnail using the unique article_id
            thumbnail_path = download_thumbnail(image_url, article_counter)

            # Store article data with metadata
            article_data = {
                "article_id": article_counter,  
                "title": title,
                "link": actual_link,
                "author": author,
                "source_publisher": source_publisher,
                "category": category,
                "time_posted": time_posted,
                "scraped_timestamp": datetime.datetime.now().isoformat(),  # Add scraping time
                "thumbnail_url": thumbnail_path
            }

            # Check for duplicate titles before adding to database
            if is_title_unique(title):      
                # collection.insert_one(article_data)  # Store in MongoDB  -------------------- collection as input params
                articles.append(article_data)
                print(f"New article added with ID: {article_counter}")
                article_counter += 1  # Increment the counter for the next article
            else:
                print(f"Duplicate title found. Skipping article: {title}")

    return articles

class JSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle MongoDB ObjectId serialization."""
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)  # Convert ObjectId to string
        return super().default(o)

def save_articles(articles, output_file):
    """Append extracted articles to a JSON file, ensuring uniqueness and proper serialization."""
    existing_articles = []

    # Load existing articles if the file exists
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as file:
                existing_articles = json.load(file)
        except json.JSONDecodeError:
            print("Warning: JSON file is empty or corrupted. Skipping the addition of new articles.")
            return  # Exit the function without adding new articles

    # Combine existing and new articles
    all_articles = existing_articles + articles

    # Remove duplicates based on title and link
    unique_articles = []
    seen = set()
    for article in all_articles:
        key = (article['title'], article['link'])
        if key not in seen:
            unique_articles.append(article)
            seen.add(key)
    
    # Sort articles by scraped_timestamp
    unique_articles.sort(key=lambda x: x['scraped_timestamp'], reverse=True)

    print(all_articles)

    # Save back to the JSON file using custom JSON encoder for ObjectId
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(unique_articles, file, indent=4, ensure_ascii=False, cls=JSONEncoder)


def main():
    parser = argparse.ArgumentParser(description="Google News Scraper with Thumbnails")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    
    # Avoid unrecognized argument error in Jupyter
    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    config = load_config(args.config)
    google_news_url = config.get("google_news_url", "https://news.google.com/")
    output_file = config.get("output_file", "news_articles.json")
    
    html = get_google_news_homepage(google_news_url)

    if html:
        articles = parse_google_news(html)
        print("Fetched HTML Preview:\n", html[:500])  # Print first 500 characters
        save_articles(articles, output_file)
        print(f"Scraped {len(articles)} articles. Data saved to {output_file}.")
        print("Scraped Articles:")
        for article in articles:
            print(f"- [{article['article_id']}] {article['title']} ({article['link']}) - Thumbnail: {article['thumbnail_url']}")

if __name__ == "__main__":
    main()
