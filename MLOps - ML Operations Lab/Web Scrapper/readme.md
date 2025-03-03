Google News Scraper - MM21B051 - Preethi

Overview
This project scrapes articles from Google News, extracts metadata (title, source, author, category, publication time, scrapped time), and stores them in a MongoDB database. It also downloads and saves article thumbnails. The scraper is scheduled to run every hour using orchestration.py.
google_news_scraper
│── orchestration.py  
│── google_news_scraper.py  
│── config.json  
│── README.md  
│── thumbnails/ (stores downloaded images)  

orchestration.py
-This is the main script scheduled to run every hour. (I am on windows, so I used windows task scheduler, howver it can be setup as a cron job too)
-It imports functions from google_news_scraper.py to perform web scraping.
-Loads configurations from config.json.
-Calls the scraping function and saves results to MongoDB and a JSON file for reference.

google_news_scraper.py
-Contains all core functions for scraping Google News.
-Uses Requests and BeautifulSoup to fetch and parse the news page.
-Uses Selenium to bypass Google News redirection and get the actual news URLs, as the urls extracted from the homepage are dynamic and change with time, hence it important to find the redirected url and store that.
-Extracts metadata such as title, source, author, category, and times posted and time scraped.
-The thumbnails are downloaded and stored locally as the urls to these images are dynamic, the mongodb has a path to these images and the image file along with article data.
-Uses TfidfVectorizer to check for duplicate titles before saving. This method was chosen as naively comparing headlines for repetition will only eliminated duplicates. However, articles with the same content by different journals can get stored, which is redundant information. To ensure each article has unique and useful information, the article headlines are compared for similarity for the articles already existing in the database. A cosine similarity of > 0.7 articles are not added.
-Uses pymongo to store data in MongoDB.
To view the MongoDb stored information, you many run this code:
'''
    from pymongo import MongoClient
    from bson import ObjectId  # Required to properly display ObjectId

    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['google_news']
    collection = db['news_articles']# Replace with your collection name

    # Retrieve and print all news articles
    def print_news_articles():
        articles = collection.find()
        for article in articles:
            print("Article ID:", article.get('article_id'))
            print("Title:", article.get('title'))
            print("Link:", article.get('link'))
            print("Author:", article.get('author'))
            print("Source/Publisher:", article.get('source_publisher'))
            print("Category:", article.get('category'))
            print("Time Posted:", article.get('time_posted'))
            print("Scraped Timestamp:", article.get('scraped_timestamp'))
            print("Thumbnail URL:", article.get('thumbnail_url'))
            print("MongoDB Object ID:", article.get('_id'))
            print("-" * 50)

    # Call the function to print the articles
    print_news_articles()
                         '''