Google News Scraper & Notification System using Airflow - Preethi MM21B051

This project consists of two Apache Airflow DAGs:
Google News Scraper DAG – Scrapes Google News, extracts articles, stores them in a PostgreSQL database, and tracks execution status.
Send Email Notification DAG – Detects new <image, headline> pairs added to the database and sends an email notification.

1. Google News Scraper DAG (google_news_scraper.py)
Overview
This DAG scrapes articles from Google News, saves them in a PostgreSQL database, and tracks execution status for triggering the second DAG.
The functions used in dag1.py are each individual functions imported from google_news_scraper.py
Workflow
Load Configuration – Reads config.json for the Google News URL and output settings.
Scrape Articles – Fetches and parses news articles from Google News.
Save to JSON – Stores scraped articles locally for reference.
Create Database Tables – Ensures the articles and images tables exist in PostgreSQL.
Insert Data – Inserts scraped articles and associated images into the database.
Track Status – Writes the number of successful inserts to a status file (status).
Scheduling (Crontab)
This DAG runs every hour using the following cron schedule:
0 * * * * airflow dags trigger google_news_scraper

2. Send Email Notification DAG (dag2.py)
Overview
This DAG monitors the articles and images tables, detects new <image, headline> tuples, and sends an email notification.

Workflow
File Sensor Trigger – Monitors the status file to detect when google_news_scraper.py completes.
Fetch New Entries – Compares the latest records in the database with previously known entries stored in prev_state.json.
Send Email Notification – Sends an email with new <image, headline> entries.
Update State – Stores the last known timestamp in prev_state.json to track new records in future runs.
Delete Status File – Removes status to reset the monitoring process.
Handling Missing prev_state.json
If prev_state.json does not exist, it is initialized with 1970-01-01 00:00:00 to ensure proper tracking.
SMTP Configuration for Email Sending
To send emails, configured SMTP with TLS/SSL. Please enter your email details in dag2.py
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USER = "your_email@gmail.com"
EMAIL_PASS = "your_app_password"

All the required modules to run this code are included in the PIP_INSTALL_REQUIREMENTS of the .yaml file, hence there is no requirements.txt
