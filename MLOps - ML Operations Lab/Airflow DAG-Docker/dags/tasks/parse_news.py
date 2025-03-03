# tasks/parse_news.py
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def parse_google_news(html):
    """Parse Google News homepage to extract headlines, URLs, and metadata."""
    soup = BeautifulSoup(html, "html.parser")
    articles = []

    for item in soup.select("a.gPFEn"):  # Targeting article links
        title = item.get_text(strip=True)
        relative_link = item.get("href")

        if relative_link.startswith("./"):
            google_news_url = urljoin("https://news.google.com", relative_link)
            articles.append({
                "title": title,
                "link": google_news_url
            })

    return articles
