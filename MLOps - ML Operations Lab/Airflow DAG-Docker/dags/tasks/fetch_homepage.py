# tasks/fetch_homepage.py
import requests

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
