# tasks/download_thumbnails.py
import os
import requests

def download_thumbnail(thumbnail_url, article_id):
    """Download the thumbnail image for a given article using the article_id for unique naming."""
    if not thumbnail_url:
        return None

    try:
        response = requests.get(thumbnail_url, stream=True)
        if response.status_code == 200:
            image_filename = f"thumbnail_{article_id}.jpg"
            image_path = os.path.join("thumbnails", image_filename)

            if not os.path.exists("thumbnails"):
                os.makedirs("thumbnails")

            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            return image_path
        else:
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None
