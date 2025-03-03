# tasks/check_unique_title.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def is_title_unique(new_title, existing_titles, threshold=0.7):
    """Check if the title is unique using TF-IDF similarity."""
    if not existing_titles:
        return True

    titles = existing_titles + [new_title]
    vectorizer = TfidfVectorizer().fit_transform(titles)
    similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    max_similarity = similarity_matrix.max()

    return max_similarity <= threshold
