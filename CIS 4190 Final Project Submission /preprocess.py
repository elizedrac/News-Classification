"""preprocess.py
    Data preprocessing for news source classification.
    Extracts text from provided URLs and prepares it for model input.
"""

# Imports
import re
from typing import List, Tuple
from urllib.parse import urlparse
import pandas as pd


# URL Parsing Functions
def extract_slug(url):
    """
    Extract the "slug" (last path segment) from a URL.
    
    Args:
        url: URL string
        
    Returns:
        Slug string or None if URL has no path
    """
    path = urlparse(url).path
    parts = path.strip("/").split("/")

    if len(parts) == 0:
        return None

    slug = parts[-1]
    return slug


def clean_slug(slug):
    """
    Clean slug by removing NBC-specific identifiers and formatting.
    
    Args:
        slug: URL slug string
        
    Returns:
        Cleaned slug string or None
    """
    if slug is None:
        return None
    slug = re.sub(r'(rcna|ncna)\d+$', '', slug)
    slug = slug.replace(".print", "")
    slug = slug.replace("-", " ")
    return slug.strip()


def slug_to_text(slug):
    """
    Convert slug to text format.
    
    Args:
        slug: URL slug string
        
    Returns:
        Text string or None
    """
    if slug is None:
        return None
    text = slug.replace("-", " ")
    return text


def extract_label(url):
    """
    Extract label from URL based on domain.
    
    Args:
        url: URL string
        
    Returns:
        0 for FoxNews, 1 for NBC, None otherwise
    """
    # FoxNew = 0
    if "foxnews.com" in url:
        return 0
    # NBX = 1
    elif "nbcnews.com" in url:
        return 1
    else:
        return None


# Preprocessing Functions
# In the notebook, we tested different preprocessing versions.
# Since Version 1 worked best for our model, we use it here.
stop_words = []


def preprocess_version1(text):
    """
    Version 1: Raw text → just lowercase
    Minimal preprocessing to preserve original text characteristics.
    This version performed best in our experiments.

    Args:
        text: Input text string

    Returns:
        Preprocessed text (lowercase only)
    """
    if pd.isna(text):
        return ""

    # Just lowercase
    text = str(text).lower().strip()

    return text


def preprocess_version2(text):
    """
    Version 2: Clean + stopwords removed
    Removes punctuation, special characters, and stopwords.
    (Not used in final model, but kept for reference)

    Args:
        text: Input text string

    Returns:
        Preprocessed text (cleaned, stopwords removed)
    """
    if pd.isna(text):
        return ""

    # Lowercase
    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove stopwords if available
    if stop_words:
        words = text.split()
        words = [w for w in words if w not in stop_words]
        text = ' '.join(words)

    return text.strip()


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    """
    Main preprocessing function.
    Reads CSV data, extracts text from URLs, and prepares inputs for the model.
    
    Args:
        path: Path to CSV file containing 'url' column
        
    Returns:
        Tuple of (X, y) where:
        - X: List of preprocessed text strings
        - y: List of integer labels (0 or 1)
    """
    df = pd.read_csv(path)

    X: List[str] = []
    y: List[int] = []

    for _, row in df.iterrows():
        url = row["url"]

        # Drop videos and any invalid URLs to increase accuracy
        if "/video" in url or "/watch" in url:
            continue

        # URL -> slug -> text
        slug = extract_slug(url)
        if slug is None:
            continue
        text = slug_to_text(clean_slug(slug))

        # Preprocess
        text = preprocess_version1(text)

        if text == "":
            continue

        X.append(text)

        # URL -> label
        label = extract_label(url)
        if label is not None:
            y.append(label)
        else:
            # Skip rows where we can't determine label
            X.pop()
    return X, y
