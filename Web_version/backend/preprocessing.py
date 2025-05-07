# preprocessing.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} was not found.")

def compute_tfidf(df, column):
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(df[column].fillna(''))