from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def recommend_movies(movie_title, df, tfidf_matrix, top_n=5):
    """Fixed version with proper output formatting"""
    if movie_title not in df['Title'].values:
        return "Movie not found in database!"

    idx = df[df['Title'] == movie_title].index[0]
    
    # 1. Fix tokenizer warning by specifying token_pattern
    genre_vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(','),
        token_pattern=None  # This suppresses the warning
    )
    genre_matrix = genre_vectorizer.fit_transform(df['Genres'])
    
    # Calculate similarities
    content_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    genre_sim = cosine_similarity(genre_matrix[idx], genre_matrix)[0]
    director_bonus = np.array([1.5 if director == df.iloc[idx]['Director'] else 1 
                             for director in df['Director']])
    
    combined_score = (0.4*content_sim + 0.3*genre_sim + 0.2*director_bonus + 0.1*(df['Rating']/10))
    
    # Get top recommendations
    scores = list(enumerate(combined_score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in scores[1:top_n+1]]
    
    # Return properly formatted recommendations
    return df.iloc[top_indices]

# Add this helper function for clean display
def format_recommendations(recommendations):
    if isinstance(recommendations, str):
        return recommendations
        
    result = []
    for i, row in recommendations.iterrows():
        entry = (
            f"{row['Title']}\n"
            f"   Genres: {row['Genres']}\n"
            f"   Director: {row['Director']}\n"
            f"   Rating: {row['Rating']}/10\n"
            f"   Description: {row['Description'][:100]}..."
        )
        result.append(entry)
    return "\n\n".join(result)