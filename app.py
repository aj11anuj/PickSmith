from preprocessing import load_data, compute_tfidf
from rec import recommend_movies, format_recommendations

def main():
    filepath = "movies.csv"
    df = load_data(filepath)
    tfidf_matrix = compute_tfidf(df, 'Description')
    
    while True:
        print("\nMovie Recommendation System")
        print("---------------------------")
        movie_title = input("Enter a movie title (or 'q' to quit): ").strip()
        
        if movie_title.lower() == 'q':
            break
            
        recommendations = recommend_movies(movie_title, df, tfidf_matrix)
        formatted_recs = format_recommendations(recommendations)
        
        print(f"\nRecommendations for '{movie_title}':\n")
        print(formatted_recs)

if __name__ == "__main__":
    main()