from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
import pandas as pd

from preprocessing import load_data, compute_tfidf
from rec import recommend_movies

# ========== FastAPI Setup ==========
app = FastAPI()

# CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Load Data Once ==========
df = load_data("movies.csv")
tfidf_matrix = compute_tfidf(df, "Description")

# ========== Request & Response Models ==========
class MovieInput(BaseModel):
    movie_name: str

class MovieRec(BaseModel):
    title: str
    genres: str
    director: str
    rating: float
    description: str

class RecResponse(BaseModel):
    movie_name: str
    recommended: Union[List[MovieRec], str]

# ========== API Endpoint ==========
@app.post("/recommend", response_model=RecResponse)
async def get_recommendations(movie: MovieInput):
    movie_title = movie.movie_name.strip()

    result = recommend_movies(movie_title, df, tfidf_matrix)

    if isinstance(result, str):  # If movie not found or error
        return {"movie_name": movie_title, "recommended": result}

    # Convert dataframe rows to list of dicts
    recommended = []
    for _, row in result.iterrows():
        recommended.append({
            "title": row["Title"],
            "genres": row["Genres"],
            "director": row["Director"],
            "rating": row["Rating"],
            "description": row["Description"]
        })

    return {"movie_name": movie_title, "recommended": recommended}
