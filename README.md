# PickSmith 🔍
PickSmith is a content-based movie recommendation system that implements TF-IDF vectorisation and cosine similarity. The Python engine processes plot descriptions, genre metadata, and director information to generate personalised suggestions. It is designed as a modular application that demonstrates production-ready ML pipelines for educational and practical implementations.

## Features
- Analyses movie plot descriptions using TF-IDF vectorisation text processing  
- Combines genre, director, and rating data for hybrid and reliable recommendations  
- Measures movie cosine similarity based on processed feature vectors  
- Easy to extend with new algorithms or data sources

## File Overview
| File               | Description                                                            |
|--------------------|------------------------------------------------------------------------|
| `app.py`           | Main CLI interface for user interaction                                |
| `preprocessing.py` | Data loading and feature engineering                                   |
| `rec.py`           | Recommendation algorithms and scoring logic                            |
| `movies.csv`       | Dataset containing movie metadata (title, plot, genres, etc.)          |
| `test.py`          | Unit tests for recommendation logic                                    |

## Requirements
- Python 3  
- scikit-learn (TF-IDF, Cosine similarity)  
- Pandas (Data processing)  
- Numpy  

## How to use
- Clone repo:
   ```bash
   git clone https://github.com/aj11anuj/SeenIt.git
   ```
- Install dependencies:
   ```bash
   pip install pandas scikit-learn numpy python
   ```
- Run:
   ```bash
   python app.py
   ```

## Sample Output
![Screenshot (1214)](https://github.com/user-attachments/assets/560b97c5-1e67-4875-8b3a-90ad31fafdfd)
