from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from hybrid_recommender import HybridRecommender
from database import engine, get_avg_rating
from sqlalchemy import text
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializare Hybrid Recommender...")
hybrid = HybridRecommender(test_size=0.2)
hybrid.train_cf_models()
print("Modele incarcate!")


def get_user_profile_from_db(db_user_id: int) -> dict:
    with engine.connect() as conn:
        user = conn.execute(text("""
            SELECT top_genres, preferred_language,
                   preferred_year_start, preferred_year_end,
                   COALESCE(csv_user_id, user_id) AS model_id
            FROM users WHERE user_id = :uid
        """), {"uid": db_user_id}).fetchone()

        if user is None:
            raise HTTPException(status_code=404, detail=f"User {db_user_id} nu exista")

        history = conn.execute(text("""
            SELECT r.isbn, r.rating
            FROM rating r
            WHERE r.user_id = :uid AND r.rating > 0
        """), {"uid": db_user_id}).fetchall()

    genres = [g.strip() for g in user[0].split(',')] if user[0] else []

    return {
        "model_id": user[4],
        "favorite_genres": genres,
        "favorite_subjects": genres,
        "preferred_year_range": (user[2], user[3]) if user[2] and user[3] else None,
        "language": user[1],
        "user_read_history": [(row[0], row[1]) for row in history]
    }


def get_popular_books(n: int) -> list:
    popular_isbns = hybrid.ratings_df['ISBN'].value_counts().head(n).index.tolist()
    results = []
    for isbn in popular_isbns:
        book_match = hybrid.books_df[hybrid.books_df['ISBN'] == isbn]
        if not book_match.empty:
            book_info = book_match.iloc[0]
            results.append({
                'isbn': str(isbn),
                'title': str(book_info['Title']) if pd.notna(book_info['Title']) else '',
                'author': str(book_info['Author']) if pd.notna(book_info['Author']) else '',
                'year': int(book_info['Year']) if pd.notna(book_info.get('Year')) else None,
                'genre': str(book_info.get('genres', '')) if pd.notna(book_info.get('genres')) else '',
                'theme': str(book_info.get('subjects', '')) if pd.notna(book_info.get('subjects')) else '',
                'rating': get_avg_rating(isbn),
                'score': 0.0,
                'imageUrl': str(book_info.get('image_url', '')) if pd.notna(book_info.get('image_url')) else '',
                'note': 'popular_fallback'
            })
    return results


@app.get("/recommend/{db_user_id}")
def recommend(db_user_id: int, n: int = 10):
    profile = get_user_profile_from_db(db_user_id)
    recs = hybrid.get_recommendations(profile, n=n)
    if not recs:
        recs = get_popular_books(n)
    return {"db_user_id": db_user_id, "method": "hybrid", "books": recs}


@app.post("/admin/reload")
def reload_models():
    global hybrid
    hybrid = HybridRecommender(test_size=0.2)
    hybrid.train_cf_models()
    return {"status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "books": len(hybrid.books_df),
        "users": len(hybrid.users_df),
        "ratings": len(hybrid.ratings_df)
    }