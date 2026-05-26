from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommended_models import RecommenderManager
from sqlalchemy import text
from database import engine, get_avg_rating
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializare modele ML...")
manager = RecommenderManager(test_size=0.2)
svd_model = manager.run_svd(n_factors=50)
knn_user_model = manager.run_knn(k=20, method='user')
knn_item_model = manager.run_knn(k=20, method='item')
print("Modele incarcate!")


def get_model_user_id(db_user_id: int) -> int:
    """Converteste user_id din DB in ID-ul folosit in modelul ML"""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COALESCE(csv_user_id, user_id) AS model_id
            FROM users 
            WHERE user_id = :uid
        """), {"uid": db_user_id}).fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail=f"User {db_user_id} nu exista in DB")
    return result[0]


def get_popular_books(n: int) -> list:
    """Returneaza cele mai populare carti ca fallback pentru useri noi"""
    popular_isbns = manager.ratings_df['ISBN'].value_counts().head(n).index.tolist()
    results = []
    for isbn in popular_isbns:
        book_match = manager.books_df[manager.books_df['ISBN'] == isbn]
        if not book_match.empty:
            book_info = book_match.iloc[0]
            results.append({
                'ISBN': str(isbn),
                'Title': str(book_info['Title']) if pd.notna(book_info['Title']) else '',
                'Author': str(book_info['Author']) if pd.notna(book_info['Author']) else '',
                'Year': int(book_info['Year']) if pd.notna(book_info.get('Year')) else None,
                'Genres': str(book_info.get('genres', '')) if pd.notna(book_info.get('genres')) else '',
                'EstimatedRating': 0.0,
                'note': 'popular_fallback'
            })
    return results


@app.get("/recommend/{db_user_id}")
def recommend(db_user_id: int, n: int = 5, method: str = "svd"):
    model_user_id = get_model_user_id(db_user_id)

    if method == "svd":
        model = svd_model
    elif method == "knn_user":
        model = knn_user_model
    elif method == "knn_item":
        model = knn_item_model
    else:
        raise HTTPException(status_code=400, detail="method trebuie sa fie svd, knn_user sau knn_item")

    recs = manager.get_recommendations(model, model_user_id, n=n)

    # Fallback pentru useri noi fara ratinguri in matrice
    if not recs:
        recs = get_popular_books(n)

    return {
        "db_user_id": db_user_id,
        "model_user_id": model_user_id,
        "method": method,
        "books": recs
    }


@app.get("/recommend/{db_user_id}/all")
def recommend_all(db_user_id: int, n: int = 5):
    model_user_id = get_model_user_id(db_user_id)

    svd_recs = manager.get_recommendations(svd_model, model_user_id, n=n) or get_popular_books(n)
    knn_user_recs = manager.get_recommendations(knn_user_model, model_user_id, n=n) or get_popular_books(n)
    knn_item_recs = manager.get_recommendations(knn_item_model, model_user_id, n=n) or get_popular_books(n)

    return {
        "db_user_id": db_user_id,
        "model_user_id": model_user_id,
        "svd": svd_recs,
        "knn_user": knn_user_recs,
        "knn_item": knn_item_recs
    }


@app.post("/admin/reload")
def reload_models():
    global manager, svd_model, knn_user_model, knn_item_model
    print("Reloading models...")
    manager = RecommenderManager(test_size=0.2)
    svd_model = manager.run_svd(n_factors=50)
    knn_user_model = manager.run_knn(k=20, method='user')
    knn_item_model = manager.run_knn(k=20, method='item')
    return {"status": "ok", "users": len(manager.users_df), "books": len(manager.books_df)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "books": len(manager.books_df),
        "users": len(manager.users_df),
        "ratings": len(manager.ratings_df)
    }