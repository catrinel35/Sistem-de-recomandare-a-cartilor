from fastapi import FastAPI, HTTPException
from recommended_models import RecommenderManager
from knn import CustomKNN
from svd import CustomSVD

app = FastAPI()

# Incarcam modelele la pornire
print("Initializare modele ML...")
manager = RecommenderManager(test_size=0.2)
svd_model = manager.run_svd(n_factors=50)
knn_user_model = manager.run_knn(k=20, method='user')
knn_item_model = manager.run_knn(k=20, method='item')
print("Modele incarcate!")


@app.get("/recommend/{csv_user_id}")
def recommend(csv_user_id: int, n: int = 5, method: str = "svd"):
    """
    Returneaza top N recomandari pentru un user.
    method: svd | knn_user | knn_item
    """
    if method == "svd":
        model = svd_model
    elif method == "knn_user":
        model = knn_user_model
    elif method == "knn_item":
        model = knn_item_model
    else:
        raise HTTPException(status_code=400, detail="method trebuie sa fie svd, knn_user sau knn_item")

    recs = manager.get_recommendations(model, csv_user_id, n=n)

    if not recs:
        raise HTTPException(status_code=404, detail=f"Nu s-au gasit recomandari pentru user {csv_user_id}")

    return {
        "user_id": csv_user_id,
        "method": method,
        "books": recs
    }


@app.get("/recommend/{csv_user_id}/all")
def recommend_all(csv_user_id: int, n: int = 5):
    """
    Returneaza recomandari din toate cele 3 modele simultan.
    """
    return {
        "user_id": csv_user_id,
        "svd": manager.get_recommendations(svd_model, csv_user_id, n=n),
        "knn_user": manager.get_recommendations(knn_user_model, csv_user_id, n=n),
        "knn_item": manager.get_recommendations(knn_item_model, csv_user_id, n=n),
    }


@app.get("/health")
def health():
    return {"status": "ok", "books": len(manager.books_df), "ratings": len(manager.ratings_df)}