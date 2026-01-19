import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Importăm implementările noastre custom (pe care le vom scrie imediat)
from svd import CustomSVD
from knn import CustomKNN

class RecommenderManager:
    def __init__(self, train_path, test_path, books_path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.books_df = pd.read_csv(books_path)
        
        # Pregătim matricea Pivot (User-Item Matrix)
        # Aceasta este inima ambelor algoritmi
        print("Pregătire matrice User-Item...")
        self.user_item_matrix = self.train_df.pivot(
            index='User-ID', 
            columns='ISBN', 
            values='Rating'
        ).fillna(0)
        
    def run_svd(self, n_factors=50):
        print(f"\n--- Rulare SVD (factors={n_factors}) ---")
        model = CustomSVD(n_factors=n_factors)
        model.fit(self.user_item_matrix)
        
        rmse, mae = self.evaluate(model)
        print(f"SVD - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return model

    def run_knn(self, k=20, method='user'):
        print(f"\n--- Rulare KNN ({method}-based, k={k}) ---")
        model = CustomKNN(k=k, method=method)
        model.fit(self.user_item_matrix)
        
        rmse, mae = self.evaluate(model)
        print(f"KNN {method} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return model

    def evaluate(self, model):
        """Evaluează modelul pe setul de test folosind RMSE și MAE."""
        y_true = []
        y_pred = []
        
        for row in self.test_df.itertuples():
            # Facem predicție pentru fiecare pereche (User, ISBN) din test
            pred = model.predict(row._1, row.ISBN) # _1 este User-ID
            
            # Surprise/Standard scara este 0-10, clipuim valorile pentru siguranță
            pred = max(0, min(10, pred))
            
            y_true.append(row.Rating)
            y_pred.append(pred)
            
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae

    def get_recommendations(self, model, user_id, n=5):
        """Obține top N recomandări pentru un utilizator."""
        print(f"\nGenerare recomandări pentru User {user_id}...")
        
        # Identificăm ISBN-urile pe care user-ul nu le-a votat în train
        user_ratings = self.train_df[self.train_df['User-ID'] == user_id]
        read_isbns = set(user_ratings['ISBN'].unique())
        all_isbns = self.user_item_matrix.columns
        
        to_predict = [isbn for isbn in all_isbns if isbn not in read_isbns]
        
        predictions = []
        for isbn in to_predict:
            score = model.predict(user_id, isbn)
            predictions.append((isbn, score))
            
        # Sortăm după scor
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]
        
        # Adăugăm metadate
        results = []
        for isbn, score in top_n:
            book_info = self.books_df[self.books_df['ISBN'] == isbn].iloc[0]
            results.append({
                'Title': book_info['Title'],
                'Author': book_info['Author'],
                'EstimatedRating': round(score, 2)
            })
        return results

if __name__ == "__main__":
    manager = RecommenderManager(
        '../data/processed/train.csv',
        '../data/processed/test.csv',
        '../data/processed/books_final.csv'
    )
    
    # Executăm SVD
    svd_model = manager.run_svd()
    
    # Exemplu recomandare
    sample_user = manager.test_df['User-ID'].iloc[0]
    recs = manager.get_recommendations(svd_model, sample_user)
    for r in recs:
        print(f"- {r['Title']} ({r['Author']}): {r['EstimatedRating']}")