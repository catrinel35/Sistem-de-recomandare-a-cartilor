from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from knn import CustomKNN
from svd import CustomSVD


class RecommenderManager:
    def __init__(self, data_dir='../data/processed', test_size=0.2):
        self.data_dir = Path(data_dir)

        # Incarcam datele
        print("Incarcare date...")
        self.books_df = self._load_books()
        self.users_df = self._load_users()
        self.ratings_df = self._load_ratings()

        # Cream train/test split
        print(f"Creare train/test split (test_size={test_size})...")
        self.train_df, self.test_df = self._create_train_test_split(test_size)

        # Pregatim matricea Pivot (User-Item Matrix)
        print("Pregatire matrice User-Item...")
        self.user_item_matrix = self.train_df.pivot(
            index='User-ID',
            columns='ISBN',
            values='Rating'
        ).fillna(0)

        print(f"Matrice User-Item: {self.user_item_matrix.shape}")
        print(f"Train: {len(self.train_df)}, Test: {len(self.test_df)}")

    def _load_books(self):
        """Incarca books_enriched.csv"""
        books_path = self.data_dir / 'books_enriched.csv'
        if not books_path.exists():
            # Fallback la books_processed.csv
            books_path = self.data_dir / 'books_processed.csv'

        if not books_path.exists():
            raise FileNotFoundError(f"Nu am gasit fisierul cu carti in {self.data_dir}")

        books = pd.read_csv(books_path)
        books['ISBN'] = books['ISBN'].astype(str)
        print(f"Books loaded: {len(books)}")
        return books

    def _load_users(self):
        """Incarca users_enriched.csv sau users_processed.csv"""
        users_path = self.data_dir / 'users_enriched.csv'
        if not users_path.exists():
            users_path = self.data_dir / 'users_processed.csv'

        if users_path.exists():
            users = pd.read_csv(users_path)
            print(f"Users loaded: {len(users)}")
            return users

        print("Users file not found, continuing without user data")
        return None

    def _load_ratings(self):
        """Incarca ratings_processed.csv"""
        ratings_path = self.data_dir / 'ratings_processed.csv'

        if not ratings_path.exists():
            raise FileNotFoundError(f"Nu am gasit {ratings_path}")

        ratings = pd.read_csv(ratings_path)
        ratings['ISBN'] = ratings['ISBN'].astype(str)

        # Filtram doar rating-uri explicite (> 0) pentru CF
        ratings = ratings[ratings['Rating'] > 0]
        print(f"Ratings loaded (explicit only): {len(ratings)}")

        # Filtram sa pastram doar ISBN-uri care exista in books_df
        valid_isbns = set(self.books_df['ISBN'].astype(str))
        ratings = ratings[ratings['ISBN'].isin(valid_isbns)]
        print(f"Ratings after filtering valid ISBNs: {len(ratings)}")

        return ratings

    def _create_train_test_split(self, test_size=0.2):
        """Creeaza train/test split din ratings"""
        train_df, test_df = train_test_split(
            self.ratings_df,
            test_size=test_size,
            random_state=42
        )

        # Salvam pentru reutilizare (optional)
        train_path = self.data_dir / 'train.csv'
        test_path = self.data_dir / 'test.csv'

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Train/Test salvate in {self.data_dir}")

        return train_df, test_df

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
        """Evalueaza modelul pe setul de test folosind RMSE si MAE."""
        y_true = []
        y_pred = []

        for row in self.test_df.itertuples():
            user_id = getattr(row, 'User-ID', row[1])  # User-ID
            isbn = str(row.ISBN)
            rating = row.Rating

            # Facem predictie
            pred = model.predict(user_id, isbn)

            # Clipuim valorile intre 0-10
            pred = max(0, min(10, pred))

            y_true.append(rating)
            y_pred.append(pred)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae

    def get_recommendations(self, model, user_id, n=5):
        """Obtine top N recomandari pentru un utilizator."""
        print(f"\nGenerare recomandari pentru User {user_id}...")

        # Identificam ISBN-urile pe care user-ul nu le-a votat in train
        user_ratings = self.train_df[self.train_df['User-ID'] == user_id]
        read_isbns = set(user_ratings['ISBN'].astype(str).unique())
        all_isbns = self.user_item_matrix.columns.astype(str)

        to_predict = [isbn for isbn in all_isbns if isbn not in read_isbns]

        predictions = []
        for isbn in to_predict:
            score = model.predict(user_id, isbn)
            predictions.append((isbn, score))

        # Sortam dupa scor
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Adaugam metadate - cu verificare
        results = []
        for isbn, score in predictions:
            if len(results) >= n:
                break

            book_match = self.books_df[self.books_df['ISBN'] == isbn]

            if not book_match.empty:
                book_info = book_match.iloc[0]
                results.append({
                    'ISBN': isbn,
                    'Title': book_info['Title'],
                    'Author': book_info['Author'],
                    'Year': book_info.get('Year', 'N/A'),
                    'Genres': book_info.get('genres', 'N/A'),
                    'EstimatedRating': round(score, 2)
                })

        return results

    def get_user_profile(self, user_id):
        """Returneaza profilul unui utilizator (carti citite)"""
        user_ratings = self.ratings_df[self.ratings_df['User-ID'] == user_id]

        profile = []
        for _, row in user_ratings.iterrows():
            isbn = str(row['ISBN'])
            book_match = self.books_df[self.books_df['ISBN'] == isbn]

            if not book_match.empty:
                book = book_match.iloc[0]
                profile.append({
                    'ISBN': isbn,
                    'Title': book['Title'],
                    'Author': book['Author'],
                    'Rating': row['Rating']
                })

        return profile


if __name__ == "__main__":
    # Initializam managerul
    manager = RecommenderManager(
        data_dir='../data/processed',
        test_size=0.2
    )

    # Executam SVD
    svd_model = manager.run_svd(n_factors=50)

    # Executam KNN User-based
    knn_user_model = manager.run_knn(k=20, method='user')

    # Executam KNN Item-based
    knn_item_model = manager.run_knn(k=20, method='item')

    # Exemplu recomandare
    print("\n" + "=" * 50)
    print("EXEMPLU RECOMANDARI")
    print("=" * 50)

    sample_user = manager.test_df['User-ID'].iloc[0]

    # Profilul userului
    print(f"\nProfilul User {sample_user}:")
    profile = manager.get_user_profile(sample_user)
    for p in profile[:5]:
        print(f"  - {p['Title']} (Rating: {p['Rating']})")

    # Recomandari SVD
    print(f"\nRecomandari SVD pentru User {sample_user}:")
    recs = manager.get_recommendations(svd_model, sample_user, n=5)
    for r in recs:
        print(f"  - {r['Title']} ({r['Author']}): {r['EstimatedRating']}")

    # Recomandari KNN
    print(f"\nRecomandari KNN User-based pentru User {sample_user}:")
    recs_knn = manager.get_recommendations(knn_user_model, sample_user, n=5)
    for r in recs_knn:
        print(f"  - {r['Title']} ({r['Author']}): {r['EstimatedRating']}")
