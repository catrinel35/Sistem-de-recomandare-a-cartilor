import pandas as pd
from surprise import SVD, KNNWithMeans, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

class RecommenderEngine:
    def __init__(self, train_path, test_path):
        # Definim scala rating-urilor (0-10 pentru Book-Crossing)
        self.reader = Reader(rating_scale=(0, 10))
        
        # Încărcăm seturile de date
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Surprise are nevoie de ordinea: User, Item, Rating
        self.train_data = Dataset.load_from_df(train_df[['User-ID', 'ISBN', 'Rating']], self.reader)
        self.test_df = test_df
        
        # Generăm Trainset-ul oficial pentru Surprise
        self.trainset = self.train_data.build_full_trainset()
        
    def train_svd(self):
        print("Antrenare SVD...")
        self.model_svd = SVD(n_factors=100, lr_all=0.005, reg_all=0.02)
        self.model_svd.fit(self.trainset)
        return self.model_svd

    def train_knn_user(self):
        print("Antrenare User-based KNN...")
        # similarity options: cosine, pearson, msd
        sim_options = {'name': 'cosine', 'user_based': True}
        self.model_knn_user = KNNWithMeans(sim_options=sim_options)
        self.model_knn_user.fit(self.trainset)
        return self.model_knn_user

    def train_knn_item(self):
        print("Antrenare Item-based KNN...")
        sim_options = {'name': 'cosine', 'user_based': False}
        self.model_knn_item = KNNWithMeans(sim_options=sim_options)
        self.model_knn_item.fit(self.trainset)
        return self.model_knn_item

    def evaluate_model(self, model):
        # Transformăm test_df în formatul de test Surprise
        testset = list(self.test_df[['User-ID', 'ISBN', 'Rating']].itertuples(index=False, name=None))
        predictions = model.test(testset)
        
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return rmse, mae
    
    def get_top_n_recommendations(self, model, user_id, n=5, books_metadata=None):
        # Obținem toate ISBN-urile unice din dataset
        all_books = self.trainset.all_items()
        all_books_isbn = [self.trainset.to_raw_iid(i) for i in all_books]
        
        # Găsim cărțile pe care user-ul le-a citit deja în train
        try:
            user_inner_id = self.trainset.to_inner_uid(user_id)
            user_read_books = [self.trainset.to_raw_iid(i) for (i, _) in self.trainset.ur[user_inner_id]]
        except ValueError:
            user_read_books = []

        # Filtrăm cărțile necitite
        books_to_predict = [b for b in all_books_isbn if b not in user_read_books]
        
        # Generăm predicții
        predictions = [model.predict(user_id, isbn) for isbn in books_to_predict]
        
        # Sortăm după rating-ul estimat
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_n = predictions[:n]
        
        # Adăugăm metadate (Titlu, Autor) dacă sunt disponibile
        results = []
        for p in top_n:
            book_info = {'ISBN': p.iid, 'EstimatedRating': round(p.est, 2)}
            if books_metadata is not None:
                meta = books_metadata[books_metadata['ISBN'] == p.iid]
                if not meta.empty:
                    book_info['Title'] = meta.iloc[0]['Title']
                    book_info['Author'] = meta.iloc[0]['Author']
            results.append(book_info)
            
        return results
    
    if __name__ == "__main__":
        engine = RecommenderEngine('../data/processed/train.csv', '../data/processed/test.csv')
        books_meta = pd.read_csv('../data/processed/books_final.csv')

        results = []
        
        # Rulăm modelele
        models = {
            "SVD": engine.train_svd(),
            "User-KNN": engine.train_knn_user(),
            "Item-KNN": engine.train_knn_item()
        }
        
        for name, model in models.items():
            rmse, mae = engine.evaluate_model(model)
            results.append({"Model": name, "RMSE": rmse, "MAE": mae})
        
        # Afișăm tabelul de metrici
        comparison_df = pd.DataFrame(results)
        print("\n--- Comparație Metrici ---")
        print(comparison_df)
        
        # Exemplu recomandare
        example_user = 276747 # Schimbă cu un ID valid din dataset-ul tău
        print(f"\nRecomandări SVD pentru User {example_user}:")
        recs = engine.get_top_n_recommendations(models["SVD"], example_user, n=5, books_metadata=books_meta)
        for r in recs:
            print(f"- {r.get('Title', 'N/A')} ({r['Author']}): {r['EstimatedRating']}")

