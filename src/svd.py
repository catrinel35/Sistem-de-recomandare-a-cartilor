import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

class SVDRecommender:
    def __init__(self, train_path, test_path):
        # 1. Definirea Reader-ului (specificăm gama de rating 0-10)
        self.reader = Reader(rating_scale=(0, 10))
        
        # 2. Încărcarea datelor salvate anterior de data_processor.py
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        # 3. Conversia în format Surprise (User, Item, Rating)
        self.train_data = Dataset.load_from_df(
            self.train_df[['User-ID', 'ISBN', 'Rating']], 
            self.reader
        )
        # Generăm setul de antrenament complet
        self.trainset = self.train_data.build_full_trainset()
        
        # 4. Inițializarea modelului SVD
        # n_factors: numărul de trăsături latente (latent factors)
        self.model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)

    def train(self):
        """Antrenarea modelului pe setul de date."""
        print("Antrenare model SVD în curs...")
        self.model.fit(self.trainset)
        print("✓ Antrenare finalizată.")

    def evaluate(self):
        """Evaluare RMSE și MAE pe setul de test."""
        # Convertim test_df într-o listă de tupluri cerută de Surprise
        testset = list(self.test_df[['User-ID', 'ISBN', 'Rating']].itertuples(index=False, name=None))
        predictions = self.model.test(testset)
        
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        return rmse, mae

    def predict_user_rating(self, user_id, isbn):
        """Predicția unui singur rating pentru un user și o carte."""
        prediction = self.model.predict(user_id, isbn)
        return prediction.est
    
    def get_top_n_recommendations(self, user_id, n=10, books_metadata=None):
        """
        Generază top N recomandări pentru un utilizator.
        """
        # 1. Identificăm toate cărțile din setul de antrenament
        all_books = self.trainset.all_items()
        all_books_raw = [self.trainset.to_raw_iid(i) for i in all_books]
        
        # 2. Identificăm cărțile pe care user-ul le-a citit deja
        try:
            user_inner_id = self.trainset.to_inner_uid(user_id)
            user_read_inner_ids = [item_id for (item_id, _) in self.trainset.ur[user_inner_id]]
            user_read_books = [self.trainset.to_raw_iid(i) for i in user_read_inner_ids]
        except ValueError:
            # Dacă user-ul nu e în trainset (cold start), returnăm listă goală sau populare
            user_read_books = []

        # 3. Filtrăm cărțile pentru a rămâne doar cele necitite
        books_to_predict = [b for b in all_books_raw if b not in user_read_books]
        
        # 4. Generăm predicții pentru toate aceste cărți
        predictions = [self.model.predict(user_id, isbn) for isbn in books_to_predict]
        
        # 5. Sortăm după rating-ul estimat (est) în ordine descrescătoare
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_predictions = predictions[:n]
        
        # 6. Formatăm rezultatele și adăugăm metadate dacă sunt disponibile
        recommendations = []
        for pred in top_predictions:
            item_info = {
                'ISBN': pred.iid,
                'EstimatedRating': round(pred.est, 2)
            }
            if books_metadata is not None:
                meta = books_metadata[books_metadata['ISBN'] == pred.iid]
                if not meta.empty:
                    item_info['Title'] = meta.iloc[0]['Title']
                    item_info['Author'] = meta.iloc[0]['Author']
                    item_info['Year'] = meta.iloc[0]['Year']
            
            recommendations.append(item_info)
            
        return recommendations
    
    if __name__ == "__main__":
        # Inițializăm și antrenăm
        svd_engine = SVDRecommender('../data/processed/train.csv', '../data/processed/test.csv')
        svd_engine.train()
        
        # Încărcăm metadatele pentru a vedea titlurile cărților, nu doar ISBN-urile
        books_metadata = pd.read_csv('../data/processed/books_final.csv')
        
        # Alegem un user din setul de test
        target_user = svd_engine.test_df['User-ID'].iloc[0]
        
        print(f"\n--- Generare Top 5 Recomandări SVD pentru User {target_user} ---")
        recommendations = svd_engine.get_top_n_recommendations(
            user_id=target_user, 
            n=5, 
            books_metadata=books_metadata
        )
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec.get('Title', 'Necunoscut')} - {rec.get('Author', 'N/A')} ({rec['EstimatedRating']})")