import pandas as pd
from surprise import KNNWithMeans, Dataset, Reader, accuracy

class KNNRecommender:
    def __init__(self, train_path, test_path):
        self.reader = Reader(rating_scale=(0, 10))
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        self.train_data = Dataset.load_from_df(
            self.train_df[['User-ID', 'ISBN', 'Rating']], 
            self.reader
        )
        self.trainset = self.train_data.build_full_trainset()
        self.testset = list(self.test_df[['User-ID', 'ISBN', 'Rating']].itertuples(index=False, name=None))

    def train_user_based(self, k=40):
        """Antrenează KNN User-based (Găsește useri similari)."""
        print(f"Antrenare User-based KNN (k={k})...")
        sim_options = {
            'name': 'cosine', # Putem folosi și 'pearson' sau 'msd'
            'user_based': True # TRUE pentru User-based
        }
        self.model_user = KNNWithMeans(k=k, sim_options=sim_options)
        self.model_user.fit(self.trainset)
        return self.model_user

    def train_item_based(self, k=40):
        """Antrenează KNN Item-based (Găsește cărți similare)."""
        print(f"Antrenare Item-based KNN (k={k})...")
        sim_options = {
            'name': 'cosine',
            'user_based': False # FALSE pentru Item-based
        }
        self.model_item = KNNWithMeans(k=k, sim_options=sim_options)
        self.model_item.fit(self.trainset)
        return self.model_item

    def evaluate(self, model):
        """Calculează RMSE și MAE."""
        predictions = model.test(self.testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return rmse, mae