import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CustomKNN:
    def __init__(self, k=20, method='user'):
        """
        Args:
            k: Numărul de vecini luați în calcul.
            method: 'user' pentru User-based sau 'item' pentru Item-based.
        """
        self.k = k
        self.method = method
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_mapping = {}
        self.item_mapping = {}

    def fit(self, user_item_matrix):
        """
        Calculează matricea de similaritate.
        """
        self.user_item_matrix = user_item_matrix.values
        self.user_mapping = {user_id: i for i, user_id in enumerate(user_item_matrix.index)}
        self.item_mapping = {isbn: i for i, isbn in enumerate(user_item_matrix.columns)}

        if self.method == 'user':
            # Similaritate între rânduri (utilizatori)
            print("Calculare similaritate User-User...")
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        else:
            # Similaritate între coloane (itemi)
            print("Calculare similaritate Item-Item...")
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
        print(f"✓ Matrice de similaritate {self.method} finalizată.")

    def predict(self, user_id, isbn):
        """
        Prezice rating-ul folosind media ponderată a celor mai apropiați K vecini.
        """
        if user_id not in self.user_mapping or isbn not in self.item_mapping:
            return 0.0

        u_idx = self.user_mapping[user_id]
        i_idx = self.item_mapping[isbn]

        if self.method == 'user':
            return self._predict_user_based(u_idx, i_idx)
        else:
            return self._predict_item_based(u_idx, i_idx)

    def _predict_user_based(self, u_idx, i_idx):
        # Similaritățile user-ului curent cu toți ceilalți
        user_sims = self.similarity_matrix[u_idx]
        
        # Rating-urile tuturor userilor pentru item-ul curent
        item_ratings = self.user_item_matrix[:, i_idx]
        
        # Filtrăm doar userii care au dat rating la acest item (rating > 0)
        idx_with_rating = np.where(item_ratings > 0)[0]
        
        if len(idx_with_rating) == 0:
            return 0.0

        # Sortăm vecinii după similaritate
        similarities = user_sims[idx_with_rating]
        ratings = item_ratings[idx_with_rating]
        
        # Luăm top K
        top_k_indices = np.argsort(similarities)[-self.k:]
        top_k_sims = similarities[top_k_indices]
        top_k_ratings = ratings[top_k_indices]

        # Media ponderată: (Sim * Rating) / Suma Sim
        sum_sims = np.sum(np.abs(top_k_sims))
        if sum_sims == 0:
            return 0.0
            
        return np.dot(top_k_sims, top_k_ratings) / sum_sims

    def _predict_item_based(self, u_idx, i_idx):
        # Similaritățile item-ului curent cu toate celelalte
        item_sims = self.similarity_matrix[i_idx]
        
        # Rating-urile date de user-ul curent tuturor item-urilor
        user_ratings = self.user_item_matrix[u_idx, :]
        
        # Filtrăm itemii pe care user-ul i-a votat deja
        idx_voted = np.where(user_ratings > 0)[0]
        
        if len(idx_voted) == 0:
            return 0.0

        similarities = item_sims[idx_voted]
        ratings = user_ratings[idx_voted]
        
        top_k_indices = np.argsort(similarities)[-self.k:]
        top_k_sims = similarities[top_k_indices]
        top_k_ratings = ratings[top_k_indices]

        sum_sims = np.sum(np.abs(top_k_sims))
        if sum_sims == 0:
            return 0.0
            
        return np.dot(top_k_sims, top_k_ratings) / sum_sims