import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

class CustomSVD:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_means = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.matrix_shape = None

    def fit(self, user_item_matrix):
        """
        Antrenează modelul prin descompunerea matricii User-Item.
        """
        # Salvăm mapările pentru a găsi indexul corect ulterior
        self.user_mapping = {user_id: i for i, user_id in enumerate(user_item_matrix.index)}
        self.item_mapping = {isbn: i for i, isbn in enumerate(user_item_matrix.columns)}
        self.matrix_shape = user_item_matrix.shape
        
        # 1. Normalizare: Scădem media fiecărui user pentru a centra datele
        # (Esențial pentru a trata diferențele de generozitate între useri)
        data_float = user_item_matrix.values.astype(float)
        self.user_means = np.mean(data_float, axis=1).reshape(-1, 1)
        matrix_demeaned = data_float - self.user_means
        
        # 2. Aplicăm SVD
        # U: Matricea userilor, sigma: valorile singulare, Vt: Matricea itemilor
        U, sigma, Vt = svds(matrix_demeaned, k=self.n_factors)
        
        # Convertim sigma în matrice diagonală
        sigma = np.diag(sigma)
        
        # 3. Reconstruim factorii pentru predicție
        # Calculăm produsul punct pentru a obține predicțiile finale (fără a reconstrui toată matricea)
        self.user_factors = U
        self.item_factors = np.dot(sigma, Vt)
        
        print(f"✓ SVD finalizat. Factori latenți: {self.n_factors}")

    def predict(self, user_id, isbn):
        """
        Prezice rating-ul pentru un anumit user și ISBN.
        """
        # Dacă user-ul sau cartea nu au fost în setul de train
        if user_id not in self.user_mapping or isbn not in self.item_mapping:
            return 0.0 # Sau media globală
            
        u_idx = self.user_mapping[user_id]
        i_idx = self.item_mapping[isbn]
        
        # Predicția este: Media Userului + (User_Factor_i * Item_Factor_j)
        # Folosim produsul scalar al vectorilor latenți corespunzători
        prediction = self.user_means[u_idx, 0] + \
                     np.dot(self.user_factors[u_idx, :], self.item_factors[:, i_idx])
        
        return float(prediction)