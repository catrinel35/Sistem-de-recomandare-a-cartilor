"""
=============================================================================
HYBRID RECOMMENDER - INTEGRARE COMPLETA
=============================================================================

Integreaza toate componentele:
1. RecommenderManager (CF: SVD, KNN) - din recommended_models.py
2. BERT Embeddings - din fisierele .npy pre-generate
3. Hybrid Scoring - din book_scoring.ipynb

Autor: Echipa Proiect
Data: 2025
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.model_selection import train_test_split

# Import custom CF models
from svd import CustomSVD
from knn import CustomKNN

# BERT (optional - doar pentru user queries noi)
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("sentence-transformers nu este instalat - doar embeddings pre-calculate")


# =============================================================================
# SCORING WEIGHTS
# =============================================================================

SCORING_WEIGHTS = {
    'bert_similarity': 0.30,
    'cf_score': 0.25,
    'genre_match': 0.20,
    'subject_match': 0.10,
    'popularity': 0.10,
    'recency': 0.05
}


class HybridRecommender:
    """
    Sistem hibrid de recomandare
    """

    def __init__(self, data_dir: str = '../data/processed', test_size: float = 0.2):
        self.data_dir = Path(data_dir)
        self.test_size = test_size

        # Date
        self.books_df = None
        self.users_df = None
        self.ratings_df = None
        self.train_df = None
        self.test_df = None
        self.user_item_matrix = None
        self.popularity = {}

        # Embeddings (.npy)
        self.book_embeddings = None      # numpy array [n_books, embedding_dim]
        self.user_embeddings = None      # numpy array [n_users, embedding_dim]
        self.isbn_to_idx = {}            # ISBN -> index in book_embeddings
        self.user_to_idx = {}            # User-ID -> index in user_embeddings
        self.bert_model = None           # pentru encoding preferinte noi

        # CF Models
        self.svd_model = None
        self.knn_user_model = None
        self.knn_item_model = None

        # Scoring weights
        self.weights = SCORING_WEIGHTS.copy()

        # Load data
        self._load_data()

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def _load_data(self):
        """Incarca toate datele necesare"""
        print("\n" + "="*50)
        print("INCARCARE DATE")
        print("="*50)

        self.books_df = self._load_books()
        self.users_df = self._load_users()
        self.ratings_df = self._load_ratings()
        self._create_train_test_split()
        self._build_user_item_matrix()
        self._compute_popularity()

    def _load_books(self) -> pd.DataFrame:
        """Incarca books_enriched.csv"""
        books_path = self.data_dir / 'books_enriched.csv'
        if not books_path.exists():
            books_path = self.data_dir / 'books_processed.csv'

        if not books_path.exists():
            raise FileNotFoundError(f"Nu am gasit fisierul cu carti in {self.data_dir}")

        books = pd.read_csv(books_path)
        books['ISBN'] = books['ISBN'].astype(str)

        # Creeaza mapping ISBN -> index
        self.isbn_to_idx = {isbn: idx for idx, isbn in enumerate(books['ISBN'])}

        print(f"Books loaded: {len(books)}")
        return books

    def _load_users(self) -> Optional[pd.DataFrame]:
        """Incarca users"""
        users_path = self.data_dir / 'users_enriched.csv'
        if not users_path.exists():
            users_path = self.data_dir / 'users_processed.csv'

        if users_path.exists():
            users = pd.read_csv(users_path)

            # Creeaza mapping User-ID -> index
            self.user_to_idx = {uid: idx for idx, uid in enumerate(users['User-ID'])}

            print(f"Users loaded: {len(users)}")
            return users

        print("Users file not found")
        return None

    def _load_ratings(self) -> pd.DataFrame:
        """Incarca ratings"""
        ratings_path = self.data_dir / 'ratings_processed.csv'

        if not ratings_path.exists():
            raise FileNotFoundError(f"Nu am gasit {ratings_path}")

        ratings = pd.read_csv(ratings_path)
        ratings['ISBN'] = ratings['ISBN'].astype(str)

        # Doar rating-uri explicite
        ratings = ratings[ratings['Rating'] > 0]
        print(f"Ratings loaded (explicit): {len(ratings)}")

        # Filtram ISBN-uri valide
        valid_isbns = set(self.books_df['ISBN'])
        ratings = ratings[ratings['ISBN'].isin(valid_isbns)]
        print(f"Ratings after ISBN filter: {len(ratings)}")

        return ratings

    def _create_train_test_split(self):
        """Creeaza train/test split"""
        self.train_df, self.test_df = train_test_split(
            self.ratings_df,
            test_size=self.test_size,
            random_state=42
        )
        print(f"Train: {len(self.train_df)}, Test: {len(self.test_df)}")

    def _build_user_item_matrix(self):
        """Construieste matricea User-Item"""
        print("Building User-Item matrix...")
        self.user_item_matrix = self.train_df.pivot(
            index='User-ID',
            columns='ISBN',
            values='Rating'
        ).fillna(0)
        print(f"Matrix shape: {self.user_item_matrix.shape}")

    def _compute_popularity(self):
        """Calculeaza popularitatea cartilor"""
        stats = self.ratings_df.groupby('ISBN').agg({
            'Rating': ['count', 'mean']
        }).reset_index()
        stats.columns = ['ISBN', 'count', 'mean']

        C = stats['mean'].mean()
        m = stats['count'].quantile(0.7)

        stats['popularity'] = (stats['count'] * stats['mean'] + m * C) / (stats['count'] + m)
        self.popularity = stats.set_index('ISBN')['popularity'].to_dict()
        print(f"Popularity computed: {len(self.popularity)} books")

    # =========================================================================
    # EMBEDDINGS (.npy files)
    # =========================================================================

    def load_embeddings(self,
                        books_emb_path: str = None,
                        users_emb_path: str = None) -> bool:
        """
        Incarca embeddings din fisiere .npy

        Args:
            books_emb_path: Calea catre books_embeddings.npy
            users_emb_path: Calea catre user_embeddings.npy
        """
        print("\n" + "="*50)
        print("INCARCARE EMBEDDINGS")
        print("="*50)

        # Books embeddings
        if books_emb_path is None:
            books_emb_path = self.data_dir / 'books_embeddings.npy'
        else:
            books_emb_path = Path(books_emb_path)

        if books_emb_path.exists():
            self.book_embeddings = np.load(books_emb_path)
            print(f"Book embeddings loaded: {self.book_embeddings.shape}")
        else:
            print(f"Book embeddings not found: {books_emb_path}")

        # User embeddings
        if users_emb_path is None:
            users_emb_path = self.data_dir / 'user_embeddings.npy'
        else:
            users_emb_path = Path(users_emb_path)

        if users_emb_path.exists():
            self.user_embeddings = np.load(users_emb_path)
            print(f"User embeddings loaded: {self.user_embeddings.shape}")
        else:
            print(f"User embeddings not found: {users_emb_path}")

        return self.book_embeddings is not None

    def load_bert_model(self, model_name: str = 'all-MiniLM-L6-v2') -> bool:
        """Incarca modelul BERT pentru encoding preferinte noi"""
        if not BERT_AVAILABLE:
            print("BERT not available")
            return False

        print(f"Loading BERT model: {model_name}")
        self.bert_model = SentenceTransformer(model_name)
        print("BERT model loaded!")
        return True

    def _get_bert_similarity_scores(self, user_profile: Dict,
                                    candidate_isbns: List[str]) -> Dict[str, float]:
        """
        Calculeaza similaritatea BERT intre preferintele user-ului si carti
        Foloseste embeddings pre-calculate
        """
        if self.book_embeddings is None:
            return {}

        # Daca avem model BERT, cream embedding din preferinte
        if self.bert_model is not None:
            user_embedding = self._create_user_preference_embedding(user_profile)
        else:
            # Fallback: folosim media embeddings-urilor cartilor citite
            user_embedding = self._create_user_embedding_from_history(user_profile)

        if user_embedding is None:
            return {}

        # Calculeaza similaritate cosinus cu toate cartile candidate
        similarities = {}

        for isbn in candidate_isbns:
            if isbn in self.isbn_to_idx:
                idx = self.isbn_to_idx[isbn]
                if idx < len(self.book_embeddings):
                    book_emb = self.book_embeddings[idx]

                    # Cosine similarity
                    sim = np.dot(user_embedding, book_emb) / (
                        np.linalg.norm(user_embedding) * np.linalg.norm(book_emb) + 1e-8
                    )
                    similarities[isbn] = float(sim)

        return similarities

    def _create_user_preference_embedding(self, user_profile: Dict) -> Optional[np.ndarray]:
        """Creeaza embedding din preferintele text ale user-ului"""
        if self.bert_model is None:
            return None

        parts = []

        genres = user_profile.get('favorite_genres', [])
        if genres:
            parts.append(f"I enjoy reading {', '.join(genres)} books.")

        subjects = user_profile.get('favorite_subjects', [])
        if subjects:
            parts.append(f"Topics I like: {', '.join(subjects)}.")

        # Titlurile cartilor apreciate
        read_history = user_profile.get('user_read_history', [])
        liked_titles = []
        for isbn, rating in read_history:
            if rating >= 4:
                book = self.books_df[self.books_df['ISBN'] == str(isbn)]
                if not book.empty:
                    liked_titles.append(book.iloc[0]['Title'])

        if liked_titles:
            parts.append(f"Books I loved: {', '.join(liked_titles[:5])}")

        if not parts:
            return self._create_user_embedding_from_history(user_profile)

        user_text = " ".join(parts)
        return self.bert_model.encode([user_text])[0]

    def _create_user_embedding_from_history(self, user_profile: Dict) -> Optional[np.ndarray]:
        """
        Creeaza user embedding ca media embeddings-urilor cartilor citite
        (pentru cazul fara model BERT)
        """
        if self.book_embeddings is None:
            return None

        read_history = user_profile.get('user_read_history', [])
        if not read_history:
            return None

        embeddings = []
        weights = []

        for isbn, rating in read_history:
            isbn = str(isbn)
            if isbn in self.isbn_to_idx:
                idx = self.isbn_to_idx[isbn]
                if idx < len(self.book_embeddings):
                    embeddings.append(self.book_embeddings[idx])
                    weights.append(rating)  # Ponderam cu rating-ul

        if not embeddings:
            return None

        # Media ponderata
        embeddings = np.array(embeddings)
        weights = np.array(weights)
        weights = weights / weights.sum()

        return np.average(embeddings, axis=0, weights=weights)

    # =========================================================================
    # COLLABORATIVE FILTERING
    # =========================================================================

    def train_cf_models(self, n_factors: int = 50, k: int = 20):
        """Antreneaza modelele CF"""
        print("\n" + "="*50)
        print("ANTRENARE MODELE CF")
        print("="*50)

        # SVD
        print(f"\nTraining SVD (factors={n_factors})...")
        self.svd_model = CustomSVD(n_factors=n_factors)
        self.svd_model.fit(self.user_item_matrix)

        # KNN User-based
        print(f"\nTraining User-KNN (k={k})...")
        self.knn_user_model = CustomKNN(k=k, method='user')
        self.knn_user_model.fit(self.user_item_matrix)

        # KNN Item-based
        print(f"\nTraining Item-KNN (k={k})...")
        self.knn_item_model = CustomKNN(k=k, method='item')
        self.knn_item_model.fit(self.user_item_matrix)

        print("\nCF models trained!")

    def save_cf_models(self, path: str = None):
        """Salveaza modelele CF"""
        if path is None:
            path = self.data_dir / 'cf_models.pkl'

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'svd': self.svd_model,
                'knn_user': self.knn_user_model,
                'knn_item': self.knn_item_model
            }, f)
        print(f"CF models saved to {path}")

    def load_cf_models(self, path: str = None) -> bool:
        """Incarca modelele CF"""
        if path is None:
            path = self.data_dir / 'cf_models.pkl'

        if not Path(path).exists():
            print(f"CF models not found: {path}")
            return False

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.svd_model = data.get('svd')
        self.knn_user_model = data.get('knn_user')
        self.knn_item_model = data.get('knn_item')

        print(f"CF models loaded from {path}")
        return True

    def _get_cf_score(self, user_history: List[Tuple], candidate_isbn: str) -> float:
        """Calculeaza scorul CF pentru o carte"""
        if not user_history or self.knn_item_model is None:
            return 0.5

        liked = [(isbn, r) for isbn, r in user_history if r >= 4]
        if not liked:
            return 0.5

        total_score = 0
        total_weight = 0

        for liked_isbn, rating in liked:
            liked_isbn = str(liked_isbn)
            candidate_isbn = str(candidate_isbn)

            if (liked_isbn in self.knn_item_model.item_mapping and
                candidate_isbn in self.knn_item_model.item_mapping):

                i_idx = self.knn_item_model.item_mapping[candidate_isbn]
                j_idx = self.knn_item_model.item_mapping[liked_isbn]

                if (i_idx < len(self.knn_item_model.similarity_matrix) and
                    j_idx < len(self.knn_item_model.similarity_matrix)):

                    sim = self.knn_item_model.similarity_matrix[i_idx, j_idx]
                    if sim > 0:
                        total_score += sim * rating
                        total_weight += sim

        if total_weight > 0:
            return (total_score / total_weight) / 10

        return 0.5

    # =========================================================================
    # SCORING
    # =========================================================================

    def _calculate_genre_match(self, book_genres: str, user_genres: List[str]) -> float:
        if not user_genres or pd.isna(book_genres):
            return 0.0

        book_set = set(g.strip().lower() for g in str(book_genres).split(','))
        user_set = set(g.lower() for g in user_genres)

        matches = len(book_set & user_set)
        return min(matches / max(len(user_set), 1), 1.0)

    def _calculate_subject_match(self, book_subjects: str, user_subjects: List[str]) -> float:
        if not user_subjects or pd.isna(book_subjects):
            return 0.0

        book_set = set(s.strip().lower() for s in str(book_subjects).split(','))
        user_set = set(s.lower() for s in user_subjects)

        matches = len(book_set & user_set)
        return min(matches / max(len(user_set), 1), 1.0)

    def _calculate_recency_score(self, year) -> float:
        if pd.isna(year):
            return 0.5

        age = 2025 - int(year)

        if age <= 5:
            return 1.0
        elif age <= 10:
            return 0.8
        elif age <= 20:
            return 0.6
        elif age <= 50:
            return 0.4
        else:
            return 0.2

    def _calculate_final_score(self, row, user_profile: Dict,
                               bert_scores: Dict, cf_scores: Dict) -> float:
        """Calculeaza scorul final hibrid"""
        isbn = str(row['ISBN'])

        bert_sim = bert_scores.get(isbn, 0.5)
        cf_score = cf_scores.get(isbn, 0.5)

        genre_match = self._calculate_genre_match(
            row.get('genres', ''),
            user_profile.get('favorite_genres', [])
        )

        subject_match = self._calculate_subject_match(
            row.get('subjects', ''),
            user_profile.get('favorite_subjects', [])
        )

        pop = self.popularity.get(isbn, 0)
        max_pop = max(self.popularity.values()) if self.popularity else 1
        pop_norm = pop / max_pop if max_pop > 0 else 0.5

        recency = self._calculate_recency_score(row.get('Year'))

        final_score = (
            self.weights['bert_similarity'] * bert_sim +
            self.weights['cf_score'] * cf_score +
            self.weights['genre_match'] * genre_match +
            self.weights['subject_match'] * subject_match +
            self.weights['popularity'] * pop_norm +
            self.weights['recency'] * recency
        )

        return final_score

    # =========================================================================
    # MAIN RECOMMENDATION
    # =========================================================================

    def get_recommendations(self, user_profile: Dict, n: int = 12) -> List[Dict]:
        """Genereaza recomandari hibride"""
        if self.books_df is None or self.books_df.empty:
            return []

        genres = user_profile.get('favorite_genres', [])
        subjects = user_profile.get('favorite_subjects', [])
        year_range = user_profile.get('preferred_year_range')
        language = user_profile.get('language')
        read_history = user_profile.get('user_read_history', [])

        exclude_isbns = set(str(isbn) for isbn, _ in read_history)

        candidates = self.books_df.copy()

        if year_range:
            candidates = candidates[
                (candidates['Year'] >= year_range[0]) &
                (candidates['Year'] <= year_range[1])
            ]

        if language and 'language' in candidates.columns:
            candidates = candidates[candidates['language'] == language]

        candidates = candidates[~candidates['ISBN'].isin(exclude_isbns)]

        if candidates.empty:
            return []

        candidate_isbns = candidates['ISBN'].tolist()

        print("Calculating scores...")

        # BERT scores
        bert_scores = self._get_bert_similarity_scores(user_profile, candidate_isbns)

        # CF scores
        cf_scores = {}
        for isbn in candidate_isbns:
            cf_scores[isbn] = self._get_cf_score(read_history, isbn)

        # Final score
        candidates['final_score'] = candidates.apply(
            lambda row: self._calculate_final_score(row, user_profile, bert_scores, cf_scores),
            axis=1
        )

        candidates = candidates.sort_values('final_score', ascending=False).head(n)

        placeholder = "https://via.placeholder.com/150x220?text=No+Cover"
        results = []

        for _, row in candidates.iterrows():
            image_url = row.get('image_url', placeholder)
            if pd.isna(image_url) or not image_url:
                image_url = placeholder

            results.append({
                'isbn': str(row['ISBN']),
                'title': row['Title'],
                'author': row['Author'],
                'year': int(row['Year']) if pd.notna(row.get('Year')) else 'N/A',
                'genres': row.get('genres', 'N/A') if pd.notna(row.get('genres')) else 'N/A',
                'image_url': image_url,
                'score': round(row['final_score'], 3)
            })

        return results

    def get_book_by_isbn(self, isbn: str) -> Optional[Dict]:
        """Gaseste o carte dupa ISBN"""
        if self.books_df is None:
            return None

        book = self.books_df[self.books_df['ISBN'] == str(isbn)]
        if book.empty:
            return None

        row = book.iloc[0]
        placeholder = "https://via.placeholder.com/150x220?text=No+Cover"
        image_url = row.get('image_url', placeholder)
        if pd.isna(image_url):
            image_url = placeholder

        return {
            'isbn': str(row['ISBN']),
            'title': row['Title'],
            'author': row['Author'],
            'year': int(row['Year']) if pd.notna(row.get('Year')) else 'N/A',
            'genres': row.get('genres', 'N/A') if pd.notna(row.get('genres')) else 'N/A',
            'image_url': image_url
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_hybrid_recommender(
    data_dir: str = '../data/processed',
    books_emb_path: str = None,
    users_emb_path: str = None,
    load_bert: bool = False,
    load_cf: bool = True
) -> HybridRecommender:
    """
    Creeaza si initializeaza sistemul de recomandare

    Args:
        data_dir: Directorul cu datele procesate
        books_emb_path: Calea catre books_embeddings.npy (optional)
        users_emb_path: Calea catre user_embeddings.npy (optional)
        load_bert: Daca sa incarce modelul BERT pentru query-uri noi
        load_cf: Daca sa antreneze/incarce modele CF

    Usage:
        recommender = create_hybrid_recommender(
            data_dir='../data/processed',
            books_emb_path='../data/processed/books_embeddings.npy',
            users_emb_path='../data/processed/user_embeddings.npy'
        )
        recs = recommender.get_recommendations(user_profile)
    """
    recommender = HybridRecommender(data_dir=data_dir)

    # Embeddings .npy
    recommender.load_embeddings(books_emb_path, users_emb_path)

    # BERT (optional)
    if load_bert and BERT_AVAILABLE:
        recommender.load_bert_model()

    # CF
    if load_cf:
        if not recommender.load_cf_models():
            print("Training CF models...")
            recommender.train_cf_models()
            recommender.save_cf_models()

    return recommender


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST HYBRID RECOMMENDER")
    print("="*60)

    # Initializare cu embeddings .npy
    recommender = create_hybrid_recommender(
        data_dir='../data/processed',
        books_emb_path='../data/embeddings/book_embeddings.npy',
        users_emb_path='../data/embeddings/user_embedding.npy',
        load_bert=False,  # Nu avem nevoie de BERT daca avem .npy
        load_cf=True
    )

    # Test profile
    test_profile = {
        "favorite_genres": ["Fiction", "Mystery & Detective", "Thrillers"],
        "favorite_subjects": ["Suspense", "Psychological"],
        "preferred_year_range": (1990, 2005),
        "age": 30,
        "language": "en",
        "user_read_history": []
    }

    print("\n--- Test Profile ---")
    print(f"Genres: {test_profile['favorite_genres']}")
    print(f"Subjects: {test_profile['favorite_subjects']}")

    # Recomandari
    print("\n--- Generating Recommendations ---")
    recommendations = recommender.get_recommendations(test_profile, n=10)

    print("\n--- Top 10 Recommendations ---")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} ({rec['year']}) - Score: {rec['score']}")