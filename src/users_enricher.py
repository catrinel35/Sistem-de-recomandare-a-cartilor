import logging
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserProfileEnricher:
    """
    Îmbogățește profilurile utilizatorilor cu features comportamentale
    """

    def __init__(self, data_dir='../data/processed'):
        self.data_dir = Path(data_dir)
        self.users = None
        self.ratings = None
        self.books = None
        self.merged = None

    def load_data(self):
        """Încarcă datele procesate"""
        self.users = pd.read_csv(self.data_dir / 'users_processed.csv')
        self.ratings = pd.read_csv(self.data_dir / 'ratings_processed.csv')
        self.books = pd.read_csv(self.data_dir / 'books_processed.csv')

        # Merge pentru analiză
        self.merged = self.ratings.merge(
            self.books[['ISBN', 'Title', 'Author', 'Year', 'Publisher']],
            on='ISBN'
        )

        logger.info(f"✓ Date încărcate: {len(self.users)} users, {len(self.ratings)} ratings")

    def create_behavioral_features(self):
        """Creează features comportamentale pentru fiecare utilizator"""

        logger.info("Creare features comportamentale...")

        # Agreg statistici per utilizator
        user_stats = self.ratings.groupby('User-ID').agg({
            'Rating': ['mean', 'std', 'count', lambda x: (x > 0).sum()],
            'ISBN': 'nunique'
        }).reset_index()

        user_stats.columns = ['User-ID', 'avg_rating', 'rating_std',
                              'total_ratings', 'explicit_ratings', 'unique_books']

        # Fill NaN std (useri cu 1 singur rating)
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)

        # Diversitate autori
        author_div = self.merged.groupby('User-ID')['Author'].nunique().reset_index()
        author_div.columns = ['User-ID', 'author_diversity']
        user_stats = user_stats.merge(author_div, on='User-ID')

        # An mediu cărți citite (recency)
        year_avg = self.merged.groupby('User-ID')['Year'].mean().reset_index()
        year_avg.columns = ['User-ID', 'avg_book_year']
        user_stats = user_stats.merge(year_avg, on='User-ID')

        # Recency score (cât de noi sunt cărțile)
        user_stats['recency_score'] = 2024 - user_stats['avg_book_year']

        # Explicit ratio
        user_stats['explicit_ratio'] = user_stats['explicit_ratings'] / user_stats['total_ratings']

        # Exploration vs Exploitation
        # Dacă author_diversity / unique_books e mare → explorează
        user_stats['exploration_score'] = user_stats['author_diversity'] / user_stats['unique_books']

        # Merge înapoi cu users
        self.users = self.users.merge(user_stats, on='User-ID', how='left')

        logger.info(f" Features create: {user_stats.shape[1] - 1} noi coloane")

    def prepare_age_for_clustering(self):
        """Pregătește vârsta pentru clustering (fill cu mediană)"""

        logger.info("Pregătire câmp Age pentru clustering...")

        # Folosește Age original, fill cu mediană doar pentru clustering
        self.users['Age_Final'] = self.users['Age'].fillna(self.users['Age'].median())

        logger.info(f" Age pregătit (median fill: {self.users['Age'].median():.1f})")

    def cluster_users(self, n_clusters=5):
        """Clusterizare utilizatori pe bază de comportament"""

        logger.info(f"Clusterizare utilizatori în {n_clusters} grupuri...")

        # Features pentru clustering
        feature_cols = ['avg_rating', 'total_ratings', 'author_diversity',
                        'recency_score', 'explicit_ratio', 'exploration_score', 'Age_Final']

        # Pregătește date
        df_cluster = self.users[feature_cols].copy()
        df_cluster.fillna(df_cluster.median(), inplace=True)

        # Standardizare
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_cluster)

        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.users['user_cluster'] = kmeans.fit_predict(X_scaled)

        # Analizează clusterele
        print("\n" + "=" * 60)
        print("ANALIZA CLUSTERELOR UTILIZATORI")
        print("=" * 60)

        for cluster_id in range(n_clusters):
            cluster_data = self.users[self.users['user_cluster'] == cluster_id]

            print(f"\n🔹 Cluster {cluster_id}:")
            print(f"   Size: {len(cluster_data)} utilizatori ({len(cluster_data) / len(self.users) * 100:.1f}%)")
            print(f"   Avg Age: {cluster_data['Age_Final'].mean():.1f} ani")
            print(f"   Avg Ratings: {cluster_data['total_ratings'].mean():.1f}")
            print(f"   Avg Diversity: {cluster_data['author_diversity'].mean():.1f} autori")
            print(f"   Recency Score: {cluster_data['recency_score'].mean():.1f}")
            print(f"   Explicit Ratio: {cluster_data['explicit_ratio'].mean():.2f}")

            # Etichetă cluster
            if cluster_data['total_ratings'].mean() > 50:
                label = "Power Readers"
            elif cluster_data['recency_score'].mean() < 15:
                label = "Classic Lovers"
            elif cluster_data['recency_score'].mean() > 25:
                label = "Trend Followers"
            elif cluster_data['exploration_score'].mean() > 0.7:
                label = "Explorers"
            else:
                label = "Casual Readers"

            print(f" Label sugerat: {label}")

        print("=" * 60 + "\n")

        logger.info(f"✓ Clusterizare completă")

    def find_similar_users(self, top_n=5):
        """Găsește cei mai similari N utilizatori pentru fiecare"""

        logger.info("Calculare similarități între utilizatori...")

        # Crează matrice user-item
        user_item = self.ratings.pivot_table(
            index='User-ID',
            columns='ISBN',
            values='Rating',
            fill_value=0
        )

        # Calculează similaritate cosinus
        similarity_matrix = cosine_similarity(user_item)

        # Pentru fiecare user, găsește top N similari
        similar_users = {}
        for i, user_id in enumerate(user_item.index):
            # Sortează și ia top N (exclude pe sine - index 0)
            similar_indices = similarity_matrix[i].argsort()[::-1][1:top_n + 1]
            similar_ids = user_item.index[similar_indices].tolist()
            similar_scores = similarity_matrix[i][similar_indices].tolist()

            similar_users[user_id] = list(zip(similar_ids, similar_scores))

        # Salvează ca JSON sau pickle pentru refolosire
        import json
        output_path = self.data_dir / 'user_similarities.json'

        similar_users_serializable = {
            int(k): [(int(uid), float(score)) for uid, score in v]
            for k, v in similar_users.items()
        }

        with open(output_path, 'w') as f:
            json.dump(similar_users_serializable, f)

        logger.info(f"Similarități calculate și salvate în {output_path}")

        return similar_users

    def enrich_all(self, n_clusters=5):
        """Pipeline complet de îmbogățire"""

        print("\n" + "=" * 60)
        print("ÎMBOGĂȚIRE PROFILURI UTILIZATORI")
        print("=" * 60 + "\n")

        self.load_data()
        self.create_behavioral_features()
        self.prepare_age_for_clustering()
        self.cluster_users(n_clusters=n_clusters)
        self.find_similar_users(top_n=5)

        # Salvează rezultatul
        output_path = self.data_dir / 'users_enriched.csv'
        self.users.to_csv(output_path, index=False)

        print(f"\nÎmbogățire completă!")
        print(f"Salvat: {output_path}")
        print(f"Coloane noi: {len(self.users.columns) - 3}")  # minus cele originale

        return self.users

    def get_statistics(self):
        """Statistici despre utilizatori îmbogățiți"""

        print("\n" + "=" * 60)
        print("STATISTICI UTILIZATORI ÎMBOGĂȚIȚI")
        print("=" * 60)

        print(f"\nTotal utilizatori: {len(self.users)}")
        print(
            f"Cu vârstă: {self.users['Age'].notna().sum()} ({self.users['Age'].notna().sum() / len(self.users) * 100:.1f}%)")
        print(
            f"Fără vârstă: {self.users['Age'].isna().sum()} ({self.users['Age'].isna().sum() / len(self.users) * 100:.1f}%)")

        print(f"\nDistribuție clustere:")
        print(self.users['user_cluster'].value_counts().sort_index())

        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Creează enricher
    enricher = UserProfileEnricher(data_dir='../data/processed')

    # Rulează pipeline complet
    users_enriched = enricher.enrich_all(n_clusters=5)

    # Statistici
    enricher.get_statistics()

    # avg_rating - cât de generos e userul rating_std - consistența evaluărilor
    # total_ratings - cât de activ e
    # explicit_ratings - câte rating - uri explicite( > 0)
    # unique_books - diversitate cărți
    # author_diversity - câți autori diferiți
    # avg_book_year - preferință pentru cărți noi / vechi
    # recency_score - 2024 - avg_book_year
    # explicit_ratio - %rating - uri explicite
    # exploration_score - raport explorare
    # user_cluster - clusterul 0 - 4

    # Exemplu: Vezi un utilizator îmbogățit
    print("\nExemplu utilizator îmbogățit:")
    print(users_enriched.iloc[0][['User-ID', 'Age',
                                  'total_ratings', 'author_diversity',
                                  'user_cluster', 'exploration_score']])
