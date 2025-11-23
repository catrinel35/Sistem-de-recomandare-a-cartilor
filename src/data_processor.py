import pandas as pd
import numpy as np
from pathlib import Path


class BookCrossingDataLoader:

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)
        self.books = None
        self.users = None
        self.ratings = None

    def load_books(self, filename='Books.csv'):

        filepath = self.data_dir / filename

        try:
            books = pd.read_csv(filepath,
                                sep=';',
                                encoding='latin-1',
                                on_bad_lines='skip',
                                low_memory=False)
        except:
            books = pd.read_csv(filepath,
                                encoding='latin-1',
                                on_bad_lines='skip')

        books.columns = books.columns.str.strip()

        if 'Year' in books.columns:
            books['Year'] = pd.to_numeric(
                books['Year'],
                errors='coerce'
            )
            books = books[
                (books['Year'] > 1000) &
                (books['Year'] <= 2025)
                ]

        if 'ISBN' in books.columns:
            books['ISBN'] = books['ISBN'].astype(str).str.strip()
            books = books.drop_duplicates(subset=['ISBN'], keep='first')

        if 'Title' in books.columns:
            books['Title'] = books['Title'].str.strip()
            books = books[books['Title'].notna()]

        if 'Author' in books.columns:
            books['Author'] = books['Author'].str.strip()

        self.books = books
        print(f"✓ Încărcat {len(books)} cărți")
        return books

    def load_users(self, filename='Users.csv'):

        filepath = self.data_dir / filename

        try:
            users = pd.read_csv(filepath,
                                sep=';',
                                encoding='latin-1',
                                on_bad_lines='skip')
        except:
            users = pd.read_csv(filepath,
                                encoding='latin-1',
                                on_bad_lines='skip')

        users.columns = users.columns.str.strip()

        if 'Age' in users.columns:
            users['Age'] = pd.to_numeric(users['Age'], errors='coerce')
            users.loc[(users['Age'] < 5) | (users['Age'] > 100), 'Age'] = np.nan

        self.users = users
        print(f"✓ Încărcat {len(users)} utilizatori")
        return users

    def load_ratings(self, filename='Ratings.csv'):

        filepath = self.data_dir / filename

        try:
            ratings = pd.read_csv(filepath,
                                  sep=';',
                                  encoding='latin-1',
                                  on_bad_lines='skip')
        except:
            ratings = pd.read_csv(filepath,
                                  encoding='latin-1',
                                  on_bad_lines='skip')

        ratings.columns = ratings.columns.str.strip()

        if 'ISBN' in ratings.columns:
            ratings['ISBN'] = ratings['ISBN'].astype(str).str.strip()

        if 'Rating' in ratings.columns:
            ratings['Rating'] = pd.to_numeric(
                ratings['Rating'],
                errors='coerce'
            )
            ratings = ratings[
                (ratings['Rating'] >= 0) &
                (ratings['Rating'] <= 10)
                ]

        self.ratings = ratings
        print(f"✓ Încărcat {len(ratings)} rating-uri")
        return ratings

    def load_all(self):
        self.load_books()
        self.load_users()
        self.load_ratings()
        return self.books, self.users, self.ratings

    def filter_dataset(self, min_book_ratings=5, min_user_ratings=3):

        if self.ratings is None:
            raise ValueError("Trebuie să încarci mai întâi ratings!")

        print(f"\nDimensiune inițială: {len(self.ratings)} rating-uri")

        prev_size = 0
        current_size = len(self.ratings)
        iteration = 0

        while prev_size != current_size:
            iteration += 1
            prev_size = current_size

            book_counts = self.ratings['ISBN'].value_counts()
            valid_books = book_counts[book_counts >= min_book_ratings].index
            self.ratings = self.ratings[self.ratings['ISBN'].isin(valid_books)]

            user_counts = self.ratings['User-ID'].value_counts()
            valid_users = user_counts[user_counts >= min_user_ratings].index
            self.ratings = self.ratings[self.ratings['User-ID'].isin(valid_users)]

            current_size = len(self.ratings)
            print(f"  Iterația {iteration}: {current_size} rating-uri")

        if self.books is not None:
            valid_isbns = self.ratings['ISBN'].unique()
            self.books = self.books[self.books['ISBN'].isin(valid_isbns)]
            print(f"✓ Cărți rămase: {len(self.books)}")

        if self.users is not None:
            valid_user_ids = self.ratings['User-ID'].unique()
            self.users = self.users[self.users['User-ID'].isin(valid_user_ids)]
            print(f"✓ Utilizatori rămași: {len(self.users)}")

        print(f"✓ Rating-uri finale: {len(self.ratings)}")

        return self.books, self.users, self.ratings

    def create_merged_dataset(self):

        if any(df is None for df in [self.books, self.users, self.ratings]):
            raise ValueError("Trebuie să încarci toate cele 3 fișiere!")

        merged = self.ratings.merge(
            self.books[['ISBN', 'Title', 'Author', 'Year', 'Publisher']],
            on='ISBN',
            how='left'
        )

        merged = merged.merge(
            self.users[['User-ID', 'Age']],
            on='User-ID',
            how='left'
        )

        print(f"✓ Dataset unificat: {len(merged)} înregistrări")
        return merged

    def get_statistics(self):

        if self.ratings is None:
            print("Încarcă mai întâi datele!")
            return

        print("\n" + "=" * 50)
        print("STATISTICI DATASET")
        print("=" * 50)

        if self.books is not None:
            print(f"\nCărți: {len(self.books)}")
            if 'Book-Author' in self.books.columns:
                print(f"  Autori unici: {self.books['Author'].nunique()}")
            if 'Year-Of-Publication' in self.books.columns:
                print(f"  An mediu publicare: {self.books['Year-Of-Publication'].mean():.0f}")

        if self.users is not None:
            print(f"\nUtilizatori: {len(self.users)}")
            if 'Age' in self.users.columns:
                print(f"  Vârstă medie: {self.users['Age'].mean():.1f}")
            if 'Country' in self.users.columns:
                top_countries = self.users['Country'].value_counts().head(5)
                print(f"  Top 5 țări: {', '.join(top_countries.index.tolist())}")

        if self.ratings is not None:
            print(f"\nRating-uri: {len(self.ratings)}")
            print(f"  Rating mediu: {self.ratings['Rating'].mean():.2f}")
            print(f"  Rating-uri explicite (>0): {(self.ratings['Rating'] > 0).sum()}")
            print(f"  Rating-uri implicite (=0): {(self.ratings['Rating'] == 0).sum()}")

            n_users = self.ratings['User-ID'].nunique()
            n_books = self.ratings['ISBN'].nunique()
            sparsity = 1 - (len(self.ratings) / (n_users * n_books))
            print(f"\n  Sparsity matrice: {sparsity:.4f} ({sparsity * 100:.2f}%)")
            print(f"  Rating-uri/utilizator: {len(self.ratings) / n_users:.1f}")
            print(f"  Rating-uri/carte: {len(self.ratings) / n_books:.1f}")

        print("=" * 50 + "\n")

    def save_processed(self, output_dir='data/processed'):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.books is not None:
            self.books.to_csv(output_path / 'books_processed.csv', index=False)
            print(f"✓ Salvat books_processed.csv")

        if self.users is not None:
            self.users.to_csv(output_path / 'users_processed.csv', index=False)
            print(f"✓ Salvat users_processed.csv")

        if self.ratings is not None:
            self.ratings.to_csv(output_path / 'ratings_processed.csv', index=False)
            print(f"✓ Salvat ratings_processed.csv")

if __name__ == "__main__":
    loader = BookCrossingDataLoader(data_dir='../data')
    books, users, ratings = loader.load_all()
    print("Coloane Books:", books.columns.tolist())
    print("Coloane Users:", users.columns.tolist())
    print("Coloane Ratings:", ratings.columns.tolist())

    loader.get_statistics()

    books, users, ratings = loader.filter_dataset(
        min_book_ratings=10,  # Fiecare carte să aibă min 10 rating-uri
        min_user_ratings=5  # Fiecare user să aibă min 5 rating-uri
    )

    loader.get_statistics()

    loader.save_processed(output_dir='../data/processed')

    merged_df = loader.create_merged_dataset()
    merged_df.to_csv('../data/processed/merged_dataset.csv', index=False)