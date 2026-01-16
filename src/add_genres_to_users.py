from collections import Counter
from pathlib import Path

import pandas as pd


def extract_user_top_genres(data_dir='../data/processed'):
    """
    Adaugă coloana 'top_genres' în users_enriched.csv
    cu top 3 genuri pentru fiecare utilizator
    """

    data_path = Path(data_dir)

    print("Încărcare date...")

    # Încarcă users enriched
    users = pd.read_csv(data_path / 'users_enriched.csv')
    print(f" Users: {len(users)}")

    # Încarcă ratings
    ratings = pd.read_csv(data_path / 'ratings_processed.csv')
    print(f" Ratings: {len(ratings)}")

    # Încarcă books enriched (cu genuri din ISBNdb)
    books_enriched_path = data_path / 'books_enriched.csv'

    if not books_enriched_path.exists():
        print(" books_enriched.csv nu există!")
        print("   Folosesc books_processed.csv (fără genuri ISBNdb)")
        books = pd.read_csv(data_path / 'books_processed.csv')
    else:
        books = pd.read_csv(books_enriched_path)
        print(f" Books enriched: {len(books)}")

    # Verifică dacă există coloane cu genuri
    genre_cols = ['genres', 'Genre']  # posibile nume
    genre_col = None

    for col in genre_cols:
        if col in books.columns:
            genre_col = col
            print(f"Găsită coloană genuri: '{col}'")
            break

    if genre_col is None:
        print("⚠Nu există coloană cu genuri în books!")
        print("   Adaug coloană goală pentru viitor.")
        users['top_genres'] = None
        users.to_csv(data_path / 'users_enriched.csv', index=False)
        return users

    # Merge ratings cu books pentru a avea genuri
    print("\n🔗 Combinare ratings cu books...")
    merged = ratings.merge(
        books[['ISBN', genre_col]],
        on='ISBN',
        how='left'
    )

    print(f"✓ Merged: {len(merged)} rânduri")
    print(
        f"   Cu genuri: {merged[genre_col].notna().sum()} ({merged[genre_col].notna().sum() / len(merged) * 100:.1f}%)")

    # Calculează top genuri pentru fiecare user
    print("\n Calculare top 3 genuri per utilizator...")

    user_top_genres = {}

    for user_id in users['User-ID']:
        # Găsește cărțile citite de user
        user_ratings = merged[merged['User-ID'] == user_id]

        if len(user_ratings) == 0:
            user_top_genres[user_id] = None
            continue

        # Extrage toate genurile (split pe virgulă dacă sunt multiple)
        all_genres = []

        for genres_str in user_ratings[genre_col].dropna():
            if isinstance(genres_str, str):
                # Split pe virgulă sau alt separator
                genres = [g.strip() for g in genres_str.split(',')]
                all_genres.extend(genres)

        if not all_genres:
            user_top_genres[user_id] = None
            continue

        # Numără genurile
        genre_counts = Counter(all_genres)

        # Top 3 cele mai frecvente
        top_3 = genre_counts.most_common(3)
        top_genres_list = [genre for genre, count in top_3]

        # Salvează ca string separat prin virgulă
        user_top_genres[user_id] = ', '.join(top_genres_list)

    # Adaugă coloana în users DataFrame
    users['top_genres'] = users['User-ID'].map(user_top_genres)

    # Salvează
    output_path = data_path / 'users_enriched.csv'
    users.to_csv(output_path, index=False)

    print(f"\n Salvat: {output_path}")

    # Statistici
    print("\n" + "=" * 60)
    print("STATISTICI TOP GENURI")
    print("=" * 60)
    print(f"Total utilizatori: {len(users)}")
    print(
        f"Cu top genuri: {users['top_genres'].notna().sum()} ({users['top_genres'].notna().sum() / len(users) * 100:.1f}%)")
    print(f"Fără genuri: {users['top_genres'].isna().sum()}")

    # Arată câteva exemple
    print("\n Exemple top genuri utilizatori:")
    print("=" * 60)

    sample = users[users['top_genres'].notna()].head(10)
    for idx, row in sample.iterrows():
        print(f"User {row['User-ID']}: {row['top_genres']}")

    print("=" * 60 + "\n")

    # Genuri cele mai populare în general
    all_top_genres = []
    for genres_str in users['top_genres'].dropna():
        genres = [g.strip() for g in str(genres_str).split(',')]
        all_top_genres.extend(genres)

    if all_top_genres:
        genre_popularity = Counter(all_top_genres)
        print("\n Top 10 genuri cele mai populare:")
        print("=" * 60)
        for genre, count in genre_popularity.most_common(10):
            print(f"{genre[:50]:<50} {count:>5} utilizatori")
        print("=" * 60 + "\n")

    return users


if __name__ == "__main__":
    print("=" * 60)
    print("ADĂUGARE TOP 3 GENURI PENTRU FIECARE UTILIZATOR")
    print("=" * 60)

    # Adaugă coloana top_genres
    users = extract_user_top_genres(data_dir='../data/processed')

    print("\nProces complet!")
    print(" Fișier actualizat: users_enriched.csv")
    print(" Coloană nouă: 'top_genres' (top 3 genuri preferate)")
