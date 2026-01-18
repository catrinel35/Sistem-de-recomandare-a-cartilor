"""
=============================================================================
INTERFATA GRADIO - SISTEM DE RECOMANDARE CARTI
=============================================================================

Flow:
1. Tab 1: Selectare preferinte + adaugare carti citite dupa ISBN
2. Click "Gaseste Recomandari" -> Tab 2
3. Tab 2: Grid de carti cu imagini, titlu, autor, an, genuri

Backend format:
user_profile = {
    "favorite_genres": ["Fantasy", "Action & Adventure"],
    "favorite_subjects": ["Romance", "Espionage"],
    "preferred_year_range": (2000, 2025),
    "age": 35,
    "language": "en",
    "user_read_history": [(isbn, rating), ...]
}

Autor: Echipa Proiect
Data: 2025
=============================================================================
"""

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# =============================================================================
# CONFIGURARE
# =============================================================================

DATA_DIR = Path('../data/processed')

# 10 Genuri populare din dataset
AVAILABLE_GENRES = [
    "Fiction",
    "Romance",
    "Mystery & Detective",
    "Historical",
    "Thrillers",
    "Fantasy",
    "Science Fiction",
    "Horror",
    "Biography & Autobiography",
    "Classics"
]

# 10 Subjects populare din dataset
AVAILABLE_SUBJECTS = [
    "Suspense",
    "Contemporary",
    "Literary",
    "Action & Adventure",
    "Women",
    "Psychological",
    "Family Life",
    "Juvenile Fiction",
    "Espionage",
    "Sagas"
]

# Limbi disponibile
AVAILABLE_LANGUAGES = {
    'en': 'English',
    'de': 'German',
    'es': 'Spanish',
    'fr': 'French',
    'it': 'Italian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'nl': 'Dutch'
}

# Year ranges
YEAR_RANGES = {
    '1920-1989': (1920, 1989),
    '1990-2005': (1990, 2005)
}

# Placeholder image pentru carti fara imagine
PLACEHOLDER_IMAGE = "https://via.placeholder.com/150x220?text=No+Cover"


# =============================================================================
# SISTEM DE RECOMANDARE
# =============================================================================

class BookRecommender:
    """Sistem de recomandare carti"""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.books = None
        self.ratings = None
        self.load_data()

    def load_data(self):
        """Incarca datele"""
        # Books
        books_path = self.data_dir / 'books_processed.csv'
        if books_path.exists():
            self.books = pd.read_csv(books_path)
            print(f"Books: {len(self.books)}")
        else:
            print(f"ATENTIE: {books_path} nu exista!")
            self.books = pd.DataFrame()

        # Books enriched (cu genuri, imagini)
        enriched_path = self.data_dir / 'books_enriched.csv'
        if enriched_path.exists() and not self.books.empty:
            enriched = pd.read_csv(enriched_path)
            merge_cols = ['ISBN']
            for col in ['genres', 'subjects', 'image_url', 'language', 'synopsis']:
                if col in enriched.columns:
                    merge_cols.append(col)

            if len(merge_cols) > 1:
                self.books = self.books.merge(
                    enriched[merge_cols].drop_duplicates(),
                    on='ISBN', how='left'
                )
            print(f"Books enriched merged")

        # Ratings
        ratings_path = self.data_dir / 'ratings_processed.csv'
        if ratings_path.exists():
            self.ratings = pd.read_csv(ratings_path)
            print(f"Ratings: {len(self.ratings)}")

    def get_recommendations(self, user_profile: Dict, n: int = 12) -> List[Dict]:
        """
        Genereaza recomandari bazate pe profilul utilizatorului

        Args:
            user_profile: {
                "favorite_genres": [...],
                "favorite_subjects": [...],
                "preferred_year_range": (min, max),
                "age": int,
                "language": "en",
                "user_read_history": [(isbn, rating), ...]
            }
            n: numar de recomandari

        Returns:
            Lista de dictionare cu info despre carti
        """
        if self.books is None or self.books.empty:
            return []

        candidates = self.books.copy()

        # Extrage din profil
        genres = user_profile.get('favorite_genres', [])
        subjects = user_profile.get('favorite_subjects', [])
        year_range = user_profile.get('preferred_year_range')
        language = user_profile.get('language')
        read_history = user_profile.get('user_read_history', [])

        # ISBN-uri citite (pentru excludere)
        read_isbns = set(str(item[0]) for item in read_history)

        # Filtru year range
        if year_range:
            min_year, max_year = year_range
            candidates = candidates[
                (candidates['Year'] >= min_year) &
                (candidates['Year'] <= max_year)
            ]

        # Filtru limba
        if language and 'language' in candidates.columns:
            candidates = candidates[candidates['language'] == language]

        # Calculeaza scor pentru fiecare carte
        def calculate_score(row):
            score = 0.0

            # Match genuri
            if pd.notna(row.get('genres')):
                book_genres = [g.strip().lower() for g in str(row['genres']).split(',')]
                for genre in genres:
                    if any(genre.lower() in bg for bg in book_genres):
                        score += 2.0

            # Match subjects
            if pd.notna(row.get('subjects')):
                book_subjects = [s.strip().lower() for s in str(row['subjects']).split(',')]
                for subject in subjects:
                    if any(subject.lower() in bs for bs in book_subjects):
                        score += 1.5

            return score

        candidates['score'] = candidates.apply(calculate_score, axis=1)

        # Filtreaza carti cu scor > 0 si exclude cele citite
        candidates = candidates[candidates['score'] > 0]
        candidates = candidates[~candidates['ISBN'].astype(str).isin(read_isbns)]

        # Sorteaza si ia top N
        candidates = candidates.sort_values('score', ascending=False).head(n)

        # Formateaza rezultate
        results = []
        for _, row in candidates.iterrows():
            image_url = row.get('image_url', PLACEHOLDER_IMAGE)
            if pd.isna(image_url) or not image_url:
                image_url = PLACEHOLDER_IMAGE

            results.append({
                'isbn': str(row['ISBN']),
                'title': row['Title'],
                'author': row['Author'],
                'year': int(row['Year']) if pd.notna(row.get('Year')) else 'N/A',
                'genres': row.get('genres', 'N/A') if pd.notna(row.get('genres')) else 'N/A',
                'image_url': image_url,
                'score': round(row['score'], 2)
            })

        return results

    def get_book_by_isbn(self, isbn: str) -> Optional[Dict]:
        """Gaseste o carte dupa ISBN"""
        if self.books is None or self.books.empty:
            return None

        book = self.books[self.books['ISBN'].astype(str) == str(isbn)]
        if book.empty:
            return None

        row = book.iloc[0]
        image_url = row.get('image_url', PLACEHOLDER_IMAGE)
        if pd.isna(image_url) or not image_url:
            image_url = PLACEHOLDER_IMAGE

        return {
            'isbn': str(row['ISBN']),
            'title': row['Title'],
            'author': row['Author'],
            'year': int(row['Year']) if pd.notna(row.get('Year')) else 'N/A',
            'genres': row.get('genres', 'N/A') if pd.notna(row.get('genres')) else 'N/A',
            'image_url': image_url
        }


# =============================================================================
# INITIALIZARE
# =============================================================================

print("\n" + "="*60)
print("INITIALIZARE SISTEM DE RECOMANDARE")
print("="*60)

recommender = BookRecommender(DATA_DIR)

# State global pentru profilul utilizatorului
user_profile = {
    "favorite_genres": [],
    "favorite_subjects": [],
    "preferred_year_range": None,
    "age": None,
    "language": None,
    "user_read_history": []
}


# =============================================================================
# FUNCTII HELPER
# =============================================================================

def create_book_card_html(book: Dict) -> str:
    """Creeaza HTML pentru un card de carte"""
    title_display = book['title'][:45] + '...' if len(book['title']) > 45 else book['title']
    author_display = book['author'][:25] + '...' if len(book['author']) > 25 else book['author']
    genres_display = book['genres'][:35] + '...' if len(str(book['genres'])) > 35 else book['genres']

    return f"""
    <div style="
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 12px;
        background: white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        width: 180px;
        text-align: center;
    ">
        <img src="{book['image_url']}" 
             style="width: 100px; height: 150px; object-fit: cover; border-radius: 6px; margin-bottom: 8px;"
             onerror="this.src='{PLACEHOLDER_IMAGE}'">
        <div style="font-weight: bold; font-size: 12px; height: 32px; overflow: hidden; margin-bottom: 4px;">
            {title_display}
        </div>
        <div style="color: #555; font-size: 11px; margin-bottom: 2px;">
            {author_display}
        </div>
        <div style="color: #777; font-size: 10px; margin-bottom: 2px;">
            {book['year']}
        </div>
        <div style="color: #888; font-size: 9px; height: 24px; overflow: hidden;">
            {genres_display}
        </div>
        <div style="color: #aaa; font-size: 9px; margin-top: 4px;">
            {book['isbn']}
        </div>
    </div>
    """


def create_books_grid_html(books: List[Dict]) -> str:
    """Creeaza HTML grid pentru toate cartile"""
    if not books:
        return "<p style='text-align: center; color: #666; padding: 40px;'>Nu am gasit carti care sa corespunda preferintelor tale. Incearca alte filtre!</p>"

    html = """
    <div style="
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 12px;
        padding: 15px;
    ">
    """

    for book in books:
        html += create_book_card_html(book)

    html += "</div>"
    return html


def format_read_history() -> str:
    """Formateaza istoricul de lectura"""
    global user_profile

    history = user_profile.get('user_read_history', [])

    if not history:
        return "Nu ai adaugat inca nicio carte citita."

    output = "**Carti citite:**\n\n"

    for isbn, rating in history:
        book = recommender.get_book_by_isbn(isbn)
        if book:
            stars = "*" * int(rating) + "." * (5 - int(rating))
            output += f"- {book['title']} [{stars}]\n"
        else:
            stars = "*" * int(rating) + "." * (5 - int(rating))
            output += f"- ISBN: {isbn} [{stars}]\n"

    return output


def get_current_profile_summary() -> str:
    """Returneaza un sumar al profilului curent"""
    global user_profile

    lines = []

    if user_profile['favorite_genres']:
        lines.append(f"Genuri: {', '.join(user_profile['favorite_genres'])}")

    if user_profile['favorite_subjects']:
        lines.append(f"Tematici: {', '.join(user_profile['favorite_subjects'])}")

    if user_profile['preferred_year_range']:
        yr = user_profile['preferred_year_range']
        lines.append(f"Perioada: {yr[0]}-{yr[1]}")

    if user_profile['language']:
        lang_name = AVAILABLE_LANGUAGES.get(user_profile['language'], user_profile['language'])
        lines.append(f"Limba: {lang_name}")

    if user_profile['age']:
        lines.append(f"Varsta: {user_profile['age']}")

    history_count = len(user_profile.get('user_read_history', []))
    lines.append(f"Carti citite: {history_count}")

    return "\n".join(lines) if lines else "Nicio preferinta setata."


# =============================================================================
# FUNCTII GRADIO
# =============================================================================

def add_book_by_isbn(isbn: str, rating: int):
    """Adauga o carte la istoric dupa ISBN"""
    global user_profile

    if not isbn or not isbn.strip():
        return "Introdu un ISBN valid!", format_read_history()

    if not rating:
        return "Selecteaza un rating!", format_read_history()

    isbn = isbn.strip()

    # Verifica daca cartea exista
    book = recommender.get_book_by_isbn(isbn)

    if not book:
        return f"ISBN '{isbn}' nu a fost gasit in baza de date!", format_read_history()

    # Verifica daca nu e deja in istoric
    existing = [item for item in user_profile['user_read_history'] if str(item[0]) == isbn]
    if existing:
        return f"Cartea '{book['title']}' este deja in istoric!", format_read_history()

    # Adauga la istoric
    user_profile['user_read_history'].append((isbn, float(rating)))

    return f"Adaugat: {book['title']} by {book['author']} (Rating: {rating})", format_read_history()


def remove_last_book():
    """Sterge ultima carte din istoric"""
    global user_profile

    if not user_profile['user_read_history']:
        return "Istoricul este deja gol!", format_read_history()

    removed = user_profile['user_read_history'].pop()
    book = recommender.get_book_by_isbn(removed[0])
    title = book['title'] if book else removed[0]

    return f"Sters: {title}", format_read_history()


def clear_history():
    """Sterge tot istoricul"""
    global user_profile
    user_profile['user_read_history'] = []
    return "Istoric sters!", format_read_history()


def generate_recommendations(
    genre1, genre2, genre3,
    subject1, subject2, subject3,
    year_range_choice,
    language_choice,
    age
):
    """Genereaza recomandari bazate pe preferinte"""
    global user_profile

    # Colecteaza genuri si subjects selectate
    genres = [g for g in [genre1, genre2, genre3] if g]
    subjects = [s for s in [subject1, subject2, subject3] if s]

    if not genres and not subjects:
        return "Selecteaza cel putin un gen sau o tematica!", ""

    # Parse year range
    year_range = YEAR_RANGES.get(year_range_choice) if year_range_choice else None

    # Parse language
    lang_code = None
    if language_choice:
        for code, name in AVAILABLE_LANGUAGES.items():
            if name == language_choice:
                lang_code = code
                break

    # Actualizeaza profilul
    user_profile['favorite_genres'] = genres
    user_profile['favorite_subjects'] = subjects
    user_profile['preferred_year_range'] = year_range
    user_profile['language'] = lang_code
    user_profile['age'] = int(age) if age else None

    # Genereaza recomandari
    recommendations = recommender.get_recommendations(user_profile, n=12)

    # Creeaza HTML pentru carti
    books_html = create_books_grid_html(recommendations)

    # Sumar profil
    profile_summary = get_current_profile_summary()

    return books_html, profile_summary


def refresh_recommendations():
    """Reincarca recomandarile cu acelasi profil"""
    global user_profile

    if not user_profile['favorite_genres'] and not user_profile['favorite_subjects']:
        return "Seteaza mai intai preferintele in Tab 1!", ""

    recommendations = recommender.get_recommendations(user_profile, n=12)
    books_html = create_books_grid_html(recommendations)
    profile_summary = get_current_profile_summary()

    return books_html, profile_summary


# =============================================================================
# CSS PERSONALIZAT
# =============================================================================

custom_css = """
.gradio-container {
    max-width: 1400px !important;
}
.book-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    justify-content: center;
}
"""


# =============================================================================
# CONSTRUIRE INTERFATA
# =============================================================================

with gr.Blocks(title="Book Recommender", css=custom_css) as app:

    gr.Markdown("# Sistem de Recomandare Carti")
    gr.Markdown("Descopera carti noi bazate pe preferintele tale")
    gr.Markdown("---")

    with gr.Tabs() as tabs:

        # =================================================================
        # TAB 1: PREFERINTE SI ISTORIC
        # =================================================================

        with gr.Tab("1. Preferinte"):

            gr.Markdown("## Selecteaza Preferintele")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Genuri Favorite (max 3)")
                    genre1 = gr.Dropdown(choices=AVAILABLE_GENRES, label="Gen 1")
                    genre2 = gr.Dropdown(choices=AVAILABLE_GENRES, label="Gen 2")
                    genre3 = gr.Dropdown(choices=AVAILABLE_GENRES, label="Gen 3")

                with gr.Column():
                    gr.Markdown("### Tematici Favorite (max 3)")
                    subject1 = gr.Dropdown(choices=AVAILABLE_SUBJECTS, label="Tematica 1")
                    subject2 = gr.Dropdown(choices=AVAILABLE_SUBJECTS, label="Tematica 2")
                    subject3 = gr.Dropdown(choices=AVAILABLE_SUBJECTS, label="Tematica 3")

            gr.Markdown("### Filtre Aditionale")

            with gr.Row():
                year_range = gr.Radio(
                    choices=list(YEAR_RANGES.keys()),
                    label="Perioada publicarii"
                )
                language = gr.Dropdown(
                    choices=list(AVAILABLE_LANGUAGES.values()),
                    label="Limba"
                )
                age = gr.Number(
                    label="Varsta ta",
                    minimum=10,
                    maximum=100,
                    step=1
                )

            gr.Markdown("---")
            gr.Markdown("## Adauga Carti Citite")
            gr.Markdown("Introdu ISBN-ul cartilor pe care le-ai citit pentru recomandari mai bune")

            with gr.Row():
                isbn_input = gr.Textbox(
                    label="ISBN",
                    placeholder="Ex: 0316769487"
                )
                rating_input = gr.Radio(
                    choices=[1, 2, 3, 4, 5],
                    label="Rating",
                    value=4
                )
                add_btn = gr.Button("Adauga", variant="secondary")

            add_status = gr.Markdown()

            gr.Markdown("### Istoric Lectura")
            history_display = gr.Markdown(value=format_read_history())

            gr.Markdown("---")

            get_recs_btn = gr.Button(
                "Gaseste Recomandari",
                variant="primary",
                size="lg"
            )

            recs_status = gr.Markdown()

        # =================================================================
        # TAB 2: RECOMANDARI
        # =================================================================

        with gr.Tab("2. Recomandari"):

            gr.Markdown("## Carti Recomandate pentru Tine")

            with gr.Row():
                with gr.Column(scale=4):
                    books_display = gr.HTML(
                        value="<p style='text-align: center; padding: 40px; color: #666;'>Seteaza preferintele in Tab 1 si apasa 'Gaseste Recomandari'</p>"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Profilul Tau")
                    profile_display = gr.Markdown(value=get_current_profile_summary())

                    refresh_btn = gr.Button("Reincarca Recomandari", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### Adauga Rapid o Carte Citita")
            gr.Markdown("Copiaza ISBN-ul unei carti de mai sus si adaug-o la istoric")

            with gr.Row():
                quick_isbn = gr.Textbox(label="ISBN", placeholder="Copiaza ISBN-ul aici")
                quick_rating = gr.Radio(choices=[1, 2, 3, 4, 5], label="Rating", value=4)
                quick_add_btn = gr.Button("Adauga si Reincarca")

            quick_status = gr.Markdown()

        # =================================================================
        # TAB 3: DESPRE
        # =================================================================

        with gr.Tab("3. Despre"):

            gr.Markdown("""
            ## Cum Functioneaza
            
            ### Pasul 1: Seteaza Preferintele
            - Selecteaza pana la 3 genuri favorite
            - Selecteaza pana la 3 tematici
            - Alege perioada de publicare (optional)
            - Selecteaza limba preferata (optional)
            - Introdu varsta ta (optional)
            
            ### Pasul 2: Adauga Carti Citite
            - Introdu ISBN-ul cartilor pe care le-ai citit
            - Da-le un rating de la 1 la 5
            - Cu cat ai mai multe carti in istoric, cu atat recomandarile sunt mai bune
            
            ### Pasul 3: Primeste Recomandari
            - Apasa "Gaseste Recomandari"
            - Vezi cartile recomandate in Tab 2
            - Poti adauga carti noi si reincarca pentru recomandari actualizate
            
            ---
            
            ## Dataset
            
            - Sursa: Book-Crossing Dataset + ISBNdb
            - Carti: ~14,610
            - Rating-uri: ~429,205
            
            ---
            
            ## Limbi Disponibile
            
            | Cod | Limba |
            |-----|-------|
            | en | English |
            | de | German |
            | es | Spanish |
            | fr | French |
            | it | Italian |
            | zh | Chinese |
            | ja | Japanese |
            | nl | Dutch |
            """)

    # =================================================================
    # EVENT HANDLERS
    # =================================================================

    # Tab 1: Adauga carte
    add_btn.click(
        fn=add_book_by_isbn,
        inputs=[isbn_input, rating_input],
        outputs=[add_status, history_display]
    )

    # Tab 1: Genereaza recomandari
    get_recs_btn.click(
        fn=generate_recommendations,
        inputs=[genre1, genre2, genre3, subject1, subject2, subject3, year_range, language, age],
        outputs=[books_display, profile_display]
    )

    # Tab 2: Refresh recomandari
    refresh_btn.click(
        fn=refresh_recommendations,
        outputs=[books_display, profile_display]
    )

    # Tab 2: Adauga rapid si reincarca
    def quick_add_and_refresh(isbn, rating):
        add_result, history = add_book_by_isbn(isbn, rating)
        books_html, profile = refresh_recommendations()
        return add_result, books_html, profile

    quick_add_btn.click(
        fn=quick_add_and_refresh,
        inputs=[quick_isbn, quick_rating],
        outputs=[quick_status, books_display, profile_display]
    )


# =============================================================================
# LANSARE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LANSARE INTERFATA GRADIO")
    print("="*60)
    print("\nAcceseaza: http://localhost:7860\n")

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )