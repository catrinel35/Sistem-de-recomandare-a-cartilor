from pathlib import Path
from typing import List, Dict

import gradio as gr
import pandas as pd

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

PLACEHOLDER_IMAGE = "https://via.placeholder.com/150x220?text=No+Cover"

try:
    from hybrid_recommender import HybridRecommender, create_hybrid_recommender

    HYBRID_AVAILABLE = True
    print("HybridRecommender importat cu succes!")
except ImportError as e:
    HYBRID_AVAILABLE = False
    print(f"HybridRecommender nu e disponibil: {e}")

# =============================================================================
# INITIALIZARE SISTEM
# =============================================================================

print("\n" + "=" * 60)
print("INITIALIZARE SISTEM DE RECOMANDARE")
print("=" * 60)

recommender = None

if HYBRID_AVAILABLE:
    try:
        recommender = create_hybrid_recommender(
            data_dir=str(DATA_DIR),
            books_emb_path=str(DATA_DIR / 'book_embeddings.npy'),
            users_emb_path=str(DATA_DIR / 'user_embedding.npy'),
            load_bert=False,  # Folosim embeddings pre-calculate
            load_cf=True
        )
        print("HybridRecommender initializat cu succes!")
    except Exception as e:
        print(f"Eroare la initializare HybridRecommender: {e}")
        recommender = None

# Fallback simplu daca HybridRecommender nu functioneaza
if recommender is None:
    print("Folosesc fallback simplu...")


    class SimpleRecommender:
        def __init__(self, data_dir):
            self.data_dir = Path(data_dir)
            self.books_df = None
            self._load_data()

        def _load_data(self):
            books_path = self.data_dir / 'books_enriched.csv'
            if books_path.exists():
                self.books_df = pd.read_csv(books_path)
                self.books_df['ISBN'] = self.books_df['ISBN'].astype(str)
                print(f"Books loaded: {len(self.books_df)}")

        def get_recommendations(self, user_profile, n=12):
            if self.books_df is None:
                return []

            candidates = self.books_df.copy()
            genres = user_profile.get('favorite_genres', [])
            subjects = user_profile.get('favorite_subjects', [])
            year_range = user_profile.get('preferred_year_range')
            language = user_profile.get('language')
            read_history = user_profile.get('user_read_history', [])

            read_isbns = set(str(item[0]) for item in read_history)

            if year_range:
                candidates = candidates[
                    (candidates['Year'] >= year_range[0]) &
                    (candidates['Year'] <= year_range[1])
                    ]

            if language and 'language' in candidates.columns:
                candidates = candidates[candidates['language'] == language]

            def calc_score(row):
                score = 0.0
                if pd.notna(row.get('genres')):
                    for g in genres:
                        if g.lower() in str(row['genres']).lower():
                            score += 2.0
                if pd.notna(row.get('subjects')):
                    for s in subjects:
                        if s.lower() in str(row['subjects']).lower():
                            score += 1.5
                return score

            candidates['score'] = candidates.apply(calc_score, axis=1)
            candidates = candidates[candidates['score'] > 0]
            candidates = candidates[~candidates['ISBN'].isin(read_isbns)]
            candidates = candidates.sort_values('score', ascending=False).head(n)

            results = []
            for _, row in candidates.iterrows():
                img = row.get('image_url', PLACEHOLDER_IMAGE)
                if pd.isna(img):
                    img = PLACEHOLDER_IMAGE
                results.append({
                    'isbn': str(row['ISBN']),
                    'title': row['Title'],
                    'author': row['Author'],
                    'year': int(row['Year']) if pd.notna(row.get('Year')) else 'N/A',
                    'genres': row.get('genres', 'N/A') if pd.notna(row.get('genres')) else 'N/A',
                    'image_url': img,
                    'score': round(row['score'], 3)
                })
            return results

        def get_book_by_isbn(self, isbn):
            if self.books_df is None:
                return None
            book = self.books_df[self.books_df['ISBN'] == str(isbn)]
            if book.empty:
                return None
            row = book.iloc[0]
            img = row.get('image_url', PLACEHOLDER_IMAGE)
            if pd.isna(img):
                img = PLACEHOLDER_IMAGE
            return {
                'isbn': str(row['ISBN']),
                'title': row['Title'],
                'author': row['Author'],
                'year': int(row['Year']) if pd.notna(row.get('Year')) else 'N/A',
                'genres': row.get('genres', 'N/A') if pd.notna(row.get('genres')) else 'N/A',
                'image_url': img
            }


    recommender = SimpleRecommender(DATA_DIR)

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
    title_display = book['title'][:50] + '...' if len(book['title']) > 50 else book['title']
    author_display = book['author'][:30] + '...' if len(book['author']) > 30 else book['author']
    genres_display = book['genres'][:45] + '...' if len(str(book['genres'])) > 45 else book['genres']

    return f"""
    <div style="
        border: 2px solid #bbb;
        border-radius: 12px;
        padding: 18px;
        background: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        width: 240px;
        text-align: center;
    ">
        <img src="{book['image_url']}" 
             style="width: 140px; height: 210px; object-fit: cover; border-radius: 8px; margin-bottom: 14px; box-shadow: 0 3px 10px rgba(0,0,0,0.2);"
             onerror="this.src='{PLACEHOLDER_IMAGE}'">
        <div style="color: #1a1a1a; font-weight: bold; font-size: 16px; line-height: 1.3; min-height: 44px; overflow: hidden; margin-bottom: 10px;">
            {title_display}
        </div>
        <div style="color: #333; font-size: 15px; font-weight: 500; margin-bottom: 8px;">
            {author_display}
        </div>
        <div style="color: #444; font-size: 14px; font-weight: 500; margin-bottom: 8px;">
            {book['year']}
        </div>
        <div style="color: #555; font-size: 13px; min-height: 40px; overflow: hidden; margin-bottom: 10px; line-height: 1.4;">
            {genres_display}
        </div>
        <div style="color: #2563eb; font-size: 15px; font-weight: bold; margin-top: 10px;">
            Score: {book.get('score', 'N/A')}
        </div>
        <div style="color: #555; font-size: 13px; margin-top: 8px; font-family: monospace; background: #f5f5f5; padding: 4px 8px; border-radius: 4px;">
            {book['isbn']}
        </div>
    </div>
    """


def create_books_grid_html(books: List[Dict]) -> str:
    if not books:
        return "<p style='text-align: center; color: #666; padding: 40px; font-size: 16px;'>Nu am gasit carti care sa corespunda preferintelor tale. Incearca alte filtre!</p>"

    html = """
    <div style="
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 20px;
        padding: 20px;
    ">
    """

    for book in books:
        html += create_book_card_html(book)

    html += "</div>"
    return html


def format_read_history() -> str:
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
    global user_profile

    lines = []

    if user_profile['favorite_genres']:
        lines.append(f"**Genuri:** {', '.join(user_profile['favorite_genres'])}")

    if user_profile['favorite_subjects']:
        lines.append(f"**Tematici:** {', '.join(user_profile['favorite_subjects'])}")

    if user_profile['preferred_year_range']:
        yr = user_profile['preferred_year_range']
        lines.append(f"**Perioada:** {yr[0]}-{yr[1]}")

    if user_profile['language']:
        lang_name = AVAILABLE_LANGUAGES.get(user_profile['language'], user_profile['language'])
        lines.append(f"**Limba:** {lang_name}")

    if user_profile['age']:
        lines.append(f"**Varsta:** {user_profile['age']}")

    history_count = len(user_profile.get('user_read_history', []))
    lines.append(f"**Carti citite:** {history_count}")

    return "\n\n".join(lines) if lines else "Nicio preferinta setata."


# =============================================================================
# FUNCTII GRADIO
# =============================================================================

def add_book_by_isbn(isbn: str, rating: int):
    global user_profile

    if not isbn or not isbn.strip():
        return "Introdu un ISBN valid!", format_read_history()

    if not rating:
        return "Selecteaza un rating!", format_read_history()

    isbn = isbn.strip()

    book = recommender.get_book_by_isbn(isbn)

    if not book:
        return f"ISBN '{isbn}' nu a fost gasit in baza de date!", format_read_history()

    existing = [item for item in user_profile['user_read_history'] if str(item[0]) == isbn]
    if existing:
        return f"Cartea '{book['title']}' este deja in istoric!", format_read_history()

    user_profile['user_read_history'].append((isbn, float(rating)))

    return f"Adaugat: {book['title']} by {book['author']} (Rating: {rating})", format_read_history()


def generate_recommendations(
        genre1, genre2, genre3,
        subject1, subject2, subject3,
        year_range_choice,
        language_choice,
        age
):
    global user_profile

    genres = [g for g in [genre1, genre2, genre3] if g]
    subjects = [s for s in [subject1, subject2, subject3] if s]

    if not genres and not subjects:
        return "Selecteaza cel putin un gen sau o tematica!", ""

    year_range = YEAR_RANGES.get(year_range_choice) if year_range_choice else None

    lang_code = None
    if language_choice:
        for code, name in AVAILABLE_LANGUAGES.items():
            if name == language_choice:
                lang_code = code
                break

    user_profile['favorite_genres'] = genres
    user_profile['favorite_subjects'] = subjects
    user_profile['preferred_year_range'] = year_range
    user_profile['language'] = lang_code
    user_profile['age'] = int(age) if age else None

    print(f"\nGenerating recommendations for profile: {user_profile}")
    recommendations = recommender.get_recommendations(user_profile, n=12)
    print(f"Got {len(recommendations)} recommendations")

    books_html = create_books_grid_html(recommendations)

    profile_summary = get_current_profile_summary()

    return books_html, profile_summary


def refresh_recommendations():
    global user_profile

    if not user_profile['favorite_genres'] and not user_profile['favorite_subjects']:
        return "Seteaza mai intai preferintele in Tab 1!", ""

    recommendations = recommender.get_recommendations(user_profile, n=12)
    books_html = create_books_grid_html(recommendations)
    profile_summary = get_current_profile_summary()

    return books_html, profile_summary


def quick_add_and_refresh(isbn, rating):
    add_result, history = add_book_by_isbn(isbn, rating)
    books_html, profile = refresh_recommendations()
    return add_result, books_html, profile


# =============================================================================
# CSS PERSONALIZAT
# =============================================================================

custom_css = """
.gradio-container {
    max-width: 1400px !important;
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

                    # refresh_btn = gr.Button("Reincarca Recomandari", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### Adauga Rapid o Carte Citita")

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

            Sistemul foloseste un **algoritm hibrid** care combina:

            1. **BERT Embeddings (30%)** - Similaritate semantica intre preferinte si carti
            2. **Collaborative Filtering (25%)** - SVD si KNN bazat pe rating-uri
            3. **Genre Matching (20%)** - Potrivire directa genuri
            4. **Subject Matching (10%)** - Potrivire directa tematici
            5. **Popularity (10%)** - Cat de populara e cartea
            6. **Recency (5%)** - Cat de recenta e cartea

            ---

            ### Pasul 1: Seteaza Preferintele
            - Selecteaza pana la 3 genuri favorite
            - Selecteaza pana la 3 tematici
            - Alege perioada de publicare (optional)
            - Selecteaza limba preferata (optional)

            ### Pasul 2: Adauga Carti Citite
            - Introdu ISBN-ul cartilor pe care le-ai citit
            - Da-le un rating de la 1 la 5
            - Cu cat ai mai multe carti in istoric, cu atat recomandarile sunt mai bune

            ### Pasul 3: Primeste Recomandari
            - Apasa "Gaseste Recomandari"
            - Vezi cartile recomandate in Tab 2

            ---

            ## Dataset

            - **Sursa:** Book-Crossing Dataset + ISBNdb API
            - **Carti:** ~14,610 cu metadate complete
            - **Rating-uri:** ~143,264 explicite
            - **Utilizatori:** ~12,294

            ---

            ## Tehnologii Folosite

            | Componenta | Tehnologie |
            |------------|------------|
            | Embeddings | BERT (all-MiniLM-L6-v2) |
            | CF - SVD | Custom implementation |
            | CF - KNN | Custom (User + Item based) |
            | Interfata | Gradio |
            | Data Processing | Pandas, NumPy |
            """)

    # =================================================================
    # EVENT HANDLERS
    # =================================================================

    add_btn.click(
        fn=add_book_by_isbn,
        inputs=[isbn_input, rating_input],
        outputs=[add_status, history_display]
    )

    get_recs_btn.click(
        fn=generate_recommendations,
        inputs=[genre1, genre2, genre3, subject1, subject2, subject3, year_range, language, age],
        outputs=[books_display, profile_display]
    )

    quick_add_btn.click(
        fn=quick_add_and_refresh,
        inputs=[quick_isbn, quick_rating],
        outputs=[quick_status, books_display, profile_display]
    )

# =============================================================================
# LANSARE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LANSARE INTERFATA GRADIO")
    print("=" * 60)
    print("\nAcceseaza: http://localhost:7860\n")

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
