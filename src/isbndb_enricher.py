import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ISBNdbEnricher:

    def __init__(self, api_key: str, data_dir='../data/processed'):

        self.api_key = api_key
        self.base_url = "https://api2.isbndb.com"
        self.data_dir = Path(data_dir)
        self.books = None

        self.requests_made = 0
        self.max_requests_per_day = 10000
        self.delay_between_requests = 1.0  # secunde

    def load_books(self):

        books_path = self.data_dir / 'books_processed.csv'
        self.books = pd.read_csv(books_path)
        logger.info(f" Încărcat {len(self.books)} cărți")
        return self.books

    def fetch_book_metadata(self, isbn: str) -> Optional[Dict]:

        if self.requests_made >= self.max_requests_per_day:
            logger.warning(" Limită zilnică atinsă! Oprire.")
            return None

        isbn_clean = str(isbn).strip().replace('-', '').replace(' ', '')

        url = f"{self.base_url}/book/{isbn_clean}"
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            self.requests_made += 1

            if response.status_code == 200:
                data = response.json()

                book_data = data.get('book', {})

                metadata = {
                    'isbn': isbn_clean,
                    'title_full': book_data.get('title'),
                    'title_long': book_data.get('title_long'),
                    'synopsis': book_data.get('synopsis'),
                    'subjects': book_data.get('subjects', []),
                    'publisher': book_data.get('publisher'),
                    'pages': book_data.get('pages'),
                    'date_published': book_data.get('date_published'),
                    'binding': book_data.get('binding'),
                    'language': book_data.get('language'),
                    'isbn13': book_data.get('isbn13'),
                    'authors': book_data.get('authors', []),
                    'image': book_data.get('image'),
                    'msrp': book_data.get('msrp'),  # price
                    'overview': book_data.get('overview')
                }

                logger.info(f"✓ Găsit: {book_data.get('title', 'N/A')}")

                # Delay pentru rate limiting
                time.sleep(self.delay_between_requests)

                return metadata

            elif response.status_code == 404:
                logger.warning(f"ISBN {isbn} nu găsit în ISBNdb")
                time.sleep(self.delay_between_requests)
                return None

            elif response.status_code == 429:
                logger.error("Rate limit exceeded! Așteptare 60s...")
                time.sleep(60)
                return self.fetch_book_metadata(isbn)  # Retry

            else:
                logger.error(f"Eroare {response.status_code}: {response.text}")
                time.sleep(self.delay_between_requests)
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Timeout pentru ISBN {isbn}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f" Eroare request: {e}")
            return None
        except json.JSONDecodeError:
            logger.error(f" Eroare parsare JSON pentru ISBN {isbn}")
            return None

    def enrich_books(self, batch_size: int = 100, start_index: int = 0):

        if self.books is None:
            self.load_books()

        new_columns = ['synopsis', 'subjects', 'genres', 'language', 'pages',
                       'binding', 'image_url', 'overview']

        for col in new_columns:
            if col not in self.books.columns:
                self.books[col] = None

        total = min(len(self.books), start_index + batch_size)

        logger.info(f"\n Începere îmbogățire: {start_index} → {total} (total: {len(self.books)})")
        logger.info(f" Progres: {self.requests_made}/{self.max_requests_per_day} requests folosite\n")

        enriched_count = 0
        not_found_count = 0

        for idx in range(start_index, total):
            if self.requests_made >= self.max_requests_per_day:
                logger.warning(f"\n Limită zilnică atinsă! Procesate {enriched_count}/{total - start_index}")
                break

            row = self.books.iloc[idx]
            isbn = row['ISBN']

            # Skip dacă deja are metadate
            if pd.notna(row.get('synopsis')):
                logger.info(f"  [{idx + 1}/{total}] ISBN {isbn} - deja îmbogățit, skip")
                continue

            logger.info(f" [{idx + 1}/{total}] Procesare ISBN: {isbn}")

            metadata = self.fetch_book_metadata(isbn)

            if metadata:

                self.books.at[idx, 'synopsis'] = metadata.get('synopsis')
                self.books.at[idx, 'subjects'] = ', '.join(metadata.get('subjects', []))
                self.books.at[idx, 'language'] = metadata.get('language')
                self.books.at[idx, 'pages'] = metadata.get('pages')
                self.books.at[idx, 'binding'] = metadata.get('binding')
                self.books.at[idx, 'image_url'] = metadata.get('image')
                self.books.at[idx, 'overview'] = metadata.get('overview')

                subjects = metadata.get('subjects', [])
                if subjects:
                    genres = subjects[:3]  # Primele 3 ca genuri principale
                    self.books.at[idx, 'genres'] = ', '.join(genres)

                enriched_count += 1

            else:
                not_found_count += 1

            if (idx + 1) % 10 == 0:
                self.save_enriched_books(suffix=f'_checkpoint_{idx + 1}')
                logger.info(f"💾 Checkpoint salvat la index {idx + 1}")

        self.save_enriched_books()

        logger.info(f"\n Îmbogățire completă!")
        logger.info(f"   Procesate: {total - start_index}")
        logger.info(f"   Găsite: {enriched_count}")
        logger.info(f"   Nu găsite: {not_found_count}")
        logger.info(f"   Requests folosite: {self.requests_made}/{self.max_requests_per_day}")

    def save_enriched_books(self, suffix=''):
        output_path = self.data_dir / f'books_enriched{suffix}.csv'
        self.books.to_csv(output_path, index=False)
        logger.info(f" Salvat: {output_path}")

    def get_enrichment_stats(self):
        if self.books is None:
            self.load_books()

        stats = {
            'total_books': len(self.books),
            'with_synopsis': self.books['synopsis'].notna().sum() if 'synopsis' in self.books.columns else 0,
            'with_subjects': self.books['subjects'].notna().sum() if 'subjects' in self.books.columns else 0,
            'with_genres': self.books['genres'].notna().sum() if 'genres' in self.books.columns else 0,
            'with_language': self.books['language'].notna().sum() if 'language' in self.books.columns else 0,
            'with_image': self.books['image_url'].notna().sum() if 'image_url' in self.books.columns else 0,
        }

        print("\n" + "=" * 60)
        print("STATISTICI ÎMBOGĂȚIRE DATASET")
        print("=" * 60)
        print(f"Total cărți:           {stats['total_books']}")
        print(
            f"Cu synopsis:           {stats['with_synopsis']} ({stats['with_synopsis'] / stats['total_books'] * 100:.1f}%)")
        print(
            f"Cu subjects:           {stats['with_subjects']} ({stats['with_subjects'] / stats['total_books'] * 100:.1f}%)")
        print(
            f"Cu genres:             {stats['with_genres']} ({stats['with_genres'] / stats['total_books'] * 100:.1f}%)")
        print(
            f"Cu limbă:              {stats['with_language']} ({stats['with_language'] / stats['total_books'] * 100:.1f}%)")
        print(f"Cu imagine:            {stats['with_image']} ({stats['with_image'] / stats['total_books'] * 100:.1f}%)")
        print("=" * 60 + "\n")

        return stats

    def extract_top_genres(self, top_n=20):
        if self.books is None or 'subjects' not in self.books.columns:
            logger.error("Nu există date de genuri!")
            return None

        all_subjects = []
        for subjects_str in self.books['subjects'].dropna():
            subjects = [s.strip() for s in subjects_str.split(',')]
            all_subjects.extend(subjects)

        subject_counts = pd.Series(all_subjects).value_counts().head(top_n)

        print(f"\n Top {top_n} Genuri/Subjects:")
        print("=" * 60)
        for idx, (subject, count) in enumerate(subject_counts.items(), 1):
            print(f"{idx:2d}. {subject[:50]:<50} {count:>5}")
        print("=" * 60 + "\n")

        return subject_counts


class ISBNdbBatchProcessor:

    def __init__(self, api_key: str, data_dir='../data/processed'):
        self.enricher = ISBNdbEnricher(api_key, data_dir)
        self.progress_file = Path(data_dir) / 'enrichment_progress.json'

    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'last_index': 0, 'requests_today': 0, 'date': None}

    def save_progress(self, last_index: int, requests_today: int):
        from datetime import date
        progress = {
            'last_index': last_index,
            'requests_today': requests_today,
            'date': str(date.today())
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)

    def process_daily_batch(self, max_requests=900):

        from datetime import date

        progress = self.load_progress()

        if progress['date'] != str(date.today()):
            progress['requests_today'] = 0

        remaining = max_requests - progress['requests_today']

        if remaining <= 0:
            logger.warning("Limită zilnică deja atinsă! Încearcă mâine.")
            return

        logger.info(f" Procesare batch: {remaining} requests disponibile")

        start_idx = progress['last_index']
        self.enricher.load_books()
        self.enricher.enrich_books(batch_size=remaining, start_index=start_idx)

        end_idx = start_idx + self.enricher.requests_made
        self.save_progress(end_idx, progress['requests_today'] + self.enricher.requests_made)

        logger.info(
            f"Progres salvat: index {end_idx}, requests azi: {progress['requests_today'] + self.enricher.requests_made}")


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    # ========== CONFIGURARE ==========


    API_KEY = "66605_39b09023b3f5eda4fae34d619fb5b2ef"

    print("Procesare Simplă")
    print("=" * 60)

    enricher = ISBNdbEnricher(api_key=API_KEY, data_dir='../data/processed')

    # Încarcă cărțile
    enricher.load_books()

    # Statistici
    enricher.get_enrichment_stats()

    print("\n\n" + "=" * 60)
    print("Procesare în Batch-uri (Automată)")
    print("=" * 60)

    batch_processor = ISBNdbBatchProcessor(
        api_key=API_KEY,
        data_dir='../data/processed'
    )
    batch_processor.process_daily_batch(max_requests=4999)

