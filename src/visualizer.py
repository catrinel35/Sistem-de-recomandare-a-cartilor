import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class BookCrossingVisualizer:

    def __init__(self, data_dir='../data/processed'):
        self.data_dir = Path(data_dir)
        self.books = None
        self.users = None
        self.ratings = None
        self.merged = None

    def load_data(self):
        print("Încărcare date...")
        self.books = pd.read_csv(self.data_dir / 'books_processed.csv')
        self.users = pd.read_csv(self.data_dir / 'users_processed.csv')
        self.ratings = pd.read_csv(self.data_dir / 'ratings_processed.csv')
        self.merged = pd.read_csv(self.data_dir / 'merged_dataset.csv')
        print("✓ Date încărcate cu succes!")

    def plot_rating_distribution(self, save_fig=False):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        rating_counts = self.ratings['Rating'].value_counts().sort_index()
        axes[0].bar(rating_counts.index, rating_counts.values, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Rating')
        axes[0].set_ylabel('Frecvență')
        axes[0].set_title('Distribuția tuturor rating-urilor (0-10)')
        axes[0].grid(axis='y', alpha=0.3)

        explicit_ratings = self.ratings[self.ratings['Rating'] > 0]['Rating']
        axes[1].hist(explicit_ratings, bins=10, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Rating')
        axes[1].set_ylabel('Frecvență')
        axes[1].set_title('Distribuția rating-urilor explicite (1-10)')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].axvline(explicit_ratings.mean(), color='red', linestyle='--',
                        label=f'Medie: {explicit_ratings.mean():.2f}')
        axes[1].legend()

        plt.tight_layout()
        if save_fig:
            plt.savefig('../outputs/rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n" + "=" * 50)
        print("STATISTICI RATING-URI")
        print("=" * 50)
        print(f"Total rating-uri: {len(self.ratings)}")
        print(
            f"Rating-uri implicite (0): {(self.ratings['Rating'] == 0).sum()} ({(self.ratings['Rating'] == 0).sum() / len(self.ratings) * 100:.1f}%)")
        print(
            f"Rating-uri explicite (>0): {(self.ratings['Rating'] > 0).sum()} ({(self.ratings['Rating'] > 0).sum() / len(self.ratings) * 100:.1f}%)")
        print(f"\nRating mediu (toate): {self.ratings['Rating'].mean():.2f}")
        print(f"Rating mediu (explicite): {explicit_ratings.mean():.2f}")
        print(f"Mediană: {explicit_ratings.median():.2f}")
        print("=" * 50 + "\n")

    def plot_user_activity(self, save_fig=False):
        user_activity = self.ratings.groupby('User-ID').size()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].hist(user_activity, bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Număr de rating-uri per utilizator')
        axes[0].set_ylabel('Număr utilizatori')
        axes[0].set_title('Distribuția activității utilizatorilor')
        axes[0].axvline(user_activity.mean(), color='red', linestyle='--',
                        label=f'Medie: {user_activity.mean():.1f}')
        axes[0].axvline(user_activity.median(), color='orange', linestyle='--',
                        label=f'Mediană: {user_activity.median():.1f}')
        axes[0].legend()
        axes[0].set_yscale('log')

        top_users = user_activity.nlargest(20)
        axes[1].barh(range(len(top_users)), top_users.values, color='purple', alpha=0.7)
        axes[1].set_xlabel('Număr de rating-uri')
        axes[1].set_ylabel('User ID')
        axes[1].set_title('Top 20 cei mai activi utilizatori')
        axes[1].invert_yaxis()

        plt.tight_layout()
        if save_fig:
            plt.savefig('../outputs/user_activity.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n" + "=" * 50)
        print("STATISTICI UTILIZATORI")
        print("=" * 50)
        print(f"Total utilizatori: {len(user_activity)}")
        print(f"Rating-uri per utilizator (medie): {user_activity.mean():.1f}")
        print(f"Rating-uri per utilizator (mediană): {user_activity.median():.1f}")
        print(f"Cel mai activ utilizator: {user_activity.max()} rating-uri")
        print(f"Cel mai puțin activ: {user_activity.min()} rating-uri")
        print("=" * 50 + "\n")

    def plot_book_popularity(self, save_fig=False):

        book_popularity = self.ratings.groupby('ISBN').size()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].hist(book_popularity, bins=50, color='teal', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Număr de rating-uri per carte')
        axes[0].set_ylabel('Număr cărți')
        axes[0].set_title('Distribuția popularității cărților')
        axes[0].axvline(book_popularity.mean(), color='red', linestyle='--',
                        label=f'Medie: {book_popularity.mean():.1f}')
        axes[0].legend()
        axes[0].set_yscale('log')

        top_books_isbn = book_popularity.nlargest(20)
        top_books_info = self.merged[self.merged['ISBN'].isin(top_books_isbn.index)][
            ['ISBN', 'Title', 'Author']
        ].drop_duplicates('ISBN').set_index('ISBN')

        top_books_df = pd.DataFrame({
            'count': top_books_isbn,
            'title': top_books_info['Title'],
            'author': top_books_info['Author']
        })

        axes[1].barh(range(len(top_books_df)), top_books_df['count'].values,
                     color='indianred', alpha=0.7)
        axes[1].set_xlabel('Număr de rating-uri')
        axes[1].set_title('Top 20 cele mai populare cărți')
        axes[1].set_yticks(range(len(top_books_df)))

        labels = [f"{title[:30]}..." if len(str(title)) > 30 else title
                  for title in top_books_df['title'].values]
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].invert_yaxis()

        plt.tight_layout()
        if save_fig:
            plt.savefig('../outputs/book_popularity.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n" + "=" * 50)
        print("TOP 10 CELE MAI POPULARE CĂRȚI")
        print("=" * 50)
        for idx, (isbn, row) in enumerate(top_books_df.head(10).iterrows(), 1):
            print(f"{idx}. {row['title'][:50]}")
            print(f"   Autor: {row['author']}")
            print(f"   Rating-uri: {row['count']}\n")
        print("=" * 50 + "\n")

    def plot_year_distribution(self, save_fig=False):
        years = self.books['Year'].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].hist(years, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('An publicare')
        axes[0].set_ylabel('Număr de cărți')
        axes[0].set_title('Distribuția anilor de publicare')
        axes[0].axvline(years.mean(), color='red', linestyle='--',
                        label=f'Medie: {years.mean():.0f}')
        axes[0].legend()

        decades = (years // 10) * 10
        decade_counts = decades.value_counts().sort_index()

        axes[1].bar(decade_counts.index, decade_counts.values,
                    width=8, color='orange', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Decadă')
        axes[1].set_ylabel('Număr de cărți')
        axes[1].set_title('Cărți pe decenii')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save_fig:
            plt.savefig('../outputs/year_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n" + "=" * 50)
        print("STATISTICI ANI PUBLICARE")
        print("=" * 50)
        print(f"Cel mai vechi: {years.min():.0f}")
        print(f"Cel mai recent: {years.max():.0f}")
        print(f"An mediu: {years.mean():.0f}")
        print(f"Mediană: {years.median():.0f}")
        print("=" * 50 + "\n")

    def plot_age_distribution(self, save_fig=False):

        ages = self.users['Age'].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].hist(ages, bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Vârstă')
        axes[0].set_ylabel('Număr utilizatori')
        axes[0].set_title('Distribuția vârstelor utilizatorilor')
        axes[0].axvline(ages.mean(), color='red', linestyle='--',
                        label=f'Medie: {ages.mean():.1f}')
        axes[0].axvline(ages.median(), color='orange', linestyle='--',
                        label=f'Mediană: {ages.median():.1f}')
        axes[0].legend()

        age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
        age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        age_categories = pd.cut(ages, bins=age_bins, labels=age_labels)
        age_counts = age_categories.value_counts().sort_index()

        axes[1].bar(range(len(age_counts)), age_counts.values,
                    color='mediumpurple', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Categorie vârstă')
        axes[1].set_ylabel('Număr utilizatori')
        axes[1].set_title('Utilizatori pe categorii de vârstă')
        axes[1].set_xticks(range(len(age_counts)))
        axes[1].set_xticklabels(age_counts.index, rotation=45)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save_fig:
            plt.savefig('../outputs/age_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n" + "=" * 50)
        print("STATISTICI VÂRSTĂ")
        print("=" * 50)
        print(f"Utilizatori cu vârstă specificată: {len(ages)} / {len(self.users)}")
        print(f"Vârstă medie: {ages.mean():.1f} ani")
        print(f"Vârstă mediană: {ages.median():.1f} ani")
        print(f"Cel mai tânăr: {ages.min():.0f} ani")
        print(f"Cel mai în vârstă: {ages.max():.0f} ani")
        print("=" * 50 + "\n")

    def plot_top_authors(self, top_n=20, save_fig=False):

        author_counts = self.books['Author'].value_counts().head(top_n)

        author_ratings = self.merged.groupby('Author').size().nlargest(top_n)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].barh(range(len(author_counts)), author_counts.values,
                     color='goldenrod', alpha=0.7)
        axes[0].set_xlabel('Număr de cărți în dataset')
        axes[0].set_title(f'Top {top_n} autori după număr de cărți')
        axes[0].set_yticks(range(len(author_counts)))
        labels = [f"{author[:25]}..." if len(str(author)) > 25 else author
                  for author in author_counts.index]
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].invert_yaxis()

        axes[1].barh(range(len(author_ratings)), author_ratings.values,
                     color='darkseagreen', alpha=0.7)
        axes[1].set_xlabel('Număr total de rating-uri')
        axes[1].set_title(f'Top {top_n} autori după popularitate')
        axes[1].set_yticks(range(len(author_ratings)))
        labels = [f"{author[:25]}..." if len(str(author)) > 25 else author
                  for author in author_ratings.index]
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].invert_yaxis()

        plt.tight_layout()
        if save_fig:
            plt.savefig('../outputs/top_authors.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_correlation_matrix(self, save_fig=False):

        user_stats = self.ratings.groupby('User-ID').agg({
            'Rating': ['mean', 'count', lambda x: (x > 0).sum()]
        }).reset_index()
        user_stats.columns = ['User-ID', 'avg_rating', 'total_ratings', 'explicit_ratings']

        user_stats = user_stats.merge(self.users[['User-ID', 'Age']], on='User-ID', how='left')

        corr_data = user_stats[['Age', 'avg_rating', 'total_ratings', 'explicit_ratings']].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Matrice de corelație - Caracteristici utilizatori')
        plt.tight_layout()

        if save_fig:
            plt.savefig('../outputs/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_all_visualizations(self, save_figs=True):

        Path('../outputs').mkdir(exist_ok=True)

        print("\n🎨 Generare vizualizări...\n")

        print("1️⃣  Distribuția rating-urilor...")
        self.plot_rating_distribution(save_fig=save_figs)

        print("2️⃣  Activitatea utilizatorilor...")
        self.plot_user_activity(save_fig=save_figs)

        print("3️⃣  Popularitatea cărților...")
        self.plot_book_popularity(save_fig=save_figs)

        print("4️⃣  Distribuția anilor de publicare...")
        self.plot_year_distribution(save_fig=save_figs)

        print("5️⃣  Distribuția vârstelor...")
        self.plot_age_distribution(save_fig=save_figs)

        print("6️⃣  Top autori...")
        self.plot_top_authors(save_fig=save_figs)

        print("7️⃣  Matrice de corelație...")
        self.plot_correlation_matrix(save_fig=save_figs)

        print("\n✅ Toate vizualizările au fost generate!")
        if save_figs:
            print("📁 Salvate în: ../outputs/")


if __name__ == "__main__":
    viz = BookCrossingVisualizer(data_dir='../data/processed')
    viz.load_data()
    viz.generate_all_visualizations(save_figs=True)
