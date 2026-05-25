from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:admin@localhost:5432/postgres"

engine = create_engine(
    DB_URL,
    connect_args={"options": "-csearch_path=book_recommendation"}
)


def get_avg_rating(isbn: str) -> float:
    """Returneaza media ratingurilor > 0 pentru un ISBN"""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT ROUND(AVG(rating)::numeric, 1)
            FROM rating
            WHERE isbn = :isbn AND rating > 0
        """), {"isbn": isbn}).fetchone()

    if result and result[0] is not None:
        return float(result[0])
    return 0.0