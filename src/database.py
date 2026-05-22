from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:admin@localhost:5432/postgres"

engine = create_engine(
    DB_URL,
    connect_args={"options": "-csearch_path=book_recommendation"}
)