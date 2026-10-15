import psycopg2, pandas as pd
import os
from dotenv import load_dotenv

# modificado para cargar credenciales del .env
def connect_db():
    load_dotenv()
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    return conn

def fetch_data():
    conn = connect_db()
    df = pd.read_sql("SELECT id, contenido FROM noticias;", conn)
    conn.close()
    return df
