import sqlite3
import pandas as pd

def get_bank_table():
    """
    Gets the bank_marketing table from 'data/bmarket.db'

    Args: None

    Returns:
        DataFrame: The bank_marketing dataframe
    
    """
    file_loc = 'data/bmarket.db'

    conn = sqlite3.connect(file_loc)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if not(tables):
        raise ConnectionError("Unable to connect to " + file_loc)
    else:
        bank_df = pd.read_sql_query("SELECT * FROM bank_marketing", conn)
        conn.close()
        return bank_df
