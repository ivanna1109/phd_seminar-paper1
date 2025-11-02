import pandas as pd
import os
import sys

sys.path.append('/home/ivana-milutinovic/Documents/Doktorske/Prva godina/Seminar1/LocalWorkspace')

def load_parquet_and_print_size(parquet_file_path):
    """
    Učitava Parquet fajl u Pandas DataFrame i ispisuje njegovu veličinu.
    """
    if not os.path.exists(parquet_file_path):
        print(f"Greška: Fajl ne postoji na putanji: {parquet_file_path}")
        return

    try:
        df = pd.read_parquet(parquet_file_path)

        print(f"\n--- Učitavanje Podataka iz Parquet Fajla ---")
        print(f"Putanja fajla: {parquet_file_path}")
        print(f"Veličina skupa (Redovi x Kolone): {df.shape}")
        print(f"Broj uzoraka (embrija): {len(df)}")
        print("\nPrvih 5 redova (za uvid u strukturu):")
        print(df.head())
        
        if 'class' in df.columns:
            print("\nBalans klasa:")
            print(df['class'].value_counts())

    except Exception as e:
        print(f"Greška pri učitavanju Parquet fajla: {e}")


TEST_PARQUET_PATH = 'prepared_data/test_images.parquet' 

load_parquet_and_print_size(TEST_PARQUET_PATH)