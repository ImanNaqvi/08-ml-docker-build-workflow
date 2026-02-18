import pandas as pd

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and drop rows with missing values.
    """
    df = pd.read_csv(csv_path)
    return df.dropna()
