
import pandas as pd
from pathlib import Path

DATA_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/"
    "datasets/bace.csv"
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATA_PATH = RAW_DIR / "bace.csv"


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        print("Downloading BACE dataset...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
        print("Dataset downloaded:", DATA_PATH)

    return DATA_PATH


def load_dataset():
    path = download_dataset()
    df = pd.read_csv(path)

    required = ["mol", "Class"]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing dataset column: {column}")

    df = df[required].copy()
    df = df.dropna()

    df["mol"] = df["mol"].astype(str)
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df.dropna(subset=["Class"])

    df["Class"] = df["Class"].astype(int)

    df = df[df["Class"].isin([0, 1])]
    df = df.drop_duplicates(subset=["mol"])

    print(f"Loaded {len(df)} molecules")
    print("Class distribution:")
    print(df["Class"].value_counts())

    return df


if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset.head())
