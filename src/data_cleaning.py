import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/munch_pricing_dataset.xlsx")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "cleaned_munch_pricing.csv"


TARGET_COLUMN = "Munch_ASP_Rs_per_10g"

DROP_COLUMNS = [
    "Source_File",
    "Report_Label",
    "Has_Munch",
    "Has_Perk",
    "Has_Snakker",

    # Leakage columns because they directly use Munch ASP
    "RelativePrice_Munch_vs_Perk",
    "RelativePrice_Munch_vs_Snakker",
    "PriceGap_Munch_Perk_Rs_per_10g",
    "PriceGap_Munch_Snakker_Rs_per_10g",
]


def load_data() -> pd.DataFrame:
    df = pd.read_excel(RAW_DATA_PATH, sheet_name="ML_Dataset")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("=" * 80)
    print("ORIGINAL DATA")
    print("=" * 80)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Remove fully empty rows
    df = df.dropna(how="all")

    # Remove rows where target is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    # Remove impossible target values
    df = df[df[TARGET_COLUMN] > 0]

    # Drop metadata and leakage columns
    existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # Remove duplicates
    before_duplicates = df.shape[0]
    df = df.drop_duplicates()
    after_duplicates = df.shape[0]

    print("\n" + "=" * 80)
    print("AFTER BASIC CLEANING")
    print("=" * 80)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Duplicates removed: {before_duplicates - after_duplicates}")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Do not fill target here
    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)

    print("\n" + "=" * 80)
    print("MISSING VALUES BEFORE IMPUTATION")
    print("=" * 80)
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False))

    # Fill categorical missing values
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Fill numeric missing values with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    print("\n" + "=" * 80)
    print("MISSING VALUES AFTER IMPUTATION")
    print("=" * 80)
    missing_after = df.isna().sum()
    print(missing_after[missing_after > 0].sort_values(ascending=False))

    return df


def save_cleaned_data(df: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("CLEANED DATA SAVED")
    print("=" * 80)
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Final rows: {df.shape[0]}")
    print(f"Final columns: {df.shape[1]}")


def show_final_summary(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("FINAL COLUMN LIST")
    print("=" * 80)
    for col in df.columns:
        print(f"- {col}")

    print("\n" + "=" * 80)
    print("TARGET SUMMARY")
    print("=" * 80)
    print(df[TARGET_COLUMN].describe())

    print("\n" + "=" * 80)
    print("CATEGORICAL COLUMNS")
    print("=" * 80)
    print(df.select_dtypes(include=["object"]).columns.tolist())

    print("\n" + "=" * 80)
    print("NUMERIC COLUMNS")
    print("=" * 80)
    print(df.select_dtypes(include=["int64", "float64"]).columns.tolist())


def main():
    df = load_data()
    df = basic_cleaning(df)
    df = handle_missing_values(df)
    save_cleaned_data(df)
    show_final_summary(df)


if __name__ == "__main__":
    main()