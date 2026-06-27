import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/munch_pricing_dataset.xlsx")


def inspect_excel_file(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print("=" * 80)
    print("EXCEL FILE INSPECTION")
    print("=" * 80)

    excel_file = pd.ExcelFile(file_path)

    print("\nSheet names:")
    for sheet in excel_file.sheet_names:
        print(f"- {sheet}")

    print("\n" + "=" * 80)
    print("SHEET DETAILS")
    print("=" * 80)

    for sheet in excel_file.sheet_names:
        print(f"\n\n--- Sheet: {sheet} ---")

        df = pd.read_excel(file_path, sheet_name=sheet)

        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        print("\nColumn names:")
        for col in df.columns:
            print(f"- {col}")

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        missing = df.isna().sum()
        print(missing[missing > 0].sort_values(ascending=False))

        print("\nFirst 5 rows:")
        print(df.head())


if __name__ == "__main__":
    inspect_excel_file(RAW_DATA_PATH)