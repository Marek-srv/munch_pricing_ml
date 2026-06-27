import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/processed/cleaned_munch_pricing.csv")
OUTPUT_PATH = Path("data/processed/featured_munch_pricing.csv")

TARGET_COLUMN = "Munch_ASP_Rs_per_10g"


def load_cleaned_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cleaned data not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    return df


def add_competitor_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Competitor_Avg_ASP"] = df[
        ["Perk_ASP_Rs_per_10g", "Snakker_ASP_Rs_per_10g"]
    ].mean(axis=1)

    df["Perk_Snakker_PriceGap"] = (
        df["Perk_ASP_Rs_per_10g"] - df["Snakker_ASP_Rs_per_10g"]
    )

    df["Perk_to_Snakker_ASP_Ratio"] = np.where(
        df["Snakker_ASP_Rs_per_10g"] > 0,
        df["Perk_ASP_Rs_per_10g"] / df["Snakker_ASP_Rs_per_10g"],
        np.nan
    )

    return df


def add_share_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Munch_vs_Perk_VolShare_Gap"] = (
        df["Munch_VolShare_pct"] - df["Perk_VolShare_pct"]
    )

    df["Munch_vs_Perk_ValShare_Gap"] = (
        df["Munch_ValShare_pct"] - df["Perk_ValShare_pct"]
    )

    df["Munch_vs_Snakker_VolShare_Gap"] = (
        df["Munch_VolShare_pct"] - df["Snakker_VolShare_pct"]
    )

    df["Munch_vs_Snakker_ValShare_Gap"] = (
        df["Munch_ValShare_pct"] - df["Snakker_ValShare_pct"]
    )

    return df


def add_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Munch_Distribution_Strength"] = (
        df["Munch_ND_pct"] * df["Munch_WD_pct"]
    ) / 100

    df["Perk_Distribution_Strength"] = (
        df["Perk_ND_pct"] * df["Perk_WD_pct"]
    ) / 100

    df["Snakker_Distribution_Strength"] = (
        df["Snakker_ND_pct"] * df["Snakker_WD_pct"]
    ) / 100

    df["Munch_vs_Perk_Distribution_Gap"] = (
        df["Munch_Distribution_Strength"] - df["Perk_Distribution_Strength"]
    )

    df["Munch_vs_Snakker_Distribution_Gap"] = (
        df["Munch_Distribution_Strength"] - df["Snakker_Distribution_Strength"]
    )

    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    market_text = (
        df["Source_Sheet"].astype(str).str.lower()
        + " "
        + df["Market_Type"].astype(str).str.lower()
        + " "
        + df["Market"].astype(str).str.lower()
    )

    df["Is_South_Market"] = market_text.str.contains(
        "south|kerala|tamil|karnataka|andhra|telangana|chennai|bangalore|kochi|trivandrum"
    ).astype(int)

    df["Is_Kerala_Market"] = market_text.str.contains(
        "kerala|kochi|trivandrum|calicut|kollam|thrissur"
    ).astype(int)

    df["Is_Metro_Market"] = market_text.str.contains(
        "metro|delhi|mumbai|bangalore|chennai|kolkata|hyderabad|pune"
    ).astype(int)

    df["Is_Channel_Market"] = df["Market_Type"].astype(str).str.lower().str.contains(
        "channel"
    ).astype(int)

    return df


def add_period_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    period_text = (
        df["Period"].astype(str).str.lower()
        + " "
        + df["Period_Type"].astype(str).str.lower()
    )

    df["Is_Fiscal_Year"] = period_text.str.contains("fy|fiscal").astype(int)
    df["Is_MAT"] = period_text.str.contains("mat").astype(int)
    df["Is_Quarter"] = period_text.str.contains("q1|q2|q3|q4|quarter").astype(int)
    df["Is_Month"] = period_text.str.contains(
        "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|month"
    ).astype(int)

    # Onam is mainly Kerala/South linked and around Aug-Sep.
    df["Is_Onam_Period"] = period_text.str.contains(
        "aug|sep|q2|q3"
    ).astype(int)

    df["Is_Onam_Relevant"] = (
        (df["Is_South_Market"] == 1) & (df["Is_Onam_Period"] == 1)
    ).astype(int)

    return df


def clean_engineered_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df


def save_featured_data(df: pd.DataFrame):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 80)
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")


def show_feature_summary(df: pd.DataFrame):
    new_features = [
        "Competitor_Avg_ASP",
        "Perk_Snakker_PriceGap",
        "Perk_to_Snakker_ASP_Ratio",
        "Munch_vs_Perk_VolShare_Gap",
        "Munch_vs_Perk_ValShare_Gap",
        "Munch_vs_Snakker_VolShare_Gap",
        "Munch_vs_Snakker_ValShare_Gap",
        "Munch_Distribution_Strength",
        "Perk_Distribution_Strength",
        "Snakker_Distribution_Strength",
        "Munch_vs_Perk_Distribution_Gap",
        "Munch_vs_Snakker_Distribution_Gap",
        "Is_South_Market",
        "Is_Kerala_Market",
        "Is_Metro_Market",
        "Is_Channel_Market",
        "Is_Fiscal_Year",
        "Is_MAT",
        "Is_Quarter",
        "Is_Month",
        "Is_Onam_Period",
        "Is_Onam_Relevant",
    ]

    print("\n" + "=" * 80)
    print("NEW FEATURE SUMMARY")
    print("=" * 80)

    for col in new_features:
        if col in df.columns:
            print(f"\n{col}")
            print(df[col].describe())


def main():
    df = load_cleaned_data()

    print("=" * 80)
    print("STARTING FEATURE ENGINEERING")
    print("=" * 80)
    print(f"Input rows: {df.shape[0]}")
    print(f"Input columns: {df.shape[1]}")

    df = add_competitor_price_features(df)
    df = add_share_features(df)
    df = add_distribution_features(df)
    df = add_market_features(df)
    df = add_period_features(df)
    df = clean_engineered_data(df)

    save_featured_data(df)
    show_feature_summary(df)


if __name__ == "__main__":
    main()