import pandas as pd
import numpy as np
from pathlib import Path
import joblib


DATA_PATH = Path("data/processed/featured_munch_pricing.csv")
MODEL_PATH = Path("models/robust_final_model.pkl")
OUTPUT_DIR = Path("reports/outputs")
OUTPUT_PATH = OUTPUT_DIR / "onam_price_recommendations.xlsx"

TARGET_COLUMN = "Munch_ASP_Rs_per_10g"

DROP_COLUMNS = [
    TARGET_COLUMN,
    "Munch_Volume_KG",
    "Munch_Value_INR000",
    "Period",
]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    return df, model


def filter_onam_market(df):
    """
    Onam-relevant pricing should focus on:
    - South markets
    - Kerala markets
    - Onam period proxy
    - Latest year, preferably 2026
    """

    latest_year = df["Report_Year"].max()

    onam_df = df[
        (df["Report_Year"] == latest_year)
        & (
            (df["Is_Onam_Relevant"] == 1)
            | (df["Is_South_Market"] == 1)
            | (df["Is_Kerala_Market"] == 1)
            | (df["Source_Sheet"].astype(str).str.lower().str.contains("south"))
        )
    ].copy()

    if onam_df.empty:
        raise ValueError("No Onam-relevant rows found. Check market and period flags.")

    return onam_df


def predict_prices(onam_df, model):
    X = onam_df.drop(columns=DROP_COLUMNS)

    onam_df = onam_df.copy()
    onam_df["Predicted_Munch_ASP"] = model.predict(X)

    return onam_df


def create_price_bands(df):
    df = df.copy()

    # Based on model MAE from robust test
    model_mae = 0.180096

    df["Conservative_ASP"] = df["Predicted_Munch_ASP"] - model_mae
    df["Aggressive_ASP"] = df["Predicted_Munch_ASP"] + model_mae

    # Rounded business-friendly prices
    df["Recommended_ASP_Rounded"] = df["Predicted_Munch_ASP"].round(2)
    df["Conservative_ASP_Rounded"] = df["Conservative_ASP"].round(2)
    df["Aggressive_ASP_Rounded"] = df["Aggressive_ASP"].round(2)

    df["Actual_ASP_vs_Predicted_Gap"] = (
        df[TARGET_COLUMN] - df["Predicted_Munch_ASP"]
    )

    return df


def summarize_recommendations(df):
    summary = {}

    summary["overall_onam_summary"] = pd.DataFrame({
        "Metric": [
            "Rows",
            "Actual ASP Mean",
            "Predicted ASP Mean",
            "Conservative ASP Mean",
            "Aggressive ASP Mean",
            "Actual ASP Min",
            "Actual ASP Max",
            "Predicted ASP Min",
            "Predicted ASP Max",
        ],
        "Value": [
            len(df),
            df[TARGET_COLUMN].mean(),
            df["Predicted_Munch_ASP"].mean(),
            df["Conservative_ASP"].mean(),
            df["Aggressive_ASP"].mean(),
            df[TARGET_COLUMN].min(),
            df[TARGET_COLUMN].max(),
            df["Predicted_Munch_ASP"].min(),
            df["Predicted_Munch_ASP"].max(),
        ]
    })

    summary["by_source_sheet"] = (
        df.groupby("Source_Sheet")
        .agg(
            rows=("Predicted_Munch_ASP", "count"),
            actual_asp_mean=(TARGET_COLUMN, "mean"),
            predicted_asp_mean=("Predicted_Munch_ASP", "mean"),
            conservative_asp_mean=("Conservative_ASP", "mean"),
            aggressive_asp_mean=("Aggressive_ASP", "mean"),
            perk_asp_mean=("Perk_ASP_Rs_per_10g", "mean"),
            snakker_asp_mean=("Snakker_ASP_Rs_per_10g", "mean"),
            competitor_avg_asp=("Competitor_Avg_ASP", "mean"),
            munch_vol_share=("Munch_VolShare_pct", "mean"),
            munch_val_share=("Munch_ValShare_pct", "mean"),
            munch_distribution_strength=("Munch_Distribution_Strength", "mean"),
        )
        .sort_values("predicted_asp_mean", ascending=False)
        .reset_index()
    )

    summary["by_market_type"] = (
        df.groupby("Market_Type")
        .agg(
            rows=("Predicted_Munch_ASP", "count"),
            actual_asp_mean=(TARGET_COLUMN, "mean"),
            predicted_asp_mean=("Predicted_Munch_ASP", "mean"),
            conservative_asp_mean=("Conservative_ASP", "mean"),
            aggressive_asp_mean=("Aggressive_ASP", "mean"),
            perk_asp_mean=("Perk_ASP_Rs_per_10g", "mean"),
            snakker_asp_mean=("Snakker_ASP_Rs_per_10g", "mean"),
            munch_vol_share=("Munch_VolShare_pct", "mean"),
            munch_distribution_strength=("Munch_Distribution_Strength", "mean"),
        )
        .sort_values("predicted_asp_mean", ascending=False)
        .reset_index()
    )

    summary["top_markets"] = (
        df.groupby(["Market_Type", "Market"])
        .agg(
            rows=("Predicted_Munch_ASP", "count"),
            actual_asp_mean=(TARGET_COLUMN, "mean"),
            predicted_asp_mean=("Predicted_Munch_ASP", "mean"),
            conservative_asp_mean=("Conservative_ASP", "mean"),
            aggressive_asp_mean=("Aggressive_ASP", "mean"),
            perk_asp_mean=("Perk_ASP_Rs_per_10g", "mean"),
            snakker_asp_mean=("Snakker_ASP_Rs_per_10g", "mean"),
            munch_vol_share=("Munch_VolShare_pct", "mean"),
            munch_val_share=("Munch_ValShare_pct", "mean"),
            munch_distribution_strength=("Munch_Distribution_Strength", "mean"),
        )
        .sort_values("predicted_asp_mean", ascending=False)
        .reset_index()
    )

    return summary


def save_to_excel(df, summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_cols = [
        "Report_Year",
        "Source_Sheet",
        "Market_Type",
        "Market",
        "Period_Type",
        "Period_Year",
        "Is_South_Market",
        "Is_Kerala_Market",
        "Is_Onam_Relevant",
        TARGET_COLUMN,
        "Predicted_Munch_ASP",
        "Recommended_ASP_Rounded",
        "Conservative_ASP_Rounded",
        "Aggressive_ASP_Rounded",
        "Perk_ASP_Rs_per_10g",
        "Snakker_ASP_Rs_per_10g",
        "Competitor_Avg_ASP",
        "Munch_VolShare_pct",
        "Munch_ValShare_pct",
        "Munch_ND_pct",
        "Munch_WD_pct",
        "Munch_Distribution_Strength",
        "Munch_PDO_Volume",
    ]

    selected_cols = [col for col in selected_cols if col in df.columns]

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        summary["overall_onam_summary"].to_excel(
            writer, sheet_name="Overall_Summary", index=False
        )

        summary["by_source_sheet"].to_excel(
            writer, sheet_name="By_Source_Sheet", index=False
        )

        summary["by_market_type"].to_excel(
            writer, sheet_name="By_Market_Type", index=False
        )

        summary["top_markets"].to_excel(
            writer, sheet_name="Top_Markets", index=False
        )

        df[selected_cols].to_excel(
            writer, sheet_name="Detailed_Recommendations", index=False
        )

    print("=" * 80)
    print("ONAM PRICE RECOMMENDATION COMPLETED")
    print("=" * 80)
    print(f"Saved to: {OUTPUT_PATH}")


def print_final_summary(df, summary):
    print("\n" + "=" * 80)
    print("OVERALL ONAM RECOMMENDATION")
    print("=" * 80)
    print(summary["overall_onam_summary"])

    print("\n" + "=" * 80)
    print("RECOMMENDATION BY SOURCE SHEET")
    print("=" * 80)
    print(summary["by_source_sheet"])

    print("\n" + "=" * 80)
    print("TOP 20 MARKETS BY RECOMMENDED ASP")
    print("=" * 80)
    print(summary["top_markets"].head(20))


def main():
    df, model = load_data()

    onam_df = filter_onam_market(df)
    onam_df = predict_prices(onam_df, model)
    onam_df = create_price_bands(onam_df)

    summary = summarize_recommendations(onam_df)

    save_to_excel(onam_df, summary)
    print_final_summary(onam_df, summary)


if __name__ == "__main__":
    main()