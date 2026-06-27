import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_PATH = Path("data/processed/featured_munch_pricing.csv")
MODEL_PATH = Path("models/decision_model_best_model.pkl")
OUTPUT_DIR = Path("reports/outputs")

TARGET_COLUMN = "Munch_ASP_Rs_per_10g"

LEAKAGE_COLUMNS = [
    "Munch_Volume_KG",
    "Munch_Value_INR000",
]


def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Data not found: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def time_based_split(df):
    train_df = df[df["Report_Year"] < 2026].copy()
    test_df = df[df["Report_Year"] == 2026].copy()
    return train_df, test_df


def create_predictions(df, model):
    train_df, test_df = time_based_split(df)

    drop_cols = [TARGET_COLUMN] + LEAKAGE_COLUMNS

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COLUMN]

    test_df = test_df.copy()
    test_df["Predicted_ASP"] = model.predict(X_test)
    test_df["Error"] = test_df["Predicted_ASP"] - test_df[TARGET_COLUMN]
    test_df["Absolute_Error"] = test_df["Error"].abs()
    test_df["Pct_Error"] = np.where(
        test_df[TARGET_COLUMN] > 0,
        test_df["Error"] / test_df[TARGET_COLUMN] * 100,
        np.nan
    )

    mae = mean_absolute_error(y_test, test_df["Predicted_ASP"])
    rmse = np.sqrt(mean_squared_error(y_test, test_df["Predicted_ASP"]))
    r2 = r2_score(y_test, test_df["Predicted_ASP"])

    print("=" * 80)
    print("DECISION MODEL DIAGNOSTICS")
    print("=" * 80)

    print("\nOverall 2026 Test Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    print("\nActual vs Predicted ASP Summary:")
    print(test_df[[TARGET_COLUMN, "Predicted_ASP", "Error", "Absolute_Error", "Pct_Error"]].describe())

    print("\nMean Actual ASP:")
    print(test_df[TARGET_COLUMN].mean())

    print("\nMean Predicted ASP:")
    print(test_df["Predicted_ASP"].mean())

    print("\nMean Error:")
    print(test_df["Error"].mean())

    return test_df


def summarize_errors(test_df):
    summaries = {}

    summaries["overall_summary"] = test_df[
        [TARGET_COLUMN, "Predicted_ASP", "Error", "Absolute_Error", "Pct_Error"]
    ].describe()

    summaries["error_by_market_type"] = (
        test_df.groupby("Market_Type")
        .agg(
            rows=(TARGET_COLUMN, "count"),
            actual_asp=(TARGET_COLUMN, "mean"),
            predicted_asp=("Predicted_ASP", "mean"),
            mae=("Absolute_Error", "mean"),
            bias=("Error", "mean"),
        )
        .sort_values("mae", ascending=False)
    )

    summaries["error_by_source_sheet"] = (
        test_df.groupby("Source_Sheet")
        .agg(
            rows=(TARGET_COLUMN, "count"),
            actual_asp=(TARGET_COLUMN, "mean"),
            predicted_asp=("Predicted_ASP", "mean"),
            mae=("Absolute_Error", "mean"),
            bias=("Error", "mean"),
        )
        .sort_values("mae", ascending=False)
    )

    summaries["error_by_onam_relevance"] = (
        test_df.groupby("Is_Onam_Relevant")
        .agg(
            rows=(TARGET_COLUMN, "count"),
            actual_asp=(TARGET_COLUMN, "mean"),
            predicted_asp=("Predicted_ASP", "mean"),
            mae=("Absolute_Error", "mean"),
            bias=("Error", "mean"),
        )
    )

    summaries["top_50_highest_errors"] = (
        test_df.sort_values("Absolute_Error", ascending=False)
        .head(50)
    )

    return summaries


def save_outputs(test_df, summaries):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "decision_model_diagnostics.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        test_df.to_excel(writer, sheet_name="2026_predictions", index=False)

        for sheet_name, data in summaries.items():
            data.to_excel(writer, sheet_name=sheet_name[:31])

    print("\n" + "=" * 80)
    print("DIAGNOSTICS SAVED")
    print("=" * 80)
    print(f"Saved to: {output_path}")


def print_key_tables(summaries):
    print("\n" + "=" * 80)
    print("ERROR BY MARKET TYPE")
    print("=" * 80)
    print(summaries["error_by_market_type"])

    print("\n" + "=" * 80)
    print("ERROR BY SOURCE SHEET")
    print("=" * 80)
    print(summaries["error_by_source_sheet"])

    print("\n" + "=" * 80)
    print("ERROR BY ONAM RELEVANCE")
    print("=" * 80)
    print(summaries["error_by_onam_relevance"])


def main():
    df = load_data()
    model = load_model()

    test_df = create_predictions(df, model)
    summaries = summarize_errors(test_df)

    print_key_tables(summaries)
    save_outputs(test_df, summaries)


if __name__ == "__main__":
    main()