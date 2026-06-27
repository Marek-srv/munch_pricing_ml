import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("data/processed/featured_munch_pricing.csv")
FIGURE_DIR = Path("reports/figures")
OUTPUT_DIR = Path("reports/outputs")

TARGET_COLUMN = "Munch_ASP_Rs_per_10g"


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Featured data not found: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)


def save_plot(filename: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_target_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.hist(df[TARGET_COLUMN], bins=40)
    plt.title("Distribution of Munch ASP")
    plt.xlabel("Munch ASP Rs per 10g")
    plt.ylabel("Frequency")
    save_plot("01_munch_asp_distribution.png")


def plot_asp_trend_by_year(df: pd.DataFrame):
    yearly = df.groupby("Report_Year")[TARGET_COLUMN].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(yearly["Report_Year"], yearly[TARGET_COLUMN], marker="o")
    plt.title("Average Munch ASP by Report Year")
    plt.xlabel("Report Year")
    plt.ylabel("Average Munch ASP Rs per 10g")
    plt.grid(True)
    save_plot("02_munch_asp_trend_by_year.png")


def plot_asp_by_market_type(df: pd.DataFrame):
    market_type_avg = (
        df.groupby("Market_Type")[TARGET_COLUMN]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    plt.bar(market_type_avg["Market_Type"], market_type_avg[TARGET_COLUMN])
    plt.title("Average Munch ASP by Market Type")
    plt.xlabel("Market Type")
    plt.ylabel("Average Munch ASP Rs per 10g")
    plt.xticks(rotation=45, ha="right")
    save_plot("03_munch_asp_by_market_type.png")


def plot_asp_by_source_sheet(df: pd.DataFrame):
    source_avg = (
        df.groupby("Source_Sheet")[TARGET_COLUMN]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    plt.bar(source_avg["Source_Sheet"], source_avg[TARGET_COLUMN])
    plt.title("Average Munch ASP by Source Sheet")
    plt.xlabel("Source Sheet")
    plt.ylabel("Average Munch ASP Rs per 10g")
    plt.xticks(rotation=45, ha="right")
    save_plot("04_munch_asp_by_source_sheet.png")


def plot_competitor_relationships(df: pd.DataFrame):
    competitor_cols = [
        ("Perk_ASP_Rs_per_10g", "05_munch_vs_perk_asp.png", "Munch ASP vs Perk ASP"),
        ("Snakker_ASP_Rs_per_10g", "06_munch_vs_snakker_asp.png", "Munch ASP vs Snakker ASP"),
        ("Competitor_Avg_ASP", "07_munch_vs_competitor_avg_asp.png", "Munch ASP vs Competitor Average ASP"),
    ]

    for col, filename, title in competitor_cols:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[col], df[TARGET_COLUMN], alpha=0.3)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Munch ASP Rs per 10g")
        plt.grid(True)
        save_plot(filename)


def plot_onam_relevant_asp(df: pd.DataFrame):
    onam_summary = (
        df.groupby("Is_Onam_Relevant")[TARGET_COLUMN]
        .mean()
        .reset_index()
    )

    onam_summary["Segment"] = onam_summary["Is_Onam_Relevant"].map({
        0: "Non-Onam Relevant",
        1: "Onam Relevant"
    })

    plt.figure(figsize=(8, 6))
    plt.bar(onam_summary["Segment"], onam_summary[TARGET_COLUMN])
    plt.title("Average Munch ASP: Onam Relevant vs Non-Onam Relevant")
    plt.xlabel("Segment")
    plt.ylabel("Average Munch ASP Rs per 10g")
    save_plot("08_onam_relevant_asp.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    selected_cols = [
        TARGET_COLUMN,
        "Report_Year",
        "Munch_VolShare_pct",
        "Munch_ValShare_pct",
        "Munch_ND_pct",
        "Munch_WD_pct",
        "Munch_PDO_Volume",
        "Perk_ASP_Rs_per_10g",
        "Snakker_ASP_Rs_per_10g",
        "Competitor_Avg_ASP",
        "Munch_vs_Perk_VolShare_Gap",
        "Munch_vs_Snakker_VolShare_Gap",
        "Munch_Distribution_Strength",
        "Munch_vs_Perk_Distribution_Gap",
        "Is_South_Market",
        "Is_Kerala_Market",
        "Is_Metro_Market",
        "Is_Onam_Relevant",
    ]

    selected_cols = [col for col in selected_cols if col in numeric_df.columns]
    corr = numeric_df[selected_cols].corr()

    plt.figure(figsize=(14, 10))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    save_plot("09_correlation_heatmap.png")


def save_eda_summary(df: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    summary["overall_target_summary"] = df[TARGET_COLUMN].describe()
    summary["asp_by_year"] = df.groupby("Report_Year")[TARGET_COLUMN].mean()
    summary["asp_by_market_type"] = df.groupby("Market_Type")[TARGET_COLUMN].mean().sort_values(ascending=False)
    summary["asp_by_source_sheet"] = df.groupby("Source_Sheet")[TARGET_COLUMN].mean().sort_values(ascending=False)
    summary["asp_by_onam_relevance"] = df.groupby("Is_Onam_Relevant")[TARGET_COLUMN].mean()

    output_path = OUTPUT_DIR / "eda_summary.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, data in summary.items():
            data.to_excel(writer, sheet_name=sheet_name[:31])

    print(f"Saved EDA summary: {output_path}")


def print_key_insights(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("KEY EDA INSIGHTS")
    print("=" * 80)

    print("\nTarget summary:")
    print(df[TARGET_COLUMN].describe())

    print("\nAverage ASP by year:")
    print(df.groupby("Report_Year")[TARGET_COLUMN].mean())

    print("\nTop Market Types by ASP:")
    print(
        df.groupby("Market_Type")[TARGET_COLUMN]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nAverage ASP: Onam Relevant vs Non-Onam Relevant:")
    print(df.groupby("Is_Onam_Relevant")[TARGET_COLUMN].mean())

    print("\nCorrelation with Munch ASP:")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    correlations = numeric_df.corr()[TARGET_COLUMN].sort_values(ascending=False)
    print(correlations.head(15))
    print("\nLowest correlations:")
    print(correlations.tail(10))


def main():
    df = load_data()

    print("=" * 80)
    print("STARTING EDA")
    print("=" * 80)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    plot_target_distribution(df)
    plot_asp_trend_by_year(df)
    plot_asp_by_market_type(df)
    plot_asp_by_source_sheet(df)
    plot_competitor_relationships(df)
    plot_onam_relevant_asp(df)
    plot_correlation_heatmap(df)
    save_eda_summary(df)
    print_key_insights(df)

    print("\nEDA completed.")


if __name__ == "__main__":
    main()