import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_PATH = Path("data/processed/featured_munch_pricing.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("reports/outputs")

TARGET_COLUMN = "Munch_ASP_Rs_per_10g"

LEAKAGE_COLUMNS = [
    "Munch_Volume_KG",
    "Munch_Value_INR000",
]

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Featured data not found: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)


def time_based_split(df: pd.DataFrame):
    """
    Time-based split:
    Train: years before 2026
    Test: 2026

    This is better than random split because we want future price prediction.
    """
    train_df = df[df["Report_Year"] < 2026].copy()
    test_df = df[df["Report_Year"] == 2026].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Check Report_Year values.")

    return train_df, test_df


def build_preprocessor(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def evaluate_model(model_name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def train_model_version(df: pd.DataFrame, version_name: str, drop_cols=None):
    if drop_cols is None:
        drop_cols = []

    print("\n" + "=" * 80)
    print(f"TRAINING: {version_name}")
    print("=" * 80)

    train_df, test_df = time_based_split(df)

    columns_to_drop = [TARGET_COLUMN] + drop_cols

    X_train = train_df.drop(columns=columns_to_drop)
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=columns_to_drop)
    y_test = test_df[TARGET_COLUMN]

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    print(f"Train rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")
    print(f"Input columns: {X_train.shape[1]}")
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    models = {
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_metrics = evaluate_model(
            f"{version_name}_{model_name}_TRAIN",
            y_train,
            train_pred,
        )

        test_metrics = evaluate_model(
            f"{version_name}_{model_name}_TEST",
            y_test,
            test_pred,
        )

        results.append(train_metrics)
        results.append(test_metrics)

        trained_pipelines[model_name] = pipeline

        print(f"\n{model_name}")
        print(f"Train MAE: {train_metrics['MAE']:.4f}")
        print(f"Train RMSE: {train_metrics['RMSE']:.4f}")
        print(f"Train R2: {train_metrics['R2']:.4f}")
        print(f"Test MAE: {test_metrics['MAE']:.4f}")
        print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
        print(f"Test R2: {test_metrics['R2']:.4f}")

    results_df = pd.DataFrame(results)

    # Save best model by test R2
    test_results = results_df[results_df["Model"].str.contains("_TEST")].copy()
    best_row = test_results.sort_values("R2", ascending=False).iloc[0]
    best_model_name = best_row["Model"].replace(f"{version_name}_", "").replace("_TEST", "")

    best_pipeline = trained_pipelines[best_model_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{version_name.lower()}_best_model.pkl"
    joblib.dump(best_pipeline, model_path)

    print("\nBest model:")
    print(best_row)
    print(f"Saved best model to: {model_path}")

    return results_df


def main():
    df = load_data()

    print("=" * 80)
    print("BASELINE MODEL TRAINING")
    print("=" * 80)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Years available: {sorted(df['Report_Year'].unique())}")

    # Model A: full model
    full_results = train_model_version(
        df=df,
        version_name="Full_Model",
        drop_cols=[],
    )

    # Model B: decision model without direct ASP leakage variables
    decision_results = train_model_version(
        df=df,
        version_name="Decision_Model",
        drop_cols=LEAKAGE_COLUMNS,
    )

    final_results = pd.concat([full_results, decision_results], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "baseline_model_results.xlsx"
    final_results.to_excel(output_path, index=False)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)
    print(final_results)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()