import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


INPUT_PATH = Path("data/processed/featured_munch_pricing.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("reports/outputs")

TARGET_COLUMN = "Munch_ASP_Rs_per_10g"

DROP_COLUMNS = [
    TARGET_COLUMN,

    # Direct leakage / weak decision variables
    "Munch_Volume_KG",
    "Munch_Value_INR000",

    # Too specific, high-cardinality period label
    "Period",
]

RANDOM_STATE = 42


def load_data():
    return pd.read_csv(INPUT_PATH)


def time_based_split(df):
    train_df = df[df["Report_Year"] < 2026].copy()
    test_df = df[df["Report_Year"] == 2026].copy()
    return train_df, test_df


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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

    return preprocessor


def evaluate(name, y_true, y_pred):
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Actual_Mean": y_true.mean(),
        "Predicted_Mean": y_pred.mean(),
        "Actual_Std": y_true.std(),
        "Predicted_Std": y_pred.std(),
        "Bias": (y_pred - y_true).mean(),
    }


def get_models():
    return {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=700,
            learning_rate=0.025,
            max_depth=4,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=700,
            learning_rate=0.025,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def save_feature_importance(best_pipeline, best_model_name):
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    output_path = OUTPUT_DIR / "robust_model_feature_importance.xlsx"
    fi_df.to_excel(output_path, index=False)

    print("\nTop 25 Feature Importances:")
    print(fi_df.head(25))
    print(f"\nSaved feature importance to: {output_path}")


def main():
    df = load_data()

    print("=" * 80)
    print("ROBUST FINAL MODEL TRAINING")
    print("=" * 80)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns before drop: {df.shape[1]}")
    print(f"Years: {sorted(df['Report_Year'].unique())}")

    train_df, test_df = time_based_split(df)

    X_train = train_df.drop(columns=DROP_COLUMNS)
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=DROP_COLUMNS)
    y_test = test_df[TARGET_COLUMN]

    print(f"\nTrain rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")
    print(f"Input columns: {X_train.shape[1]}")
    print(f"Columns used:")
    for col in X_train.columns:
        print(f"- {col}")

    preprocessor = build_preprocessor(X_train)

    models = get_models()

    results = []
    trained_models = {}
    predictions = test_df.copy()

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(f"Training {model_name}")
        print("=" * 80)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_metrics = evaluate(f"{model_name}_TRAIN", y_train, train_pred)
        test_metrics = evaluate(f"{model_name}_TEST", y_test, test_pred)

        results.append(train_metrics)
        results.append(test_metrics)

        trained_models[model_name] = pipeline

        predictions[f"{model_name}_Prediction"] = test_pred
        predictions[f"{model_name}_Error"] = test_pred - y_test
        predictions[f"{model_name}_Abs_Error"] = predictions[f"{model_name}_Error"].abs()

        print(f"Train MAE: {train_metrics['MAE']:.4f}")
        print(f"Train RMSE: {train_metrics['RMSE']:.4f}")
        print(f"Train R2: {train_metrics['R2']:.4f}")

        print(f"Test MAE: {test_metrics['MAE']:.4f}")
        print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
        print(f"Test R2: {test_metrics['R2']:.4f}")
        print(f"Test Bias: {test_metrics['Bias']:.4f}")
        print(f"Predicted Std: {test_metrics['Predicted_Std']:.4f}")

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "robust_model_results.xlsx"
    predictions_path = OUTPUT_DIR / "robust_model_predictions.xlsx"

    results_df.to_excel(results_path, index=False)
    predictions.to_excel(predictions_path, index=False)

    test_results = results_df[results_df["Model"].str.contains("_TEST")].copy()
    best_row = test_results.sort_values("MAE", ascending=True).iloc[0]

    best_model_name = best_row["Model"].replace("_TEST", "")
    best_pipeline = trained_models[best_model_name]

    model_path = MODEL_DIR / "robust_final_model.pkl"
    joblib.dump(best_pipeline, model_path)

    print("\n" + "=" * 80)
    print("ROBUST MODEL COMPARISON")
    print("=" * 80)
    print(results_df)

    print("\n" + "=" * 80)
    print("BEST ROBUST MODEL")
    print("=" * 80)
    print(best_row)
    print(f"Saved model to: {model_path}")

    save_feature_importance(best_pipeline, best_model_name)

    print(f"\nSaved results to: {results_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()