import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


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
    train_df = df[df["Report_Year"] < 2026].copy()
    test_df = df[df["Report_Year"] == 2026].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty. Check Report_Year.")

    return train_df, test_df


def prepare_xy(train_df, test_df):
    drop_cols = [TARGET_COLUMN] + LEAKAGE_COLUMNS

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test


def build_preprocessor(X_train):
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

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

    return preprocessor, numeric_cols, categorical_cols


def evaluate_predictions(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Actual_Mean": y_true.mean(),
        "Predicted_Mean": y_pred.mean(),
        "Actual_Std": y_true.std(),
        "Predicted_Std": y_pred.std(),
        "Bias": (y_pred - y_true).mean(),
    }


def get_models():
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return models


def save_predictions(test_df, predictions_dict):
    output_df = test_df.copy()

    for model_name, preds in predictions_dict.items():
        output_df[f"{model_name}_Prediction"] = preds
        output_df[f"{model_name}_Error"] = preds - output_df[TARGET_COLUMN]
        output_df[f"{model_name}_Abs_Error"] = output_df[f"{model_name}_Error"].abs()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "advanced_model_predictions.xlsx"
    output_df.to_excel(output_path, index=False)

    print(f"\nSaved predictions to: {output_path}")


def save_feature_importance(best_pipeline, X_train, model_name):
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        output_path = OUTPUT_DIR / f"{model_name.lower()}_feature_importance.xlsx"
        fi_df.to_excel(output_path, index=False)

        print(f"Saved feature importance to: {output_path}")
        print("\nTop 20 Feature Importances:")
        print(fi_df.head(20))
    else:
        print(f"Feature importance not available for {model_name}.")


def main():
    df = load_data()

    print("=" * 80)
    print("ADVANCED MODEL TRAINING")
    print("=" * 80)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Years: {sorted(df['Report_Year'].unique())}")

    train_df, test_df = time_based_split(df)
    X_train, y_train, X_test, y_test = prepare_xy(train_df, test_df)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    print(f"\nTrain rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")
    print(f"Input columns: {X_train.shape[1]}")
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    models = get_models()

    results = []
    trained_pipelines = {}
    predictions_dict = {}

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

        train_metrics = evaluate_predictions(
            f"{model_name}_TRAIN",
            y_train,
            train_pred,
        )

        test_metrics = evaluate_predictions(
            f"{model_name}_TEST",
            y_test,
            test_pred,
        )

        results.append(train_metrics)
        results.append(test_metrics)

        trained_pipelines[model_name] = pipeline
        predictions_dict[model_name] = test_pred

        print(f"Train MAE: {train_metrics['MAE']:.4f}")
        print(f"Train RMSE: {train_metrics['RMSE']:.4f}")
        print(f"Train R2: {train_metrics['R2']:.4f}")

        print(f"Test MAE: {test_metrics['MAE']:.4f}")
        print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
        print(f"Test R2: {test_metrics['R2']:.4f}")
        print(f"Test Bias: {test_metrics['Bias']:.4f}")
        print(f"Actual Std: {test_metrics['Actual_Std']:.4f}")
        print(f"Predicted Std: {test_metrics['Predicted_Std']:.4f}")

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "advanced_model_results.xlsx"
    results_df.to_excel(results_path, index=False)

    print("\n" + "=" * 80)
    print("ADVANCED MODEL COMPARISON")
    print("=" * 80)
    print(results_df)

    test_results = results_df[results_df["Model"].str.contains("_TEST")].copy()
    best_row = test_results.sort_values("MAE", ascending=True).iloc[0]

    best_model_name = best_row["Model"].replace("_TEST", "")
    best_pipeline = trained_pipelines[best_model_name]

    best_model_path = MODEL_DIR / "final_decision_model.pkl"
    joblib.dump(best_pipeline, best_model_path)

    print("\n" + "=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(best_row)
    print(f"Saved final model to: {best_model_path}")

    save_predictions(test_df, predictions_dict)
    save_feature_importance(best_pipeline, X_train, best_model_name)

    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()