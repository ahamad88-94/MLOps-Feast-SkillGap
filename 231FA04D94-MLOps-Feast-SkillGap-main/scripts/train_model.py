"""
train_model.py
-----------------
Trains a simple classifier that predicts high_priority_gap (whether a
skill needs urgent curriculum attention) using features pulled from
Feast via get_historical_features(). Saves the trained model and its
accuracy metrics.

Run from the feature_repo/ directory:
    python ../scripts/train_model.py
"""

import json

import joblib
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ENTITY_CSV_PATH = "../data/skill_gap_entities.csv"
MODEL_PATH = "../outputs/model.joblib"
METRICS_PATH = "../outputs/model_metrics.json"
CATEGORY_ENCODER_PATH = "../outputs/category_encoder.joblib"

FEATURES = [
    "skill_gap_features_view:curriculum_hours",
    "skill_gap_features_view:curriculum_coverage_score",
    "skill_gap_features_view:industry_demand_score",
    "skill_gap_features_view:num_job_postings",
    "skill_gap_features_view:avg_salary_premium_pct",
    "skill_gap_features_view:curriculum_coverage_norm",
    "skill_gap_features_view:industry_demand_norm",
    "skill_gap_features_view:job_postings_log",
    "skill_gap_features_view:category",
]

NUMERIC_COLUMNS = [
    "curriculum_hours",
    "curriculum_coverage_score",
    "industry_demand_score",
    "num_job_postings",
    "avg_salary_premium_pct",
    "curriculum_coverage_norm",
    "industry_demand_norm",
    "job_postings_log",
]


def main():
    store = FeatureStore(repo_path=".")
    entity_df = pd.read_csv(ENTITY_CSV_PATH, parse_dates=["event_timestamp"])

    df = store.get_historical_features(entity_df=entity_df, features=FEATURES).to_df()

    # Note: gap_score / demand_supply_gap are intentionally excluded from
    # training features because high_priority_gap is derived directly
    # from gap_score (that would leak the label). Everything else Feast
    # serves is fair game.
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])

    feature_cols = NUMERIC_COLUMNS + ["category_encoded"]
    X = df[feature_cols]
    y = df["high_priority_gap"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": round(accuracy, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": feature_cols,
        "classification_report": report,
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, CATEGORY_ENCODER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
