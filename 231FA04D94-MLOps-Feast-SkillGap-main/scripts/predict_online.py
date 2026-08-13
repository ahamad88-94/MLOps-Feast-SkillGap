"""
predict_online.py
--------------------
End-to-end online inference demo: pulls a skill's features from the
Feast online store (low latency) and feeds them into the trained model
to produce a live prediction, exactly as a real prediction service would.

Run from the feature_repo/ directory:
    python ../scripts/predict_online.py
"""

import json

import joblib
import pandas as pd
from feast import FeatureStore

MODEL_PATH = "../outputs/model.joblib"
CATEGORY_ENCODER_PATH = "../outputs/category_encoder.joblib"
OUTPUT_PATH = "../outputs/final_prediction.json"

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

# The skill we want a live curriculum-review prediction for.
TARGET_SKILL_ID = 29  # Git/Version Control


def main():
    store = FeatureStore(repo_path=".")
    model = joblib.load(MODEL_PATH)
    le = joblib.load(CATEGORY_ENCODER_PATH)

    online_features = store.get_online_features(
        features=FEATURES,
        entity_rows=[{"skill_id": TARGET_SKILL_ID}],
    ).to_dict()

    row = {k: v[0] for k, v in online_features.items()}
    row["category_encoded"] = le.transform([row["category"]])[0]

    X = pd.DataFrame([{col: row[col] for col in NUMERIC_COLUMNS + ["category_encoded"]}])
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    result = {
        "skill_id": TARGET_SKILL_ID,
        "input_features": row,
        "predicted_high_priority_gap": prediction,
        "probability_high_priority_gap": round(probability, 4),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(json.dumps(result, indent=2, default=str))
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
