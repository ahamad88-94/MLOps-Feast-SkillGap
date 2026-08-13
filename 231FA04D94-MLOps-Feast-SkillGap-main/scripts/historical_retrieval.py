"""
historical_retrieval.py
-------------------------
Demonstrates Feast's point-in-time correct historical feature retrieval.

Reads the entity dataframe (skill_id + label + event_timestamp), joins in
the engineered features from the offline store, and writes the joined
training dataframe to outputs/historical_features.csv.

Run from the feature_repo/ directory:
    python ../scripts/historical_retrieval.py
"""

import pandas as pd
from feast import FeatureStore

ENTITY_CSV_PATH = "../data/skill_gap_entities.csv"
OUTPUT_CSV_PATH = "../outputs/historical_features.csv"

FEATURES = [
    "skill_gap_features_view:curriculum_hours",
    "skill_gap_features_view:curriculum_coverage_score",
    "skill_gap_features_view:industry_demand_score",
    "skill_gap_features_view:num_job_postings",
    "skill_gap_features_view:avg_salary_premium_pct",
    "skill_gap_features_view:curriculum_coverage_norm",
    "skill_gap_features_view:industry_demand_norm",
    "skill_gap_features_view:demand_supply_gap",
    "skill_gap_features_view:gap_score",
    "skill_gap_features_view:job_postings_log",
    "skill_gap_features_view:category",
]


def main():
    store = FeatureStore(repo_path=".")

    entity_df = pd.read_csv(ENTITY_CSV_PATH, parse_dates=["event_timestamp"])

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURES,
    ).to_df()

    training_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Retrieved {len(training_df)} rows of historical features -> {OUTPUT_CSV_PATH}")
    print(training_df.head(10).to_string())


if __name__ == "__main__":
    main()
