"""
online_retrieval.py
----------------------
Demonstrates low-latency online feature retrieval via
Feast's get_online_features(), the same call pattern a live
prediction service would use.

Run from the feature_repo/ directory:
    python ../scripts/online_retrieval.py
"""

import json

from feast import FeatureStore

OUTPUT_JSON_PATH = "../outputs/online_features.json"

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

# A handful of skill_ids to simulate an incoming prediction request for
# (e.g. skills a curriculum committee is currently reviewing).
SAMPLE_SKILL_IDS = [1, 16, 29, 41]


def main():
    store = FeatureStore(repo_path=".")

    entity_rows = [{"skill_id": sid} for sid in SAMPLE_SKILL_IDS]

    online_features = store.get_online_features(
        features=FEATURES,
        entity_rows=entity_rows,
    ).to_dict()

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(online_features, f, indent=2, default=str)

    print(f"Retrieved online features for skill_ids={SAMPLE_SKILL_IDS} -> {OUTPUT_JSON_PATH}")
    print(json.dumps(online_features, indent=2, default=str))


if __name__ == "__main__":
    main()
