"""
feature_engineering.py
-----------------------
Reads data/skill_gap_raw.csv and engineers the features that Feast will
serve, plus the ML label. Writes a Parquet file into feature_repo/data/
(the offline store Feast reads from) and a labeled entity dataframe into
data/skill_gap_entities.csv (used to drive historical retrieval).

Run:
    python data/feature_engineering.py
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/skill_gap_raw.csv"
FEATURE_PARQUET_PATH = "feature_repo/data/skill_gap_features.parquet"
ENTITY_CSV_PATH = "data/skill_gap_entities.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Normalisation (0-1 scale) -----------------------------------
    df["curriculum_coverage_norm"] = df["curriculum_coverage_score"] / 100.0
    df["industry_demand_norm"] = df["industry_demand_score"] / 100.0

    # --- Core gap metric ------------------------------------------------
    # Positive => industry demands the skill more than the curriculum
    # currently teaches it (a "gap"). Negative => curriculum over-invests
    # relative to current industry demand.
    df["demand_supply_gap"] = df["industry_demand_norm"] - df["curriculum_coverage_norm"]
    df["gap_score"] = (df["demand_supply_gap"] * 100).round(2)

    # --- Evidence-strength feature ---------------------------------------
    # Job postings are heavy-tailed, so a log transform makes the feature
    # more model-friendly.
    df["job_postings_log"] = np.log1p(df["num_job_postings"]).round(4)

    # --- ML label ---------------------------------------------------------
    # A skill is flagged "high_priority_gap" (1) if its gap_score sits in
    # the top quartile of all skills, i.e. it is one of the most
    # under-taught-relative-to-demand skills in the catalogue.
    threshold = df["gap_score"].quantile(0.75)
    df["high_priority_gap"] = (df["gap_score"] >= threshold).astype(int)

    # Feast requires a timestamp column already present in event_timestamp.
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    # created_timestamp is used by Feast for de-duplication when multiple
    # rows share the same entity + event_timestamp.
    df["created_timestamp"] = pd.Timestamp.utcnow().tz_localize(None)

    return df


def main():
    raw = pd.read_csv(RAW_PATH, parse_dates=["event_timestamp"])
    features = engineer_features(raw)

    feature_columns = [
        "skill_id",
        "skill_name",
        "category",
        "curriculum_hours",
        "curriculum_coverage_score",
        "industry_demand_score",
        "num_job_postings",
        "avg_salary_premium_pct",
        "curriculum_coverage_norm",
        "industry_demand_norm",
        "demand_supply_gap",
        "gap_score",
        "job_postings_log",
        "event_timestamp",
        "created_timestamp",
    ]
    features[feature_columns].to_parquet(FEATURE_PARQUET_PATH, index=False)
    print(f"Wrote feature table ({len(features)} rows) -> {FEATURE_PARQUET_PATH}")

    # Entity dataframe (entity key + label + timestamp) used later to pull
    # historical features via feast.get_historical_features().
    entity_df = features[["skill_id", "skill_name", "high_priority_gap", "event_timestamp"]]
    entity_df.to_csv(ENTITY_CSV_PATH, index=False)
    print(f"Wrote entity/label table ({len(entity_df)} rows) -> {ENTITY_CSV_PATH}")


if __name__ == "__main__":
    main()
