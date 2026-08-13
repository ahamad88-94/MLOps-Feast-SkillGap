"""
materialize.py
----------------
Materializes the latest feature values from the offline store (Parquet)
into the online store (SQLite) so they can be served with low latency
via get_online_features().

Run from the feature_repo/ directory:
    python ../scripts/materialize.py
"""

from datetime import datetime, timedelta, timezone

from feast import FeatureStore


def main():
    store = FeatureStore(repo_path=".")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365)

    store.materialize(start_date=start_date, end_date=end_date)
    print(f"Materialized features from {start_date} to {end_date} into the online store.")


if __name__ == "__main__":
    main()
