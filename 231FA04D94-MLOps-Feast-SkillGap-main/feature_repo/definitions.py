"""
definitions.py
----------------
Feast object definitions for the Curriculum-Industry Skill-Gap feature
store: one Entity (skill), one FileSource pointing at the offline Parquet
data, and one FeatureView exposing the engineered features.

This file is picked up automatically by `feast apply` when run from
inside feature_repo/.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
# The entity is a single "skill" (e.g. Python, Docker, Communication).
# skill_id is the join key used everywhere features are requested.
skill = Entity(
    name="skill",
    join_keys=["skill_id"],
    value_type=ValueType.INT64,
    description="A single curriculum/industry skill, identified by skill_id.",
)

# ---------------------------------------------------------------------------
# Data source (offline store)
# ---------------------------------------------------------------------------
skill_gap_source = FileSource(
    name="skill_gap_source",
    path="data/skill_gap_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ---------------------------------------------------------------------------
# FeatureView
# ---------------------------------------------------------------------------
skill_gap_features_view = FeatureView(
    name="skill_gap_features_view",
    entities=[skill],
    ttl=timedelta(days=365),
    schema=[
        Field(name="curriculum_hours", dtype=Int64),
        Field(name="curriculum_coverage_score", dtype=Float32),
        Field(name="industry_demand_score", dtype=Float32),
        Field(name="num_job_postings", dtype=Int64),
        Field(name="avg_salary_premium_pct", dtype=Float32),
        Field(name="curriculum_coverage_norm", dtype=Float32),
        Field(name="industry_demand_norm", dtype=Float32),
        Field(name="demand_supply_gap", dtype=Float32),
        Field(name="gap_score", dtype=Float32),
        Field(name="job_postings_log", dtype=Float32),
        Field(name="category", dtype=String),
    ],
    online=True,
    source=skill_gap_source,
    tags={"team": "curriculum-analytics"},
)
