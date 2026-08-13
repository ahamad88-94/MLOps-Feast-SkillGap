# Curriculum-Industry Skill Feature Store Using Feast

## Student Details
- **Name:** Shaik Nissar Ahamad
- **Register Number:** 231FA04D94
- **Section:** 3

---

## Problem Statement

Colleges design curricula on a multi-year cycle, but industry skill demand shifts
much faster than that. As a result, some skills are **over-taught** relative to
how much the market currently wants them, while other, often newer, skills are
**under-taught** even though industry demand for them is high. This is the
*curriculum-industry skill gap*.

This project represents each skill as an **entity** with two sides of evidence —
how much the curriculum currently covers it, and how much industry currently
demands it — computes a **gap score** between the two, and serves that
information through a **Feast feature store** so it can be reused consistently
for analytics, dashboards, and machine-learning model training/inference,
instead of being recalculated ad hoc every time.

## Dataset

- **Number of skills (entities):** 60
- **Categories (6 × 10 skills each):** Programming, Data Science, Cloud & DevOps,
  Cybersecurity, Soft Skills, Emerging Tech
- **Dataset columns (raw, `data/skill_gap_raw.csv`):**
  - `skill_id` — unique entity key
  - `skill_name` — human-readable skill name
  - `category` — skill category
  - `curriculum_hours` — simulated hours of instruction dedicated to the skill
  - `curriculum_coverage_score` (0–100) — how well the curriculum covers the skill
  - `num_job_postings` — simulated job postings mentioning the skill
  - `industry_demand_score` (0–100) — how strongly industry currently demands the skill
  - `avg_salary_premium_pct` — simulated salary premium associated with the skill
  - `event_timestamp` — when this evidence snapshot was recorded (last 90 days)
- **Target:** `high_priority_gap` (binary) — 1 if a skill's `gap_score` falls in
  the top quartile of all skills (i.e. industry demand is significantly ahead
  of curriculum coverage), 0 otherwise.
- **How the entries were created:** `data/generate_dataset.py` builds a fixed
  catalogue of 60 real-world skill names across 6 categories, then simulates
  curriculum-side and industry-side metrics per skill using category-level
  base tendencies (e.g. "Emerging Tech" skills are biased toward low
  curriculum coverage / high industry demand) plus Gaussian noise, with a
  fixed random seed (42) for reproducibility.

## Feature Engineering

`data/feature_engineering.py` reads the raw dataset and produces the feature
table that Feast serves (`feature_repo/data/skill_gap_features.parquet`):

| Feature | Meaning |
|---|---|
| `curriculum_hours` | Raw teaching hours dedicated to the skill |
| `curriculum_coverage_score` | Raw 0–100 curriculum coverage score |
| `industry_demand_score` | Raw 0–100 industry demand score |
| `num_job_postings` | Raw count of job postings mentioning the skill |
| `avg_salary_premium_pct` | Raw salary premium associated with the skill |
| `curriculum_coverage_norm` | `curriculum_coverage_score / 100`, scaled to 0–1 |
| `industry_demand_norm` | `industry_demand_score / 100`, scaled to 0–1 |
| `demand_supply_gap` | `industry_demand_norm − curriculum_coverage_norm`; positive means under-taught relative to demand |
| `gap_score` | `demand_supply_gap × 100`, the same gap on a 0–100-style scale, used to derive the label |
| `job_postings_log` | `log1p(num_job_postings)`, a log transform so the heavy-tailed posting counts are model-friendly |
| `category` | Skill category, kept as a feature for the model (label-encoded before training) |

`high_priority_gap` (the label) is computed in the same script but is **not**
served as a training feature, since it is derived directly from `gap_score`
and including it would leak the label.

### Example: how one feature was calculated

For `Git/Version Control` (skill_id = 29): `curriculum_coverage_score = 34.45`,
`industry_demand_score = 79.85`.

1. Normalise: `curriculum_coverage_norm = 34.45 / 100 = 0.3445`,
   `industry_demand_norm = 79.85 / 100 = 0.7985`
2. Gap: `demand_supply_gap = 0.7985 − 0.3445 = 0.4540`
3. Scaled gap score: `gap_score = 0.4540 × 100 = 45.40`

Because `45.40` sits in the top quartile of `gap_score` across all 60 skills,
`high_priority_gap = 1` for this skill — the model later predicts this same
outcome from live online features (see **Results**).

## Feast Architecture

```
Original Dataset
      ↓
Feature Engineering
      ↓
Parquet Offline Data
      ↓
Feast FeatureView
      ↓
 ┌─────────────────────┐
 ↓                     ↓
Historical Features   Materialization
 ↓                     ↓
Model Training       Online Store
                       ↓
                  Online Retrieval
                       ↓
                    Prediction
```

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── run_all.sh                     # reproduces the whole pipeline end to end
├── data/
│   ├── generate_dataset.py        # generates the raw skill-gap dataset
│   ├── skill_gap_raw.csv          # generated raw dataset
│   ├── feature_engineering.py     # builds the Feast-ready feature table + labels
│   └── skill_gap_entities.csv     # entity + label + timestamp table
├── feature_repo/
│   ├── feature_store.yaml         # Feast project config (local file offline store + SQLite online store)
│   ├── definitions.py             # Entity, FileSource, FeatureView definitions
│   └── data/
│       ├── skill_gap_features.parquet  # offline feature data
│       ├── registry.db            # Feast registry (created by `feast apply`)
│       └── online_store.db        # SQLite online store (created by materialize)
├── scripts/
│   ├── historical_retrieval.py    # get_historical_features() demo
│   ├── materialize.py             # materializes offline -> online store
│   ├── online_retrieval.py        # get_online_features() demo
│   ├── train_model.py             # trains a model on Feast historical features
│   └── predict_online.py          # live prediction using Feast online features
└── outputs/
    ├── historical_features.csv
    ├── online_features.json
    ├── model.joblib
    ├── category_encoder.joblib
    ├── model_metrics.json
    └── final_prediction.json
```

## Implementation

- **Entity:** `skill`, joined on `skill_id` (int). Represents a single curriculum/
  industry skill such as "Python" or "Kubernetes".
- **Data source:** a Feast `FileSource` pointing at
  `feature_repo/data/skill_gap_features.parquet`, with `event_timestamp` as
  the timestamp field and `created_timestamp` for de-duplication.
- **FeatureView:** `skill_gap_features_view`, defined in
  `feature_repo/definitions.py`, exposes the 10 engineered features listed
  above with a 365-day TTL, and is registered with `feast apply`.
- **Historical retrieval:** `scripts/historical_retrieval.py` loads the
  entity/label table (`data/skill_gap_entities.csv`) and calls
  `store.get_historical_features()` to point-in-time-correctly join in the
  Feast features, producing the training dataframe.
- **Model:** `scripts/train_model.py` trains a `RandomForestClassifier`
  (scikit-learn) on the historical features to predict `high_priority_gap`,
  using an 75/25 train/test split (`random_state=42`, stratified).
- **Materialization:** `scripts/materialize.py` calls `store.materialize()`
  to push the latest feature values into the SQLite online store.
- **Online retrieval:** `scripts/online_retrieval.py` and
  `scripts/predict_online.py` call `store.get_online_features()` to fetch
  low-latency feature values for specific `skill_id`s, exactly as a live
  prediction service would, and feed them into the trained model for a
  final prediction.

## How to Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh
```

This runs, in order: dataset generation → feature engineering → `feast apply`
→ historical retrieval → materialization → online retrieval → model training
→ online prediction, writing all results into `outputs/`.

## Results

### Historical feature output (sample, full file in `outputs/historical_features.csv`)

| skill_id | skill_name | category | curriculum_coverage_score | industry_demand_score | gap_score | high_priority_gap |
|---|---|---|---|---|---|---|
| 29 | Git/Version Control | Cloud & DevOps | 34.45 | 79.85 | 45.40 | 1 |
| 16 | Feature Engineering | Data Science | 33.16 | 67.03 | 33.87 | 1 |
| 1 | Python | Programming | 62.93 | 76.32 | 13.39 | 0 |
| 41 | Communication | Soft Skills | 98.00 | 65.90 | -32.10 | 0 |

### Model accuracy

`RandomForestClassifier` trained on Feast historical features:

- **Accuracy:** 0.60 (on a held-out 15-skill test split, from a 60-skill dataset)
- Full precision/recall/F1 breakdown is in `outputs/model_metrics.json`.
  With only 60 total skills, the test set is small (15 rows), so this
  accuracy should be read as a pipeline demonstration rather than a
  production-grade result (see Limitations below).

### Online feature output (sample, full file in `outputs/online_features.json`)

Requested live for `skill_id = 29` (Git/Version Control) from the SQLite
online store:

```json
{
  "skill_id": 29,
  "category": "Cloud & DevOps",
  "curriculum_coverage_score": 34.45,
  "industry_demand_score": 79.85,
  "gap_score": 45.40,
  "num_job_postings": 962,
  "avg_salary_premium_pct": 17.24
}
```

### Final prediction

Using the online features above as model input
(`scripts/predict_online.py`, full output in `outputs/final_prediction.json`):

- **Predicted `high_priority_gap`:** `1`
- **Predicted probability:** `0.77`

This matches the ground-truth label for Git/Version Control, correctly
flagging it as a skill where the curriculum should be prioritised for
review because industry demand is well ahead of current coverage.

---

## Required Analysis

**1. What is the entity in your Feast implementation?**
`skill`, joined on `skill_id`. Each row represents one skill in the
curriculum-industry catalogue (e.g. Python, Docker, Communication).

**2. List the features stored in your FeatureView.**
`curriculum_hours`, `curriculum_coverage_score`, `industry_demand_score`,
`num_job_postings`, `avg_salary_premium_pct`, `curriculum_coverage_norm`,
`industry_demand_norm`, `demand_supply_gap`, `gap_score`, `job_postings_log`,
`category`.

**3. Explain how one feature was calculated.**
See "Example: how one feature was calculated" above — `gap_score` for
Git/Version Control is computed as
`(industry_demand_norm − curriculum_coverage_norm) × 100 = (0.7985 − 0.3445) × 100 = 45.40`.

**4. What is the difference between your original dataset and the feature dataset?**
The original dataset (`data/skill_gap_raw.csv`) contains only the raw,
directly-observed metrics (curriculum hours/coverage, job postings, industry
demand, salary premium). The feature dataset
(`feature_repo/data/skill_gap_features.parquet`) additionally contains
derived, model-ready features — normalised scores, the demand-supply gap, a
log-transformed posting count — plus the `created_timestamp` column Feast
needs for offline/online serving. The raw dataset is human-readable evidence;
the feature dataset is what the feature store actually serves.

**5. What is the purpose of the offline store?**
The offline store (a Parquet file in this project) holds the full historical
feature data and supports point-in-time-correct joins for generating training
datasets via `get_historical_features()`. It's optimised for large batch
reads, not for low-latency single-entity lookups.

**6. What is the purpose of the online store?**
The online store (SQLite here) holds only the *latest* feature value per
entity and is optimised for fast, low-latency lookups
(`get_online_features()`) — the kind of lookup a real-time prediction
service needs when it receives a request for one specific skill.

**7. What is the purpose of `feast apply`?**
`feast apply` reads the Python object definitions in `definitions.py`
(entities, data sources, feature views), validates them, and registers them
into the Feast registry (`registry.db`). It also provisions the
corresponding online-store tables. Nothing is queryable through Feast until
`feast apply` has run.

**8. What does materialization do?**
Materialization (`store.materialize()` / `feast materialize`) copies the
latest feature values for a given time range from the offline store into the
online store, so that `get_online_features()` has current data to serve.
Without materialization, the online store stays empty even if the offline
data is up to date.

**9. What is the advantage of retrieving features through Feast instead of manually calculating them separately during training and prediction?**
Feast guarantees the *same* feature definitions and transformation logic are
used both at training time (historical retrieval) and at serving time
(online retrieval), eliminating training/serving skew. It also centralises
feature definitions so multiple models or teams can reuse the same features
instead of re-implementing the same calculation (and risking subtle bugs or
inconsistencies) in different places, and it handles point-in-time
correctness automatically during historical joins.

**10. State two limitations of your current dataset.**
1. It is synthetically generated with category-level heuristics and random
   noise rather than sourced from real curriculum documents and live job
   postings, so the specific scores don't reflect an actual institution's
   situation.
2. With only 60 skills and a single snapshot-style timestamp window per
   skill, the dataset is too small and too static to support robust
   model evaluation or to capture how skill demand trends change over
   time (only 90 days of simulated recency is represented).

**11. State two ways your feature store could be improved when more curriculum and industry evidence becomes available.**
1. Replace the file-based offline store with a scalable warehouse source
   (e.g. BigQuery/Snowflake `FileSource` alternative) once real curriculum
   syllabi and live job-board data are ingested continuously, and add a
   streaming source so `industry_demand_score` reflects near-real-time
   market signals instead of a static snapshot.
2. Add additional FeatureViews with a shorter TTL for fast-changing signals
   (e.g. weekly job-posting counts) versus a longer TTL for slow-changing
   ones (e.g. curriculum hours), and register feature services so different
   consumers (a dashboard vs. a prediction API) can request only the
   feature subset they need.
