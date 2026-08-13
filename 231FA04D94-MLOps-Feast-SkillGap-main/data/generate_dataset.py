"""
generate_dataset.py
--------------------
Generates the raw Curriculum-Industry Skill-Gap dataset.

The dataset simulates, for a fixed catalogue of technical / soft skills,
how much a college curriculum covers each skill versus how much that
skill is currently demanded by industry. This raw dataset is the input
to feature_engineering.py, which turns it into the Feast feature table.

Run:
    python data/generate_dataset.py
Output:
    data/skill_gap_raw.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Skill catalogue (entities)
# ---------------------------------------------------------------------------
# Each category contributes a fixed set of representative skills so the
# dataset is reproducible and easy to explain in the README.
SKILL_CATALOGUE = {
    "Programming": [
        "Python", "Java", "C++", "JavaScript", "SQL",
        "Go", "TypeScript", "R", "Shell Scripting", "Rust",
    ],
    "Data Science": [
        "Machine Learning", "Deep Learning", "Data Visualization",
        "Statistics", "Pandas/NumPy", "Feature Engineering",
        "NLP", "Computer Vision", "MLOps", "A/B Testing",
    ],
    "Cloud & DevOps": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes",
        "CI/CD", "Terraform", "Linux Administration",
        "Git/Version Control", "Monitoring & Logging",
    ],
    "Cybersecurity": [
        "Network Security", "Penetration Testing", "Cryptography",
        "Cloud Security", "Identity & Access Management",
        "Security Compliance", "Incident Response",
        "Application Security", "Threat Modeling", "SOC Operations",
    ],
    "Soft Skills": [
        "Communication", "Teamwork", "Problem Solving",
        "Time Management", "Leadership", "Adaptability",
        "Critical Thinking", "Presentation Skills",
        "Conflict Resolution", "Agile Collaboration",
    ],
    "Emerging Tech": [
        "Generative AI", "LLM Fine-tuning", "Prompt Engineering",
        "Blockchain", "IoT", "Edge Computing", "AR/VR",
        "Quantum Computing Basics", "Robotics", "Web3",
    ],
}


def build_skill_rows():
    rows = []
    skill_id = 1
    for category, skills in SKILL_CATALOGUE.items():
        for skill_name in skills:
            rows.append({"skill_id": skill_id, "skill_name": skill_name, "category": category})
            skill_id += 1
    return rows


def simulate_metrics(rows):
    """Attach curriculum-side and industry-side raw metrics to every skill."""

    # Category-level base tendencies: some categories are historically
    # over-taught (soft skills, core programming) and some are under-taught
    # relative to industry demand (emerging tech, cloud, security).
    category_bias = {
        "Programming": {"curriculum": 65, "industry": 55},
        "Data Science": {"curriculum": 45, "industry": 70},
        "Cloud & DevOps": {"curriculum": 30, "industry": 75},
        "Cybersecurity": {"curriculum": 25, "industry": 65},
        "Soft Skills": {"curriculum": 55, "industry": 50},
        "Emerging Tech": {"curriculum": 15, "industry": 60},
    }

    for row in rows:
        cat = row["category"]
        base = category_bias[cat]

        # Curriculum-side: how many teaching hours + how well the curriculum
        # covers the skill (0-100 coverage score).
        curriculum_hours = max(2, int(np.random.normal(loc=base["curriculum"] / 2.2, scale=8)))
        curriculum_coverage_score = float(
            np.clip(np.random.normal(loc=base["curriculum"], scale=15), 2, 98)
        )

        # Industry-side: job postings mentioning the skill (evidence of
        # demand) + an industry demand score (0-100) + salary premium.
        num_job_postings = max(5, int(np.random.normal(loc=base["industry"] * 12, scale=250)))
        industry_demand_score = float(
            np.clip(np.random.normal(loc=base["industry"], scale=14), 5, 99)
        )
        avg_salary_premium_pct = float(
            np.clip(np.random.normal(loc=industry_demand_score * 0.25, scale=4), 0, 40)
        )

        # event_timestamp: when this evidence point was collected. Spread
        # over the last 90 days so Feast's point-in-time joins have
        # something meaningful to do.
        days_ago = int(np.random.randint(0, 90))
        event_timestamp = datetime.utcnow() - timedelta(days=days_ago)

        row.update(
            {
                "curriculum_hours": curriculum_hours,
                "curriculum_coverage_score": round(curriculum_coverage_score, 2),
                "num_job_postings": num_job_postings,
                "industry_demand_score": round(industry_demand_score, 2),
                "avg_salary_premium_pct": round(avg_salary_premium_pct, 2),
                "event_timestamp": event_timestamp,
            }
        )
    return rows


def main():
    rows = build_skill_rows()
    rows = simulate_metrics(rows)
    df = pd.DataFrame(rows)
    df = df.sort_values("skill_id").reset_index(drop=True)

    out_path = "data/skill_gap_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} skills -> {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
