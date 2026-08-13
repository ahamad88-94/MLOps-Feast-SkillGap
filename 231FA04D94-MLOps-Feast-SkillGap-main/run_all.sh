#!/usr/bin/env bash
# Reproduces the entire pipeline end to end:
# dataset -> features -> feast apply -> historical retrieval ->
# materialize -> online retrieval -> train model -> online prediction
set -e

echo "== 1. Generating raw dataset =="
python3 data/generate_dataset.py

echo "== 2. Engineering features =="
python3 data/feature_engineering.py

cd feature_repo

echo "== 3. feast apply =="
feast apply

echo "== 4. Historical feature retrieval =="
python3 ../scripts/historical_retrieval.py

echo "== 5. Materializing to online store =="
python3 ../scripts/materialize.py

echo "== 6. Online feature retrieval =="
python3 ../scripts/online_retrieval.py

echo "== 7. Training model on historical features =="
python3 ../scripts/train_model.py

echo "== 8. Online prediction demo =="
python3 ../scripts/predict_online.py

echo "== Done. See outputs/ for all results. =="
