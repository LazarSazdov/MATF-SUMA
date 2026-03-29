import sys
import os
import time

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import run_preprocessing, INSURERS, PRICE_COLS
from train import train_all_insurers

from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np


def generate_predictions(test_df, feature_cols, cat_cols, model_dir, output_path):
    """Load trained models and generate predictions for a test set."""
    cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    predictions = pd.DataFrame({'quote_id': test_df['quote_id']})

    for insurer in INSURERS:
        model_path = os.path.join(model_dir, f"catboost_insurer_{insurer}.cbm")
        model = CatBoostRegressor()
        model.load_model(model_path)

        X_test = test_df[feature_cols]
        preds = model.predict(X_test)
        predictions[f"Insurer_{insurer}_price"] = preds

        print(f"  Insurer {insurer}: mean={preds.mean():.2f}, std={preds.std():.2f}")
        del model

    # Save with semicolon separator (matches baseline format)
    predictions.to_csv(output_path, sep=';', index=False)
    print(f"Saved {output_path}: {predictions.shape}")
    return predictions


def main():
    start = time.time()

    # ---- STEP 1: PREPROCESS ----
    print("=" * 70)
    print("STEP 1: PREPROCESSING")
    print("=" * 70)
    train_part, val_part, test_b2, test_b3, feature_cols, cat_cols = run_preprocessing()

    print(f"\nPreprocessing done in {time.time()-start:.0f}s")
    print(f"Features: {len(feature_cols)}, Categoricals: {len(cat_cols)}")

    # ---- STEP 2: TRAIN ----
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING 11 CATBOOST REGRESSORS")
    print("=" * 70)

    results = train_all_insurers(
        train_df=train_part,
        val_df=val_part,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        model_dir="models",
        n_jobs=11,
    )

    # ---- STEP 3: GENERATE SUBMISSIONS ----
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING PREDICTIONS")
    print("=" * 70)

    print("\n--- Block 2 predictions ---")
    generate_predictions(test_b2, feature_cols, cat_cols, "models",
                         "submissions/submission_block2.csv")

    print("\n--- Block 3 predictions ---")
    generate_predictions(test_b3, feature_cols, cat_cols, "models",
                         "submissions/submission_block3.csv")

    print(f"\nTotal runtime: {(time.time()-start)/60:.1f} minutes")


if __name__ == '__main__':
    main()
