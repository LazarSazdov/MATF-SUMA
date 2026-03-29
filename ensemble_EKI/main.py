"""
Ensemble pipeline for Insurers E, I, K only.
Uses CatBoost + LightGBM blend with harsher preprocessing.
Optimized for Google Colab (GPU training, sequential to save RAM).

After running, use merge_submissions.py to patch E/I/K predictions
into the existing full submissions.
"""
import sys
import os
import time
import gc

import pandas as pd
import numpy as np

# Ensure this directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import (
    run_preprocessing, TARGET_INSURERS, TARGET_COLS, INSURERS_ALL,
    DEDUCTIBLE_COLS, CATEGORICAL_COLS
)
from train import train_all_target_insurers


# ============================================================
# PREDICTION
# ============================================================
def generate_predictions(test_df, feature_cols, cat_cols, model_dir, results):
    """
    Generate ensemble predictions for E, I, K on a test set.
    Returns a dict: {insurer: predictions_array}
    """
    from catboost import CatBoostRegressor
    import lightgbm as lgb

    cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    cat_col_names = [c for c in cat_cols if c in feature_cols]

    # Prepare LightGBM-compatible test data
    X_test_lgb = test_df[feature_cols].copy()
    for col in cat_col_names:
        if col in X_test_lgb.columns:
            X_test_lgb[col] = X_test_lgb[col].astype('category')

    preds = {}
    for r in results:
        insurer = r['insurer']
        w_cb = r['blend_weight_cb']
        w_lgb = r['blend_weight_lgb']

        # CatBoost prediction
        cb_model = CatBoostRegressor()
        cb_model.load_model(r['cb_model_path'])
        cb_preds = cb_model.predict(test_df[feature_cols])

        # LightGBM prediction
        lgb_model = lgb.Booster(model_file=r['lgb_model_path'])
        lgb_preds = lgb_model.predict(X_test_lgb)

        # Blend
        blended = w_cb * cb_preds + w_lgb * lgb_preds
        blended = np.clip(blended, 1.0, None)  # no zero/negative premiums

        preds[insurer] = blended
        print(f"  Insurer {insurer}: mean={blended.mean():.2f}, "
              f"std={blended.std():.2f}, min={blended.min():.2f}, max={blended.max():.2f}")

        del cb_model, lgb_model
        gc.collect()

    del X_test_lgb
    gc.collect()

    return preds


# ============================================================
# MAIN
# ============================================================
def main():
    start = time.time()

    # Resolve base dir (one level up from this script)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    model_dir = os.path.join(base_dir, 'models')
    submissions_dir = os.path.join(base_dir, 'submissions')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)

    # ---- STEP 1: PREPROCESS ----
    print("=" * 70)
    print("STEP 1: PREPROCESSING (with harsher feature engineering)")
    print("=" * 70)
    train_part, val_part, test_b2, test_b3, feature_cols, cat_cols = run_preprocessing()
    print(f"\nPreprocessing done in {time.time()-start:.0f}s")
    print(f"Features: {len(feature_cols)}, Categoricals: {len(cat_cols)}")

    gc.collect()

    # ---- STEP 2: TRAIN ENSEMBLE ----
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING ENSEMBLE (CatBoost + LightGBM) FOR E, I, K")
    print("=" * 70)

    results = train_all_target_insurers(
        train_df=train_part,
        val_df=val_part,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        model_dir=model_dir,
    )

    # Free training data
    del train_part, val_part
    gc.collect()

    # ---- STEP 3: GENERATE PREDICTIONS ----
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING ENSEMBLE PREDICTIONS FOR E, I, K")
    print("=" * 70)

    print("\n--- Block 2 predictions ---")
    preds_b2 = generate_predictions(test_b2, feature_cols, cat_cols, model_dir, results)

    print("\n--- Block 3 predictions ---")
    preds_b3 = generate_predictions(test_b3, feature_cols, cat_cols, model_dir, results)

    # Save standalone E/I/K submissions for merging
    for block_name, test_df, preds in [('block2', test_b2, preds_b2), ('block3', test_b3, preds_b3)]:
        out_df = pd.DataFrame({'quote_id': test_df['quote_id']})
        for insurer in TARGET_INSURERS:
            out_df[f"Insurer_{insurer}_price"] = preds[insurer]
        out_path = os.path.join(submissions_dir, f'ensemble_EKI_{block_name}.csv')
        out_df.to_csv(out_path, sep=';', index=False)
        print(f"Saved {out_path}: {out_df.shape}")

    print(f"\nTotal runtime: {(time.time()-start)/60:.1f} minutes")
    print("\nNext step: run merge_submissions.py to patch E/I/K into full submissions.")


if __name__ == '__main__':
    main()
