import numpy as np
import os
import json
import gc

# ============================================================
# CONFIG
# ============================================================
TARGET_INSURERS = ['E', 'I', 'K']


# ============================================================
# CATBOOST TRAINING
# ============================================================
def train_catboost(X_train, y_train, X_val, y_val, cat_indices, insurer):
    from catboost import CatBoostRegressor, Pool

    print(f"  [CatBoost] Training Insurer {insurer}...")

    model = CatBoostRegressor(
        iterations=10000,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        loss_function='RMSE',
        eval_metric='MAE',
        cat_features=cat_indices,
        random_seed=42,
        verbose=500,
        early_stopping_rounds=300,
        use_best_model=True,
        thread_count=-1,
        task_type='GPU',
        min_data_in_leaf=50,
        random_strength=2,
        bagging_temperature=1,
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

    model.fit(train_pool, eval_set=val_pool)

    val_preds = model.predict(X_val)
    val_mae = float(np.mean(np.abs(val_preds - y_val.values)))
    best_iter = model.get_best_iteration()

    print(f"  [CatBoost] Insurer {insurer}: MAE={val_mae:.2f}, best_iter={best_iter}")

    del train_pool, val_pool
    gc.collect()

    return model, val_mae, val_preds


# ============================================================
# LABEL ENCODING FOR LIGHTGBM GPU
# ============================================================
def _label_encode_for_lgb(df, cat_col_names, label_maps=None):
    """Label-encode categoricals to integers for LightGBM GPU compatibility."""
    df = df.copy()
    if label_maps is None:
        label_maps = {}
    for col in cat_col_names:
        if col not in df.columns:
            continue
        if col not in label_maps:
            uniques = df[col].unique()
            label_maps[col] = {v: i for i, v in enumerate(uniques)}
        mapping = label_maps[col]
        df[col] = df[col].map(mapping).fillna(-1).astype(int)
    return df, label_maps


# ============================================================
# LIGHTGBM TRAINING
# ============================================================
def train_lightgbm(X_train, y_train, X_val, y_val, cat_col_names, insurer):
    import lightgbm as lgb

    print(f"  [LightGBM] Training Insurer {insurer}...")

    # Label-encode categoricals (GPU can't handle high-cardinality native categoricals)
    X_train_lgb, label_maps = _label_encode_for_lgb(X_train, cat_col_names)
    X_val_lgb, _ = _label_encode_for_lgb(X_val, cat_col_names, label_maps)

    train_ds = lgb.Dataset(X_train_lgb, label=y_train)
    val_ds = lgb.Dataset(X_val_lgb, label=y_val, reference=train_ds)

    params = {
        'objective': 'huber',
        'huber_delta': 15.0,
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_data_in_leaf': 30,
        'max_bin': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'lambda_l1': 0.5,
        'lambda_l2': 3.0,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
        'device': 'gpu',
        'gpu_use_dp': False,
    }

    callbacks = [
        lgb.log_evaluation(period=500),
        lgb.early_stopping(stopping_rounds=300),
    ]

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=10000,
        valid_sets=[val_ds],
        valid_names=['val'],
        callbacks=callbacks,
    )

    val_preds = model.predict(X_val_lgb)
    val_mae = float(np.mean(np.abs(val_preds - y_val.values)))

    print(f"  [LightGBM] Insurer {insurer}: MAE={val_mae:.2f}, best_iter={model.best_iteration}")

    del train_ds, val_ds, X_train_lgb, X_val_lgb
    gc.collect()

    return model, val_mae, val_preds, label_maps


# ============================================================
# ENSEMBLE: TRAIN BOTH AND FIND BEST BLEND
# ============================================================
def train_ensemble_insurer(insurer, train_df, val_df, feature_cols, cat_cols, model_dir):
    target_col = f"Insurer_{insurer}_price"

    # Filter to quoted rows
    train_mask = train_df[target_col].notna()
    val_mask = val_df[target_col].notna()

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, target_col]
    X_val = val_df.loc[val_mask, feature_cols]
    y_val = val_df.loc[val_mask, target_col]

    cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    cat_col_names = [c for c in cat_cols if c in feature_cols]

    print(f"\n{'='*60}")
    print(f"INSURER {insurer} — ENSEMBLE TRAINING")
    print(f"  Train: {len(X_train):,} rows ({100*train_mask.mean():.1f}% quoted)")
    print(f"  Val:   {len(X_val):,} rows ({100*val_mask.mean():.1f}% quoted)")
    print(f"  Price range: [{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"  Price median: {y_train.median():.0f}, mean: {y_train.mean():.0f}")
    print(f"{'='*60}")

    # Train CatBoost
    cb_model, cb_mae, cb_val_preds = train_catboost(
        X_train, y_train, X_val, y_val, cat_indices, insurer
    )

    # Train LightGBM
    lgb_model, lgb_mae, lgb_val_preds, label_maps = train_lightgbm(
        X_train, y_train, X_val, y_val, cat_col_names, insurer
    )

    # Find optimal blend weight via grid search on validation
    best_w = 0.5
    best_mae = float('inf')
    for w in np.arange(0.0, 1.05, 0.05):
        blended = w * cb_val_preds + (1 - w) * lgb_val_preds
        mae = float(np.mean(np.abs(blended - y_val.values)))
        if mae < best_mae:
            best_mae = mae
            best_w = w

    print(f"\n  ENSEMBLE Insurer {insurer}:")
    print(f"    CatBoost MAE:  {cb_mae:.2f}")
    print(f"    LightGBM MAE:  {lgb_mae:.2f}")
    print(f"    Best blend:    w_cb={best_w:.2f}, w_lgb={1-best_w:.2f}")
    print(f"    Blended MAE:   {best_mae:.2f}")

    # Save CatBoost model
    cb_path = os.path.join(model_dir, f"ensemble_cb_insurer_{insurer}.cbm")
    cb_model.save_model(cb_path)

    # Save LightGBM model
    lgb_path = os.path.join(model_dir, f"ensemble_lgb_insurer_{insurer}.txt")
    lgb_model.save_model(lgb_path)

    # Feature importance from CatBoost
    importance = cb_model.get_feature_importance()
    feat_imp = sorted(zip(feature_cols, importance.tolist()), key=lambda x: -x[1])[:20]

    result = {
        'insurer': insurer,
        'cb_mae': cb_mae,
        'lgb_mae': lgb_mae,
        'ensemble_mae': best_mae,
        'blend_weight_cb': best_w,
        'blend_weight_lgb': 1 - best_w,
        'cb_model_path': cb_path,
        'lgb_model_path': lgb_path,
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'cb_best_iteration': int(cb_model.get_best_iteration()),
        'top_features': feat_imp,
    }

    print(f"  Top 5 features (CatBoost):")
    for fname, fimp in feat_imp[:5]:
        print(f"    {fname}: {fimp:.2f}")

    del cb_model, lgb_model, X_train, y_train, X_val, y_val
    gc.collect()

    return result


# ============================================================
# TRAIN ALL TARGET INSURERS
# ============================================================
def train_all_target_insurers(train_df, val_df, feature_cols, cat_cols, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    results = []
    for insurer in TARGET_INSURERS:
        r = train_ensemble_insurer(
            insurer, train_df, val_df, feature_cols, cat_cols, model_dir
        )
        results.append(r)
        gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING SUMMARY — INSURERS E, I, K")
    print("=" * 70)

    total_weighted_mae = 0
    total_val_rows = 0

    for r in sorted(results, key=lambda x: x['insurer']):
        print(f"  Insurer {r['insurer']}: "
              f"CB={r['cb_mae']:.2f} | LGB={r['lgb_mae']:.2f} | "
              f"Ensemble={r['ensemble_mae']:.2f} (w_cb={r['blend_weight_cb']:.2f}) | "
              f"val={r['n_val']:,}")
        total_weighted_mae += r['ensemble_mae'] * r['n_val']
        total_val_rows += r['n_val']

    pooled_mae = total_weighted_mae / total_val_rows
    print(f"\n  >>> POOLED VALIDATION MAE (E,I,K): {pooled_mae:.2f} <<<")
    print("=" * 70)

    # Save results
    results_path = os.path.join(model_dir, 'ensemble_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    return results
