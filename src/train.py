from catboost import CatBoostRegressor, Pool
import numpy as np
import os
import json
import gc

INSURERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']


def train_single_insurer(insurer, train_df, val_df, feature_cols, cat_cols, model_dir):
    target_col = f"Insurer_{insurer}_price"

    # Filter to rows where this insurer quoted
    train_mask = train_df[target_col].notna()
    val_mask = val_df[target_col].notna()

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, target_col]
    X_val = val_df.loc[val_mask, feature_cols]
    y_val = val_df.loc[val_mask, target_col]

    cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    print(f"\n{'='*60}")
    print(f"INSURER {insurer}")
    print(f"  Train: {len(X_train):,} rows ({100*train_mask.mean():.1f}% quoted)")
    print(f"  Val:   {len(X_val):,} rows ({100*val_mask.mean():.1f}% quoted)")
    print(f"  Price range: [{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"  Price median: {y_train.median():.0f}, mean: {y_train.mean():.0f}")
    print(f"{'='*60}")

    model = CatBoostRegressor(
        iterations=5000,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=5,
        loss_function='MAE',
        eval_metric='MAE',
        cat_features=cat_indices,
        random_seed=42,
        verbose=500,
        early_stopping_rounds=200,
        use_best_model=True,
        thread_count=2,
	task_type='GPU'
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

    model.fit(train_pool, eval_set=val_pool)

    # Evaluate
    val_preds = model.predict(X_val)
    val_mae = float(np.mean(np.abs(val_preds - y_val.values)))

    # Capture results before cleanup
    best_iter = model.get_best_iteration()
    importance = model.get_feature_importance()
    feat_imp = sorted(zip(feature_cols, importance.tolist()), key=lambda x: -x[1])[:20]

    # Save model
    model_path = os.path.join(model_dir, f"catboost_insurer_{insurer}.cbm")
    model.save_model(model_path)

    result = {
        'insurer': insurer,
        'val_mae': val_mae,
        'model_path': model_path,
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'best_iteration': int(best_iter),
        'top_features': feat_imp,
    }

    print(f"\n  Insurer {insurer} -- Val MAE: {val_mae:.2f}, best iter: {best_iter}")
    print(f"  Top 5 features:")
    for fname, fimp in feat_imp[:5]:
        print(f"    {fname}: {fimp:.2f}")

    del train_pool, val_pool, model
    gc.collect()

    return result


def train_all_insurers(train_df, val_df, feature_cols, cat_cols, model_dir, n_jobs=11):
    os.makedirs(model_dir, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                train_single_insurer, insurer, train_df, val_df, feature_cols, cat_cols, model_dir
            ): insurer
            for insurer in INSURERS
        }
        for future in as_completed(futures):
            insurer = futures[future]
            try:
                r = future.result()
                results.append(r)
            except Exception as e:
                print(f"  ERROR training insurer {insurer}: {e}")
                raise

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY -- ALL INSURERS")
    print("=" * 70)

    total_weighted_mae = 0
    total_val_rows = 0

    for r in sorted(results, key=lambda x: x['insurer']):
        print(f"  Insurer {r['insurer']}: "
              f"MAE={r['val_mae']:.2f} | "
              f"train={r['n_train']:,} | "
              f"val={r['n_val']:,} | "
              f"best_iter={r['best_iteration']}")
        total_weighted_mae += r['val_mae'] * r['n_val']
        total_val_rows += r['n_val']

    pooled_mae = total_weighted_mae / total_val_rows
    print(f"\n  >>> POOLED VALIDATION MAE: {pooled_mae:.2f} <<<")
    print("=" * 70)

    # Save results to JSON
    results_path = os.path.join(model_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    return results
