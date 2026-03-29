"""
=== SOTA Insurance Premium Prediction Pipeline ===
Level-2 Stacking Ensemble for all Insurers A-K.

Architecture:
  Level-1: CatBoost (Tweedie) + LightGBM (Tweedie) + XGBoost (Tweedie) + MLP (log-target)
  Level-2: Ridge Regression Meta-Learner on OOF predictions (with polynomial features)

Key Innovations:
  - Stratified K-Fold CV on decile-binned targets (preserves price distribution)
  - Out-of-Fold Target Encoding for high-cardinality spatial features
  - K-Means Geospatial Risk Zones (lat/lon → 100 risk clusters)
  - Tweedie loss for tree models (actuarial standard for right-skewed premiums)
  - Model diversity: trees + MLP → uncorrelated errors → better ensemble
  - Ridge meta-learner learns dynamic per-price-range weighting

Before running:
  1. Upload data/ folder to Colab (or mount Google Drive)
  2. Upload existing submissions/ folder
  3. pip install catboost lightgbm xgboost scikit-learn

Usage in Colab:
  !pip install catboost lightgbm xgboost scikit-learn
"""

import pandas as pd
import numpy as np
import os
import time
import gc
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# COLAB SETUP — adjust paths here
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
BASE_DIR = '/content/drive/MyDrive/matf_data'

# If files uploaded directly to Colab:
# BASE_DIR = '.'

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
INSURERS_ALL = list('ABCDEFGHIJK')
TARGET_INSURERS = list('ABCDEFGHIJK')

N_FOLDS = 5
N_RISK_ZONES = 100
TOP_N_FEATURES = 100
N_BAGS = 1  # Single bag per fold — 5-model ensemble already provides diversity

# High-cardinality columns to target-encode (OOF, per-insurer)
TARGET_ENCODE_COLS = [
    'postal_code', 'municipality', 'province', 'vehicle_maker', 'vehicle_model',
    'coverage_x_age_bucket',  # composite: coverage type × driver age band
]

PRICE_COLS = [f"Insurer_{i}_price" for i in INSURERS_ALL]
DEDUCTIBLE_COLS = [f"Insurer_{i}_deductible" for i in INSURERS_ALL]

EXCLUDE_COLS = {'quote_id', 'vehicle_number_plate', 'Unnamed: 0'} | set(PRICE_COLS)

DATE_COLS = [
    'contractor_birthdate', 'second_driver_birthdate',
    'vehicle_first_registration_date', 'vehicle_country_first_registration_date',
    'vehicle_last_registration_date', 'vehicle_inspection_report_date',
    'vehicle_inspection_expiry_date',
]

CATEGORICAL_COLS = [
    'coverage', 'payment_frequency', 'is_driver_owner', 'usage',
    'vehicle_maker', 'vehicle_model', 'vehicle_fuel_type',
    'vehicle_primary_color', 'vehicle_odometer_verdict_code',
    'vehicle_is_imported', 'vehicle_is_imported_within_last_12_months',
    'vehicle_can_be_registered', 'vehicle_has_open_recall',
    'vehicle_is_marked_for_export', 'vehicle_is_taxi',
    'postal_code', 'province', 'municipality',
    'postal_code_urban_category',
]

REFERENCE_DATE = pd.Timestamp('2025-01-01')


# ============================================================
# PREPROCESSING
# ============================================================
def _to_dt(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, dayfirst=True, errors='coerce')


def convert_dtypes(df):
    df = df.copy()
    skip = set(DATE_COLS) | {'quote_id', 'vehicle_number_plate'} | set(CATEGORICAL_COLS)
    for col in df.columns:
        if col in skip:
            continue
        if hasattr(df[col].dtype, 'name') and df[col].dtype.name in ('str', 'string', 'object'):
            converted = pd.to_numeric(df[col], errors='coerce')
            orig_non_null = df[col].notna().sum()
            conv_non_null = converted.notna().sum()
            if orig_non_null > 0 and conv_non_null / orig_non_null > 0.8:
                df[col] = converted
    return df


def engineer_features(df, precomputed_stats=None):
    """Phase 2: Actuarial ratios, date features, domain interactions.

    Args:
        precomputed_stats: If None (train), computes and returns stats.
                          If provided (test), uses train-derived stats for
                          frequency encoding and crime threshold.
    Returns:
        (df, stats) tuple.
    """
    df = convert_dtypes(df.copy())
    new_cols = {}
    stats = precomputed_stats if precomputed_stats is not None else {}
    is_train = precomputed_stats is None

    # --- Date-derived features ---
    cb = _to_dt(df['contractor_birthdate'])
    new_cols['contractor_age'] = (REFERENCE_DATE - cb).dt.days / 365.25

    sb = _to_dt(df['second_driver_birthdate'])
    new_cols['second_driver_age'] = (REFERENCE_DATE - sb).dt.days / 365.25
    new_cols['has_second_driver'] = df['second_driver_birthdate'].notna().astype(int)

    insp_report = _to_dt(df['vehicle_inspection_report_date'])
    new_cols['days_since_inspection'] = (REFERENCE_DATE - insp_report).dt.days

    insp_expiry = _to_dt(df['vehicle_inspection_expiry_date'])
    new_cols['inspection_days_remaining'] = (insp_expiry - REFERENCE_DATE).dt.days

    vlr = _to_dt(df['vehicle_last_registration_date'])
    new_cols['years_since_last_registration'] = (REFERENCE_DATE - vlr).dt.days / 365.25

    vfr = _to_dt(df['vehicle_first_registration_date'])
    new_cols['years_since_first_registration'] = (REFERENCE_DATE - vfr).dt.days / 365.25

    vcfr = _to_dt(df['vehicle_country_first_registration_date'])
    new_cols['years_since_country_first_reg'] = (REFERENCE_DATE - vcfr).dt.days / 365.25

    # --- Actuarial features ---
    contractor_age = new_cols['contractor_age']
    new_cols['contractor_age_sq'] = contractor_age ** 2
    new_cols['second_driver_age_sq'] = new_cols['second_driver_age'] ** 2

    new_cols['contractor_age_bucket'] = pd.cut(
        contractor_age, bins=[0, 25, 35, 50, 65, 120],
        labels=['very_young', 'young', 'mid', 'senior', 'elderly'], ordered=False
    ).astype(str).fillna('_MISSING_')

    # Claim-free years interactions (actuarial ratios)
    claim_free = df.get('claim_free_years', pd.Series(0, index=df.index))
    new_cols['claim_free_x_age'] = claim_free * contractor_age
    # KEY: Claim-Free Years / Driver Age ratio
    new_cols['claim_free_per_age'] = claim_free / contractor_age.clip(lower=18)

    if 'vehicle_age' in df.columns:
        new_cols['claim_free_x_vehicle_age'] = claim_free * df['vehicle_age']

    # Second driver interactions
    if 'second_driver_claim_free_years' in df.columns:
        sd_claim_free = df['second_driver_claim_free_years']
        new_cols['second_driver_claim_free_x_age'] = sd_claim_free * new_cols['second_driver_age']
        new_cols['total_claim_free_years'] = claim_free.fillna(0) + sd_claim_free.fillna(0)
        new_cols['claim_free_diff'] = claim_free.fillna(0) - sd_claim_free.fillna(0)

    # Vehicle Power-to-Weight Ratio (actuarial)
    if 'vehicle_power' in df.columns and 'vehicle_net_weight' in df.columns:
        new_cols['weight_per_power'] = df['vehicle_net_weight'] / (df['vehicle_power'] + 1)
        new_cols['power_per_weight'] = df['vehicle_power'] / (df['vehicle_net_weight'] + 1)

    # Vehicle Value / Vehicle Age (actuarial: depreciation proxy)
    if 'vehicle_value_new' in df.columns:
        vage = df.get('vehicle_age', pd.Series(1, index=df.index)).clip(lower=0.5)
        new_cols['value_per_age'] = df['vehicle_value_new'] / vage
        new_cols['log_vehicle_value'] = np.log1p(df['vehicle_value_new'].clip(lower=0))

    # Crime / density features
    if 'municipality_crimes_per_1000' in df.columns:
        if is_train:
            stats['crime_threshold'] = float(df['municipality_crimes_per_1000'].quantile(0.75))
        new_cols['high_crime_area'] = (
            df['municipality_crimes_per_1000'] > stats.get('crime_threshold', 0)
        ).astype(int)

    if 'postal_code_address_density' in df.columns:
        new_cols['log_address_density'] = np.log1p(df['postal_code_address_density'].clip(lower=0))

    # Deductible statistics across all insurers
    deductible_present = [c for c in DEDUCTIBLE_COLS if c in df.columns]
    if deductible_present:
        ded_df = df[deductible_present]
        new_cols['mean_deductible'] = ded_df.mean(axis=1)
        new_cols['std_deductible'] = ded_df.std(axis=1)
        new_cols['max_deductible'] = ded_df.max(axis=1)
        new_cols['min_deductible'] = ded_df.min(axis=1)
        mean_ded = new_cols['mean_deductible']
        for ins in TARGET_INSURERS:
            ded_col = f"Insurer_{ins}_deductible"
            if ded_col in df.columns:
                new_cols[f'{ins}_ded_vs_mean'] = df[ded_col] - mean_ded

    if 'vehicle_inspection_number_of_deficiencies_found' in df.columns:
        new_cols['has_deficiencies'] = (
            df['vehicle_inspection_number_of_deficiencies_found'] > 0
        ).astype(int)

    if 'vehicle_planned_annual_mileage' in df.columns:
        new_cols['log_mileage'] = np.log1p(df['vehicle_planned_annual_mileage'].clip(lower=0))

    # --- Coverage interaction features ---
    # Coverage type determines the entire premium band. Interacting it with
    # age/value captures patterns like "young driver + Casco = very expensive"
    if 'coverage' in df.columns:
        cov_str = df['coverage'].fillna('_MISSING_').astype(str)
        # Composite categorical for target encoding (captures coverage × age nonlinearity)
        new_cols['coverage_x_age_bucket'] = cov_str + '_' + new_cols['contractor_age_bucket']

    # --- Deductible / Vehicle Value ratio (actuarial standard) ---
    # A $500 deductible on a $50K car means very different risk than on a $10K car
    if 'vehicle_value_new' in df.columns and deductible_present:
        vval = df['vehicle_value_new'].clip(lower=1)
        new_cols['mean_deductible_ratio'] = new_cols['mean_deductible'] / vval
        for ins in TARGET_INSURERS:
            ded_col = f"Insurer_{ins}_deductible"
            if ded_col in df.columns:
                new_cols[f'{ins}_ded_ratio'] = df[ded_col] / vval

    # --- Vehicle cost per HP ---
    if 'vehicle_value_new' in df.columns and 'vehicle_power' in df.columns:
        new_cols['value_per_power'] = df['vehicle_value_new'] / (df['vehicle_power'] + 1)

    # --- Frequency encoding (how common each category is — rare = riskier) ---
    # Use train frequencies for all datasets to avoid train/test distribution mismatch
    for col in ['postal_code', 'municipality', 'vehicle_maker', 'vehicle_model']:
        if col in df.columns:
            freq_key = f'{col}_freq_map'
            if is_train:
                stats[freq_key] = df[col].value_counts().to_dict()
            freq_map = stats.get(freq_key, {})
            new_cols[f'{col}_freq'] = df[col].map(freq_map).fillna(1).astype(int)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df, stats


def get_feature_and_cat_columns(df):
    exclude = EXCLUDE_COLS | set(DATE_COLS)
    feature_cols = [c for c in df.columns if c not in exclude]

    cat_cols = []
    for col in feature_cols:
        dtype_name = df[col].dtype.name if hasattr(df[col].dtype, 'name') else str(df[col].dtype)
        if (col in CATEGORICAL_COLS or col == 'contractor_age_bucket'
                or col == 'risk_zone' or col == 'coverage_x_age_bucket'
                or dtype_name in ('object', 'str', 'string', 'category')):
            if col not in cat_cols:
                cat_cols.append(col)

    return feature_cols, cat_cols


def prepare_categoricals(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('_MISSING_').astype(str)
    return df


# ============================================================
# GEOSPATIAL RISK ZONES (K-Means Clustering)
# ============================================================
def add_risk_zones(train_df, test_dfs, n_clusters=50):
    """Cluster lat/lon into risk zones. Captures geographic risk patterns."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.impute import SimpleImputer

    lat_col = 'postal_code_latitude'
    lon_col = 'postal_code_longitude'

    if lat_col not in train_df.columns or lon_col not in train_df.columns:
        print("  Warning: lat/lon columns not found, skipping risk zones")
        return

    coords_cols = [lat_col, lon_col]
    imputer = SimpleImputer(strategy='median')
    train_coords = imputer.fit_transform(train_df[coords_cols])

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=10000)
    train_df['risk_zone'] = km.fit_predict(train_coords).astype(str)

    for tdf in test_dfs:
        test_coords = imputer.transform(tdf[coords_cols])
        tdf['risk_zone'] = km.predict(test_coords).astype(str)

    print(f"  Created {n_clusters} geospatial risk zones via K-Means on lat/lon")


# ============================================================
# OUT-OF-FOLD TARGET ENCODING
# ============================================================
def oof_target_encode(train_series, y, fold_indices, test_series_list, smoothing=20):
    """
    Out-of-fold target encoding with Bayesian smoothing.
    Each sample is encoded using only OTHER folds' data → no leakage.
    Test data is encoded using full training set statistics.
    """
    global_mean = float(y.mean())
    train_encoded = pd.Series(np.nan, index=train_series.index, dtype=float)

    for train_idx, val_idx in fold_indices:
        fold_cats = train_series.iloc[train_idx]
        fold_targets = y.iloc[train_idx]

        stats = pd.DataFrame({
            'cat': fold_cats, 'y': fold_targets
        }).groupby('cat')['y'].agg(['mean', 'count'])

        smoothed = (
            (stats['count'] * stats['mean'] + smoothing * global_mean)
            / (stats['count'] + smoothing)
        )

        train_encoded.iloc[val_idx] = train_series.iloc[val_idx].map(smoothed).fillna(global_mean)

    # Test: full training set statistics
    full_stats = pd.DataFrame({
        'cat': train_series, 'y': y
    }).groupby('cat')['y'].agg(['mean', 'count'])
    full_smoothed = (
        (full_stats['count'] * full_stats['mean'] + smoothing * global_mean)
        / (full_stats['count'] + smoothing)
    )

    test_encoded = [ts.map(full_smoothed).fillna(global_mean) for ts in test_series_list]
    return train_encoded, test_encoded


# ============================================================
# FEATURE SELECTION (Quick CatBoost scout)
# ============================================================
def select_features(X_train, y_train, X_val, y_val, feature_cols, cat_cols, top_n=80):
    from catboost import CatBoostRegressor, Pool

    cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    print(f"  [Feature Selection] Scouting {len(feature_cols)} features...")
    scout = CatBoostRegressor(
        iterations=1200,
        depth=6,
        learning_rate=0.1,
        loss_function='MAE',
        eval_metric='RMSE',
        cat_features=cat_indices,
        random_seed=42,
        verbose=0,
        task_type='GPU',
        devices='0',
        border_count=254,
        early_stopping_rounds=100,
        use_best_model=True,
    )
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)
    scout.fit(train_pool, eval_set=val_pool)

    importance = scout.get_feature_importance()
    ranked = sorted(zip(feature_cols, importance), key=lambda x: -x[1])

    keep = min(top_n, len(ranked))
    selected = [name for name, _ in ranked[:keep]]

    print(f"  [Feature Selection] Keeping {keep}/{len(feature_cols)} features")
    print(f"  Top 10: {[f'{n}({v:.1f})' for n, v in ranked[:10]]}")

    del scout, train_pool, val_pool
    gc.collect()

    selected_cat = [c for c in cat_cols if c in selected]
    return selected, selected_cat


# ============================================================
# LEVEL-1 MODEL: CatBoost (Tweedie)
# ============================================================
def train_catboost_fold(X_train, y_train, X_val, y_val, cat_indices, insurer,
                        fold_idx, n_bags=2):
    """CatBoost with Tweedie loss — actuarial standard for insurance pricing."""
    from catboost import CatBoostRegressor, Pool

    val_pool = Pool(X_val, y_val, cat_features=cat_indices)
    models = []
    val_preds_sum = np.zeros(len(X_val))

    for bag in range(n_bags):
        rng = np.random.RandomState(42 + bag + fold_idx * 100)
        idx = rng.choice(len(X_train), size=int(len(X_train) * 0.85), replace=False)
        X_bag = X_train.iloc[idx]
        y_bag = y_train.iloc[idx]

        model = CatBoostRegressor(
            iterations=8000,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=4,
            loss_function='Tweedie:variance_power=1.5',
            eval_metric='RMSE',
            cat_features=cat_indices,
            random_seed=42 + bag,
            verbose=500,
            early_stopping_rounds=300,
            use_best_model=True,
            task_type='GPU',
            devices='0',
            border_count=254,
            gpu_ram_part=0.8,
            min_data_in_leaf=20,
            random_strength=0.6,
            bagging_temperature=0.3,
            max_ctr_complexity=3,
        )

        train_pool = Pool(X_bag, y_bag, cat_features=cat_indices)
        model.fit(train_pool, eval_set=val_pool)

        preds = model.predict(X_val)
        bag_mae = float(np.mean(np.abs(preds - y_val.values)))
        print(f"    CB Bag {bag+1}/{n_bags}: MAE={bag_mae:.2f}, iter={model.get_best_iteration()}")

        val_preds_sum += preds
        models.append(model)

        del train_pool, X_bag, y_bag
        gc.collect()

    val_preds = val_preds_sum / n_bags
    del val_pool
    gc.collect()
    return models, val_preds


# ============================================================
# LEVEL-1 MODEL: LightGBM (Tweedie)
# ============================================================
def _label_encode_for_lgb(df, cat_col_names, label_maps=None):
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


def train_lightgbm_fold(X_train, y_train, X_val, y_val, cat_col_names, insurer,
                         fold_idx, n_bags=2, sample_weights=None):
    """LightGBM with Tweedie loss — fast, handles numeric features beautifully."""
    import lightgbm as lgb

    X_train_lgb, label_maps = _label_encode_for_lgb(X_train, cat_col_names)
    X_val_lgb, _ = _label_encode_for_lgb(X_val, cat_col_names, label_maps)

    models = []
    val_preds_sum = np.zeros(len(X_val))

    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 96,
        'max_depth': 8,
        'min_data_in_leaf': 25,
        'max_bin': 255,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'lambda_l1': 0.8,
        'lambda_l2': 4.0,
        'path_smooth': 5.0,
        'verbose': -1,
        'n_jobs': -1,
    }

    for bag in range(n_bags):
        rng = np.random.RandomState(42 + bag + fold_idx * 100)
        idx = rng.choice(len(X_train_lgb), size=int(len(X_train_lgb) * 0.85), replace=False)
        X_bag = X_train_lgb.iloc[idx]
        y_bag = y_train.iloc[idx]
        w_bag = sample_weights[idx] if sample_weights is not None else None

        train_ds = lgb.Dataset(X_bag, label=y_bag, weight=w_bag)
        val_ds = lgb.Dataset(X_val_lgb, label=y_val, reference=train_ds)

        bag_params = {**params, 'seed': 42 + bag}

        model = lgb.train(
            bag_params, train_ds, num_boost_round=8000,
            valid_sets=[val_ds], valid_names=['val'],
            callbacks=[lgb.log_evaluation(500), lgb.early_stopping(300)],
        )

        preds = model.predict(X_val_lgb)
        bag_mae = float(np.mean(np.abs(preds - y_val.values)))
        print(f"    LGB Bag {bag+1}/{n_bags}: MAE={bag_mae:.2f}, iter={model.best_iteration}")

        val_preds_sum += preds
        models.append(model)

        del train_ds, val_ds, X_bag, y_bag
        gc.collect()

    val_preds = val_preds_sum / n_bags
    del X_train_lgb, X_val_lgb
    gc.collect()
    return models, val_preds, label_maps


# ============================================================
# LEVEL-1 MODEL: XGBoost (Tweedie)
# ============================================================
def train_xgboost_fold(X_train, y_train, X_val, y_val, cat_col_names, insurer,
                        fold_idx, sample_weights=None):
    """
    XGBoost with Tweedie loss — different tree-building algorithm (approximate
    greedy) than CatBoost (ordered) and LightGBM (leaf-wise).
    No bagging needed: XGBoost has strong built-in subsample + colsample stochasticity.
    """
    import xgboost as xgb

    X_train_xgb, label_maps = _label_encode_for_lgb(X_train, cat_col_names)
    X_val_xgb, _ = _label_encode_for_lgb(X_val, cat_col_names, label_maps)

    dtrain = xgb.DMatrix(X_train_xgb, label=y_train, weight=sample_weights, enable_categorical=False)
    dval = xgb.DMatrix(X_val_xgb, label=y_val, enable_categorical=False)

    params = {
        'objective': 'reg:tweedie',
        'tweedie_variance_power': 1.5,
        'eval_metric': 'mae',
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_child_weight': 20,
        'subsample': 0.85,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.8,
        'reg_alpha': 0.8,
        'reg_lambda': 4.0,
        'max_bin': 256,
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': 42 + fold_idx,
        'verbosity': 0,
    }

    model = xgb.train(
        params, dtrain, num_boost_round=8000,
        evals=[(dval, 'val')],
        early_stopping_rounds=300,
        verbose_eval=500,
    )

    val_preds = model.predict(dval)
    mae = float(np.mean(np.abs(val_preds - y_val.values)))
    print(f"    XGB: MAE={mae:.2f}, iter={model.best_iteration}")

    del dtrain, dval, X_train_xgb, X_val_xgb
    gc.collect()
    return model, val_preds, label_maps


# ============================================================
# LEVEL-1 MODEL: CatBoost (MAE — directly optimizes competition metric)
# ============================================================
def train_catboost_mae_fold(X_train, y_train, X_val, y_val, cat_indices, insurer,
                            fold_idx):
    """
    CatBoost with MAE loss — directly optimizes the competition metric.
    No bagging (single model per fold) to stay within time budget.
    Complements Tweedie models: Tweedie captures distribution shape,
    MAE captures the median — meta-learner blends both signals.
    """
    from catboost import CatBoostRegressor, Pool

    model = CatBoostRegressor(
        iterations=6000,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5,
        loss_function='MAE',
        eval_metric='RMSE',
        cat_features=cat_indices,
        random_seed=42 + fold_idx,
        verbose=500,
        early_stopping_rounds=250,
        use_best_model=True,
        task_type='GPU',
        devices='0',
        border_count=254,
        min_data_in_leaf=25,
        max_ctr_complexity=3,
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)
    model.fit(train_pool, eval_set=val_pool)

    val_preds = model.predict(X_val)
    mae = float(np.mean(np.abs(val_preds - y_val.values)))
    print(f"    CB-MAE: MAE={mae:.2f}, iter={model.get_best_iteration()}")

    del train_pool, val_pool
    gc.collect()
    return model, val_preds


# ============================================================
# LEVEL-1 MODEL: MLP (Neural Network — PyTorch GPU)
# ============================================================
def _build_mlp(n_features):
    """Build a 2-layer MLP with BatchNorm — lightweight for speed."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.15),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


def _mlp_predict(model, X_np, device):
    """Run MLP inference on numpy array, return numpy array (log-space → real)."""
    import torch
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X_np, dtype=torch.float32, device=device)
        preds = model(t).squeeze(-1).cpu().numpy()
        del t
    return np.clip(np.expm1(preds), 1.0, None)


def train_mlp_fold(X_train, y_train, X_val, y_val, cat_cols):
    """
    MLP with standardized numeric features and log1p(target).
    Runs on GPU via PyTorch — fully utilises the H100.
    Draws smooth, non-orthogonal decision boundaries — uncorrelated
    errors with tree models boost the ensemble.
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MLP uses only numeric features (including target-encoded columns)
    numeric_cols = [c for c in X_train.columns if c not in cat_cols]

    if len(numeric_cols) < 5:
        X_tr = X_train.copy()
        X_v = X_val.copy()
        for col in cat_cols:
            if col in X_tr.columns:
                uniques = list(X_tr[col].unique())
                mapping = {v: i for i, v in enumerate(uniques)}
                X_tr[col] = X_tr[col].map(mapping).fillna(-1).astype(float)
                X_v[col] = X_v[col].map(mapping).fillna(-1).astype(float)
        numeric_cols = list(X_tr.columns)
    else:
        X_tr = X_train[numeric_cols]
        X_v = X_val[numeric_cols]

    imputer = SimpleImputer(strategy='median')
    X_tr_imp = imputer.fit_transform(X_tr)
    X_v_imp = imputer.transform(X_v)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_v_scaled = scaler.transform(X_v_imp)

    # Train on log1p(target) — handles right-skew, ensures positive predictions
    y_tr_log = np.log1p(y_train.values)
    y_v_log = np.log1p(y_val.values)

    # PyTorch tensors
    X_train_t = torch.tensor(X_tr_scaled, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_tr_log, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_v_scaled, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_v_log, dtype=torch.float32, device=device)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4096, shuffle=True)

    model = _build_mlp(X_tr_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    best_val_loss = float('inf')
    patience = 15
    no_improve = 0
    best_state = None

    for epoch in range(200):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb).squeeze(-1)
            loss = nn.functional.l1_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.l1_loss(
                model(X_val_t).squeeze(-1), y_val_t
            ).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)

    val_preds = _mlp_predict(model, X_v_scaled, device)
    mae = float(np.mean(np.abs(val_preds - y_val.values)))
    print(f"    MLP (GPU): MAE={mae:.2f}, epochs={epoch+1}")

    del X_train_t, y_train_t, X_val_t, y_val_t, train_ds, train_loader
    torch.cuda.empty_cache()

    return model, val_preds, imputer, scaler, numeric_cols, device


# ============================================================
# LEVEL-2 META-LEARNER (Ridge Regression with polynomial features)
# ============================================================
def _build_meta_features(oof_lgb, oof_xgb):
    """
    Build meta-features from 2 Level-1 OOF predictions.
    Includes raw predictions + disagreement signal.
    """
    raw = np.column_stack([oof_lgb, oof_xgb])
    diff = np.abs(oof_lgb - oof_xgb).reshape(-1, 1)
    return np.hstack([raw, diff])


def train_meta_learner(oof_lgb, oof_xgb, y_true):
    """
    Ridge regression on OOF predictions from 2 Level-1 models.
    """
    from sklearn.linear_model import Ridge

    X_meta = _build_meta_features(oof_lgb, oof_xgb)
    meta = Ridge(alpha=1.0)
    meta.fit(X_meta, y_true.values)

    meta_preds = meta.predict(X_meta)
    meta_mae = float(np.mean(np.abs(meta_preds - y_true.values)))

    lgb_mae_score = float(np.mean(np.abs(oof_lgb - y_true.values)))
    xgb_mae_score = float(np.mean(np.abs(oof_xgb - y_true.values)))

    print(f"\n  Level-2 Meta-Learner:")
    print(f"    LGB-Tweedie OOF MAE: {lgb_mae_score:.2f}")
    print(f"    XGB-Tweedie OOF MAE: {xgb_mae_score:.2f}")
    print(f"    Meta-Learner MAE:    {meta_mae:.2f}")
    w = meta.coef_
    print(f"    Weights: LGB={w[0]:.3f}, XGB={w[1]:.3f}, Disagreement={w[2]:.3f}")

    return meta, meta_mae


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline():
    from sklearn.model_selection import StratifiedKFold

    start = time.time()

    # ---- LOAD ----
    print("Loading data...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, 'block1_train.parquet'))
    test_b2 = pd.read_parquet(os.path.join(DATA_DIR, 'block2_test.parquet'))
    test_b3 = pd.read_parquet(os.path.join(DATA_DIR, 'block3_test.parquet'))
    print(f"Train: {train_df.shape}, Test B2: {test_b2.shape}, Test B3: {test_b3.shape}")

    # ---- TEMPORAL / RECENCY FEATURE ----
    # Data is chronologically ordered: train=weeks 1-4, b2=week 5, b3=week 6
    n_train = len(train_df)
    train_df['quote_recency'] = np.arange(n_train) / n_train          # 0 → ~1
    test_b2['quote_recency'] = 1.0   # week 5 — immediately after training
    test_b3['quote_recency'] = 1.2   # week 6 — further out

    # ---- PHASE 2: FEATURE ENGINEERING ----
    print("\nPhase 2: Feature engineering (actuarial ratios, interactions)...")
    train_df, feat_stats = engineer_features(train_df)
    test_b2, _ = engineer_features(test_b2, precomputed_stats=feat_stats)
    test_b3, _ = engineer_features(test_b3, precomputed_stats=feat_stats)

    # ---- GEOSPATIAL RISK ZONES ----
    print("\nAdding geospatial risk zones...")
    add_risk_zones(train_df, [test_b2, test_b3], n_clusters=N_RISK_ZONES)

    # ---- COLUMN SETUP ----
    feature_cols, cat_cols = get_feature_and_cat_columns(train_df)
    print(f"Features: {len(feature_cols)}, Categoricals: {len(cat_cols)}")

    train_df = prepare_categoricals(train_df, cat_cols)
    test_b2 = prepare_categoricals(test_b2, cat_cols)
    test_b3 = prepare_categoricals(test_b3, cat_cols)

    print(f"\nPreprocessing done in {time.time()-start:.0f}s")

    # ---- RESULTS (with checkpoint resume) ----
    checkpoint_path = os.path.join(MODEL_DIR, 'checkpoint_completed.json')
    completed_insurers = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            completed_insurers = json.load(f)
        print(f"\n  Resuming from checkpoint. Already completed: {completed_insurers}")

    if completed_insurers and os.path.exists(os.path.join(MODEL_DIR, 'checkpoint_b2.csv')):
        preds_b2 = pd.read_csv(os.path.join(MODEL_DIR, 'checkpoint_b2.csv'), sep=';')
        preds_b3 = pd.read_csv(os.path.join(MODEL_DIR, 'checkpoint_b3.csv'), sep=';')
    else:
        preds_b2 = pd.DataFrame({'quote_id': test_b2['quote_id']})
        preds_b3 = pd.DataFrame({'quote_id': test_b3['quote_id']})
    all_results = []

    # ================================================================
    # PER-INSURER TRAINING LOOP
    # ================================================================
    for insurer in TARGET_INSURERS:
        if insurer in completed_insurers:
            print(f"\n  Skipping Insurer {insurer} (already checkpointed)")
            continue

        target_col = f"Insurer_{insurer}_price"

        # Only rows where this insurer quoted
        ins_mask = train_df[target_col].notna()
        ins_data = train_df[ins_mask].reset_index(drop=True)
        y_all = ins_data[target_col].copy()

        # Clip outliers at 99.5th percentile
        threshold = y_all.quantile(0.995)
        n_clipped = (y_all > threshold).sum()
        y_all = y_all.clip(upper=threshold)

        print(f"\n{'='*60}")
        print(f"INSURER {insurer}: {len(ins_data):,} samples, {N_FOLDS}-fold Stratified CV")
        print(f"  Price: median={y_all.median():.0f}, mean={y_all.mean():.0f}, "
              f"range=[{y_all.min():.0f}, {y_all.max():.0f}]")
        # Recency sample weights: recent rows (week 4) get 2× weight of oldest (week 1)
        n_ins = len(ins_data)
        recency_weights = 0.5 + 0.5 * (np.arange(n_ins) / max(n_ins - 1, 1))  # 0.5 → 1.0

        print(f"  Clipped {n_clipped} rows at {threshold:.2f}")
        print(f"  Recency weights: oldest={recency_weights[0]:.2f}, newest={recency_weights[-1]:.2f}")
        print(f"{'='*60}")

        # --- PHASE 1: Stratified K-Fold on decile-binned targets ---
        y_bins = pd.qcut(y_all, q=10, labels=False, duplicates='drop')
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_indices = list(skf.split(ins_data, y_bins))

        # --- PHASE 2: Out-of-Fold Target Encoding ---
        print("  Computing OOF target encoding...")
        te_train_dict = {}
        te_b2_dict = {}
        te_b3_dict = {}

        te_cols_to_encode = TARGET_ENCODE_COLS + (['risk_zone'] if 'risk_zone' in ins_data.columns else [])

        for col in te_cols_to_encode:
            if col not in ins_data.columns:
                continue
            train_te, [b2_te, b3_te] = oof_target_encode(
                ins_data[col], y_all, fold_indices,
                [test_b2[col], test_b3[col]],
                smoothing=20,
            )
            te_name = f"{col}_te"
            te_train_dict[te_name] = train_te.values
            te_b2_dict[te_name] = b2_te.values
            te_b3_dict[te_name] = b3_te.values
            print(f"    {col} → {te_name}")

        # Build augmented feature DataFrames
        te_train_df = pd.DataFrame(te_train_dict, index=ins_data.index)
        te_b2_df = pd.DataFrame(te_b2_dict, index=test_b2.index)
        te_b3_df = pd.DataFrame(te_b3_dict, index=test_b3.index)

        aug_feature_cols = feature_cols + list(te_train_dict.keys())

        ins_features = pd.concat([ins_data[feature_cols], te_train_df], axis=1)
        test_b2_features = pd.concat([test_b2[feature_cols], te_b2_df], axis=1)
        test_b3_features = pd.concat([test_b3[feature_cols], te_b3_df], axis=1)

        # --- Feature Selection (using first fold split) ---
        first_train_idx, first_val_idx = fold_indices[0]
        sel_features, sel_cats = select_features(
            ins_features.iloc[first_train_idx],
            y_all.iloc[first_train_idx],
            ins_features.iloc[first_val_idx],
            y_all.iloc[first_val_idx],
            aug_feature_cols, cat_cols, top_n=TOP_N_FEATURES,
        )
        sel_cat_indices = [sel_features.index(c) for c in sel_cats]

        # --- OOF & Test Prediction Accumulators ---
        oof_lgb = np.zeros(len(ins_data))
        oof_xgb = np.zeros(len(ins_data))

        test_lgb_b2 = np.zeros(len(test_b2))
        test_lgb_b3 = np.zeros(len(test_b3))
        test_xgb_b2 = np.zeros(len(test_b2))
        test_xgb_b3 = np.zeros(len(test_b3))

        # --- PHASE 3: STRATIFIED K-FOLD TRAINING (Level-1) ---
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            print(f"\n  --- Fold {fold_idx+1}/{N_FOLDS} ---")

            X_train_fold = ins_features.iloc[train_idx][sel_features]
            y_train_fold = y_all.iloc[train_idx]
            X_val_fold = ins_features.iloc[val_idx][sel_features]
            y_val_fold = y_all.iloc[val_idx]

            # ---- LightGBM (Tweedie) ----
            print(f"  [LightGBM Tweedie] Insurer {insurer}...")
            fold_weights = recency_weights[train_idx]
            lgb_models, lgb_val, label_maps = train_lightgbm_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                sel_cats, insurer, fold_idx, n_bags=N_BAGS,
                sample_weights=fold_weights,
            )
            oof_lgb[val_idx] = lgb_val

            # LightGBM test predictions
            test_b2_lgb_enc, _ = _label_encode_for_lgb(test_b2_features[sel_features], sel_cats, label_maps)
            test_b3_lgb_enc, _ = _label_encode_for_lgb(test_b3_features[sel_features], sel_cats, label_maps)
            lgb_fold_b2 = np.mean([m.predict(test_b2_lgb_enc) for m in lgb_models], axis=0)
            lgb_fold_b3 = np.mean([m.predict(test_b3_lgb_enc) for m in lgb_models], axis=0)
            test_lgb_b2 += lgb_fold_b2 / N_FOLDS
            test_lgb_b3 += lgb_fold_b3 / N_FOLDS

            # ---- XGBoost (Tweedie) ----
            print(f"  [XGBoost Tweedie] Insurer {insurer}...")
            import xgboost as xgb
            xgb_model, xgb_val, xgb_label_maps = train_xgboost_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                sel_cats, insurer, fold_idx, sample_weights=fold_weights,
            )
            oof_xgb[val_idx] = xgb_val

            # XGBoost test predictions
            test_b2_xgb_enc, _ = _label_encode_for_lgb(test_b2_features[sel_features], sel_cats, xgb_label_maps)
            test_b3_xgb_enc, _ = _label_encode_for_lgb(test_b3_features[sel_features], sel_cats, xgb_label_maps)
            test_xgb_b2 += xgb_model.predict(xgb.DMatrix(test_b2_xgb_enc)) / N_FOLDS
            test_xgb_b3 += xgb_model.predict(xgb.DMatrix(test_b3_xgb_enc)) / N_FOLDS

            # Save last fold's models
            if fold_idx == N_FOLDS - 1:
                lgb_models[0].save_model(os.path.join(MODEL_DIR, f"sota_lgb_{insurer}.txt"))
                xgb_model.save_model(os.path.join(MODEL_DIR, f"sota_xgb_{insurer}.json"))

            del lgb_models, xgb_model
            del X_train_fold, y_train_fold, X_val_fold, y_val_fold
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

        # --- Clip OOF predictions (Tweedie can produce negatives on edge cases) ---
        oof_lgb = np.clip(oof_lgb, 1.0, None)
        oof_xgb = np.clip(oof_xgb, 1.0, None)

        # --- PHASE 4: Level-2 Meta-Learner ---
        print(f"\n  Training Level-2 Meta-Learner for Insurer {insurer}...")
        meta, meta_mae = train_meta_learner(oof_lgb, oof_xgb, y_all)

        # Final test predictions via meta-learner
        X_meta_b2 = _build_meta_features(test_lgb_b2, test_xgb_b2)
        X_meta_b3 = _build_meta_features(test_lgb_b3, test_xgb_b3)
        test_meta_b2 = meta.predict(X_meta_b2)
        test_meta_b3 = meta.predict(X_meta_b3)

        preds_b2[f"Insurer_{insurer}_price"] = np.clip(test_meta_b2, 1.0, None)
        preds_b3[f"Insurer_{insurer}_price"] = np.clip(test_meta_b3, 1.0, None)

        all_results.append({
            'insurer': insurer,
            'lgb_mae': float(np.mean(np.abs(oof_lgb - y_all.values))),
            'xgb_mae': float(np.mean(np.abs(oof_xgb - y_all.values))),
            'meta_mae': meta_mae,
            'meta_weights': {
                'lgb': float(meta.coef_[0]),
                'xgb': float(meta.coef_[1]),
            },
        })

        del ins_data, ins_features, te_train_df, te_b2_df, te_b3_df
        del test_b2_features, test_b3_features
        gc.collect()

        # Checkpoint after each insurer (Colab can disconnect)
        preds_b2.to_csv(os.path.join(MODEL_DIR, 'checkpoint_b2.csv'), sep=';', index=False)
        preds_b3.to_csv(os.path.join(MODEL_DIR, 'checkpoint_b3.csv'), sep=';', index=False)
        completed_insurers.append(insurer)
        with open(os.path.join(MODEL_DIR, 'checkpoint_completed.json'), 'w') as f:
            json.dump(completed_insurers, f)
        print(f"  [Checkpoint] Insurer {insurer} saved. Completed: {[r['insurer'] for r in all_results]}")

    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print(f"SOTA ENSEMBLE SUMMARY — {N_FOLDS}-FOLD STRATIFIED CV + TWEEDIE + META-LEARNER")
    print("=" * 70)
    for r in all_results:
        w = r['meta_weights']
        print(f"  {r['insurer']}: LGB={r['lgb_mae']:.2f}, XGB={r['xgb_mae']:.2f} "
              f"→ Meta={r['meta_mae']:.2f}")
        print(f"         [w: LGB={w['lgb']:.2f}, XGB={w['xgb']:.2f}]")

    # ---- SAVE SUBMISSIONS ----
    for block, df in [('block2', preds_b2), ('block3', preds_b3)]:
        path = os.path.join(SUBMISSIONS_DIR, f'sota_ensemble_{block}.csv')
        df.to_csv(path, sep=';', index=False)
        print(f"\nSaved {path}")

    # ---- MERGE INTO EXISTING SUBMISSIONS ----
    print("\n" + "=" * 70)
    print("MERGING INTO EXISTING SUBMISSIONS")
    print("=" * 70)

    for block in ['block2', 'block3']:
        base_path = os.path.join(SUBMISSIONS_DIR, f'submission_{block}.csv')
        if not os.path.exists(base_path):
            print(f"  {base_path} not found, skipping merge")
            continue

        base = pd.read_csv(base_path, sep=';')
        new_preds = preds_b2 if block == 'block2' else preds_b3

        # Merge on quote_id to avoid row-order misalignment
        merge_cols = ['quote_id'] + [f"Insurer_{ins}_price" for ins in TARGET_INSURERS]
        new_rounded = new_preds[merge_cols].copy()
        for ins in TARGET_INSURERS:
            new_rounded[f"Insurer_{ins}_price"] = new_rounded[f"Insurer_{ins}_price"].round(3)

        # Determine quote_id column name in base
        base_qid = 'quote_id'
        if base_qid not in base.columns:
            for c in base.columns:
                if 'quote' in c.lower() and 'id' in c.lower():
                    base_qid = c
                    break

        merged = base.merge(new_rounded, left_on=base_qid, right_on='quote_id',
                            how='left', suffixes=('_old', '_new'))

        for ins in TARGET_INSURERS:
            raw_col = f"Insurer_{ins}_price"
            fmt_col = f"Insurer {ins} price"
            new_col = f"{raw_col}_new" if f"{raw_col}_new" in merged.columns else raw_col
            if fmt_col in merged.columns:
                merged[fmt_col] = merged[new_col]
            elif f"{raw_col}_old" in merged.columns:
                merged[raw_col] = merged[new_col]

        # Drop helper columns from the merge
        drop_cols = [c for c in merged.columns if c.endswith('_new') or c.endswith('_old')]
        merged = merged.drop(columns=drop_cols, errors='ignore')

        out_path = os.path.join(SUBMISSIONS_DIR, f'submission_{block}_sota.csv')
        merged.to_csv(out_path, sep=';', index=False)
        print(f"  Merged -> {out_path}")

    # ---- SAVE RESULTS ----
    with open(os.path.join(MODEL_DIR, 'sota_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nTotal runtime: {(time.time()-start)/60:.1f} minutes")
    print("Done!")


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    run_pipeline()
