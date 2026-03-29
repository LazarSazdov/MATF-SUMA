"""
Quick train-set prediction for bias analysis.
Trains lightweight LGB + XGB on full training data per insurer,
predicts on the same data, saves CSV with actual vs predicted.
"""

import pandas as pd
import numpy as np
import os
import gc
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')
BASE_DIR = '/content/drive/MyDrive/matf_data'
DATA_DIR = os.path.join(BASE_DIR, 'data')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

INSURERS_ALL = list('ABCDEFGHIJK')
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
# PREPROCESSING (same as main pipeline)
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


def engineer_features(df):
    df = convert_dtypes(df.copy())
    new_cols = {}

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

    contractor_age = new_cols['contractor_age']
    new_cols['contractor_age_sq'] = contractor_age ** 2
    new_cols['second_driver_age_sq'] = new_cols['second_driver_age'] ** 2

    new_cols['contractor_age_bucket'] = pd.cut(
        contractor_age, bins=[0, 25, 35, 50, 65, 120],
        labels=['very_young', 'young', 'mid', 'senior', 'elderly'], ordered=False
    ).astype(str).fillna('_MISSING_')

    claim_free = df.get('claim_free_years', pd.Series(0, index=df.index))
    new_cols['claim_free_x_age'] = claim_free * contractor_age
    new_cols['claim_free_per_age'] = claim_free / contractor_age.clip(lower=18)

    if 'vehicle_age' in df.columns:
        new_cols['claim_free_x_vehicle_age'] = claim_free * df['vehicle_age']

    if 'second_driver_claim_free_years' in df.columns:
        sd_claim_free = df['second_driver_claim_free_years']
        new_cols['second_driver_claim_free_x_age'] = sd_claim_free * new_cols['second_driver_age']
        new_cols['total_claim_free_years'] = claim_free.fillna(0) + sd_claim_free.fillna(0)
        new_cols['claim_free_diff'] = claim_free.fillna(0) - sd_claim_free.fillna(0)

    if 'vehicle_power' in df.columns and 'vehicle_net_weight' in df.columns:
        new_cols['weight_per_power'] = df['vehicle_net_weight'] / (df['vehicle_power'] + 1)
        new_cols['power_per_weight'] = df['vehicle_power'] / (df['vehicle_net_weight'] + 1)

    if 'vehicle_value_new' in df.columns:
        vage = df.get('vehicle_age', pd.Series(1, index=df.index)).clip(lower=0.5)
        new_cols['value_per_age'] = df['vehicle_value_new'] / vage
        new_cols['log_vehicle_value'] = np.log1p(df['vehicle_value_new'].clip(lower=0))

    if 'municipality_crimes_per_1000' in df.columns:
        new_cols['high_crime_area'] = (
            df['municipality_crimes_per_1000'] > df['municipality_crimes_per_1000'].quantile(0.75)
        ).astype(int)

    if 'postal_code_address_density' in df.columns:
        new_cols['log_address_density'] = np.log1p(df['postal_code_address_density'].clip(lower=0))

    deductible_present = [c for c in DEDUCTIBLE_COLS if c in df.columns]
    if deductible_present:
        ded_df = df[deductible_present]
        new_cols['mean_deductible'] = ded_df.mean(axis=1)
        new_cols['std_deductible'] = ded_df.std(axis=1)
        new_cols['max_deductible'] = ded_df.max(axis=1)
        new_cols['min_deductible'] = ded_df.min(axis=1)
        mean_ded = new_cols['mean_deductible']
        for ins in INSURERS_ALL:
            ded_col = f"Insurer_{ins}_deductible"
            if ded_col in df.columns:
                new_cols[f'{ins}_ded_vs_mean'] = df[ded_col] - mean_ded

    if 'vehicle_inspection_number_of_deficiencies_found' in df.columns:
        new_cols['has_deficiencies'] = (
            df['vehicle_inspection_number_of_deficiencies_found'] > 0
        ).astype(int)

    if 'vehicle_planned_annual_mileage' in df.columns:
        new_cols['log_mileage'] = np.log1p(df['vehicle_planned_annual_mileage'].clip(lower=0))

    if 'coverage' in df.columns:
        cov_str = df['coverage'].fillna('_MISSING_').astype(str)
        new_cols['coverage_x_age_bucket'] = cov_str + '_' + new_cols['contractor_age_bucket']

    if 'vehicle_value_new' in df.columns and deductible_present:
        vval = df['vehicle_value_new'].clip(lower=1)
        new_cols['mean_deductible_ratio'] = new_cols['mean_deductible'] / vval
        for ins in INSURERS_ALL:
            ded_col = f"Insurer_{ins}_deductible"
            if ded_col in df.columns:
                new_cols[f'{ins}_ded_ratio'] = df[ded_col] / vval

    if 'vehicle_value_new' in df.columns and 'vehicle_power' in df.columns:
        new_cols['value_per_power'] = df['vehicle_value_new'] / (df['vehicle_power'] + 1)

    for col in ['postal_code', 'municipality', 'vehicle_maker', 'vehicle_model']:
        if col in df.columns:
            freq = df[col].value_counts()
            new_cols[f'{col}_freq'] = df[col].map(freq).fillna(1).astype(int)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# ============================================================
# MAIN
# ============================================================
print("Loading training data...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, 'block1_train.parquet'))
print(f"Train: {train_df.shape}")

print("Feature engineering...")
train_df = engineer_features(train_df)

# Risk zones
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer

lat_col, lon_col = 'postal_code_latitude', 'postal_code_longitude'
if lat_col in train_df.columns:
    imputer = SimpleImputer(strategy='median')
    coords = imputer.fit_transform(train_df[[lat_col, lon_col]])
    km = MiniBatchKMeans(n_clusters=100, random_state=42, n_init=3, batch_size=10000)
    train_df['risk_zone'] = km.fit_predict(coords).astype(str)
    print("  Added 100 risk zones")

# Column setup
exclude = EXCLUDE_COLS | set(DATE_COLS)
feature_cols = [c for c in train_df.columns if c not in exclude]
cat_cols = []
for col in feature_cols:
    dtype_name = train_df[col].dtype.name if hasattr(train_df[col].dtype, 'name') else str(train_df[col].dtype)
    if (col in CATEGORICAL_COLS or col == 'contractor_age_bucket'
            or col == 'risk_zone' or col == 'coverage_x_age_bucket'
            or dtype_name in ('object', 'str', 'string', 'category')):
        cat_cols.append(col)

# Fill categoricals
for col in cat_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna('_MISSING_').astype(str)

print(f"Features: {len(feature_cols)}, Categoricals: {len(cat_cols)}")

# ============================================================
# PER-INSURER: quick LGB + XGB on full data → predict on same
# ============================================================
import lightgbm as lgb
import xgboost as xgb

results = pd.DataFrame({'quote_id': train_df['quote_id']})

for insurer in INSURERS_ALL:
    target_col = f"Insurer_{insurer}_price"
    mask = train_df[target_col].notna()
    ins_data = train_df[mask].reset_index(drop=True)
    y = ins_data[target_col].copy()

    # Clip at 99.5th percentile (same as training pipeline)
    threshold = y.quantile(0.995)
    y = y.clip(upper=threshold)

    # Target encoding on full set (slight leakage — fine for bias check)
    te_cols = ['postal_code', 'municipality', 'province', 'vehicle_maker',
               'vehicle_model', 'coverage_x_age_bucket', 'risk_zone']
    te_dict = {}
    global_mean = float(y.mean())
    smoothing = 20
    for col in te_cols:
        if col not in ins_data.columns:
            continue
        stats = pd.DataFrame({'cat': ins_data[col], 'y': y}).groupby('cat')['y'].agg(['mean', 'count'])
        smoothed = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        te_dict[f'{col}_te'] = ins_data[col].map(smoothed).fillna(global_mean).values

    te_df = pd.DataFrame(te_dict, index=ins_data.index)
    aug_features = feature_cols + list(te_dict.keys())
    X = pd.concat([ins_data[feature_cols], te_df], axis=1)

    # Label encode categoricals
    label_maps = {}
    X_enc = X.copy()
    for col in cat_cols:
        if col not in X_enc.columns:
            continue
        uniques = X_enc[col].unique()
        label_maps[col] = {v: i for i, v in enumerate(uniques)}
        X_enc[col] = X_enc[col].map(label_maps[col]).fillna(-1).astype(int)

    print(f"\nInsurer {insurer}: {len(ins_data):,} samples")

    # --- LightGBM ---
    lgb_ds = lgb.Dataset(X_enc, label=y)
    lgb_params = {
        'objective': 'tweedie', 'tweedie_variance_power': 1.5,
        'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 96,
        'max_depth': 8, 'min_data_in_leaf': 25, 'feature_fraction': 0.7,
        'bagging_fraction': 0.85, 'bagging_freq': 1,
        'lambda_l1': 0.8, 'lambda_l2': 4.0, 'verbose': -1, 'n_jobs': -1,
        'seed': 42,
    }
    lgb_model = lgb.train(lgb_params, lgb_ds, num_boost_round=3000,
                          callbacks=[lgb.log_evaluation(0)])
    lgb_preds = lgb_model.predict(X_enc)
    lgb_mae = float(np.mean(np.abs(lgb_preds - y.values)))

    # --- XGBoost ---
    dmat = xgb.DMatrix(X_enc, label=y)
    xgb_params = {
        'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5,
        'eval_metric': 'mae', 'learning_rate': 0.05, 'max_depth': 8,
        'min_child_weight': 20, 'subsample': 0.85, 'colsample_bytree': 0.7,
        'reg_alpha': 0.8, 'reg_lambda': 4.0,
        'tree_method': 'hist', 'device': 'cuda',
        'seed': 42, 'verbosity': 0,
    }
    xgb_model = xgb.train(xgb_params, dmat, num_boost_round=3000, verbose_eval=0)
    xgb_preds = xgb_model.predict(dmat)
    xgb_mae = float(np.mean(np.abs(xgb_preds - y.values)))

    # Average
    avg_preds = (lgb_preds + xgb_preds) / 2
    avg_mae = float(np.mean(np.abs(avg_preds - y.values)))

    print(f"  LGB MAE: {lgb_mae:.4f}, XGB MAE: {xgb_mae:.4f}, Avg MAE: {avg_mae:.4f}")

    # Store in results (NaN for rows where insurer didn't quote)
    pred_col = f"Insurer_{insurer}_predicted"
    actual_col = f"Insurer_{insurer}_actual"
    results[actual_col] = np.nan
    results[pred_col] = np.nan
    results.loc[mask.values, actual_col] = y.values
    results.loc[mask.values, pred_col] = avg_preds

    del lgb_model, xgb_model, dmat, lgb_ds, X_enc, X, te_df, ins_data
    gc.collect()

# Save
out_path = os.path.join(SUBMISSIONS_DIR, 'train_predictions_bias_check.csv')
results.to_csv(out_path, sep=';', index=False)
print(f"\nSaved to {out_path}")
print(f"Shape: {results.shape}")
print("Done!")
