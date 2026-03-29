"""
Combined prediction: 3 standard CatBoost (A,H,J) + 8 ensemble CatBoost (B,C,D,E,F,G,I,K).
Outputs: submissions/comb_block2.csv, submissions/comb_block3.csv
"""

import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor

# ============================================================
# CONFIG
# ============================================================
INSURERS_ALL = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
STD_INSURERS = ['A', 'H', 'J']
ENS_INSURERS = ['B', 'C', 'D', 'E', 'F', 'G', 'I', 'K']

DATA_DIR = 'data'
MODEL_DIR = 'new_models'
OUT_DIR = 'submissions'

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
]

REFERENCE_DATE = pd.Timestamp('2025-01-01')


# ============================================================
# SHARED HELPERS
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


def _base_date_features(df):
    """Date-derived features shared by both preprocessing paths."""
    cb = _to_dt(df['contractor_birthdate'])
    df['contractor_age'] = (REFERENCE_DATE - cb).dt.days / 365.25

    sb = _to_dt(df['second_driver_birthdate'])
    df['second_driver_age'] = (REFERENCE_DATE - sb).dt.days / 365.25
    df['has_second_driver'] = df['second_driver_birthdate'].notna().astype(int)

    insp = _to_dt(df['vehicle_inspection_report_date'])
    df['days_since_inspection'] = (REFERENCE_DATE - insp).dt.days

    exp = _to_dt(df['vehicle_inspection_expiry_date'])
    df['inspection_days_remaining'] = (exp - REFERENCE_DATE).dt.days

    vlr = _to_dt(df['vehicle_last_registration_date'])
    df['years_since_last_registration'] = (REFERENCE_DATE - vlr).dt.days / 365.25

    vfr = _to_dt(df['vehicle_first_registration_date'])
    df['years_since_first_registration'] = (REFERENCE_DATE - vfr).dt.days / 365.25

    vcfr = _to_dt(df['vehicle_country_first_registration_date'])
    df['years_since_country_first_reg'] = (REFERENCE_DATE - vcfr).dt.days / 365.25

    return df


def get_feature_and_cat_columns(df):
    exclude = EXCLUDE_COLS | set(DATE_COLS)
    feature_cols = [c for c in df.columns if c not in exclude]

    cat_cols = []
    for col in feature_cols:
        dtype_name = df[col].dtype.name if hasattr(df[col].dtype, 'name') else str(df[col].dtype)
        if col in CATEGORICAL_COLS or col == 'contractor_age_bucket' or dtype_name in ('object', 'str', 'string', 'category'):
            cat_cols.append(col)

    return feature_cols, cat_cols


def prepare_categoricals(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna('_MISSING_').astype(str)
    return df


# ============================================================
# STANDARD PREPROCESSING (matches src/preprocess.py)
# ============================================================
def standard_preprocess(df):
    df = convert_dtypes(df.copy())
    df = _base_date_features(df)
    return df.copy()


# ============================================================
# ENSEMBLE PREPROCESSING (matches ensemble_EKI/preprocess.py)
# ============================================================
def ensemble_preprocess(df):
    df = convert_dtypes(df.copy())
    df = _base_date_features(df)

    # --- Extra ensemble features ---
    df['contractor_age_sq'] = df['contractor_age'] ** 2
    df['second_driver_age_sq'] = df['second_driver_age'] ** 2

    df['contractor_age_bucket'] = pd.cut(
        df['contractor_age'], bins=[0, 25, 35, 50, 65, 120],
        labels=['very_young', 'young', 'mid', 'senior', 'elderly'], ordered=False
    ).astype(str).fillna('_MISSING_')

    df['claim_free_x_age'] = df['claim_free_years'] * df['contractor_age']
    df['claim_free_x_vehicle_age'] = df['claim_free_years'] * df.get('vehicle_age', 0)

    if 'vehicle_power' in df.columns and 'vehicle_net_weight' in df.columns:
        df['weight_per_power'] = df['vehicle_net_weight'] / (df['vehicle_power'] + 1)

    if 'vehicle_value_new' in df.columns:
        df['value_per_age'] = df['vehicle_value_new'] / df.get('vehicle_age', 1).clip(lower=0.5)
        df['log_vehicle_value'] = np.log1p(df['vehicle_value_new'].clip(lower=0))

    if 'municipality_crimes_per_1000' in df.columns:
        df['high_crime_area'] = (df['municipality_crimes_per_1000'] > df['municipality_crimes_per_1000'].quantile(0.75)).astype(int)

    if 'postal_code_address_density' in df.columns:
        df['log_address_density'] = np.log1p(df['postal_code_address_density'].clip(lower=0))

    deductible_present = [c for c in DEDUCTIBLE_COLS if c in df.columns]
    if deductible_present:
        df['mean_deductible'] = df[deductible_present].mean(axis=1)
        df['std_deductible'] = df[deductible_present].std(axis=1)
        df['max_deductible'] = df[deductible_present].max(axis=1)
        df['min_deductible'] = df[deductible_present].min(axis=1)
        for ins in ENS_INSURERS:
            ded_col = f"Insurer_{ins}_deductible"
            if ded_col in df.columns:
                df[f'{ins}_ded_vs_mean'] = df[ded_col] - df['mean_deductible']

    if 'vehicle_inspection_number_of_deficiencies_found' in df.columns:
        df['has_deficiencies'] = (df['vehicle_inspection_number_of_deficiencies_found'] > 0).astype(int)

    if 'vehicle_planned_annual_mileage' in df.columns:
        df['log_mileage'] = np.log1p(df['vehicle_planned_annual_mileage'].clip(lower=0))

    return df.copy()


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading test data...")
    test_b2_raw = pd.read_parquet(os.path.join(DATA_DIR, 'block2_test.parquet'))
    test_b3_raw = pd.read_parquet(os.path.join(DATA_DIR, 'block3_test.parquet'))
    print(f"  Block 2: {test_b2_raw.shape}, Block 3: {test_b3_raw.shape}")

    # --- Standard preprocessing (for A,B,C,D,G,H,J) ---
    print("\nStandard preprocessing...")
    test_b2_std = standard_preprocess(test_b2_raw)
    test_b3_std = standard_preprocess(test_b3_raw)
    std_features, std_cats = get_feature_and_cat_columns(test_b2_std)
    test_b2_std = prepare_categoricals(test_b2_std, std_cats)
    test_b3_std = prepare_categoricals(test_b3_std, std_cats)
    print(f"  Features: {len(std_features)}, Categoricals: {len(std_cats)}")

    # --- Ensemble preprocessing (for E,F,I,K) ---
    print("\nEnsemble preprocessing...")
    test_b2_ens = ensemble_preprocess(test_b2_raw)
    test_b3_ens = ensemble_preprocess(test_b3_raw)
    ens_features, ens_cats = get_feature_and_cat_columns(test_b2_ens)
    test_b2_ens = prepare_categoricals(test_b2_ens, ens_cats)
    test_b3_ens = prepare_categoricals(test_b3_ens, ens_cats)
    print(f"  Features: {len(ens_features)}, Categoricals: {len(ens_cats)}")

    # --- Predictions ---
    preds_b2 = pd.DataFrame({'quote_id': test_b2_raw['quote_id']})
    preds_b3 = pd.DataFrame({'quote_id': test_b3_raw['quote_id']})

    # Standard CatBoost: A, H, J
    print("\n" + "=" * 60)
    print("STANDARD CATBOOST — A, H, J")
    print("=" * 60)
    for ins in STD_INSURERS:
        model_path = os.path.join(MODEL_DIR, f'catboost_insurer_{ins}.cbm')
        model = CatBoostRegressor()
        model.load_model(model_path)

        # Use model's own feature names if available, else fall back
        feat = model.feature_names_ if model.feature_names_ else std_features
        p2 = model.predict(test_b2_std[feat])
        p3 = model.predict(test_b3_std[feat])

        preds_b2[f'Insurer {ins} price'] = p2
        preds_b3[f'Insurer {ins} price'] = p3
        print(f"  Insurer {ins}: B2 mean={p2.mean():.2f}, B3 mean={p3.mean():.2f}")
        del model

    # Ensemble CatBoost: B, C, D, E, F, G, I, K
    print("\n" + "=" * 60)
    print("ENSEMBLE CATBOOST — B, C, D, E, F, G, I, K")
    print("=" * 60)
    for ins in ENS_INSURERS:
        model_path = os.path.join(MODEL_DIR, f'ensemble_cb_insurer_{ins}.cbm')
        model = CatBoostRegressor()
        model.load_model(model_path)

        # Use model's own feature names if available, else fall back
        feat = model.feature_names_ if model.feature_names_ else ens_features
        cat_feats = [feat[i] for i in model.get_cat_feature_indices()] if model.get_cat_feature_indices() is not None else []
        
        # Make sure all features expected by the model exist in the DataFrame
        for f in feat:
            if f not in test_b2_ens.columns:
                val = '_MISSING_' if f in cat_feats else np.nan
                test_b2_ens[f] = val
            elif f in cat_feats:
                test_b2_ens[f] = test_b2_ens[f].fillna('_MISSING_').astype(str)
            else:
                test_b2_ens[f] = pd.to_numeric(test_b2_ens[f], errors='coerce')
                
            if f not in test_b3_ens.columns:
                val = '_MISSING_' if f in cat_feats else np.nan
                test_b3_ens[f] = val
            elif f in cat_feats:
                test_b3_ens[f] = test_b3_ens[f].fillna('_MISSING_').astype(str)
            else:
                test_b3_ens[f] = pd.to_numeric(test_b3_ens[f], errors='coerce')

        # Ensemble models were trained on log1p(price) targets — inverse transform
        p2 = np.expm1(model.predict(test_b2_ens[feat]))
        p3 = np.expm1(model.predict(test_b3_ens[feat]))

        preds_b2[f'Insurer {ins} price'] = np.clip(p2, 1.0, None)
        preds_b3[f'Insurer {ins} price'] = np.clip(p3, 1.0, None)
        print(f"  Insurer {ins}: B2 mean={p2.mean():.2f}, B3 mean={p3.mean():.2f}")
        del model

    # --- Save ---
    os.makedirs(OUT_DIR, exist_ok=True)

    # Column order: quote_id, then A through K (matching existing submission format)
    col_order = ['quote_id'] + [f'Insurer {ins} price' for ins in INSURERS_ALL]
    preds_b2 = preds_b2[col_order]
    preds_b3 = preds_b3[col_order]

    for ins in INSURERS_ALL:
        col = f'Insurer {ins} price'
        preds_b2[col] = preds_b2[col].round(3)
        preds_b3[col] = preds_b3[col].round(3)

    b2_path = os.path.join(OUT_DIR, 'comb_block2.csv')
    b3_path = os.path.join(OUT_DIR, 'comb_block3.csv')
    preds_b2.to_csv(b2_path, sep=';', index=False)
    preds_b3.to_csv(b3_path, sep=';', index=False)

    print(f"\nSaved {b2_path}: {preds_b2.shape}")
    print(f"Saved {b3_path}: {preds_b3.shape}")
    print("Done!")


if __name__ == '__main__':
    main()
