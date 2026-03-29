import pandas as pd
import numpy as np
import gc

# ============================================================
# CONFIG
# ============================================================
INSURERS_ALL = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
TARGET_INSURERS = ['E', 'I', 'K']

PRICE_COLS = [f"Insurer_{i}_price" for i in INSURERS_ALL]
DEDUCTIBLE_COLS = [f"Insurer_{i}_deductible" for i in INSURERS_ALL]
TARGET_COLS = [f"Insurer_{i}_price" for i in TARGET_INSURERS]

EXCLUDE_COLS = {
    'quote_id',
    'vehicle_number_plate',
} | set(PRICE_COLS)

DATE_COLS = [
    'contractor_birthdate',
    'second_driver_birthdate',
    'vehicle_first_registration_date',
    'vehicle_country_first_registration_date',
    'vehicle_last_registration_date',
    'vehicle_inspection_report_date',
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
# LOAD
# ============================================================
def load_data(path):
    df = pd.read_parquet(path)
    print(f"Loaded {path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ============================================================
# CONVERT ARROW STRING COLUMNS
# ============================================================
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


# ============================================================
# HELPERS
# ============================================================
def _to_dt(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, dayfirst=True, errors='coerce')


# ============================================================
# FEATURE ENGINEERING — HARSHER / MORE AGGRESSIVE
# ============================================================
def engineer_features(df):
    df = df.copy()
    df = convert_dtypes(df)

    # --- DATE-DERIVED FEATURES ---
    df['contractor_birthdate'] = _to_dt(df['contractor_birthdate'])
    df['contractor_age'] = (REFERENCE_DATE - df['contractor_birthdate']).dt.days / 365.25

    df['second_driver_birthdate'] = _to_dt(df['second_driver_birthdate'])
    df['second_driver_age'] = (REFERENCE_DATE - df['second_driver_birthdate']).dt.days / 365.25

    df['has_second_driver'] = df['second_driver_birthdate'].notna().astype(int)

    df['vehicle_inspection_report_date'] = _to_dt(df['vehicle_inspection_report_date'])
    df['days_since_inspection'] = (REFERENCE_DATE - df['vehicle_inspection_report_date']).dt.days

    df['vehicle_inspection_expiry_date'] = _to_dt(df['vehicle_inspection_expiry_date'])
    df['inspection_days_remaining'] = (df['vehicle_inspection_expiry_date'] - REFERENCE_DATE).dt.days

    df['vehicle_last_registration_date'] = _to_dt(df['vehicle_last_registration_date'])
    df['years_since_last_registration'] = (REFERENCE_DATE - df['vehicle_last_registration_date']).dt.days / 365.25

    df['vehicle_first_registration_date'] = _to_dt(df['vehicle_first_registration_date'])
    df['years_since_first_registration'] = (REFERENCE_DATE - df['vehicle_first_registration_date']).dt.days / 365.25

    df['vehicle_country_first_registration_date'] = _to_dt(df['vehicle_country_first_registration_date'])
    df['years_since_country_first_reg'] = (REFERENCE_DATE - df['vehicle_country_first_registration_date']).dt.days / 365.25

    # --- HARSHER FEATURE ENGINEERING ---

    # Age polynomial — U-shaped pricing curve
    df['contractor_age_sq'] = df['contractor_age'] ** 2

    # Age buckets (young/mid/old) — helps trees split faster
    df['contractor_age_bucket'] = pd.cut(
        df['contractor_age'],
        bins=[0, 25, 35, 50, 65, 120],
        labels=['very_young', 'young', 'mid', 'senior', 'elderly'],
        ordered=False
    ).astype(str).fillna('_MISSING_')

    # Second driver age polynomial
    df['second_driver_age_sq'] = df['second_driver_age'] ** 2

    # Claim-free years interactions — very important for E, I, K
    df['claim_free_x_age'] = df['claim_free_years'] * df['contractor_age']
    df['claim_free_x_vehicle_age'] = df['claim_free_years'] * df.get('vehicle_age', 0)

    # Vehicle risk proxies
    if 'vehicle_power' in df.columns and 'vehicle_net_weight' in df.columns:
        # Power-to-weight already exists but let's add inverse
        df['weight_per_power'] = df['vehicle_net_weight'] / (df['vehicle_power'] + 1)

    if 'vehicle_value_new' in df.columns:
        df['value_per_age'] = df['vehicle_value_new'] / (df.get('vehicle_age', 1).clip(lower=0.5))
        df['log_vehicle_value'] = np.log1p(df['vehicle_value_new'].clip(lower=0))

    # Geographic risk
    if 'municipality_crimes_per_1000' in df.columns:
        df['high_crime_area'] = (df['municipality_crimes_per_1000'] > df['municipality_crimes_per_1000'].quantile(0.75)).astype(int)

    if 'postal_code_address_density' in df.columns:
        df['log_address_density'] = np.log1p(df['postal_code_address_density'].clip(lower=0))

    # Deductible interactions — mean deductible across all insurers
    deductible_present = [c for c in DEDUCTIBLE_COLS if c in df.columns]
    if deductible_present:
        df['mean_deductible'] = df[deductible_present].mean(axis=1)
        df['std_deductible'] = df[deductible_present].std(axis=1)
        df['max_deductible'] = df[deductible_present].max(axis=1)
        df['min_deductible'] = df[deductible_present].min(axis=1)
        # Per-insurer deductible relative to mean
        for ins in TARGET_INSURERS:
            ded_col = f"Insurer_{ins}_deductible"
            if ded_col in df.columns:
                df[f'{ins}_ded_vs_mean'] = df[ded_col] - df['mean_deductible']

    # Inspection quality proxy
    if 'vehicle_inspection_number_of_deficiencies_found' in df.columns:
        df['has_deficiencies'] = (df['vehicle_inspection_number_of_deficiencies_found'] > 0).astype(int)

    # Mileage risk
    if 'vehicle_planned_annual_mileage' in df.columns:
        df['log_mileage'] = np.log1p(df['vehicle_planned_annual_mileage'].clip(lower=0))

    df = df.copy()  # defragment
    return df


# ============================================================
# COLUMN SELECTION
# ============================================================
def get_feature_and_cat_columns(df):
    exclude = EXCLUDE_COLS | set(DATE_COLS)
    feature_cols = [c for c in df.columns if c not in exclude]

    cat_cols = []
    for col in feature_cols:
        dtype_name = df[col].dtype.name if hasattr(df[col].dtype, 'name') else str(df[col].dtype)
        if col in CATEGORICAL_COLS or col == 'contractor_age_bucket' or dtype_name in ('object', 'str', 'string', 'category'):
            cat_cols.append(col)

    print(f"Feature columns: {len(feature_cols)}")
    print(f"Categorical columns: {len(cat_cols)} -> {cat_cols}")
    return feature_cols, cat_cols


# ============================================================
# PREPARE CATEGORICALS
# ============================================================
def prepare_categoricals(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna('_MISSING_').astype(str)
    return df


# ============================================================
# TEMPORAL SPLIT
# ============================================================
def temporal_split(df, val_fraction=0.2):
    split_idx = int(len(df) * (1 - val_fraction))
    train_part = df.iloc[:split_idx].reset_index(drop=True)
    val_part = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train split: {len(train_part):,} rows")
    print(f"Val split:   {len(val_part):,} rows")
    return train_part, val_part


# ============================================================
# OUTLIER CLIPPING ON TARGETS (training only)
# ============================================================
def clip_target_outliers(df, insurer, percentile=99.5):
    """Clip extreme prices for a given insurer at the given percentile."""
    target_col = f"Insurer_{insurer}_price"
    mask = df[target_col].notna()
    if mask.sum() == 0:
        return df
    threshold = df.loc[mask, target_col].quantile(percentile / 100.0)
    n_clipped = (df[target_col] > threshold).sum()
    df.loc[df[target_col] > threshold, target_col] = threshold
    print(f"  Insurer {insurer}: clipped {n_clipped} rows at {threshold:.2f} (p{percentile})")
    return df


# ============================================================
# FULL PIPELINE
# ============================================================
def run_preprocessing():
    train_df = load_data("data/block1_train.parquet")
    test_b2 = load_data("data/block2_test.parquet")
    test_b3 = load_data("data/block3_test.parquet")

    # Quote rates for target insurers
    print("\n--- Quote rates (target insurers) ---")
    for ins in TARGET_INSURERS:
        col = f"Insurer_{ins}_price"
        pct = (1 - train_df[col].isna().mean()) * 100
        print(f"  Insurer {ins}: {pct:.1f}% quoted")

    # Engineer features identically
    train_df = engineer_features(train_df)
    test_b2 = engineer_features(test_b2)
    test_b3 = engineer_features(test_b3)

    # Column lists
    feature_cols, cat_cols = get_feature_and_cat_columns(train_df)

    # Prepare categoricals
    train_df = prepare_categoricals(train_df, cat_cols)
    test_b2 = prepare_categoricals(test_b2, cat_cols)
    test_b3 = prepare_categoricals(test_b3, cat_cols)

    # Temporal split
    train_part, val_part = temporal_split(train_df)

    # Clip outliers on training portion only (not validation!)
    print("\n--- Clipping outliers on train split ---")
    for ins in TARGET_INSURERS:
        train_part = clip_target_outliers(train_part, ins, percentile=99.5)

    gc.collect()

    return train_part, val_part, test_b2, test_b3, feature_cols, cat_cols
