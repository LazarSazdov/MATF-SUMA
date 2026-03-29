import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
INSURERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

PRICE_COLS = [f"Insurer_{i}_price" for i in INSURERS]
DEDUCTIBLE_COLS = [f"Insurer_{i}_deductible" for i in INSURERS]

# Columns to ALWAYS exclude from features
EXCLUDE_COLS = {
    'quote_id',              # unique identifier
    'vehicle_number_plate',  # unique identifier
} | set(PRICE_COLS)          # targets, not features

# Raw date columns -- extract numeric features then drop originals
DATE_COLS = [
    'contractor_birthdate',
    'second_driver_birthdate',
    'vehicle_first_registration_date',
    'vehicle_country_first_registration_date',
    'vehicle_last_registration_date',
    'vehicle_inspection_report_date',
    'vehicle_inspection_expiry_date',
]

# Columns that are truly categorical (text labels, not numeric)
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
# CONVERT ARROW STRING COLUMNS TO PROPER TYPES
# ============================================================
def convert_dtypes(df):
    """Convert Arrow str columns: numeric-like -> float64, rest stays as str."""
    df = df.copy()
    skip = set(DATE_COLS) | {'quote_id', 'vehicle_number_plate'} | set(CATEGORICAL_COLS)
    for col in df.columns:
        if col in skip:
            continue
        if hasattr(df[col].dtype, 'name') and df[col].dtype.name in ('str', 'string', 'object'):
            converted = pd.to_numeric(df[col], errors='coerce')
            # If >80% of non-null values converted successfully, treat as numeric
            orig_non_null = df[col].notna().sum()
            conv_non_null = converted.notna().sum()
            if orig_non_null > 0 and conv_non_null / orig_non_null > 0.8:
                df[col] = converted
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def _to_dt(series):
    """Convert a series to datetime if it isn't already."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, dayfirst=True, errors='coerce')


def engineer_features(df):
    df = df.copy()

    # Convert numeric-like string columns to float
    df = convert_dtypes(df)

    # -- 1. CONTRACTOR AGE --
    df['contractor_birthdate'] = _to_dt(df['contractor_birthdate'])
    df['contractor_age'] = (REFERENCE_DATE - df['contractor_birthdate']).dt.days / 365.25

    # -- 2. SECOND DRIVER AGE --
    df['second_driver_birthdate'] = _to_dt(df['second_driver_birthdate'])
    df['second_driver_age'] = (REFERENCE_DATE - df['second_driver_birthdate']).dt.days / 365.25

    # -- 3. HAS SECOND DRIVER --
    df['has_second_driver'] = df['second_driver_birthdate'].notna().astype(int)

    # -- 4. DAYS SINCE INSPECTION --
    df['vehicle_inspection_report_date'] = _to_dt(df['vehicle_inspection_report_date'])
    df['days_since_inspection'] = (REFERENCE_DATE - df['vehicle_inspection_report_date']).dt.days

    # -- 5. INSPECTION VALIDITY REMAINING --
    df['vehicle_inspection_expiry_date'] = _to_dt(df['vehicle_inspection_expiry_date'])
    df['inspection_days_remaining'] = (df['vehicle_inspection_expiry_date'] - REFERENCE_DATE).dt.days

    # -- 6. YEARS SINCE LAST REGISTRATION --
    df['vehicle_last_registration_date'] = _to_dt(df['vehicle_last_registration_date'])
    df['years_since_last_registration'] = (REFERENCE_DATE - df['vehicle_last_registration_date']).dt.days / 365.25

    # -- 7. YEARS SINCE FIRST REGISTRATION --
    df['vehicle_first_registration_date'] = _to_dt(df['vehicle_first_registration_date'])
    df['years_since_first_registration'] = (REFERENCE_DATE - df['vehicle_first_registration_date']).dt.days / 365.25

    # -- 8. YEARS SINCE COUNTRY FIRST REGISTRATION --
    df['vehicle_country_first_registration_date'] = _to_dt(df['vehicle_country_first_registration_date'])
    df['years_since_country_first_reg'] = (REFERENCE_DATE - df['vehicle_country_first_registration_date']).dt.days / 365.25

    # Defragment after many column additions
    df = df.copy()
    return df


# ============================================================
# COLUMN SELECTION
# ============================================================
def get_feature_and_cat_columns(df):
    exclude = EXCLUDE_COLS | set(DATE_COLS)
    feature_cols = [c for c in df.columns if c not in exclude]

    # Categorical = explicitly listed + any remaining string/object columns
    cat_cols = []
    for col in feature_cols:
        dtype_name = df[col].dtype.name if hasattr(df[col].dtype, 'name') else str(df[col].dtype)
        if col in CATEGORICAL_COLS or dtype_name in ('object', 'str', 'string', 'category'):
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
# FULL PIPELINE
# ============================================================
def run_preprocessing():
    train_df = load_data("data/block1_train.parquet")
    test_b2 = load_data("data/block2_test.parquet")
    test_b3 = load_data("data/block3_test.parquet")

    # NaN rates
    print("\n--- Quote rates ---")
    for ins in INSURERS:
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

    return train_part, val_part, test_b2, test_b3, feature_cols, cat_cols
