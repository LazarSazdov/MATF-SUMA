"""Compare ensemble CatBoost vs standard CatBoost for Insurer A."""
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from predict_combined import (
    standard_preprocess, get_feature_and_cat_columns, prepare_categoricals,
    DATA_DIR, EXCLUDE_COLS, DATE_COLS, CATEGORICAL_COLS,
    DEDUCTIBLE_COLS, REFERENCE_DATE, convert_dtypes, _to_dt, _base_date_features,
)

INSURERS_ALL = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']


def full_ensemble_preprocess(df):
    """Matches colab_notebook.py engineer_features exactly."""
    df = convert_dtypes(df.copy())
    new_cols = {}

    cb = _to_dt(df['contractor_birthdate'])
    new_cols['contractor_age'] = (REFERENCE_DATE - cb).dt.days / 365.25
    sb = _to_dt(df['second_driver_birthdate'])
    new_cols['second_driver_age'] = (REFERENCE_DATE - sb).dt.days / 365.25
    new_cols['has_second_driver'] = df['second_driver_birthdate'].notna().astype(int)
    insp = _to_dt(df['vehicle_inspection_report_date'])
    new_cols['days_since_inspection'] = (REFERENCE_DATE - insp).dt.days
    exp = _to_dt(df['vehicle_inspection_expiry_date'])
    new_cols['inspection_days_remaining'] = (exp - REFERENCE_DATE).dt.days
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
    if 'vehicle_age' in df.columns:
        new_cols['claim_free_x_vehicle_age'] = claim_free * df['vehicle_age']

    if 'second_driver_claim_free_years' in df.columns:
        sd_claim_free = df['second_driver_claim_free_years']
        new_cols['second_driver_claim_free_x_age'] = sd_claim_free * new_cols['second_driver_age']
        new_cols['total_claim_free_years'] = claim_free.fillna(0) + sd_claim_free.fillna(0)
        new_cols['claim_free_diff'] = claim_free.fillna(0) - sd_claim_free.fillna(0)

    if 'vehicle_power' in df.columns and 'vehicle_net_weight' in df.columns:
        new_cols['weight_per_power'] = df['vehicle_net_weight'] / (df['vehicle_power'] + 1)
    if 'vehicle_value_new' in df.columns:
        vage = df.get('vehicle_age', pd.Series(1, index=df.index)).clip(lower=0.5)
        new_cols['value_per_age'] = df['vehicle_value_new'] / vage
        new_cols['log_vehicle_value'] = np.log1p(df['vehicle_value_new'].clip(lower=0))
    if 'municipality_crimes_per_1000' in df.columns:
        new_cols['high_crime_area'] = (df['municipality_crimes_per_1000'] > df['municipality_crimes_per_1000'].quantile(0.75)).astype(int)
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
        new_cols['has_deficiencies'] = (df['vehicle_inspection_number_of_deficiencies_found'] > 0).astype(int)
    if 'vehicle_planned_annual_mileage' in df.columns:
        new_cols['log_mileage'] = np.log1p(df['vehicle_planned_annual_mileage'].clip(lower=0))

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def ens_get_feature_and_cat_columns(df):
    exclude = EXCLUDE_COLS | set(DATE_COLS)
    feature_cols = [c for c in df.columns if c not in exclude]
    cat_cols = []
    for col in feature_cols:
        dtype_name = df[col].dtype.name if hasattr(df[col].dtype, 'name') else str(df[col].dtype)
        if col in CATEGORICAL_COLS or col == 'contractor_age_bucket' or col == 'postal_code_urban_category' or dtype_name in ('object', 'str', 'string', 'category'):
            cat_cols.append(col)
    return feature_cols, cat_cols


def add_pca_features(df, feature_cols, cat_cols, n_components=15):
    numeric_cols = [c for c in feature_cols if c not in cat_cols]
    n_components = min(n_components, len(numeric_cols))
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    train_numeric = imputer.fit_transform(df[numeric_cols])
    train_scaled = scaler.fit_transform(train_numeric)
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    pca_names = [f'pca_{i}' for i in range(n_components)]
    for i, name in enumerate(pca_names):
        df[name] = train_pca[:, i]
    return pca_names


def main():
    print("Loading models...")
    
    std_model = CatBoostRegressor()
    std_model.load_model('ensemble_cb_insurer_J.cbm')
    std_feat = std_model.feature_names_
    std_cats = [std_feat[i] for i in std_model.get_cat_feature_indices()]

    print("Loading training data...")
    raw_df = pd.read_parquet(os.path.join(DATA_DIR, 'block1_train.parquet'))
    print(f"  Shape: {raw_df.shape}")

    # --- Full ensemble preprocessing ---
    print("Ensemble preprocessing...")
    ens_df = full_ensemble_preprocess(raw_df.copy())
    
    # Fill missing columns that the models expect based on their extracted features
    missing_cols = {}
    
    # First `std_model` features
    for c in std_cats:
        if c not in ens_df.columns and c not in missing_cols:
            missing_cols[c] = '_MISSING_'
        elif c in ens_df.columns:
            ens_df[c] = ens_df[c].fillna('_MISSING_').astype(str)
            
    for c in std_feat:
        if c not in std_cats and c not in ens_df.columns and c not in missing_cols:
            missing_cols[c] = np.nan

    if missing_cols:
        ens_df = pd.concat([ens_df, pd.DataFrame(missing_cols, index=ens_df.index)], axis=1)
            
    ens_df = prepare_categoricals(ens_df, list(set(std_cats)))
    
    # Ensure all categorical columns are string type to avoid CatBoost errors
    for c in set(std_cats):
        if c in ens_df.columns:
            ens_df[c] = ens_df[c].astype(str)
            
    for c in std_feat:
        if c not in std_cats:
            ens_df[c] = pd.to_numeric(ens_df[c], errors='coerce')

    del raw_df

    # 80/20 temporal split
    split = int(len(ens_df) * 0.8)
    val_ens = ens_df.iloc[split:].reset_index(drop=True)
    del ens_df
    print(f"  Validation: {len(val_ens):,} rows\n")

    target_col = "Insurer_J_price"
    mask_ens = val_ens[target_col].notna()

    # Predict
    std_preds = std_model.predict(val_ens.loc[mask_ens, std_feat])
    std_preds = np.expm1(std_preds)
    std_mae = float(np.mean(np.abs(std_preds - val_ens.loc[mask_ens, target_col].values)))
    print(f"Ensemble CatBoost J MAE = {std_mae:.4f}")
    del std_model

if __name__ == '__main__':
    main()

