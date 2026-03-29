# TASK: Reverse-Engineer 11 Insurance Pricing Models

## WHAT YOU ARE DOING

You must build machine learning models that predict car insurance prices. There are 11 competitor insurers (Insurer A through Insurer K). Given a customer profile — their age, car, location, coverage type, deductible — you predict the exact premium each insurer would charge.

---

## HOW YOU ARE SCORED

**Metric: Pooled Mean Absolute Error (MAE)**

For every row in the test set and every insurer:
1. Compute `|your_predicted_price - actual_price|`
2. **Only count rows where the insurer actually provided a quote** (price is not NaN)
3. Sum all absolute errors across all 11 insurers and all valid rows
4. Divide by total count of valid predictions

Lower is better. A pooled MAE of 50 means your predictions are on average $50 away from the real price.

**Three rules that follow from this metric:**

1. **Train with MAE/L1 loss, never MSE/L2.** Insurance premiums are right-skewed — most quotes cluster around a median with a long tail of expensive ones. MSE would make your model chase the $8,000 outliers and ruin accuracy on the thousands of $800 quotes. MAE treats all errors equally, matching the evaluation metric.

2. **Declined quotes don't affect your score.** When an insurer's price is NaN it means they refused to quote. The scorer completely skips that cell. You are never penalized for what you predict on declined rows. Predict a number for every cell — the scorer ignores what it doesn't need.

3. **Every valid prediction is weighted equally in the pool.** An insurer that quotes 90% of customers contributes more predictions to the pool than one that quotes 30%, but each individual error counts the same regardless of which insurer it belongs to.

---

## THE DATA

### Files (all Parquet format — use `pd.read_parquet()`)

| File | Size | Contents |
|------|------|----------|
| `data/block1_train.parquet` | ~88MB | Training data. 4 weeks of quotes. Contains customer profiles + deductibles + **actual prices** |
| `data/block2_test.parquet` | ~23MB | Public test. Week 5. Contains profiles + deductibles. **No prices** — you predict these |
| `data/block3_test.parquet` | ~20MB | Private test. Week 6. Same structure as block 2. **No prices** — you predict these |
| `data/baseline_submission_block2.csv` | ~34MB | Example submission format. Inspect to understand required output structure |

### Temporal Structure

Data is chronologically ordered. Block 1 = weeks 1–4, Block 2 = week 5, Block 3 = week 6. The test sets are the future relative to training. **Never use random cross-validation.** Use a temporal split — train on weeks 1–3, validate on week 4 — to simulate real evaluation.

### What One Row Represents

One specific driver, with a specific car, asking for a specific coverage type. The comparison platform sent this request to all 11 insurers. Each insurer independently decided whether to quote and at what price.

---

## COLUMNS (155 total)

### Exclude From Features
| Column | Reason |
|--------|--------|
| `quote_id` | Unique row ID — no predictive value |
| `vehicle_number_plate` | Unique vehicle ID — no predictive value |
| `Insurer_A_price` ... `Insurer_K_price` | These are the targets. Only present in training data |

### Targets (11 columns) — What You Predict
```
Insurer_A_price, Insurer_B_price, Insurer_C_price, Insurer_D_price,
Insurer_E_price, Insurer_F_price, Insurer_G_price, Insurer_H_price,
Insurer_I_price, Insurer_J_price, Insurer_K_price
```
NaN = insurer declined to quote. Train each insurer's model ONLY on rows where that insurer's price is not NaN.

### Deductible Features (11 columns) — Most Powerful Features
```
Insurer_A_deductible, Insurer_B_deductible, Insurer_C_deductible,
Insurer_D_deductible, Insurer_E_deductible, Insurer_F_deductible,
Insurer_G_deductible, Insurer_H_deductible, Insurer_I_deductible,
Insurer_J_deductible, Insurer_K_deductible
```
Present in **both train and test**. Higher deductible → lower premium. Each insurer's own deductible is the strongest single predictor of that insurer's price. Include all 11 deductible columns as features for every insurer's model.

### Policyholder / Contract Features
| Column | Type | Notes |
|--------|------|-------|
| `coverage` | Categorical | MTPL / Limited Casco / Casco. Determines entire premium band |
| `claim_free_years` | Numeric | No-claims discount. Most powerful pricing factor after deductibles |
| `payment_frequency` | Categorical | Monthly/quarterly/annual. Monthly often has surcharges |
| `contractor_birthdate` | Date | **Must convert to age in years.** U-shaped pricing: young and old pay more |
| `is_driver_owner` | Boolean | Whether policyholder owns the vehicle |
| `usage` | Categorical | Private/commute/business |
| `second_driver_birthdate` | Date | Convert to age. NaN if no second driver — leave NaN, trees handle it |
| `second_driver_claim_free_years` | Numeric | NaN if no second driver |
| `vehicle_ownership_duration` | Numeric | How long current owner has had the vehicle |

### Vehicle Features
| Column | Type | Notes |
|--------|------|-------|
| `vehicle_maker` | Categorical | Manufacturer. High cardinality |
| `vehicle_model` | Categorical | Model name. Very high cardinality |
| `vehicle_fuel_type` | Categorical | Petrol/diesel/electric/hybrid |
| `vehicle_engine_size` | Numeric | Engine displacement |
| `vehicle_power` | Numeric | Power output |
| `vehicle_net_weight` | Numeric | Curb weight |
| `vehicle_gross_weight` | Numeric | Max loaded weight |
| `vehicle_length`, `vehicle_width`, `vehicle_height` | Numeric | Dimensions |
| `vehicle_number_of_cylinders` | Numeric | |
| `vehicle_number_of_doors`, `vehicle_number_of_seats`, `vehicle_number_of_wheels` | Numeric | |
| `vehicle_primary_color` | Categorical | |
| `vehicle_value_new` | Numeric | Original retail price. Determines max total-loss payout |
| `vehicle_net_max_power` | Numeric | Maximum power output |
| `vehicle_net_max_power_electric` | Numeric | EV motor power. NaN for non-EVs |
| `vehicle_nominal_continuous_max_power` | Numeric | Continuous power rating |
| `vehicle_power_to_net_weight_ratio` | Numeric | Already computed. High = sportier/riskier |
| `vehicle_age` | Numeric | **Already in the dataset. Do not recompute** |
| `vehicle_first_registration_date` | Date | First registration anywhere |
| `vehicle_country_first_registration_date` | Date | First domestic registration |
| `vehicle_last_registration_date` | Date | Most recent registration |
| `vehicle_years_since_country_first_registration` | Numeric | Already computed |
| `vehicle_inspection_report_date` | Date | Date of last inspection |
| `vehicle_inspection_expiry_date` | Date | When current inspection expires |
| `vehicle_inspection_number_of_deficiencies_found` | Numeric | Defects found. Proxy for maintenance quality |
| `vehicle_year_of_last_odometer_report` | Numeric | |
| `vehicle_odometer_verdict_code` | Categorical | Odometer integrity verdict |
| `vehicle_planned_annual_mileage` | Numeric | Expected yearly distance |
| `vehicle_is_imported` | Boolean | |
| `vehicle_is_imported_within_last_12_months` | Boolean | |
| `vehicle_can_be_registered` | Boolean | |
| `vehicle_has_open_recall` | Boolean | Active safety recall |
| `vehicle_is_marked_for_export` | Boolean | |
| `vehicle_is_taxi` | Boolean | |

### Geographic / Socio-Demographic Features
| Column | Type | Notes |
|--------|------|-------|
| `postal_code` | Categorical | High cardinality |
| `province` | Categorical | Lower cardinality |
| `municipality` | Categorical | Medium-high cardinality |
| `municipality_crimes_per_1000` | Numeric | Crime rate. Drives theft/vandalism pricing |
| `postal_code_latitude`, `postal_code_longitude` | Numeric | GPS coordinates |
| `postal_code_distance_to_border` | Numeric | |
| `postal_code_urban_category` | Categorical | Urban/suburban/rural |
| `postal_code_population`, `postal_code_households`, `postal_code_houses` | Numeric | |
| `postal_code_average_household_size` | Numeric | |
| `postal_code_*_inhabitants_ratio` (4 columns) | Numeric | Age distribution: 0–15, 25–45, 45–65, 65+ |
| `postal_code_single_person_households_ratio` | Numeric | |
| `postal_code_multi_person_households_without_children_ratio` | Numeric | |
| `postal_code_two_parent_households_ratio` | Numeric | |
| `postal_code_social_benefit_recipients_ratio` | Numeric | Welfare ratio. Socioeconomic proxy |
| `postal_code_address_density` | Numeric | Proxy for traffic congestion |
| `postal_code_average_property_value` | Numeric | Socioeconomic proxy |
| `postal_code_owner_occupied_houses_ratio`, `postal_code_rental_houses_ratio` | Numeric | |
| `postal_code_houses_owned_by_rental_association_ratio` | Numeric | Social housing |
| `postal_code_multi_family_houses_ratio` | Numeric | |
| `postal_code_houses_built_*_ratio` (7 columns) | Numeric | Housing age brackets |
| **Distance-to-nearest features** (20 columns) | Numeric | Distance to nearest train station, hospital, school, supermarket, fire station, pharmacy, library, museum, cinema, etc. |
| **Count-within-radius features** (17 columns) | Numeric | Hospitals within 10km, schools within 3–5km, restaurants within 3km, etc. |

---

## HANDLING MISSING PRICES

When `Insurer_X_price` is NaN, the insurer **refused to quote**. This is a deliberate underwriting decision.

- **Training:** For Insurer A's model, filter to rows where `Insurer_A_price` is not NaN. Train only on real quotes. Each insurer has different decline rates — some quote 95% of customers, others 40%. Each model trains on a different subset and row count. This is correct.
- **Prediction:** Predict a numeric price for every row. The scorer ignores your prediction where the insurer declined. Never predict NaN — it may cause parsing errors.
- **Never impute missing prices** with 0, mean, median, or any value. This corrupts the training distribution.

---

## APPROACH

### Architecture
- 11 independent CatBoost regressors, one per insurer
- Each trains only on rows where that insurer quoted
- Loss function: `loss_function='MAE'`
- Pass categoricals natively via `cat_features` — no encoding
- All 11 deductible columns as features for every model

### Feature Engineering (minimal)
1. `contractor_age` from `contractor_birthdate` — determine reference date from data
2. `second_driver_age` from `second_driver_birthdate` — leave NaN where no second driver
3. `has_second_driver` — binary flag
4. `days_since_inspection` from `vehicle_inspection_report_date`
5. `inspection_days_remaining` from `vehicle_inspection_expiry_date`
6. `years_since_last_registration` from `vehicle_last_registration_date`
7. `vehicle_age` — already exists, do not recompute

Drop raw date columns after extracting numeric features. Apply identical transformations to train and test.

### Categorical Columns
```
coverage, payment_frequency, usage, vehicle_maker, vehicle_model,
vehicle_fuel_type, vehicle_primary_color, vehicle_odometer_verdict_code,
postal_code, province, municipality, postal_code_urban_category
```
Fill NaN with `'_MISSING_'` and cast to `str` before training. CatBoost requires string-typed categoricals.

### Validation
- Temporal split: train on first ~80% of Block 1, validate on last ~20%
- Never random K-fold
- Track per-insurer MAE individually — focus tuning on worst-performing insurers

### Hyperparameters (starting point)
```python
CatBoostRegressor(
    iterations=5000,
    depth=7,
    learning_rate=0.05,
    l2_leaf_reg=5,
    loss_function='MAE',
    eval_metric='MAE',
    early_stopping_rounds=200,
    use_best_model=True,
    random_seed=42,
    verbose=500,
    thread_count=-1,
)
```

### Postprocessing
- Clip predictions: `max(prediction, 1.0)` — premiums cannot be zero or negative
- Match exact format of `baseline_submission_block2.csv`
- Predict a number for every cell — never NaN

---

## MISTAKES THAT DESTROY YOUR SCORE

1. **MSE/L2 loss instead of MAE/L1** — chases outliers, ruins bulk accuracy
2. **Imputing missing prices** — corrupts the training distribution
3. **Random cross-validation** — leaks future data, gives false confidence
4. **Ignoring deductible columns** — these are by far the most predictive features
5. **One-hot encoding high-cardinality categoricals** — explodes memory; use CatBoost native handling
6. **Single multi-output model** — each insurer has different logic; always train separately
7. **Predicting NaN in submissions** — predict numbers everywhere; scorer ignores declined rows
8. **Over-engineering before having a baseline** — get a clean CatBoost submission working first
