import pandas as pd
import lightgbm as lgb
import numpy as np

# 1. Load the Data
print("Loading data...")
# Replace with your actual parquet file names
train_df = pd.read_parquet('data/block1_train.parquet') 
test_df = pd.read_parquet('data/block2_test.parquet')

# 2. Identify our columns
# Find all the target price columns (Insurer_A_price, etc.)
insurers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
target_cols = [f'Insurer_{i}_price' for i in insurers]

# Features are everything else EXCEPT the prices and the quote_id
features = [col for col in train_df.columns if col not in target_cols and col != 'quote_id']

# 3. Quick Preprocessing: Let LightGBM handle categories automatically
print("Preprocessing features...")
for col in features:
    # If a column is text (object), convert it to a pandas 'category' type
    # LightGBM loves this and will process it without needing One-Hot Encoding
    if pd.api.types.is_string_dtype(train_df[col]) or pd.api.types.is_object_dtype(train_df[col]) or train_df[col].dtype.name == 'category':
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

# 4. Prepare the Submission File
# Initialize it with the quote_ids from the test set
submission = pd.DataFrame({'quote_id': test_df['quote_id']})

# 5. Train 11 Models (One for each insurer)
print("Training models...")

# Basic LightGBM parameters (we keep it simple for now)
lgb_params = {
    'objective': 'regression_l1', # MAE is L1 loss! This optimizes directly for your hackathon metric
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'random_state': 42
}

for insurer in insurers:
    target = f'Insurer_{insurer}_price'
    print(f"--- Training model for {target} ---")
    
    # Filter the training data: ONLY keep rows where this specific insurer gave a quote
    train_subset = train_df[train_df[target].notnull()]
    
    X_train = train_subset[features]
    y_train = train_subset[target]
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train the model (using 100 trees for a quick baseline)
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=100
    )
    
    # Predict on the ENTIRE test set
    # (The hackathon scoring ignores the ones they didn't actually quote, 
    # but the submission requires a number in every row)
    predictions = model.predict(test_df[features])
    
    # Add predictions to our submission dataframe
    submission[target] = predictions

# 6. Save the Submission File
print("Saving submission file...")
# The hackathon requires ";" as separator and "." for decimals
submission.to_csv('data/baseline_submission_block2.csv', sep=';', decimal='.', index=False)
print("Done! You are ready to upload baseline_submission_block2.csv to the leaderboard.")