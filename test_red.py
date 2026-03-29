import pandas as pd

# Load the Parquet file into a Pandas DataFrame
# Replace 'block_1_training_data.parquet' with your actual file name
df1 = pd.read_parquet('data/block1_train.parquet', engine='pyarrow')
df2 = pd.read_parquet('data/block2_test.parquet', engine='pyarrow')
df3 = pd.read_parquet('data/block3_test.parquet', engine='pyarrow')

# Check the columns and data types (useful for your features like vehicle specs)
print(df1.info())
print(df2.info())
print(df3.info())

df1.head(50000).to_csv('data/block1_train_head10.csv')
df2.head(50000).to_csv('data/block2_test_head10.csv')
df3.head(50000).to_csv('data/block3_test_head10.csv')