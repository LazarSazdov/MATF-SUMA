import pandas as pd

INSURERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
PRICE_COLS_OLD = [f"Insurer_{ins}_price" for ins in INSURERS]
PRICE_COLS_NEW = [f"Insurer {ins} price" for ins in INSURERS]

for block in ['block2', 'block3']:
    path = f"submissions/submission_{block}.csv"
    df = pd.read_csv(path, sep=';')

    # Rename if needed
    if PRICE_COLS_OLD[0] in df.columns:
        df = df.rename(columns=dict(zip(PRICE_COLS_OLD, PRICE_COLS_NEW)))

    # Round prices to 3 decimals
    df[PRICE_COLS_NEW] = df[PRICE_COLS_NEW].round(3)

    df.to_csv(path, sep=';', index=False)
    print(f"Formatted {path}: {df.shape}")
    print(df.head(2).to_string())
    print()
