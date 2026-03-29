"""
Merge ensemble E/I/K predictions into existing full submissions.
Keeps the old CatBoost predictions for A, B, C, D, F, G, H, J
and replaces E, I, K with the new ensemble predictions.
"""
import pandas as pd
import os

INSURERS_ALL = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
TARGET_INSURERS = ['E', 'I', 'K']

# Column names in the formatted submissions (space-separated)
PRICE_COLS_FORMATTED = [f"Insurer {ins} price" for ins in INSURERS_ALL]
# Column names in raw submissions (underscore-separated)
PRICE_COLS_RAW = [f"Insurer_{ins}_price" for ins in INSURERS_ALL]


def merge_block(base_submission_path, ensemble_eik_path, output_path):
    """
    Read the base full submission, replace E/I/K columns with ensemble predictions.
    """
    base = pd.read_csv(base_submission_path, sep=';')
    eik = pd.read_csv(ensemble_eik_path, sep=';')

    print(f"Base submission: {base.shape}")
    print(f"Ensemble EIK:   {eik.shape}")

    # Detect column naming convention in base
    if PRICE_COLS_FORMATTED[0] in base.columns:
        # Space-separated names (post format_submission.py)
        for ins in TARGET_INSURERS:
            formatted_col = f"Insurer {ins} price"
            raw_col = f"Insurer_{ins}_price"
            base[formatted_col] = eik[raw_col].values
            print(f"  Replaced '{formatted_col}' with ensemble predictions")
    elif PRICE_COLS_RAW[0] in base.columns:
        # Underscore-separated names (raw)
        for ins in TARGET_INSURERS:
            raw_col = f"Insurer_{ins}_price"
            base[raw_col] = eik[raw_col].values
            print(f"  Replaced '{raw_col}' with ensemble predictions")
    else:
        raise ValueError(f"Unexpected column names in {base_submission_path}: {base.columns.tolist()[:5]}")

    # Round to 3 decimals
    price_cols = [c for c in base.columns if c != 'quote_id']
    base[price_cols] = base[price_cols].round(3)

    base.to_csv(output_path, sep=';', index=False)
    print(f"Saved merged submission: {output_path} ({base.shape})")
    return base


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    submissions_dir = os.path.join(base_dir, 'submissions')

    for block in ['block2', 'block3']:
        print(f"\n{'='*60}")
        print(f"MERGING {block.upper()}")
        print(f"{'='*60}")

        base_path = os.path.join(submissions_dir, f'submission_{block}.csv')
        eik_path = os.path.join(submissions_dir, f'ensemble_EKI_{block}.csv')
        output_path = os.path.join(submissions_dir, f'submission_{block}_merged.csv')

        if not os.path.exists(base_path):
            print(f"  WARNING: {base_path} not found, skipping")
            continue
        if not os.path.exists(eik_path):
            print(f"  WARNING: {eik_path} not found, skipping")
            continue

        merge_block(base_path, eik_path, output_path)


if __name__ == '__main__':
    main()
