import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLS = [
    'cash_ratio', 'quick_ratio', 'current_ratio',
    'lt_debt_to_equity', 'lt_debt_to_assets',
    'liab_to_equity', 'liab_to_assets',
    'st_debt_to_equity', 'st_debt_to_assets',
    'ap_to_equity', 'ap_to_assets',
    'assets_to_liab', 'ebitda_coverage',
    'pe_basic', 'pe_diluted', 'pb_ratio', 'ps_ratio', 'pb_tangible',
    'market_cap', 'price_to_cfo',
    'ev', 'ev_to_revenue', 'ev_to_ebitda', 'ev_to_ebit',
    'eps_diluted',
]
TARGET_COL = "Financial_Distress"
GROUP_COL = "Company"


def load_data(path):
    df = pd.read_csv(path, encoding="latin-1")
    return df


def split_data(df, test_size=0.3, random_state=42):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    groups = df[GROUP_COL]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    # Impute NaN bang median cua train (fit tren train, apply ca train+test)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test  = X_test.fillna(train_medians)

    return X_train, X_test, y_train, y_test
