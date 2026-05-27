"""
Build dataset cuoi tu data da crawl.

Input: data/vnstock/pkl/<symbol>.pkl
Output: data/vnstock/dataset_v2.csv

Logic:
1. Loai 3 cty chung khoan (HCM, SSI, VND)
2. Cho moi (cty, nam):
   - Build label theo rule Tran et al. (3 tieu chi)
   - Tinh 25 features (17 tu BS+IS + 8 tu Quote)
3. Merge thanh 1 DataFrame
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

PKL_DIR = Path('data/vnstock/pkl')
OUTPUT_PATH = Path('data/vnstock/dataset_v2.csv')

EXCLUDE = ['HCM', 'SSI', 'VND', 'VCI', 'VIX', 'MBS', 'FTS', 'CTS',
           'BSI', 'ORS', 'AGR', 'TVS', 'APS', 'BVS', 'EVS', 'HBS',
           'VCB', 'BID', 'CTG', 'TCB', 'MBB', 'ACB', 'VPB', 'STB',
           'HDB', 'TPB', 'VIB', 'SHB', 'EIB', 'MSB', 'OCB', 'LPB',
           'BVH', 'BMI', 'MIG', 'PVI', 'PGI', 'VNR', 'PTI', 'BIC',
           # Securities phat hien them (BS co 200+ items, schema khac)
           'APG', 'DSC', 'DSE']


def is_valid_company(bs, is_df):
    """Kiem tra cty co du data de build features khong."""
    if bs is None or is_df is None:
        return False
    critical_bs = ['Total Assets', "Owner's Equity", 'Liabilities']
    bs_items = set(bs['item_en'].dropna().tolist()) if 'item_en' in bs.columns else set()
    if not all(item in bs_items for item in critical_bs):
        return False
    years = [c for c in bs.columns if str(c).isdigit() and len(str(c)) == 4]
    if len(years) < 5:
        return False
    # Securities thường có 200+ items thay vì ~122 của cty thường
    if len(bs) > 180:
        return False
    return True


def get_value(df, item_name, year):
    """Lay gia tri tu wide-format DataFrame (item x year)."""
    row = df[df['item_en'] == item_name]
    if len(row) == 0:
        return np.nan
    year_str = str(year)
    if year_str not in df.columns:
        return np.nan
    val = row[year_str].values[0]
    if pd.isna(val):
        return np.nan
    return val


def safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def get_year_end_price(quote_df, year):
    """Lay gia dong cua cuoi nam tu quote DataFrame."""
    if quote_df is None or len(quote_df) == 0:
        return np.nan

    quote_df = quote_df.copy()
    quote_df['time'] = pd.to_datetime(quote_df['time'])
    quote_df['year'] = quote_df['time'].dt.year

    year_data = quote_df[quote_df['year'] == year]
    if len(year_data) == 0:
        return np.nan

    return year_data.sort_values('time').iloc[-1]['close']


def get_outstanding_shares(bs_df, year):
    """So co phieu luu hanh = Paid-in capital / menh gia (10,000 VND)."""
    paid_in = get_value(bs_df, 'Paid-in capital', year)
    if pd.isna(paid_in):
        return np.nan
    return paid_in / 10000


def build_label(bs_df, is_df, year):
    """
    Build label theo rule Tran et al.
    Label = 1 neu THOA it nhat 1:
    1. Von chu am
    2. Operating profit / |Interest| < 1 trong 2 nam lien tiep
    3. Operating profit am 3 nam lien tiep
    """
    equity = get_value(bs_df, "Owner's Equity", year)
    if pd.notna(equity) and equity < 0:
        return 1

    crit2 = True
    for y in [year, year - 1]:
        op = get_value(is_df, 'Operating profit/(loss)', y)
        interest = get_value(is_df, 'Interest expenses', y)
        if pd.isna(op) or pd.isna(interest):
            crit2 = False
            break
        ratio = safe_div(op, abs(interest))
        if pd.isna(ratio) or ratio >= 1:
            crit2 = False
            break
    if crit2:
        return 1

    ops = [get_value(is_df, 'Operating profit/(loss)', y) for y in [year, year - 1, year - 2]]
    if all(pd.notna(x) and x < 0 for x in ops):
        return 1

    return 0


def build_features(bs_df, is_df, quote_df, year):
    """Build 25 features cho 1 (cty, nam)."""
    cash           = get_value(bs_df, 'Cash and cash equivalents', year)
    inventory      = get_value(bs_df, 'Inventories, Net', year)
    current_assets = get_value(bs_df, 'CURRENT ASSETS', year)
    current_liab   = get_value(bs_df, 'Current liabilities', year)
    lt_debt        = get_value(bs_df, 'Long-term borrowings', year)
    st_debt        = get_value(bs_df, 'Short-term borrowings', year)
    ap             = get_value(bs_df, 'Trade accounts payable', year)
    total_assets   = get_value(bs_df, 'Total Assets', year)
    total_liab     = get_value(bs_df, 'Liabilities', year)
    equity         = get_value(bs_df, "Owner's Equity", year)
    intangible     = get_value(bs_df, 'Intangible fixed assets', year)
    goodwill       = get_value(bs_df, 'Goodwill', year)

    lt_debt    = lt_debt    if pd.notna(lt_debt)    else 0
    st_debt    = st_debt    if pd.notna(st_debt)    else 0
    ap         = ap         if pd.notna(ap)         else 0
    intangible = intangible if pd.notna(intangible) else 0
    goodwill   = goodwill   if pd.notna(goodwill)   else 0

    net_sales  = get_value(is_df, 'Net sales', year)
    interest   = get_value(is_df, 'Interest expenses', year)
    op_profit  = get_value(is_df, 'Operating profit/(loss)', year)
    eps_diluted = get_value(is_df, 'EPS diluted (VND)', year)
    eps_basic   = get_value(is_df, 'EPS basic (VND)', year)

    eps = eps_diluted if (pd.notna(eps_diluted) and eps_diluted != 0) else eps_basic

    close_price = get_year_end_price(quote_df, year)
    shares      = get_outstanding_shares(bs_df, year)
    # vnstock tra gia CP don vi nghin VND (e.g. 19.81 = 19,810 VND/CP), nhan 1000 de ra VND
    market_cap  = close_price * 1000 * shares if (pd.notna(close_price) and pd.notna(shares)) else np.nan

    f = {}

    # Liquidity
    f['cash_ratio']    = safe_div(cash, current_liab)
    f['quick_ratio']   = safe_div(current_assets - (inventory if pd.notna(inventory) else 0), current_liab)
    f['current_ratio'] = safe_div(current_assets, current_liab)

    # Financial risk
    f['lt_debt_to_equity'] = safe_div(lt_debt, equity)
    f['lt_debt_to_assets'] = safe_div(lt_debt, total_assets)
    f['liab_to_equity']    = safe_div(total_liab, equity)
    f['liab_to_assets']    = safe_div(total_liab, total_assets)
    f['st_debt_to_equity'] = safe_div(st_debt, equity)
    f['st_debt_to_assets'] = safe_div(st_debt, total_assets)

    # Business risk
    f['ap_to_equity']  = safe_div(ap, equity)
    f['ap_to_assets']  = safe_div(ap, total_assets)
    f['assets_to_liab'] = safe_div(total_assets, total_liab)

    # EBITDA coverage: uoc tinh khau hao = chenh lech accumulated dep giua year va year-1
    dep_rows = bs_df[bs_df['item_en'] == 'Accumulated depreciation']
    if len(dep_rows) > 0 and str(year) in bs_df.columns and str(year - 1) in bs_df.columns:
        dep_cur  = dep_rows[str(year)].abs().sum()
        dep_prev = dep_rows[str(year - 1)].abs().sum()
        da = dep_cur - dep_prev if (pd.notna(dep_cur) and pd.notna(dep_prev)) else 0.0
    else:
        da = 0.0

    ebitda = op_profit + da if pd.notna(op_profit) else np.nan
    denom_ebitda = st_debt + (abs(interest) if pd.notna(interest) else 0)
    f['ebitda_coverage'] = safe_div(ebitda, denom_ebitda)

    # Market factors — close_price unit: nghìn VND, EPS unit: VND
    f['pe_basic']   = safe_div(close_price, eps_basic   / 1000) if (pd.notna(close_price) and pd.notna(eps_basic)   and eps_basic   != 0) else np.nan
    f['pe_diluted'] = safe_div(close_price, eps_diluted / 1000) if (pd.notna(close_price) and pd.notna(eps_diluted) and eps_diluted != 0) else np.nan
    f['pb_ratio']   = safe_div(market_cap, equity)
    f['ps_ratio']   = safe_div(market_cap, net_sales)

    tangible_equity = equity - intangible - goodwill if pd.notna(equity) else np.nan
    f['pb_tangible'] = safe_div(market_cap, tangible_equity)

    f['market_cap']    = market_cap
    f['price_to_cfo']  = safe_div(market_cap, op_profit)

    # EV-based
    ev = (market_cap + lt_debt + st_debt - cash) \
         if (pd.notna(market_cap) and pd.notna(cash)) else np.nan
    f['ev']            = ev
    f['ev_to_revenue'] = safe_div(ev, net_sales)
    f['ev_to_ebitda']  = safe_div(ev, ebitda)
    f['ev_to_ebit']    = safe_div(ev, op_profit)
    f['eps_diluted']   = eps

    return f


def process_company(symbol, data=None):
    if data is None:
        pkl_path = PKL_DIR / f'{symbol}.pkl'
        if not pkl_path.exists():
            return []
        with open(pkl_path, 'rb') as fh:
            data = pickle.load(fh)

    bs    = data.get('bs')
    is_df = data.get('is')
    quote = data.get('quote')

    if bs is None or is_df is None:
        return []

    years = [int(c) for c in bs.columns if str(c).isdigit() and len(str(c)) == 4]

    results = []
    for year in sorted(years):
        if year == 2025:
            continue
        features = build_features(bs, is_df, quote, year)
        label    = build_label(bs, is_df, year)
        row = {'Company': symbol, 'year': year}
        row.update(features)
        row['Financial_Distress'] = label
        results.append(row)

    return results


def main():
    pkl_files = sorted(PKL_DIR.glob('*.pkl'))
    if not pkl_files:
        print('Chua co data trong data/vnstock/pkl/')
        return

    all_rows = []
    for pkl in pkl_files:
        symbol = pkl.stem
        if symbol in EXCLUDE:
            print(f'{symbol}: SKIP (bank/securities/insurance)')
            continue

        with open(pkl, 'rb') as fh:
            data = pickle.load(fh)

        if not is_valid_company(data.get('bs'), data.get('is')):
            print(f'{symbol}: SKIP (invalid format or insufficient years)')
            continue

        rows = process_company(symbol, data=data)
        all_rows.extend(rows)
        n_distress = sum(1 for r in rows if r['Financial_Distress'] == 1)
        print(f'{symbol}: {len(rows)} years, {n_distress} distress')

    feature_cols = [
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
    cols = ['Company', 'year'] + feature_cols + ['Financial_Distress']
    df = pd.DataFrame(all_rows)[cols]

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f'DATASET SUMMARY')
    print(f"{'='*60}")
    print(f'Shape: {df.shape}')
    print(f'Companies: {df["Company"].nunique()}')
    print(f'Years: {df["year"].min()} - {df["year"].max()}')
    print(f'Distress: {df["Financial_Distress"].sum()} ({df["Financial_Distress"].mean()*100:.2f}%)')

    print(f'\nMissing rate per feature:')
    missing = df[feature_cols].isnull().mean().sort_values(ascending=False) * 100
    for col, pct in missing.items():
        if pct > 0:
            print(f'  {col}: {pct:.1f}%')

    distress_by_co = df.groupby('Company')['Financial_Distress'].agg(['sum', 'count'])
    distress_by_co = distress_by_co[distress_by_co['sum'] > 0]
    if not distress_by_co.empty:
        print(f'\nDistress by company:')
        print(distress_by_co)

    print(f'\nSaved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
