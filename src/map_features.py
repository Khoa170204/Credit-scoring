import numpy as np
import pandas as pd


def _lookup_ratio(df, item_id):
    """
    Ratio DataFrame is long format: rows = metrics, columns = years (all named '2018'
    due to vnstock duplicate-column issue). Use iloc to avoid duplicate-column ambiguity.
    Returns the last non-zero, non-nan value (most recent Q4 available).
    """
    if df.empty or 'item_id' not in df.columns:
        return np.nan
    mask = df['item_id'] == item_id
    if not mask.any():
        return np.nan
    row_idx = df[mask].index[0]
    # Metadata occupies first 3 cols (item, item_en, item_id); data starts at col 3
    for col_pos in range(len(df.columns) - 1, 2, -1):
        try:
            val = float(df.iloc[row_idx, col_pos])
            if not (np.isnan(val) or val == 0):
                return val
        except (ValueError, TypeError):
            continue
    return np.nan


def _lookup_annual(df, item_id, year):
    """
    Balance sheet / income statement: rows = items, columns = years (2018, 2019, ...).
    Falls back to most recent available year if requested year is missing.
    """
    if df.empty or 'item_id' not in df.columns:
        return np.nan
    mask = df['item_id'] == item_id
    if not mask.any():
        return np.nan
    row = df[mask].iloc[0]
    year_str = str(year)
    if year_str in df.columns:
        try:
            val = float(row[year_str])
            return val if not np.isnan(val) else np.nan
        except (ValueError, TypeError):
            pass
    # fallback: most recent year with a non-nan value
    avail = sorted([c for c in df.columns if str(c).isdigit()], reverse=True)
    for y in avail:
        try:
            val = float(row[y])
            if not np.isnan(val):
                return val
        except (ValueError, TypeError):
            continue
    return np.nan


def _div(a, b):
    if any(pd.isna(x) for x in [a, b]) or b == 0:
        return np.nan
    return a / b


def map_to_x_features(ratio_df, bs_df, is_df, year=2022):
    r = lambda iid: _lookup_ratio(ratio_df, iid)
    b = lambda iid: _lookup_annual(bs_df, iid, year)
    s = lambda iid: _lookup_annual(is_df, iid, year)

    # --- liquidity (from ratio, TTM) ---
    cash_ratio    = r('cash_ratio')
    quick_ratio   = r('quick_ratio')
    current_ratio = r('current_ratio')

    # --- balance sheet items ---
    LT_debt    = b('long_term_borrowings')
    ST_debt    = b('short_term_borrowings')
    AP         = b('trade_accounts_payable')
    Total_A    = b('total_assets')
    Equity     = b('owners_equity')
    Cash       = b('cash_and_cash_equivalents')
    Intangible = b('intangible_fixed_assets')
    Goodwill   = b('goodwill')
    Intangible = Intangible if not pd.isna(Intangible) else 0
    Goodwill   = Goodwill   if not pd.isna(Goodwill)   else 0

    # --- leverage (computed from balance sheet) ---
    total_liab        = Total_A - Equity if not any(pd.isna(v) for v in [Total_A, Equity]) else np.nan
    lt_debt_to_equity = _div(LT_debt, Equity)
    lt_debt_to_assets = _div(LT_debt, Total_A)
    liab_to_equity    = _div(total_liab, Equity)
    liab_to_assets    = _div(total_liab, Total_A)
    st_debt_to_equity = _div(ST_debt, Equity)
    st_debt_to_assets = _div(ST_debt, Total_A)
    ap_to_equity      = _div(AP, Equity)
    ap_to_assets      = _div(AP, Total_A)
    assets_to_liab    = _div(Total_A, total_liab)

    # --- coverage ---
    EBITDA        = r('ebitda')
    interest_raw  = s('interest_expenses')
    Interest      = abs(interest_raw) if not pd.isna(interest_raw) else np.nan
    denom_cov     = (ST_debt + Interest) if not any(pd.isna(v) for v in [ST_debt, Interest]) else np.nan
    ebitda_coverage = _div(EBITDA, denom_cov)

    # --- valuation from ratio (TTM, may not be exactly year 2022) ---
    pe_basic     = r('pe_ratio')
    pe_diluted   = pe_basic                  # vnstock ratio API khong co item_id diluted PE rieng
    pb_ratio     = r('pb_ratio')
    ps_ratio     = r('ps_ratio')
    market_cap   = r('market_cap')           # VND
    tangible_eq  = Equity - Intangible - Goodwill if not pd.isna(Equity) else np.nan
    pb_tangible  = _div(market_cap, tangible_eq)
    price_to_cfo = r('price_to_cash_flow')
    ev_to_ebitda = r('ev_to_ebitda')

    # --- EV-based ---
    EBIT    = r('ebit')
    Revenue = s('net_sales')

    eps = s('eps_basic_vnd')
    if pd.isna(eps) or eps == 0:
        eps = s('eps_diluted_vnd')
    eps_diluted = eps

    ev = (market_cap + LT_debt + ST_debt - Cash) \
          if not any(pd.isna(v) for v in [market_cap, LT_debt, ST_debt, Cash]) else np.nan
    ev_to_revenue = _div(ev, Revenue)
    ev_to_ebit    = _div(ev, EBIT)

    return {
        'cash_ratio': cash_ratio, 'quick_ratio': quick_ratio, 'current_ratio': current_ratio,
        'lt_debt_to_equity': lt_debt_to_equity, 'lt_debt_to_assets': lt_debt_to_assets,
        'liab_to_equity': liab_to_equity, 'liab_to_assets': liab_to_assets,
        'st_debt_to_equity': st_debt_to_equity, 'st_debt_to_assets': st_debt_to_assets,
        'ap_to_equity': ap_to_equity, 'ap_to_assets': ap_to_assets,
        'assets_to_liab': assets_to_liab, 'ebitda_coverage': ebitda_coverage,
        'pe_basic': pe_basic, 'pe_diluted': pe_diluted,
        'pb_ratio': pb_ratio, 'ps_ratio': ps_ratio, 'pb_tangible': pb_tangible,
        'market_cap': market_cap, 'price_to_cfo': price_to_cfo,
        'ev': ev, 'ev_to_revenue': ev_to_revenue,
        'ev_to_ebitda': ev_to_ebitda, 'ev_to_ebit': ev_to_ebit,
        'eps_diluted': eps_diluted,
    }


def build_features_df(crawl_results, year=2022):
    rows = []
    for symbol, data in crawl_results.items():
        ratio_df = data.get('ratio', pd.DataFrame())
        bs_df    = data.get('bs',    pd.DataFrame())
        is_df    = data.get('is',    pd.DataFrame())

        feats = map_to_x_features(ratio_df, bs_df, is_df, year=year)
        feats['symbol'] = symbol
        rows.append(feats)

    cols = ['symbol'] + [
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
    return pd.DataFrame(rows, columns=cols)
