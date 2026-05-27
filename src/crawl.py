"""Crawl BS + IS + Quote cho danh sach cty trong symbols.txt."""

import os
import pickle
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from vnstock.api.financial import Finance
from vnstock.api.quote import Quote

BASE_DIR     = Path('data/vnstock')
BS_DIR       = BASE_DIR / 'bs'
IS_DIR       = BASE_DIR / 'is'
QUOTE_DIR    = BASE_DIR / 'quote'
PKL_DIR      = BASE_DIR / 'pkl'
SYMBOLS_FILE = BASE_DIR / 'symbols.txt'

for d in [BS_DIR, IS_DIR, QUOTE_DIR, PKL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SLEEP      = 2
START_DATE = '2018-01-01'
END_DATE   = datetime.now().strftime('%Y-%m-%d')


def load_symbols():
    if not SYMBOLS_FILE.exists():
        print(f'Chua co {SYMBOLS_FILE}. Chay "python src/get_symbols.py" truoc.')
        return []
    with open(SYMBOLS_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def crawl_one(symbol):
    data = {'bs': None, 'is': None, 'quote': None, 'ratio': None}

    try:
        f    = Finance(source='vci', symbol=symbol, period='year')
        bs   = f.balance_sheet(period='year', lang='en')
        data['bs'] = bs
        bs.to_csv(BS_DIR / f'{symbol}.csv', index=False, encoding='utf-8-sig')
        time.sleep(SLEEP)

        is_df = f.income_statement(period='year', lang='en')
        data['is'] = is_df
        is_df.to_csv(IS_DIR / f'{symbol}.csv', index=False, encoding='utf-8-sig')
        time.sleep(SLEEP)

        try:
            ratio_df = f.ratio(period='year', lang='en')
            data['ratio'] = ratio_df
            time.sleep(SLEEP)
        except Exception as e:
            print(f'  Ratio failed: {e}')
            data['ratio'] = None
            time.sleep(SLEEP)
    except Exception as e:
        print(f'  Finance failed: {e}')
        time.sleep(SLEEP)

    try:
        q      = Quote(symbol=symbol, show_log=False)
        prices = q.history(start=START_DATE, end=END_DATE, interval='1D', show_log=False)
        data['quote'] = prices
        prices.to_csv(QUOTE_DIR / f'{symbol}.csv', index=False, encoding='utf-8-sig')
        time.sleep(SLEEP)
    except Exception as e:
        print(f'  Quote failed: {e}')
        time.sleep(SLEEP)

    if any(v is not None for v in data.values()):
        with open(PKL_DIR / f'{symbol}.pkl', 'wb') as f_out:
            pickle.dump(data, f_out)
        return True
    return False


def crawl_batch(symbols, sleep=SLEEP):
    """
    Crawl nhieu cty, dung cho demo/inference (khong luu pkl).
    Tra ve (results, failed) trong do results = {symbol: data_dict}.
    """
    results = {}
    failed  = []

    for i, symbol in enumerate(symbols, 1):
        print(f'[{i}/{len(symbols)}] {symbol}...')
        try:
            f = Finance(source='vci', symbol=symbol, period='year')

            data = {}
            data['ratio'] = f.ratio(period='year', lang='en')
            time.sleep(sleep)
            data['bs']    = f.balance_sheet(period='year', lang='en')
            time.sleep(sleep)
            data['is']    = f.income_statement(period='year', lang='en')
            time.sleep(sleep)

            q = Quote(symbol=symbol, show_log=False)
            data['quote'] = q.history(start=START_DATE, end=END_DATE, interval='1D', show_log=False)
            time.sleep(sleep)

            results[symbol] = data
            print(f'  OK')
        except Exception as e:
            print(f'  FAILED: {e}')
            failed.append(symbol)

    return results, failed


def main():
    if not os.getenv('VNSTOCK_API_KEY'):
        print('Loi: API key chua co trong .env')
        return

    symbols = load_symbols()
    if not symbols:
        return

    print(f'Crawl {len(symbols)} cty')
    print(f'Est. time: ~{len(symbols) * 4 * SLEEP / 60:.1f} phut')

    success = []
    failed  = []

    for i, symbol in enumerate(symbols, 1):
        if (PKL_DIR / f'{symbol}.pkl').exists():
            print(f'[{i}/{len(symbols)}] {symbol}... cached')
            success.append(symbol)
            continue

        print(f'[{i}/{len(symbols)}] {symbol}...')
        if crawl_one(symbol):
            success.append(symbol)
        else:
            failed.append(symbol)

    print(f'\nSuccess: {len(success)}/{len(symbols)}')
    print(f'Failed:  {len(failed)}')


if __name__ == '__main__':
    main()
