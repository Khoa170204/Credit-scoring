"""Lay danh sach ma CK tu vnstock (HOSE + HNX), loai banking/securities/insurance/financial."""

from pathlib import Path
from vnstock.api.listing import Listing

OUTPUT_PATH = Path('data/vnstock/symbols.txt')

EXCLUDE_SYMBOLS = {
    # Banks
    'VCB', 'BID', 'CTG', 'TCB', 'MBB', 'ACB', 'VPB', 'STB',
    'HDB', 'TPB', 'VIB', 'SHB', 'EIB', 'MSB', 'OCB', 'LPB',
    'SSB', 'NAB', 'KLB', 'BAB', 'NVB', 'PGB', 'VBB', 'ABB',
    # Securities
    'HCM', 'SSI', 'VND', 'VCI', 'VIX', 'MBS', 'FTS', 'CTS',
    'BSI', 'ORS', 'AGR', 'TVS', 'APS', 'BVS', 'EVS', 'HBS',
    # Insurance
    'BVH', 'BMI', 'MIG', 'PVI', 'PGI', 'VNR', 'PTI', 'BIC',
    # Cong ty tai chinh tieu dung (schema ngan hang, khong phai cty thuong)
    'EVF',
}


def main():
    print('Lay listing tu vnstock...')
    listing = Listing()

    df = listing.symbols_by_exchange()
    print(f'Total symbols available: {len(df)}')
    print(f'Columns: {df.columns.tolist()}')
    print(f'Sample:')
    print(df.head())

    if 'exchange' in df.columns:
        df = df[df['exchange'].isin(['HSX', 'HOSE', 'HNX'])]
        print(f'\nSau filter HOSE+HNX: {len(df)}')

    if 'type' in df.columns:
        df = df[df['type'] == 'stock']
        print(f'Sau filter type==1 (stock only): {len(df)}')

    symbol_col = 'symbol' if 'symbol' in df.columns else df.columns[0]
    all_symbols = df[symbol_col].dropna().unique().tolist()

    filtered = [s for s in all_symbols if s not in EXCLUDE_SYMBOLS and len(s) == 3]
    print(f'Sau khi loai banking/securities: {len(filtered)} ma')

    selected = sorted(filtered)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(selected))

    print(f'\nDa save {len(selected)} ma vao {OUTPUT_PATH}')
    print(f'First 20: {selected[:20]}')


if __name__ == '__main__':
    main()
