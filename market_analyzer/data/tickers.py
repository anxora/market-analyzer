"""
Ticker list utilities - Get complete S&P 500 and NASDAQ listings.
"""

import pandas as pd
import requests
from typing import List
from functools import lru_cache


@lru_cache(maxsize=1)
def get_sp500_tickers() -> List[str]:
    """
    Get all S&P 500 tickers from Wikipedia.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return get_sp500_backup()


@lru_cache(maxsize=1)
def get_nasdaq100_tickers() -> List[str]:
    """
    Get NASDAQ-100 tickers from Wikipedia.
    """
    try:
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        tables = pd.read_html(url)
        # Find the table with tickers
        for table in tables:
            if 'Ticker' in table.columns or 'Symbol' in table.columns:
                col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                tickers = table[col].str.replace('.', '-', regex=False).tolist()
                return tickers
        return get_nasdaq100_backup()
    except Exception as e:
        print(f"Error fetching NASDAQ-100 list: {e}")
        return get_nasdaq100_backup()


def get_sp500_backup() -> List[str]:
    """Backup hardcoded S&P 500 list."""
    return [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'BRK-B', 'LLY', 'AVGO',
        'JPM', 'TSLA', 'V', 'UNH', 'XOM', 'MA', 'JNJ', 'COST', 'HD', 'PG', 'WMT',
        'ABBV', 'NFLX', 'CRM', 'BAC', 'CVX', 'MRK', 'KO', 'PEP', 'AMD', 'TMO',
        'ADBE', 'ACN', 'LIN', 'MCD', 'CSCO', 'ABT', 'WFC', 'DHR', 'ORCL', 'PM',
        'TXN', 'NEE', 'INTU', 'DIS', 'ISRG', 'VZ', 'AMGN', 'CAT', 'IBM', 'NOW',
        'QCOM', 'CMCSA', 'PFE', 'SPGI', 'GE', 'AMAT', 'HON', 'UNP', 'RTX', 'LOW',
        'BKNG', 'T', 'BLK', 'GS', 'ELV', 'SYK', 'TJX', 'ADP', 'SBUX', 'MDLZ',
        'PLD', 'VRTX', 'MMC', 'GILD', 'ADI', 'SCHW', 'DE', 'REGN', 'CB', 'LRCX',
        'ETN', 'CI', 'BMY', 'ZTS', 'SO', 'MO', 'PANW', 'CME', 'TMUS', 'BSX',
        'FI', 'DUK', 'KLAC', 'SNPS', 'CDNS', 'SHW', 'ICE', 'CL', 'MCK', 'PNC',
        'EQIX', 'ITW', 'MU', 'MDVP', 'PYPL', 'APD', 'MSI', 'AON', 'EMR', 'WELL',
        'NSC', 'APH', 'CTAS', 'CCI', 'EOG', 'MAR', 'USB', 'ORLY', 'MCO', 'NXPI',
        'ABNB', 'FCX', 'TT', 'CSX', 'PCAR', 'GM', 'SLB', 'AJG', 'OXY', 'AZO',
        'WM', 'TGT', 'PSX', 'DHI', 'HLT', 'AFL', 'FTNT', 'SRE', 'MCHP', 'MPC',
        'ROST', 'PH', 'AEP', 'CARR', 'COR', 'O', 'TRV', 'KMB', 'SPG', 'NEM',
        'KDP', 'F', 'LHX', 'PSA', 'AIG', 'JCI', 'ALL', 'VLO', 'CPRT', 'D',
        'BK', 'NUE', 'HES', 'FDX', 'PAYX', 'DXCM', 'FAST', 'ODFL', 'CMG', 'EW',
        'KMI', 'MSCI', 'IQV', 'OKE', 'IDXX', 'TEL', 'EXC', 'HUM', 'PRU', 'KR',
        'ACGL', 'RSG', 'CTVA', 'RCL', 'GWW', 'VRSK', 'A', 'LEN', 'PCG', 'MNST',
        'IR', 'DOW', 'GEHC', 'AME', 'DAL', 'EA', 'VMC', 'EXR', 'YUM', 'MLM',
        'ED', 'XYL', 'HPQ', 'DD', 'GIS', 'KHC', 'BKR', 'ON', 'WAB', 'FANG',
        'CBRE', 'WEC', 'HPE', 'CDW', 'EIX', 'CHTR', 'ROK', 'HSY', 'AVB', 'XEL',
        'CAH', 'VICI', 'NVR', 'PWR', 'MTD', 'RMD', 'PPG', 'DVN', 'IT', 'WBD',
        'ANSS', 'UAL', 'BIIB', 'KEYS', 'STZ', 'EBAY', 'DLTR', 'TSCO', 'WST', 'EQR',
        'BRO', 'TROW', 'DOV', 'ETR', 'PEG', 'HIG', 'FITB', 'AWK', 'HAL', 'CSGP',
        'GPN', 'MTB', 'EFX', 'DECK', 'LYB', 'CHD', 'MPWR', 'IFF', 'FTV', 'VTR',
        'HBAN', 'HUBB', 'PPL', 'BALL', 'WTW', 'DTE', 'PHM', 'TYL', 'WY', 'WRB',
        'CBOE', 'RJF', 'STE', 'RF', 'CLX', 'AEE', 'NTAP', 'SBAC', 'ARE', 'ES',
        'COO', 'DRI', 'ULTA', 'HOLX', 'CNC', 'LH', 'TDY', 'INVH', 'K', 'MOH',
        'IRM', 'LVS', 'EL', 'SYY', 'WAT', 'OMC', 'EXPD', 'ZBH', 'PKG', 'STLD',
        'CNP', 'FE', 'J', 'TSN', 'IP', 'BLDR', 'MAA', 'FSLR', 'AXON', 'NTRS',
        'WDC', 'CF', 'LUV', 'L', 'BBY', 'SNA', 'TER', 'MKC', 'EG', 'KEY',
        'CINF', 'NI', 'GPC', 'DGX', 'RVTY', 'VLTO', 'DG', 'CAG', 'JBHT', 'CFG',
        'TRGP', 'TTWO', 'POOL', 'AKAM', 'LNT', 'LDOS', 'AES', 'CMS', 'AMCR', 'BAX',
        'PNR', 'EVRG', 'CPT', 'MAS', 'HRL', 'BG', 'VTRS', 'SWK', 'NDSN', 'TAP',
        'ALB', 'EMN', 'CE', 'UDR', 'PFG', 'KIM', 'JKHY', 'HST', 'REG', 'ATO',
        'SWKS', 'TXT', 'WRK', 'GL', 'IPG', 'FMC', 'AOS', 'CHRW', 'CRL', 'IEX',
        'APA', 'ALLE', 'HII', 'ROL', 'TECH', 'PNW', 'BXP', 'PODD', 'INCY', 'PAYC',
        'CTLT', 'NWS', 'NWSA', 'CDAY', 'FFIV', 'MKTX', 'WYNN', 'CPB', 'BEN', 'HSIC',
        'LKQ', 'AIZ', 'QRVO', 'BBWI', 'BWA', 'MOS', 'HAS', 'MGM', 'DVA', 'CZR',
        'ETSY', 'PARA', 'GNRC', 'AAL', 'MTCH', 'BIO', 'FRT', 'EPAM', 'RL', 'NCLH',
        'WHR', 'XRAY', 'DAY', 'RHI', 'MHK', 'ENPH', 'VFC', 'SEE', 'CTSH', 'TPR',
    ]


def get_nasdaq100_backup() -> List[str]:
    """Backup hardcoded NASDAQ-100 list."""
    return [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'AVGO', 'GOOGL', 'GOOG', 'COST', 'TSLA',
        'NFLX', 'AMD', 'ADBE', 'PEP', 'CSCO', 'TMUS', 'INTC', 'CMCSA', 'INTU', 'QCOM',
        'TXN', 'AMAT', 'AMGN', 'ISRG', 'HON', 'BKNG', 'VRTX', 'SBUX', 'ADP', 'GILD',
        'MDLZ', 'ADI', 'REGN', 'LRCX', 'PANW', 'KLAC', 'SNPS', 'CDNS', 'PYPL', 'MAR',
        'ORLY', 'ASML', 'MRVL', 'ABNB', 'CTAS', 'NXPI', 'PCAR', 'MCHP', 'CPRT', 'FTNT',
        'AEP', 'ROST', 'KDP', 'PAYX', 'DXCM', 'ODFL', 'MNST', 'KHC', 'LULU', 'EA',
        'EXC', 'FAST', 'IDXX', 'VRSK', 'CHTR', 'CSGP', 'XEL', 'GEHC', 'CDW', 'CTSH',
        'TTD', 'ON', 'ANSS', 'BIIB', 'EBAY', 'ILMN', 'DLTR', 'WBD', 'GFS', 'ZS',
        'FANG', 'CEG', 'DDOG', 'TEAM', 'CRWD', 'DASH', 'WDAY', 'ALGN', 'MRNA', 'ZM',
        'SIRI', 'LCID', 'RIVN', 'OKTA', 'DOCU', 'SPLK', 'MDB', 'NET', 'ROKU', 'COIN',
    ]


def get_all_us_tickers() -> List[str]:
    """
    Get combined unique list of S&P 500 and NASDAQ-100 tickers.
    """
    sp500 = get_sp500_tickers()
    nasdaq = get_nasdaq100_tickers()

    # Combine and deduplicate
    all_tickers = list(set(sp500 + nasdaq))

    return sorted(all_tickers)


def get_growth_stocks() -> List[str]:
    """
    Get a curated list of high-growth stocks (typically better for CAN SLIM).
    """
    return [
        # AI & Semiconductors
        'NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'SMCI', 'AMAT', 'LRCX', 'KLAC', 'ASML',
        # Software & Cloud
        'MSFT', 'CRM', 'NOW', 'ADBE', 'INTU', 'PANW', 'CRWD', 'ZS', 'DDOG', 'NET',
        'SNOW', 'MDB', 'TEAM', 'WDAY', 'OKTA', 'SPLK', 'PLTR', 'PATH', 'DOCU', 'APP',
        # E-commerce & Digital
        'AMZN', 'SHOP', 'MELI', 'BKNG', 'ABNB', 'DASH', 'UBER', 'LYFT', 'COIN', 'SQ',
        # Social & Entertainment
        'META', 'GOOGL', 'NFLX', 'DIS', 'RBLX', 'TTWO', 'EA', 'ROKU', 'SPOT', 'TTD',
        # Healthcare & Biotech
        'LLY', 'NVO', 'ISRG', 'DXCM', 'VEEV', 'ALGN', 'IDXX', 'MRNA', 'VRTX', 'REGN',
        # EV & Clean Energy
        'TSLA', 'RIVN', 'LCID', 'ENPH', 'SEDG', 'FSLR', 'RUN', 'PLUG', 'CHPT', 'QS',
        # Fintech
        'V', 'MA', 'PYPL', 'AFRM', 'HOOD', 'SOFI', 'UPST', 'BILL', 'TOST', 'NU',
        # Consumer
        'COST', 'CMG', 'SBUX', 'LULU', 'NKE', 'DECK', 'ULTA', 'ORLY', 'AZO', 'MNST',
        'CELH', 'DUOL', 'PTON', 'CHWY', 'ETSY',
        # Industrial Tech
        'CAT', 'DE', 'AXON', 'TT', 'ETN', 'GNRC', 'PWR', 'BLDR',
    ]


if __name__ == "__main__":
    print("S&P 500 Tickers:", len(get_sp500_tickers()))
    print("NASDAQ-100 Tickers:", len(get_nasdaq100_tickers()))
    print("All US Tickers:", len(get_all_us_tickers()))
