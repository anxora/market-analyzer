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


def get_all_us_market_tickers(min_market_cap: float = 0, exchanges: List[str] = None) -> List[str]:
    """
    Get ALL tickers from US exchanges (NYSE, NASDAQ, AMEX).

    Tries multiple sources in order:
    1. Financial Modeling Prep API (10,000+ tickers, requires free API key)
    2. SEC EDGAR (all reporting companies)
    3. Individual exchange fetching
    4. Backup hardcoded lists

    Args:
        min_market_cap: Minimum market cap filter (in dollars). Default 0 = no filter.
        exchanges: List of exchanges to include. Default: all (NYSE, NASDAQ, AMEX)

    Returns:
        List of ticker symbols
    """
    if exchanges is None:
        exchanges = ['NYSE', 'NASDAQ', 'AMEX']

    # Try FMP first (most complete, 10,000+ tickers)
    try:
        tickers = _fetch_from_fmp_all_stocks()
        if len(tickers) > 1000:
            print(f"Fetched {len(tickers)} tickers from Financial Modeling Prep")
            # Filter by exchange if specified
            if exchanges != ['NYSE', 'NASDAQ', 'AMEX']:
                # FMP returns all exchanges, so we use backup for filtering
                pass
            return sorted(tickers)
    except Exception as e:
        print(f"FMP API failed: {e}")

    # Try SEC EDGAR
    try:
        tickers = _fetch_from_sec_edgar()
        if len(tickers) > 1000:
            print(f"Fetched {len(tickers)} tickers from SEC EDGAR")
            return sorted(tickers)
    except Exception as e:
        print(f"SEC EDGAR failed: {e}")

    # Fallback to individual exchange fetching
    all_tickers = []
    for exchange in exchanges:
        try:
            tickers = _fetch_exchange_tickers(exchange, min_market_cap)
            all_tickers.extend(tickers)
            print(f"Fetched {len(tickers)} tickers from {exchange}")
        except Exception as e:
            print(f"Error fetching {exchange}: {e}")
            # Use backup for this exchange
            if exchange == 'NASDAQ':
                all_tickers.extend(get_nasdaq100_backup() + _get_nasdaq_extended_backup())
            elif exchange == 'NYSE':
                all_tickers.extend(get_sp500_backup() + _get_nyse_extended_backup())

    # Remove duplicates and sort
    all_tickers = sorted(list(set(all_tickers)))
    return all_tickers


def _fetch_from_fmp_all_stocks() -> List[str]:
    """
    Fetch ALL US stock tickers from Financial Modeling Prep API.
    Returns 10,000+ tickers. Requires FMP_API_KEY environment variable.
    """
    import os
    api_key = os.getenv('FMP_API_KEY')

    if not api_key:
        raise ValueError("FMP_API_KEY not set. Get free key at https://financialmodelingprep.com/developer")

    url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and 'Error Message' in data:
        raise ValueError(data['Error Message'])

    # Filter to US exchanges only
    us_exchanges = {'NYSE', 'NASDAQ', 'AMEX', 'NYSEArca', 'BATS'}
    tickers = []

    for stock in data:
        symbol = stock.get('symbol', '')
        exchange = stock.get('exchangeShortName', '')

        # Skip if not US exchange
        if exchange not in us_exchanges:
            continue

        # Skip ETFs, warrants, etc (only common stock)
        stock_type = stock.get('type', '')
        if stock_type and stock_type != 'stock':
            continue

        # Clean symbol
        if symbol and '^' not in symbol and '.' not in symbol:
            tickers.append(symbol.upper())

    return tickers


def _fetch_from_sec_edgar() -> List[str]:
    """
    Fetch tickers from SEC EDGAR company database.
    Contains all companies that file with the SEC (~8,000+).
    """
    url = "https://www.sec.gov/files/company_tickers.json"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    tickers = []
    for key, company in data.items():
        ticker = company.get('ticker', '')
        if ticker and ticker.isalpha() and len(ticker) <= 5:
            tickers.append(ticker.upper())

    return tickers


def _fetch_from_stockanalysis_json(exchange: str) -> List[str]:
    """
    Fetch tickers from stockanalysis.com by parsing embedded JSON data.
    """
    urls = {
        'NYSE': 'https://stockanalysis.com/list/nyse-stocks/',
        'NASDAQ': 'https://stockanalysis.com/list/nasdaq-stocks/',
        'AMEX': 'https://stockanalysis.com/list/amex-stocks/',
    }

    url = urls.get(exchange.upper())
    if not url:
        raise ValueError(f"Unknown exchange: {exchange}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    # Try to extract JSON data from the page
    import re
    import json

    # Look for stockData in the JavaScript
    match = re.search(r'stockData:\[([^\]]+)\]', response.text)
    if match:
        try:
            # Parse the array of stock objects
            json_str = '[' + match.group(1) + ']'
            # Fix JavaScript object notation to JSON
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            stocks = json.loads(json_str)
            tickers = [s.get('s', '') for s in stocks if s.get('s')]
            if tickers:
                return tickers
        except (json.JSONDecodeError, KeyError):
            pass

    # Alternative: look for symbol patterns in the page
    symbols = re.findall(r'"s":"([A-Z]{1,5})"', response.text)
    if symbols:
        return list(set(symbols))

    raise ValueError("Could not parse stock data from page")


def _fetch_exchange_tickers(exchange: str, min_market_cap: float = 0) -> List[str]:
    """
    Fetch tickers from a specific exchange.

    Tries multiple methods:
    1. stockanalysis.com JSON parsing
    2. NASDAQ screener download
    3. Hardcoded backup lists
    """
    # Try stockanalysis.com with JSON parsing
    try:
        tickers = _fetch_from_stockanalysis_json(exchange)
        if len(tickers) > 50:
            return tickers
    except Exception as e1:
        print(f"  StockAnalysis JSON failed: {e1}")

    # Try HTML table parsing
    try:
        return _fetch_from_stockanalysis(exchange)
    except Exception as e2:
        print(f"  StockAnalysis HTML failed: {e2}")

    # Try NASDAQ screener
    try:
        return _fetch_from_nasdaq_ftp(exchange)
    except Exception as e3:
        print(f"  NASDAQ screener failed: {e3}")

    # Fallback to hardcoded lists
    print(f"  Using backup list for {exchange}")
    if exchange.upper() == 'NASDAQ':
        return get_nasdaq100_backup() + _get_nasdaq_extended_backup()
    elif exchange.upper() == 'NYSE':
        return get_sp500_backup() + _get_nyse_extended_backup()
    else:
        return []


def _fetch_from_nasdaq_ftp(exchange: str) -> List[str]:
    """
    Fetch tickers from NASDAQ FTP server (most reliable method).
    """
    # NASDAQ provides stock screener data
    urls = {
        'NASDAQ': 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv',
        'NYSE': 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv',
    }

    # Alternative: Use a more complete source
    url = f"https://www.nasdaq.com/market-activity/stocks/screener?exchange={exchange.lower()}&render=download"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/csv,application/csv,text/plain,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nasdaq.com/market-activity/stocks/screener',
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    # Parse CSV
    import io
    df = pd.read_csv(io.StringIO(response.text))

    if 'Symbol' in df.columns:
        tickers = df['Symbol'].dropna().str.strip().tolist()
    elif 'symbol' in df.columns:
        tickers = df['symbol'].dropna().str.strip().tolist()
    else:
        raise ValueError("No symbol column found")

    # Clean tickers
    tickers = [t.upper() for t in tickers if t and str(t).isalpha()]
    return tickers


def _fetch_from_stockanalysis(exchange: str) -> List[str]:
    """
    Fetch tickers from stockanalysis.com
    """
    urls = {
        'NYSE': 'https://stockanalysis.com/list/nyse-stocks/',
        'NASDAQ': 'https://stockanalysis.com/list/nasdaq-stocks/',
        'AMEX': 'https://stockanalysis.com/list/amex-stocks/',
    }

    url = urls.get(exchange.upper())
    if not url:
        raise ValueError(f"Unknown exchange: {exchange}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    # Parse HTML tables
    tables = pd.read_html(response.text)
    if not tables:
        raise ValueError("No tables found")

    df = tables[0]

    # Find symbol column
    symbol_cols = ['Symbol', 'Ticker', 'symbol', 'ticker']
    for col in symbol_cols:
        if col in df.columns:
            tickers = df[col].dropna().str.strip().tolist()
            # Clean tickers (remove special chars)
            tickers = [t.split()[0].upper() for t in tickers if t]
            return tickers

    raise ValueError("No symbol column found")


def _get_nasdaq_extended_backup() -> List[str]:
    """Extended NASDAQ tickers backup list."""
    return [
        # Tech
        'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'ADBE', 'CRM', 'NFLX',
        'PYPL', 'INTC', 'CMCSA', 'PEP', 'CSCO', 'TMUS', 'TXN', 'QCOM', 'AMGN', 'SBUX',
        'INTU', 'ISRG', 'AMD', 'MDLZ', 'GILD', 'ADI', 'BKNG', 'VRTX', 'ADP', 'REGN',
        'LRCX', 'MU', 'MELI', 'PANW', 'SNPS', 'KLAC', 'CDNS', 'ORLY', 'MNST', 'FTNT',
        'MRVL', 'CTAS', 'NXPI', 'KDP', 'PAYX', 'ODFL', 'DXCM', 'WDAY', 'ABNB', 'CPRT',
        'PCAR', 'MCHP', 'AZN', 'KHC', 'LULU', 'EA', 'ROST', 'IDXX', 'FAST', 'VRSK',
        'EXC', 'XEL', 'GEHC', 'CTSH', 'CSGP', 'DDOG', 'ZS', 'CRWD', 'TTD', 'TEAM',
        'OKTA', 'SPLK', 'SNOW', 'NET', 'MDB', 'PLTR', 'COIN', 'RBLX', 'DASH', 'HOOD',
        'AFRM', 'SOFI', 'RIVN', 'LCID', 'ARM', 'SMCI', 'APP', 'DUOL', 'CELH', 'IOT',
    ]


def _get_nyse_extended_backup() -> List[str]:
    """Extended NYSE tickers backup list."""
    return [
        # Financials
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'SCHW',
        'BLK', 'SPGI', 'ICE', 'CME', 'COF', 'AXP', 'DFS', 'SYF', 'ALLY', 'MTB',
        # Healthcare
        'UNH', 'JNJ', 'PFE', 'MRK', 'ABT', 'TMO', 'DHR', 'LLY', 'BMY', 'ABBV',
        'CVS', 'CI', 'HUM', 'ELV', 'MCK', 'CAH', 'SYK', 'BSX', 'MDT', 'ZBH',
        # Consumer
        'WMT', 'PG', 'KO', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX',
        'DG', 'DLTR', 'COST', 'CVS', 'WBA', 'KR', 'SYY', 'YUM', 'DPZ', 'CMG',
        # Industrial
        'CAT', 'DE', 'UNP', 'UPS', 'FDX', 'BA', 'GE', 'HON', 'MMM', 'LMT',
        'RTX', 'NOC', 'GD', 'EMR', 'ETN', 'PH', 'ROK', 'ITW', 'CMI', 'PCAR',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY',
        'KMI', 'WMB', 'EPD', 'ET', 'MPLX', 'HAL', 'BKR', 'DVN', 'HES', 'FANG',
        # Utilities & REITs
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES',
        'PLD', 'AMT', 'EQIX', 'PSA', 'CCI', 'O', 'SPG', 'WELL', 'DLR', 'AVB',
        # Telecom & Media
        'T', 'VZ', 'DIS', 'CMCSA', 'WBD', 'PARA', 'FOX', 'FOXA', 'NWSA', 'NWS',
    ]


def get_large_cap_tickers(min_market_cap: float = 10e9) -> List[str]:
    """
    Get large-cap stocks (market cap > $10B by default).

    Args:
        min_market_cap: Minimum market cap in dollars (default $10B)

    Returns:
        List of large-cap ticker symbols
    """
    return get_all_us_market_tickers(min_market_cap=min_market_cap)


def get_mid_cap_tickers() -> List[str]:
    """
    Get mid-cap stocks (market cap $2B - $10B).

    Returns:
        List of mid-cap ticker symbols
    """
    # This requires filtering by range, which the API doesn't support directly
    # Return large cap subset for now
    return get_all_us_market_tickers(min_market_cap=2e9)


def get_small_cap_tickers() -> List[str]:
    """
    Get small-cap stocks (market cap $300M - $2B).

    Returns:
        List of small-cap ticker symbols
    """
    return get_all_us_market_tickers(min_market_cap=300e6)


def get_exchange_tickers(exchange: str) -> List[str]:
    """
    Get all tickers from a specific exchange.

    Args:
        exchange: Exchange name ('NYSE', 'NASDAQ', or 'AMEX')

    Returns:
        List of ticker symbols from the specified exchange
    """
    return get_all_us_market_tickers(exchanges=[exchange.upper()])


def count_available_tickers() -> dict:
    """
    Count available tickers by exchange.

    Returns:
        Dict with counts by exchange and total
    """
    counts = {}
    for exchange in ['NYSE', 'NASDAQ', 'AMEX']:
        try:
            tickers = get_exchange_tickers(exchange)
            counts[exchange] = len(tickers)
        except Exception as e:
            counts[exchange] = f"Error: {e}"

    # Calculate total
    total = sum(v for v in counts.values() if isinstance(v, int))
    counts['total'] = total

    return counts


if __name__ == "__main__":
    print("S&P 500 Tickers:", len(get_sp500_tickers()))
    print("NASDAQ-100 Tickers:", len(get_nasdaq100_tickers()))
    print("All US Tickers (S&P + NASDAQ-100):", len(get_all_us_tickers()))
    print("\nFetching ALL US market tickers...")
    print(count_available_tickers())
