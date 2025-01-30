import pandas as pd
import numpy as np
from vnstock3 import Vnstock

class DataLoader:
    '''
    Fetching data using VNSTOCK Python library.

    INPUT
    -----------------
    ticker :: stock or index symbol
    type :: if symbol is a stock, type is stock; if symbol is index, type is index
    resolution :: timeframe of data (e.g., 'D' for daily)
    startDate :: starting date of the data
    endDate :: ending date of the data
    data :: fetched data

    OUTPUT
    -------------
    dataframe :: OHCLV dataframe
    '''
    
    def __init__(self, ticker, resolution, startDate, endDate):
        self.ticker = ticker
        self.resolution = resolution
        self.startDate = startDate
        self.endDate = endDate

    @staticmethod
    def get_stock_data(ticker, startDate, endDate):
        '''
        Fetch stock data using VNSTOCK.
        '''
        try:
            vnstock = Vnstock()
            stock = vnstock.stock(symbol=ticker, source="TCBS")
            price_data = stock.quote.history(start=startDate, end=endDate, interval="1D")
            price_data = price_data.rename(columns={"close": "Close", "volume": "Volume"})
            return price_data
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None

    @staticmethod
    def get_index_data(index_symbol, startDate, endDate):
        '''
        Fetch index data using VNSTOCK.
        '''
        try:
            vnstock = Vnstock()
            index_data = vnstock.stock(symbol=index_symbol, source="TCBS").quote.history(
                start=startDate, end=endDate, interval="1D"
            )
            index_data = index_data.rename(columns={"close": "Close", "volume": "Volume"})
            return index_data
        except Exception as e:
            print(f"Error fetching index data: {e}")
            return None


# Load stock data
def load_stock_data(symbol, start, end):
    '''
    Stock data loading function using VNSTOCK.

    INPUT
    ---------
    symbol :: stock symbol
    start  :: starting date
    end    :: ending date

    OUTPUT
    ---------
    dataframe :: stock price OHCLV dataframe with percentage return and logarithm return
    '''
    price_data = DataLoader.get_stock_data(symbol, start, end)
    if price_data is not None:
        # Calculate percentage return and logarithmic return
        price_data['pct_return'] = price_data['Close'].pct_change()
        price_data['log_return'] = np.log(price_data['Close']) - np.log(price_data['Close'].shift(1))
        price_data.set_index('time', inplace=True)  # Set time as the index
        price_data.index = pd.to_datetime(price_data.index).strftime('%Y-%m-%d')
        return price_data[1:]
    else:
        print("No stock data available.")
        return None


# Load index data
def load_index_data(index_symbol, start, end):
    '''
    Index data loading function using VNSTOCK.

    INPUT
    ---------
    index_symbol :: index symbol
    start  :: starting date
    end    :: ending date

    OUTPUT
    ---------
    dataframe :: index price OHCLV dataframe with percentage return and logarithm return
    '''
    index_data = DataLoader.get_index_data(index_symbol, start, end)
    if index_data is not None:
        # Calculate percentage return and logarithmic return
        index_data['pct_return'] = index_data['Close'].pct_change()
        index_data['log_return'] = np.log(index_data['Close']) - np.log(index_data['Close'].shift(1))
        index_data.set_index('time', inplace=True)  # Set time as the index
        index_data.index = pd.to_datetime(index_data.index).strftime('%Y-%m-%d')
        return index_data[1:]
    else:
        print("No index data available.")
        return None


# Load macroeconomic data (custom logic or placeholder)
def load_macro_data(symbol, start, end):
    '''
    Placeholder for macroeconomic data loading.
    Replace with VNSTOCK if similar functionality is available.
    '''
    print(f"Fetching macro data for {symbol} is not implemented in VNSTOCK yet.")
    return None


# Example Usage
if __name__ == "__main__":
    stock_data = load_stock_data(symbol="VCB", start="2023-01-01", end="2023-12-31")
    if stock_data is not None:
        print("Stock Data:\n", stock_data.head())

    index_data = load_index_data(index_symbol="VNINDEX", start="2023-01-01", end="2023-12-31")
    if index_data is not None:
        print("Index Data:\n", index_data.head())
