# FILE to download recent tickers and save to pickle file containing list of strings of tickers

import wget
import pandas as pd
import pickle
import os

def get_tickers():
    ## TODO : DELETE OLD TICKERS
    try:
        os.remove('/home/lau/GIT/Complex Systems Stock Market/stock_data/tickers.csv')
    except:
        print('No file present yet')
    url = "https://dumbstockapi.com/stock?format=csv&exchanges=NASDAQ"
    filename = '/home/lau/GIT/Complex Systems Stock Market/stock_data/tickers.csv'
    wget.download(url, filename)
    ticker_df = pd.read_csv(filename)
    tickerlist = ticker_df['ticker'].tolist()
    pickle.dump(tickerlist, open('/home/lau/GIT/Complex Systems Stock Market/stock_data/tickers.pkl','wb'))

# def get_stock_history(dates):
#     # function to download all stock data from yahoo finance


if __name__ == "__main__":
    get_tickers()

