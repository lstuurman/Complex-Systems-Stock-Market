### SCRIPT FOR SAVING CRYPTO DATA USING ALPHAVANTAGE API###

# crypto valuta tradable via Plus500:
CRPT_VALUTA = [
    'BTC','ETH','LTC','NEO',
    'XRP','IOTA','XLM','EOS',
    'BCH','ADA','TRX','XMR'
    ]

def save_crypto(valutas,daily = True):

    # try 500 first because of max API calls : 
    for tck in valutas:
        print(tck)
        if daily:
            ts = TimeSeries(key='2N7U5OBZO5MQT4IL',output_format = 'pandas')
            # Get json object with the intraday data and another with  the call's metadata
            data,_ = ts.get_daily_adjusted(tck, outputsize='full')
            tosave = data[['5. adjusted close', '6. volume']]
            #np.savetxt('../stock_data/NASDAQ/' + tck,tosave)
            #tosave.to_csv('../stock_data/NASDAQ2/' + tck)
            tosave.sort_index().to_csv('../stock_data/NASDAQ_DAILY/' + tck)
            time.sleep(10)