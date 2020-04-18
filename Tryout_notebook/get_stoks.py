from alpha_vantage.timeseries import TimeSeries
import pickle 
import numpy as np 
import pandas as pd
import time
import glob

# ts = TimeSeries(key='2N7U5OBZO5MQT4IL',output_format = 'pandas')
# # Get json object with the intraday data and another with  the call's metadata
# data, meta_data = ts.get_daily_adjusted('GOOGL', outputsize='compact')

# print(data.columns)
#print(data['6. volume'])

### Plan ; ^GSPC 
    # for every tick in Nasdaq
    # get :     [adj_close,volume]
    #           [adj_open, volume]
    # save to text file 

def save_NSDAQ():
    # ticks : 
    ticks = pd.read_csv('../stock_data/constituents_csv.csv')

    donefiles = glob.glob('../stock_data/NASDAQ/*')
    done_ticks = [f.split('/')[-1] for f in donefiles]
    not_working = ['BF.B','CA','CSRA','DPS','EVHC','GGP','LLL','MON','NFX','PX']
    print(len(done_ticks))

    for tck in ticks.Symbol.values:
        if tck not in done_ticks and tck not in not_working:
            print(tck)
            ts = TimeSeries(key='2N7U5OBZO5MQT4IL',output_format = 'pandas')
            # Get json object with the intraday data and another with  the call's metadata
            data,_ = ts.get_daily_adjusted(tck, outputsize='full')
            tosave = data[['5. adjusted close', '6. volume']]
            np.savetxt('../stock_data/NASDAQ/' + tck,tosave)
            time.sleep(10)

save_NSDAQ()