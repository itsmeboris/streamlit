import os
from concurrent import futures
from datetime import datetime
from glob import glob

import pandas as pd
import pandas_datareader.data as web

""" set the download window """
now_time = datetime.now()
start_time = datetime(now_time.year - 5, now_time.month, now_time.day)


def download_stock(stock):
    """ try to query the iex for a stock, if failed note with print """
    try:
        # print(stock)
        stock_df = web.DataReader(stock, 'yahoo', start_time, now_time)
        stock_df['Name'] = stock
        output_name = os.path.join('individual_stocks_5yr', f'{stock}_data.csv')
        stock_df.to_csv(output_name)
    except:
        pass
        # bad_names.append(stock)
        # print('bad: %s' % stock)


def pull():
    """ list of s_anp_p companies """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    s_and_p = html[0]
    s_and_p = list(s_and_p['Symbol'])

    """here we use the concurrent.futures module's ThreadPoolExecutor
      to speed up the downloads buy doing them in parallel 
      as opposed to sequentially """

    # set the maximum thread number
    max_workers = 50
    os.makedirs('individual_stocks_5yr', exist_ok=True)

    workers = min(max_workers, len(s_and_p))  # in case a smaller number of stocks than threads was passed in
    with futures.ThreadPoolExecutor(workers) as executor:
        executor.map(download_stock, s_and_p)

    # timing:
    finish_time = datetime.now()
    duration = finish_time - now_time
    minutes, seconds = divmod(duration.seconds, 60)
    print('getSandP_threaded.py')
    print(f'The threaded script took {minutes} minutes and {seconds} seconds to run.')
    files = glob(os.path.join('individual_stocks_5yr', '*.csv'))
    df = pd.concat([pd.DataFrame(pd.read_csv(file, index_col='Date')) for file in files])
    df.to_csv('individual_stocks_5yr.csv')


pull()
