import base64
from io import BytesIO

import yahoo_fin.stock_info
from flask import Blueprint, render_template, request, url_for, flash, redirect
from matplotlib.figure import Figure
from yahoo_fin.stock_info import *
import dateutil.relativedelta
import datetime
import numpy as np

def get_days_for_est(data_est_days):
    i = 0
    day = datetime.datetime.today()
    while i < 5:
        day = day + datetime.timedelta(days=1)
        if day.weekday() != 5 and day.weekday() != 6:
            # print(day.weekday())
            data_est_days[i, 0] = day.strftime("%d.%m")
            # print(data_est_days[i, 0])
            i += 1

def list_spy_holdings() -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/75845569/
    # Source: https://www.ssga.com/us/en/intermediary/etfs/funds/spdr-sp-500-etf-trust-spy
    # Note: One of the included holdings is CASH_USD.
    url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
    return pd.read_excel(url, engine='openpyxl', index_col='Ticker', skiprows=4).dropna()

views = Blueprint(__name__, "views")

tickers = list_spy_holdings()
tickers = tickers.index.to_numpy()

@views.route("/", methods=('GET', 'POST'))
def home():
    if request.method == 'POST':
        ticker = request.form['title']
        print(ticker)
        if ticker not in tickers:
            flash("enter a valid ticker!")
        else:
            flash(ticker)
            myobj = {'ticker': ticker}
            x = requests.post('http://127.0.0.1:8000/api/ticker', json=myobj)
            if x.status_code==200:
                return redirect("/")
    date = datetime.datetime.now()
    date = date + dateutil.relativedelta.relativedelta(months=-2)
    date = date.strftime("%d/%m/%Y")
    stock_info = get_data(ticker="^GSPC", start_date=date)
    stock_info = stock_info[['close']].tail(22)
    recent_data = np.zeros((22, 1))
    recent_data[:, 0] = stock_info['close'].tolist()
    stock_info.index = stock_info.index.strftime("%d/%m/%Y")
    recent_data_dates = np.chararray((22, 1), itemsize=5)
    recent_data_dates[:, 0] = stock_info.index.tolist()

    resp = requests.get('http://127.0.0.1:8000/api/data_dates')
    if resp.status_code!=200:
        return "main server unavaliable!"
    data_dates = np.asarray(resp.json())
    data_dates = data_dates.astype(str)
    resp = requests.get('http://127.0.0.1:8000/api/data_numeric')
    if resp.status_code!=200:
        return "main server unavaliable!"
    data = np.asarray(resp.json())
    if resp.status_code!=200:
        return "main server unavaliable!"
    resp = requests.get('http://127.0.0.1:8000/api/estimation')
    data_est = resp.json()
    data_est_days = np.chararray((5, 1), itemsize=5)
    get_days_for_est(data_est_days)

    fig = Figure(figsize=(14, 6))
    # fig.gcf().subplots_adjust(left=1, right=1)
    ax = fig.subplots()
    ax.plot(data_dates[:, 0], data.tolist(), 'bo-')
    ax.plot(data_est_days[:, 0], data_est, 'mo-')
    ax.tick_params(axis='x', which='major', labelsize=7)
    ax.grid()
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template("index.html", plot_url=plot_url)
