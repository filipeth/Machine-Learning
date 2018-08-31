import quandl
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

desired_width=400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

quandl.ApiConfig.api_key = 'ENpvycjfywqr84MiW5js'

stock_list = ['AAPL','AMZN', 'MSFT', 'GOOG']
start = datetime(2016, 12, 31)
end = datetime.now()

file = quandl.get_table('WIKI/PRICES', ticker = stock_list,
                        #qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, ## select the columns you might want or select all
                        date = { 'gte': start, 'lte': end },
                        paginate=True).set_index("date")



# print(file.head())

AAPL = file[file.ticker=="AAPL"]
AMZN = file[file.ticker=="AMZN"]
MSFT = file[file.ticker=="MSFT"]
GOOG = file[file.ticker=="GOOG"]
# print(AAPL.info())
df_APPL = AAPL["adj_close"]

print(df_APPL.head())

df_APPL.plot(legend=True, figsize=(10,4))
plt.title('Closing price for Apple')
plt.ylabel('Price')
plt.show()

AAPL['adj_volume'].plot(legend=True,figsize=(10,4))
plt.title('Volume trade per day')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(AAPL['adj_close'],'r',label="Apple",linewidth=1)
plt.plot(AMZN['adj_close'],'b',label="Amazon",linewidth=1)
plt.plot(MSFT['adj_close'],'g',label="Micorsoft",linewidth=1)
plt.plot(GOOG['adj_close'],'m',label="Google",linewidth=1)
plt.legend()
plt.title('Closing price')
plt.show()

AAPL['Daily_Return'] = AAPL['adj_close'].pct_change()
AAPL['Daily_Return'].plot(figsize=(15,6),legend=True,linestyle='--',marker='o')
plt.title('Percentage change')
plt.show()

closing_price_stock = quandl.get_table('WIKI/PRICES', ticker = stock_list,
                        qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, ## select the columns you might want or select all
                        date = { 'gte': start, 'lte': end },
                        paginate=True).pivot(index="date",columns='ticker', values='adj_close')
print(closing_price_stock.head())

stock_returns = closing_price_stock.pct_change()
sns.jointplot('GOOG','AAPL',stock_returns,kind='scatter')
plt.title('Daily return between Google and Apple compared')
plt.show()

sns.pairplot(stock_returns.dropna())
plt.show()

returns = stock_returns.dropna()
area = np.pi*20
plt.figure(figsize=(10,4))
plt.scatter(returns.mean(),returns.std(),alpha=0.5,s=area)
plt.xlabel('Expected Returns')
plt.ylabel('Risk')
for label, x, y in zip(returns.columns, returns.mean(), returns.std()):
    plt.annotate(
        label,
        xy = (x,y), xytext = (50,50),
        textcoords = 'offset points', ha='center', va='bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))


plt.show()
