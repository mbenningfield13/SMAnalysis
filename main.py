import requests
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load & fix the data from A.V.
KEY = 'WNVEV8F0NHWN16DR'
SYM = 'AMZN'

TICKERS = ['AAPL', 'JNJ', 'JPM']
TSD = 'Time Series (Daily)'
dfNA = pd.DataFrame()

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={SYM}&apikey={KEY}'
response = requests.get(url)
data = json.loads(response.text)    
dfNA = pd.DataFrame(data[TSD])
j=0
for i in dfNA.columns:
    dfNA.columns.values[j] = i + 'AMZN'
    j+=1

tests = pd.DataFrame()
tests['AMZN'] = dfNA.pop(dfNA.columns.values[0])

for i in TICKERS:
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={i}&apikey={KEY}'
    response = requests.get(url)
    data = json.loads(response.text) 
    dfTemp = pd.DataFrame(data[TSD])
    j=0
    for m in dfTemp.columns:
        dfTemp.columns.values[j] = m + i
        j+=1
    tests[i] = dfTemp.pop(dfTemp.columns.values[0])
    dfNA = pd.concat([dfNA, dfTemp], axis=1)

df = dfNA.transpose()
df = dfNA.drop(['5. adjusted close', '7. dividend amount', '8. split coefficient'])
tests = tests.drop(['5. adjusted close', '7. dividend amount', '8. split coefficient'])

for i in df:
    if i == '6. volume':
        df[i] = df[i].astype('int64')
    else:
        df[i] = df[i].astype('float64')

#spilt the data
Xtrain, Xtest, ytrain, ytest = train_test_split(
    df.loc[['1. open', '2. high', '3. low', '6. volume']].T.values, 
    df.loc[['4. close']].T.values,
    test_size=.15,
    random_state=13
)

#train & test the model
model = LinearRegression(fit_intercept=True)
model.fit(Xtrain, ytrain)
test = model.predict(Xtest)
r21 = model.score(Xtrain, ytrain)
r22 = model.score(Xtest, ytest)








