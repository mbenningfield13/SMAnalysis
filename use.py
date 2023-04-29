from finance import model, r21, r22, tests
import pandas as pd

print(r21, r22)

for i in tests:
    guess = model.predict(tests[i].loc[['1. open', '2. high', '3. low', '6. volume']].T.values.reshape(1, -1))
    print(f'Predicted Closing Value of {i}: ${round(guess[0][0], 2)}, Actual Closing Value: $ {tests[i].loc["4. close"]}')
    print(f'\tError in estimate = {round(guess[0][0], 2)-float(tests[i].loc["4. close"])}')

#Scrapes data for a few tickers
#Use the latest date for predicting data 
#Find the difference in predicted vs actual across multiple tickers

