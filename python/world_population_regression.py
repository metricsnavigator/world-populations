'''
Author: Eric Reschke
Cite: https://metricsnavigator.org/world-population-growth/
Last Reviewed: 2022-10-02
License: Open to all
'''

import numpy as np
import pandas as pd
from decimal import Decimal
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.metrics import mean_absolute_error,mean_squared_error
import statsmodels.api as sm

dataImport = pd.read_csv('5. future_regression_summary.csv')
pop_df = pd.DataFrame.copy(dataImport)

# convert data to arrays to set up the regression model
X = pop_df.iloc[:,0:-1].values
Y = pop_df.iloc[:,-1].values

lin = sklearn.linear_model.LinearRegression()
lin.fit(X,Y)

prediction = lin.predict(X)
FinalPrediction = []
for i in prediction:
    x = round(Decimal(i),0)
    FinalPrediction.append(x)

y_export = pd.DataFrame(FinalPrediction)
y_export.columns=['Prediction']
pop_df['Prediction'] = y_export['Prediction']

# -------------------------------------------------- #

## regression graph
plt.scatter(pop_df['WLD_Pop'],pop_df['Prediction'],color ='yellow')
plt.plot(pop_df['WLD_Pop'],pop_df['Prediction'],color ='black')
plt.show()

# mean-squared error
mse = round(mean_squared_error(y_true=Y,y_pred=pop_df['Prediction']),0)
print('\n')
print('MSE:',mse)

# r-squared value
r2 = round(lin.score(X,Y),4)
print('R2 Value:',r2)

# create dataframe for coefficients and intercept
coeff = []
for i in lin.coef_:
    x = round(Decimal(i),4)
    coeff.append(x)

coeff.append(round(Decimal(lin.intercept_),4))
cNames = ('Year','WLD_Exp','WLD_Fert','Intercept')
coeff = pd.DataFrame(coeff,index=cNames)
coeff.columns=['Results']
print(coeff,'\n')

# general summary of t-stat and p-values using the statsmodels library
model = sm.OLS.from_formula("WLD_Pop ~ Year+WLD_Exp+WLD_Fert",data=pop_df)
result = model.fit()
print(result.summary())

# use this command to list out the possibilities of data extraction from the model
# dir(result)

# use this loop to get the full list of coefficients rounded
for i in result.params:
    print(round(i,0))


#################################
# regression on world life expectancy rates
model = sm.OLS.from_formula("WLD_Exp ~ Year",data=pop_df)
result = model.fit()
print(result.summary())


## end of script

