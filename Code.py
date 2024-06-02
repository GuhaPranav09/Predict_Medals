import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np


teams = pd.read_csv("teams.csv")
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
sns.lmplot(x='athletes', y='medals',data=teams, fit_reg=True, ci=None)
sns.lmplot(x='age', y='medals', data=teams, fit_reg=True, ci=None) 
teams.plot.hist(y="medals")
plt.show()

print1 = teams[teams.isnull().any(axis=1)]
teams = teams.dropna()

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

reg = LinearRegression()

predictors=["athletes","prev_medals"]
target = "medals"

reg.fit(train[predictors],train["medals"])
LinearRegression()

predictions = reg.predict(test[predictors])
test["predictions"] = predictions

test.loc[test["predictions"]<0, "predictions"] = 0
test["predictions"] = test["predictions"].round()
print(test)

error = mean_absolute_error(test["medals"], test["predictions"])

print2 = (teams.describe()["medals"])
#error should be below standard deviation

print3 = (test[test["team"] == "USA"]) #go team by  team

errors = (test["medals"] - test["predictions"]).abs()
print4 = (errors)

error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team 

error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio.plot.hist()
plt.show()
