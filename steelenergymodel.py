import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

steeldata = pd.read_csv("C:/Users/Creacion Tech/Documents/Steel_industry_data.csv")

steeldata.head(5)

steeldata.info()

#EDA: data analysis: correlation and heatmap
import seaborn as sns

df_corr = steeldata.corr()
sns.heatmap(df_corr, annot=True, cmap='coolwarm')

#EDA: pairplot

sns.pairplot(steeldata)

# Assign X and Y variables

X = steeldata.drop(['Usage_kWh','Leading_Current_Reactive_Power_kVarh','Day_of_week','Load_Type','WeekStatus','date'],axis=1)
y = steeldata['Usage_kWh']

X


steeldata.columns

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with training data

regressor.fit(X, y)

regressor.intercept_
# Find X coefficients

regressor.coef_


#Saving Modelto disk

pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.95, 0.0, 73.21, 100.0, 900]]))

print(model.predict([[2, 0, 73, 100, 900]]))
