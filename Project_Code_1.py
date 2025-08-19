#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from matplotlib.pyplot import figure
from scipy import stats

import plotly as py
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing, svm
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.filterwarnings('ignore')

path=r"C:\Sushma - MSc Degree Apprenticeship\Final Project\dataset1.csv"
df=pd.read_csv(path)
print(df.shape)
print("\n")
print(df.info())
print("\n")
print(df.head())

df['Year'] = df['Year'].astype('str')

print(df.info())
print(df.head())

#change data types as required
df['RPK%'] = df['RPK%'].str.replace('%', '').astype(float)

print(df.info())

df['ROIC %'] = df['ROIC %'].str.replace('%', '').astype(float)
print(df.info())
#correcting data entries, removing spaces in data

df['Scheduled Passengers'] = df['Scheduled Passengers'].str.replace(',', '').astype(float)

print(df.info())

#removing special characters in data
df['Passenger load factor%'] = df['Passenger load factor%'].str.replace('%', '').astype(float)
print(df.info())

#Divide the dataset in dependent and Independent variables
X= df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Create training and test set
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.20,
                                                    train_size=0.80,
                                                    random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Train Score: ', regressor.score(X_train, y_train))
print('Test Score: ', regressor.score(X_test, y_test))


#dropping duplicates
print("Total number of duplicate rows: ", df.duplicated().sum())
df=df.drop_duplicates()
print("Total number of duplicate rows: ", df.duplicated().sum())

print(df.head())
df.nunique()

# Check for null values in data
nullcount = df.isnull().sum()
print('Total number of null values in dataset:', nullcount.sum())
print(df.head())

#correlation matrix
plt.figure(figsize=(10,10))
heatmap = sns.heatmap(df.corr(),cmap=sns.diverging_palette(20, 220, n=200),center = 0,annot=True)
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=90,horizontalalignment='right');

#scatterplot for selected columns (defined in cols)
sns.set()
cols = ['Total Revenue','RPK%','Flights' ,'NET PROFIT' ,'ROIC %', 'GDP','Inflation Rate']

sns.pairplot(df[cols], size = 2.5)
plt.show();

#scatterplot for selected columns (defined in cols)
sns.set()
cols = ['Total Revenue','Flights' ,'NET PROFIT' ,'ROIC %', 'GDP','Inflation Rate']

sns.pairplot(df[cols], size = 2.5)
plt.show();

#histogram
sns.distplot(df['GDP']);

plt.show()

sns.distplot(df['Inflation Rate']);
plt.show()

sns.distplot(df['Total Revenue']);
plt.show()

#outliers
plt.figure(figsize = (15, 10))

ax=plt.subplot(231)
plt.boxplot(df['Total Revenue'])
ax.set_title('Total Revenue')

ax=plt.subplot(232)
plt.boxplot(df['Flights'])
ax.set_title('Flights')

ax=plt.subplot(233)
plt.boxplot(df['NET PROFIT'])
ax.set_title('NET PROFIT')

ax=plt.subplot(234)
plt.boxplot(df['ROIC %'])
ax.set_title('ROIC %')

ax=plt.subplot(235)
plt.boxplot(df['GDP'])
ax.set_title('GDP')

ax=plt.subplot(236)
plt.boxplot(df['Inflation Rate'])
ax.set_title('Inflation Rate')

plt.suptitle('Boxplots showing outliers')

#to show how balanced the dataset is
fig, axs = plt.subplots(figsize=(15,3))
sns.countplot(x='Total Revenue', data=df, ax=axs)

fig, axs = plt.subplots(figsize=(15,3))
sns.countplot(x='Flights', data=df, ax=axs)

fig, axs = plt.subplots(figsize=(15,3))
sns.countplot(x='Inflation Rate', data=df, ax=axs)

fig, axs = plt.subplots(figsize=(15,3))
sns.countplot(x='GDP', data=df, ax=axs)

fig, axs = plt.subplots(figsize=(15,3))
sns.countplot(x='ROIC %', data=df, ax=axs)

fig, axs = plt.subplots(figsize=(15,3))
sns.countplot(x='NET PROFIT', data=df, ax=axs)

# plotting graphs of selected variables

plt.figure(figsize=(15,6))
plt.plot(df['Year'], df['GDP']/10000)
plt.plot(df['Year'], df['Inflation Rate'])
plt.plot(df['Year'], df['Total Revenue']/100)
plt.plot(df['Year'], df['ROIC %']/10)
plt.plot(df['Year'], df['Flights']/10)
plt.plot(df['Year'], df['NET PROFIT']/10)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("Revenue over Years")  # add title
plt.show()

#implementing linear regression model

x = df[['GDP','Inflation Rate']]
y = df['Flights']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)



print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()

predictions = model.predict(x) 
 
print_model = model.summary()
print("\n")
print("\n")
print(print_model)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Train Score: ', regressor.score(X_train, y_train))
print('Test Score: ', regressor.score(X_test, y_test))

#remove outliers

#print(np.where((df['Inflation Rate']>5)))
 
z = np.abs(stats.zscore(df['Inflation Rate']))
print(z)
print(np.where(z > 3))

#Divide the dataset in dependent and Independent variables
X= df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Create training and test set
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.20,
                                                    train_size=0.80,
                                                    random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Train Score: ', regressor.score(X_train, y_train))
print('Test Score: ', regressor.score(X_test, y_test))



x = df[['GDP','Inflation Rate']]
y = df['Flights']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print("\n")
print("\n")
print(print_model)

#removing outliers
cols = ['Inflation Rate'] # one or more

Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
#Divide the dataset in dependent and Independent variables
X= df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Create training and test set
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.20,
                                                    train_size=0.80,
                                                    random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Train Score: ', regressor.score(X_train, y_train))
print('Test Score: ', regressor.score(X_test, y_test))
#reran model after removing outliers
x = df[['GDP','Inflation Rate']]
y = df['Flights']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print("\n")
print("\n")
print(print_model)

