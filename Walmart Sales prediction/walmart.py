# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/opt/conda/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.5
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
/kaggle/input/walmart-dataset/Walmart.csv
df=pd.read_csv("/kaggle/input/walmart-dataset/Walmart.csv")
df.head()
Store	Date	Weekly_Sales	Holiday_Flag	Temperature	Fuel_Price	CPI	Unemployment
0	1	05-02-2010	1643690.90	0	42.31	2.572	211.096358	8.106
1	1	12-02-2010	1641957.44	1	38.51	2.548	211.242170	8.106
2	1	19-02-2010	1611968.17	0	39.93	2.514	211.289143	8.106
3	1	26-02-2010	1409727.59	0	46.63	2.561	211.319643	8.106
4	1	05-03-2010	1554806.68	0	46.50	2.625	211.350143	8.106
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6435 entries, 0 to 6434
Data columns (total 8 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Store         6435 non-null   int64  
 1   Date          6435 non-null   object 
 2   Weekly_Sales  6435 non-null   float64
 3   Holiday_Flag  6435 non-null   int64  
 4   Temperature   6435 non-null   float64
 5   Fuel_Price    6435 non-null   float64
 6   CPI           6435 non-null   float64
 7   Unemployment  6435 non-null   float64
dtypes: float64(5), int64(2), object(1)
memory usage: 402.3+ KB
df["Date"]=pd.to_datetime(df['Date'])
/tmp/ipykernel_20/3629038854.py:1: UserWarning: Parsing dates in DD/MM/YYYY format when dayfirst=False (the default) was specified. This may lead to inconsistently parsed dates! Specify a format to ensure consistent parsing.
  df["Date"]=pd.to_datetime(df['Date'])
df.isna().sum()
Store           0
Date            0
Weekly_Sales    0
Holiday_Flag    0
Temperature     0
Fuel_Price      0
CPI             0
Unemployment    0
dtype: int64
df.shape
(6435, 8)
df['Date'].nunique()
143
df['Date'].describe()
/tmp/ipykernel_20/3134759576.py:1: FutureWarning: Treating datetime data as categorical rather than numeric in `.describe` is deprecated and will be removed in a future version of pandas. Specify `datetime_is_numeric=True` to silence this warning and adopt the future behavior now.
  df['Date'].describe()
count                    6435
unique                    143
top       2010-05-02 00:00:00
freq                       45
first     2010-01-10 00:00:00
last      2012-12-10 00:00:00
Name: Date, dtype: object
date_sales=pd.DataFrame(df.groupby('Date')['Weekly_Sales'].sum()).reset_index()
date_sales['cumulative']=date_sales['Weekly_Sales'].cumsum()
px.histogram(date_sales,x='Date',y='Weekly_Sales')
px.line(date_sales,x='Date',y='cumulative')
df['Month']=df['Date'].dt.month
df['Year']=df['Date'].dt.year
df.head()
Store	Date	Weekly_Sales	Holiday_Flag	Temperature	Fuel_Price	CPI	Unemployment	Month	Year
0	1	2010-05-02	1643690.90	0	42.31	2.572	211.096358	8.106	5	2010
1	1	2010-12-02	1641957.44	1	38.51	2.548	211.242170	8.106	12	2010
2	1	2010-02-19	1611968.17	0	39.93	2.514	211.289143	8.106	2	2010
3	1	2010-02-26	1409727.59	0	46.63	2.561	211.319643	8.106	2	2010
4	1	2010-05-03	1554806.68	0	46.50	2.625	211.350143	8.106	5	2010
fig=df.groupby(['Year'])['Weekly_Sales'].mean().plot(kind='bar',color=['green'])
fig.set_xlabel("Year")
fig.set_title("Sales per year")

fig.set_xticklabels(fig.get_xticklabels(), rotation=20)
[Text(0, 0, '2010'), Text(1, 0, '2011'), Text(2, 0, '2012')]

fig=df.groupby(['Month'])['Weekly_Sales'].mean().plot(kind='bar',color=['green'])
fig.set_xlabel("Month")
fig.set_title("Sales per Monthes")
fig.set_xticklabels(fig.get_xticklabels(), rotation=20)
[Text(0, 0, '1'),
 Text(1, 0, '2'),
 Text(2, 0, '3'),
 Text(3, 0, '4'),
 Text(4, 0, '5'),
 Text(5, 0, '6'),
 Text(6, 0, '7'),
 Text(7, 0, '8'),
 Text(8, 0, '9'),
 Text(9, 0, '10'),
 Text(10, 0, '11'),
 Text(11, 0, '12')]

 
def month_to_season(month):
    if month in [12,1,2]:
        return 'winter'
    if month in [3,4,5]:
        return 'spring'
    if month in [6,7,8]:
        return 'summer'
    return 'Fall'
    
df['Season']=df['Month'].apply(month_to_season)
df=df.drop(columns=['Date'],axis=1)
df.head()
Store	Weekly_Sales	Holiday_Flag	Temperature	Fuel_Price	CPI	Unemployment	Month	Year	Season
0	1	1643690.90	0	42.31	2.572	211.096358	8.106	5	2010	spring
1	1	1641957.44	1	38.51	2.548	211.242170	8.106	12	2010	winter
2	1	1611968.17	0	39.93	2.514	211.289143	8.106	2	2010	winter
3	1	1409727.59	0	46.63	2.561	211.319643	8.106	2	2010	winter
4	1	1554806.68	0	46.50	2.625	211.350143	8.106	5	2010	spring
fig=df.groupby(['Season'])['Weekly_Sales'].mean().plot(kind='bar',color='green',)
fig.set_title("Weekly_Sales Per seasons")
fig.set_ylabel("Sales")
Text(0, 0.5, 'Sales')

cols=['Weekly_Sales','Temperature','Fuel_Price','CPI','Unemployment']

fig,ax=plt.subplots(2,5,figsize=(20,4))
for idx,col in enumerate(cols):
    
    sns.histplot(data=df,x=col,ax=ax[0][idx])
    ax[0][idx].set_title(col)

for idx,col in enumerate(cols):
    sns.boxplot(data=df,x=col,ax=ax[1][idx],color='green')
    ax[1][idx].set_title(col)
    


fig.show()

    
    
    

flag=(df['Weekly_Sales']<3e6)&(df['Unemployment']<11)&(df['Unemployment']>4)
df=df[flag]
df['Store']=df['Store'].astype('object') 
from category_encoders import BinaryEncoder
encoder=BinaryEncoder()
encoder.fit(df)
df=encoder.transform(df)
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2,random_state=5)
train.shape
train_X=train.drop(columns=['Weekly_Sales'],axis=1)
train_y=train['Weekly_Sales']
test_X=test.drop(columns=['Weekly_Sales'],axis=1)
test_y=test['Weekly_Sales']
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge,Lasso,LinearRegression

reg=make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=3),
    Ridge(),
)

param_grid={'polynomialfeatures__degree':[2,3,4,5]}

grd=GridSearchCV(reg,param_grid,cv=4)
grd.fit(train_X,train_y)
#
GridSearchCV
estimator: Pipeline

StandardScaler

PolynomialFeatures

Ridge
grd.best_params_
{'polynomialfeatures__degree': 3}
grd.score(train_X,train_y),grd.score(test_X,test_y)
(0.9763428702502118, 0.9541693032914947)
from sklearn.svm import SVR

reg=make_pipeline(
    StandardScaler(),
    SVR(kernel="poly",C=1,epsilon=0.1),
)

param_grid={'svr__C':[1000,10000,100000,100000],'svr__degree':[2,3,4],}

grd=GridSearchCV(reg,param_grid,cv=4)
grd.fit(train_X,train_y)
grd.best_params_
{'svr__C': 100000, 'svr__degree': 3}
grd.best_score_
0.7104270282536789
 