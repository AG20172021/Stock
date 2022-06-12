#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import warnings
import pandas_datareader.data as web
warnings.filterwarnings('ignore')



# In[41]:


import pyEX as p


# In[2]:


HDFC_df = pd.read_csv("/Users/ethan/Desktop/FMC-2.csv") 
HDFC_df = HDFC_df.round(2)
HDFC_df.head(2)


# In[3]:


HDFC_df.shape


# In[4]:


HDFC_df.isnull().sum()


# In[5]:


HDFC_df.dropna(inplace = True, axis = 0)


# In[6]:


HDFC_df.dtypes


# In[7]:


HDFC_df['Date'] = pd.to_datetime(HDFC_df['Date'])
HDFC_df.head(2)


# In[8]:


HDFC_df['Date'].max() - HDFC_df['Date'].min()


# In[9]:


HDFC_df.iloc[-90:].describe().astype(int)


# In[10]:


HDFC_df.index = HDFC_df['Date']


# In[11]:


HDFC_df['Adj Close'].plot(figsize = (15,8))
plt.show()


# In[12]:


HDFC_df['Day_Perc_Change'] = HDFC_df['Adj Close'].pct_change()*100
HDFC_df.head()


# In[13]:


HDFC_df.dropna(axis = 0, inplace = True)


# In[14]:


HDFC_df['Day_Perc_Change'].plot(figsize = (12, 6), fontsize = 12)


# In[15]:


HDFC_df['Day_Perc_Change'].hist(bins = 50, figsize = (10,5)) 
plt.xlabel('Daily returns')
plt.ylabel('Frequency')
plt.show()
#satistics
HDFC_df.Day_Perc_Change.describe()


# In[16]:


def trend(x):
    if x > -0.5 and x <= 0.5:
        return 'Slight or No change'
    elif x > 0.5 and x <= 1:
        return 'Slight Positive'
    elif x > -1 and x <= -0.5:
        return 'Slight Negative'
    elif x > 1 and x <= 3:
        return 'Positive'
    elif x > -3 and x <= -1:
        return 'Negative'
    elif x > 3 and x <= 7:
        return 'Among top gainers'
    elif x > -7 and x <= -3:
        return 'Among top losers'
    elif x > 7:
        return 'Bull run'
    elif x <= -7:
        return 'Bear drop'
HDFC_df['Trend']= np.zeros(HDFC_df['Day_Perc_Change'].count())
HDFC_df['Trend']= HDFC_df['Day_Perc_Change'].apply(lambda x:trend(x))
HDFC_df.head()


# In[17]:


HDFC_pie_data = HDFC_df.groupby('Trend')
pie_label = sorted([i for i in HDFC_df.loc[:, 'Trend'].unique()])
plt.pie(HDFC_pie_data['Trend'].count(), labels = pie_label, autopct = '%1.1f%%', radius = 2)

plt.show()


# In[18]:


plt.stem(HDFC_df['Date'], HDFC_df['Day_Perc_Change'])
(HDFC_df['Volume']/1000000).plot(figsize = (15, 7.5), color = 'green', alpha = 0.5)


# In[19]:



# set start and end dates 
start = datetime.datetime(2015, 11, 15)
end = datetime.datetime(2020, 11, 14) 
# extract the closing price data
combined_df = web.DataReader(['FMC', '^IXIC', '^DJI', '^GSPC'],
'yahoo', start = start, end = end)['Adj Close']


# In[20]:


# drop null values
combined_df.dropna(inplace = True, axis = 0)
# display first few rows
combined_df.head()


# In[21]:


# store daily returns of all above stocks in a new dataframe 
pct_chg_df = combined_df.pct_change()*100
pct_chg_df.dropna(inplace = True, how = 'any', axis = 0)
# plotting pairplot  
import seaborn as sns
sns.set(style = 'ticks', font_scale = 1.25)
sns.pairplot(pct_chg_df)


# In[22]:


HDFC_vol = pct_chg_df['FMC'].rolling(7).std()*np.sqrt(7)
HDFC_vol.plot(figsize = (15, 7))


# In[23]:


volatility = pct_chg_df[['FMC', '^DJI', '^IXIC', '^GSPC']].rolling(7).std()*np.sqrt(7)
volatility.plot(figsize = (15, 7))


# In[24]:


total_spent = 0
total_shares = 0
total_profit = 0
for index, value in HDFC_df['Adj Close'].items():
    total_spent += value *100
    total_shares += 100
    if str(index) == "2020-11-16 00:00:00":
        total_profit =value*total_shares - total_spent
        
print(total_spent)
print(value*total_shares)
print(total_profit)
        
    


# In[25]:


arr = HDFC_df['Adj Close'].array
l = len(arr)
max = 1
first = 0 
last = 0
maxFirst=0
maxLast=0
num = 0
while num<l:
    if num+1 < l and arr[num+1] > arr[num]:
        last +=1
        if (last-first+1) > max:
            max = last - first +1
            maxFirst = first
            maxLast = last
    else:
        first = num + 1
        last = num + 1
    num +=1

print( arr[maxFirst:maxLast+1])
        
    


# In[26]:


total_spent = 0
total_shares = 0
total_profit = 0
for i in arr[maxFirst:maxLast+1]:
    total_spent += i *100
    total_shares += 100
    print(i)
    if i == arr[maxLast]:
        total_profit =value*total_shares - total_spent
        
print(total_profit)


# In[27]:


DAL_df = pd.read_csv("/Users/ethan/Desktop/DAL.csv") 
DAL_df = DAL_df.round(2)
DAL_df.head()


# In[28]:


LUV_df = pd.read_csv("/Users/ethan/Desktop/LUV.csv") 
LUV_df = LUV_df.round(2)
LUV_df.head()


# In[29]:


DAL_df['Adj Close'].plot(figsize = (15,8))
plt.show()


# In[30]:


LUV_df['Adj Close'].plot(figsize = (15,8))
plt.show()


# In[31]:


DAL_df.dropna(inplace = True, axis = 0)
DAL_df['Day_Perc_Change'] = DAL_df['Adj Close'].pct_change()*100
DAL_df.dropna(inplace = True, axis = 0)


# In[32]:


DAL_df['Trend']= np.zeros(DAL_df['Day_Perc_Change'].count())
DAL_df['Trend']= DAL_df['Day_Perc_Change'].apply(lambda x:trend(x))
DAL_df.head()


# In[33]:


LUV_df.dropna(inplace = True, axis = 0)
LUV_df['Day_Perc_Change'] = LUV_df['Adj Close'].pct_change()*100
LUV_df.dropna(inplace = True, axis = 0)
LUV_df['Trend']= np.zeros(LUV_df['Day_Perc_Change'].count())
LUV_df['Trend']= LUV_df['Day_Perc_Change'].apply(lambda x:trend(x))
LUV_df.head()


# In[34]:


DAL_pie_data = DAL_df.groupby('Trend')
pie_label = sorted([i for i in DAL_df.loc[:, 'Trend'].unique()])
plt.pie(DAL_pie_data['Trend'].count(), labels = pie_label, autopct = '%1.1f%%', radius = 2)

plt.show()
LUV_pie_data = LUV_df.groupby('Trend')
pie_label = sorted([i for i in LUV_df.loc[:, 'Trend'].unique()])
plt.pie(LUV_pie_data['Trend'].count(), labels = pie_label, autopct = '%1.1f%%', radius = 2)

plt.show()


# In[35]:


print(DAL_df['Day_Perc_Change'].sum())
print(LUV_df['Day_Perc_Change'].sum())


# In[36]:


total_spent = 0
total_shares = 0
total_profit = 0
for index, value in LUV_df['Adj Close'].items():
    total_spent += value *100
    total_shares += 100
    if str(index) == "2020-11-27 00:00:00":
        total_profit =value*total_shares - total_spent
        
print(total_spent)
print(value*total_shares)
print(total_profit)


# In[37]:


total_spent = 0
total_shares = 0
total_profit = 0
for index, value in DAL_df['Adj Close'].items():
    total_spent += value *100
    total_shares += 100
    if str(index) == "2020-11-27 00:00:00":
        total_profit =value*total_shares - total_spent
        
print(total_spent)
print(value*total_shares)
print(total_profit)


# In[46]:


ticker = 'DAL'
timeframe = '5y'
df = p.chartDF(ticker, timeframe)
df = DAL_df[['Adj Close']]
df.reset_index(level=0, inplace=True)
df.columns=['ds','y']
plt.plot(df.ds, df.y)
plt.show()


# In[45]:


df = DAL_df[['Adj Close']]
rolling_mean = df.y.rolling(window=20).mean()
rolling_mean2 = df.y.rolling(window=50).mean()
plt.plot(df.ds, df.y, label='AMD')
plt.plot(df.ds, rolling_mean, label='AMD 20 Day SMA', color='orange')
plt.plot(df.ds, rolling_mean2, label='AMD 50 Day SMA', color='magenta')
plt.legend(loc='upper left')
plt.show()


# In[ ]:




