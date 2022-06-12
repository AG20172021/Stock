#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import datetime
import warnings
import pandas_datareader.data as web
warnings.filterwarnings('ignore')
import yahoo_fin.stock_info as ya
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests 
import numpy as np


# In[2]:


large = pd.read_csv("/Users/ethan/Desktop/large.csv")
large = large["Symbol"]
medium = pd.read_csv("/Users/ethan/Desktop/medium.csv")
medium = medium["Symbol"]
small = pd.read_csv("/Users/ethan/Desktop/small.csv")
small = small["Symbol"]
micro = pd.read_csv("/Users/ethan/Desktop/micro.csv")
micro = micro["Symbol"]
start1 = datetime.datetime(2000,1,1)
end1 = datetime.datetime(2022,1,1)


# In[3]:


str5 = ""
for x in large:
    str5 += x + " "
lar = yf.download(str5, start=start1, end=end1)


# In[4]:


str4 = ""
for x in medium:
    str4 += x + " "
med = yf.download(str4, start=start1, end=end1)


# In[5]:


str3 = ""
for x in small:
    str3 += x + " "
sma = yf.download(str3, start=start1, end=end1)


# In[6]:


str1 = ""
for x in micro:
    str1 += x + " "
mic = yf.download(str1, start=start1, end=end1)


# In[9]:


sp = yf.download("^GSPC",start=start1,end=end1)
spc= sp['Close'].to_numpy()


# In[7]:


larV = lar['Volume'].to_numpy()
larC = lar['Close'].to_numpy()
medV = med['Volume'].to_numpy()
medC = med['Close'].to_numpy()
smaV = sma['Volume'].to_numpy()
smaC = sma['Close'].to_numpy()
micV = mic['Volume'].to_numpy()
micC = mic['Close'].to_numpy()


# # Stop Loss

# In[ ]:


med1 = yf.download(str4, start ="2022-01-01", end ="2022-02-28" , interval="1h")
sma1 = yf.download(str3, start ="2022-01-01", end ="2022-02-28" , interval="1h")
lar1 = yf.download(str5, start ="2022-01-01", end ="2022-02-28" , interval="1h")
mic1 = yf.download(str1, start ="2022-01-01", end ="2022-02-28" , interval="1h")
larV1 = lar1['Volume'].to_numpy()
micV1 = mic1['Volume'].to_numpy()
smaV1 = sma1['Volume'].to_numpy()
medV1 = med1['Volume'].to_numpy()

larC1 = lar1['Close'].to_numpy()
micC1 = mic1['Close'].to_numpy()
smaC1 = sma1['Close'].to_numpy()
medC1 = med1['Close'].to_numpy()

med2 = yf.download(str4, start ="2021-11-01", end ="2021-12-30" , interval="1h")
sma2 = yf.download(str3, start ="2021-11-01", end ="2021-12-30" , interval="1h")
lar2 = yf.download(str5, start ="2021-11-01", end ="2021-12-30" , interval="1h")
mic2 = yf.download(str1, start ="2021-11-01", end ="2021-12-30" , interval="1h")
larV2 = lar2['Volume'].to_numpy()
micV2 = mic2['Volume'].to_numpy()
smaV2 = sma2['Volume'].to_numpy()
medV2 = med2['Volume'].to_numpy()

larC2 = lar2['Close'].to_numpy()
micC2 = mic2['Close'].to_numpy()
smaC2 = sma2['Close'].to_numpy()
medC2 = med2['Close'].to_numpy()

med3 = yf.download(str4, start ="2021-09-01", end ="2021-10-30" , interval="1h")
sma3 = yf.download(str3, start ="2021-09-01", end ="2021-10-30" , interval="1h")
lar3 = yf.download(str5, start ="2021-09-01", end ="2021-10-30" , interval="1h")
mic3 = yf.download(str1, start ="2021-09-01", end ="2021-10-30" , interval="1h")
larV3 = lar3['Volume'].to_numpy()
micV3 = mic3['Volume'].to_numpy()
smaV3 = sma3['Volume'].to_numpy()
medV3 = med3['Volume'].to_numpy()

larC3 = lar3['Close'].to_numpy()
micC3 = mic3['Close'].to_numpy()
smaC5 = sma3['Close'].to_numpy()
medC3 = med3['Close'].to_numpy()

med4 = yf.download(str4, start ="2021-07-01", end ="2021-08-30" , interval="1h")
sma4 = yf.download(str3, start ="2021-07-01", end ="2021-08-30" , interval="1h")
lar4 = yf.download(str5, start ="2021-07-01", end ="2021-08-30" , interval="1h")
mic4 = yf.download(str1, start ="2021-07-01", end ="2021-08-30" , interval="1h")
larV4 = lar4['Volume'].to_numpy()
micV4 = mic4['Volume'].to_numpy()
smaV4 = sma4['Volume'].to_numpy()
medV4 = med4['Volume'].to_numpy()

larC4 = lar4['Close'].to_numpy()
micC4 = mic4['Close'].to_numpy()
smaC4 = sma4['Close'].to_numpy()


# In[2]:


def nostoploss(data,percent,vol):
    arr = []
    lose_change=0
    countTotalBigJumps=0
    countNeg = 0
    
    

    for i in range(6,len(data)-14,7):
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+7][j]) or math.isnan(data[i+14][j]) or math.isnan(vol[i+14][j]) or math.isnan(vol[i+7][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+7][j] < data[i][j]*(1-percent) and data[i+7][j] * vol[i+7][j] > 50000):
                    countTotalBigJumps+=1
                    val = (data[i+7+7][j]-data[i+7][j])/data[i+7][j]
                    lose_change+=round(val,4)
                    if(val<0):
                        countNeg += 1
                
    
    #arr.append(lose_change)
    #arr.append(countTotalBigJumps)
    #return (arr)
    print(countTotalBigJumps)  
    print(countNeg/countTotalBigJumps)
    if(not(countTotalBigJumps==0)):
        print(lose_change/countTotalBigJumps)


# In[3]:


def stoploss(data,percent,vol,loss):
    lose_change=0
    countTotalBigJumps=0
    countNeg = 0
    
    

    for i in range(6,len(data)-14,7):
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+7][j]) or math.isnan(data[i+14][j]) or math.isnan(vol[i+14][j]) or math.isnan(vol[i+7][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+7][j] < data[i][j]*(1-percent) and data[i+7][j] * vol[i+7][j] > 50000):
                    countTotalBigJumps+=1
                    stoppedCount = 0
                    for x in range(6):
                        if(((data[i+7+x][j]-data[i+7][j])/data[i+7][j]) < (-loss)):
                            val = (data[i+7+x][j]-data[i+7][j])/data[i+7][j]
                            lose_change+=round(val,4)
                            stoppedCount = 1
                            break
                    if(stoppedCount == 0):
                        val = (data[i+7+7][j]-data[i+7][j])/data[i+7][j]
                        lose_change+=round(val,4)
                    if(val<0):
                        countNeg += 1
                

    return (lose_change/countTotalBigJumps)
    #print(countTotalBigJumps)  
    #print(countNeg/countTotalBigJumps)
    #if(not(countTotalBigJumps==0)):
    #    print(lose_change/countTotalBigJumps)


# In[4]:


def hr2(data,percent,vol):
    arr = []
    lose_change=0
    countTotalBigJumps=0
    countNeg = 0
    
    

    for i in range(6,len(data)-14,7):
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+7][j]) or math.isnan(data[i+14][j]) or math.isnan(vol[i+14][j]) or math.isnan(vol[i+7][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+7][j] < data[i][j]*(1-percent) and data[i+5][j] < data[i][j]*(.9) and data[i+7][j] * vol[i+7][j] > 50000):
                    countTotalBigJumps+=1
                    stoppedCount = 0

                    if(stoppedCount == 0):
                        val = (data[i+7+7][j]-data[i+7][j])/data[i+7][j]
                        lose_change+=round(val,4)
                    if(val<0):
                        countNeg += 1
                
    #arr.append(lose_change)
    #arr.append(countTotalBigJumps)
    #return (arr)
    print(countTotalBigJumps)  
    print(countNeg/countTotalBigJumps)
    if(not(countTotalBigJumps==0)):
        print(lose_change/countTotalBigJumps)


# In[26]:


dates = list(med.index)
dates2 = list(sp.index)
for i in range(len(dates)-2):
    if(dates2[i]!=dates[i] ):
        dates2.insert(i,dates[i])
        print(dates[i])
        print(dates2[i])
        print()
        #spc.insert(i,spc[i])
        continue
print(len(dates))
print(len(dates2))


# # Market Corr

# In[77]:


import math
def marketCorr(data,percent,days,frame,vol,market,ind,change):
    dates =list(frame.index)
    dates2 = list(ind.index)
    market = list(market)
    lose_change=0
    lose_change2=0
    lose_change3=0
    countTotalBigJumps=0
    countTotalBigJumps2=0
    countTotalBigJumps3=0
    countNeg = 0
    countNeg2 = 0
    countNeg3 = 0
    day =0
    
    

    for i in range(len(dates2)-(1+days)):
        if(dates2[i]!=dates[i] ):
            dates2.insert(i,dates[i])
            market.insert(i,market[i])
            continue
        if(market[i+1]<market[i]*(1-change)):
            day +=1
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j]) or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent) and data[i+1][j] * vol[i+1][j] > 200000 and market[i+1]<market[i]*(1-change)):
                    countTotalBigJumps2+=1
                    val2 = (data[i+1+days][j]-data[i+1][j])/data[i+1][j]
                    lose_change2+=round(val2,4)
                    
                    if(val2<0):
                        countNeg2 += 1

                     
    print(day)
    
    print(countTotalBigJumps2)  
    if(not(countTotalBigJumps2==0)): 
        print(countNeg2/countTotalBigJumps2)
        print(lose_change2/countTotalBigJumps2)
        

        


# In[129]:


count = 0
for i in range(len(sp)-466,len(sp)-444):
    if(spc[i+1]<spc[i]*(1-.01)):

        print(sp.index[i+1])
        


# In[155]:



def marketCorr2(data,percent,days,frame,vol,market,ind,change):
    col = frame.columns.values
    dates =list(frame.index)
    dates2 = list(ind.index)
    market = list(market)
    lose_change2=0
    countTotalBigJumps2=0
    countNeg2 = 0
    countNeg = 0
    day =0
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    most_neg = list()
    change2 =0
    tradesperday = list()
    

    for i in range(len(dates2)-2530,len(dates2)-(1+days)-506):
        if(dates2[i]!=dates[i] ):
            dates2.insert(i,dates[i])
            market.insert(i,market[i])
            continue
        if(market[i+1]<market[i]*(1-change)):
            day +=1
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j]) or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent) and data[i+1][j] * vol[i+1][j] > 200000 and market[i+1]<market[i]*(1-change)): #
                    countTotalBigJumps2+=1
                    val2 = (data[i+1+days][j]-data[i+1][j])/data[i+1][j]
                    lose_change2+=round(val2,4)
                    change2 +=val2
                    
                    
                    if(change2<=0):
                        most_neg.append(change2)
                            
                    if(change2>0):
                        most_neg.append(0)
                        change2 = 0
                    
                    if(val2<0):
                        countNeg += 1
                        countNeg2 += 1
                        list_changes_neg.append(countNeg)
                        neg.append(val2+neg[countNeg2-1])
                    
                    if(val2>=0):
                        list_changes_neg.append(0)
                        countNeg=0
                        neg.append(val2+neg[countNeg2-1])
                    
                        
                    print(col[j])
                    print(dates2[i+1])
                    print(market[i])
                    print(market[i+1])
                    print(data[i+1][j])
                    print(data[i+2][j])
                    print()
                     
    print()
    print(day)
    print(countTotalBigJumps2) 
    print("most possible lose: " + str(min(most_neg)))
    print(max(list_changes_neg))
    print(min(neg))
    if(not(countTotalBigJumps2==0)): 
        print(countNeg2/countTotalBigJumps2)
        print(lose_change2/countTotalBigJumps2)
        

        


# In[156]:



print("large")
marketCorr2(larC,0.15,1,lar,larV,spc,sp,0.02)
print("medium")
marketCorr2(medC,0.15,1,med,medV,spc,sp,0.02)
print("small")
marketCorr2(smaC,0.15,1,sma,smaV,spc,sp,0.02)
print("micro")
marketCorr2(micC,0.15,1,mic,micV,spc,sp,0.02)


# In[79]:


marketCorr(larC,0.15,1,lar,larV,spc,sp,0.01)
marketCorr(medC,0.15,1,med,medV,spc,sp,0.01)
marketCorr(smaC,0.15,1,sma,smaV,spc,sp,0.01)
marketCorr(micC,0.15,1,mic,micV,spc,sp,0.01)


# In[66]:


marketCorr(larC,0.1,1,lar,larV,spc,sp,0.01)
marketCorr(medC,0.1,1,med,medV,spc,sp,0.01)
marketCorr(smaC,0.1,1,sma,smaV,spc,sp,0.01)
marketCorr(micC,0.1,1,mic,micV,spc,sp,0.01)


# In[69]:


marketCorr(larC,0.15,1,lar,larV,spc,sp,0.02)
marketCorr(medC,0.15,1,med,medV,spc,sp,0.02)
marketCorr(smaC,0.15,1,sma,smaV,spc,sp,0.02)
marketCorr(micC,0.15,1,mic,micV,spc,sp,0.02)


# In[70]:


marketCorr(larC,0.1,1,lar,larV,spc,sp,0.02)
marketCorr(medC,0.1,1,med,medV,spc,sp,0.02)
marketCorr(smaC,0.1,1,sma,smaV,spc,sp,0.02)
marketCorr(micC,0.1,1,mic,micV,spc,sp,0.02)


# In[71]:


marketCorr(larC,0.1,1,lar,larV,spc,sp,0.01)
marketCorr(medC,0.1,1,med,medV,spc,sp,0.01)
marketCorr(smaC,0.1,1,sma,smaV,spc,sp,0.01)
marketCorr(micC,0.1,1,mic,micV,spc,sp,0.01)


# # Filter by Industry

# In[91]:


consumer = pd.read_csv("/Users/ethan/Desktop/Consumer.csv")
consumer = consumer["Symbol"]
capital = pd.read_csv("/Users/ethan/Desktop/CapitalGoods.csv")
capital = capital["Symbol"]
basic = pd.read_csv("/Users/ethan/Desktop/Basic.csv")
basic = basic["Symbol"]
energy = pd.read_csv("/Users/ethan/Desktop/Energy.csv")
energy = energy["Symbol"]
finance = pd.read_csv("/Users/ethan/Desktop/Finance.csv")
finance = finance["Symbol"]
health = pd.read_csv("/Users/ethan/Desktop/Health.csv")
health = health["Symbol"]
trans = pd.read_csv("/Users/ethan/Desktop/Transportation.csv")
trans = trans["Symbol"]
tech = pd.read_csv("/Users/ethan/Desktop/Technology.csv")
tech = tech["Symbol"]
util = pd.read_csv("/Users/ethan/Desktop/Utilities.csv")
util = util["Symbol"]
misc = pd.read_csv("/Users/ethan/Desktop/Misc.csv")
misc = misc["Symbol"]
str4 = ""
for x in consumer:
    str4 += x + " "
str3 = ""
for x in capital:
    str3 += x + " "
str2 = ""
for x in basic:
    str2 += x + " "
str1 = ""
for x in energy:
    str1 += x + " "
str5 = ""
for x in finance:
    str5 += x + " "
str6 = ""
for x in health:
    str6 += x + " "
str7 = ""
for x in trans:
    str7 += x + " "
str8 = ""
for x in tech:
    str8 += x + " "
str9 = ""
for x in util:
    str9 += x + " "
str10 = ""
for x in misc:
    str10 += x + " "
   
   
consumer = yf.download(str4, start=start1, end=end1)
capital = yf.download(str3, start=start1, end=end1)
basic = yf.download(str2, start=start1, end=end1)
enrg = yf.download(str1, start=start1, end=end1)
fin = yf.download(str5, start=start1, end=end1)
health = yf.download(str6, start=start1, end=end1)
trans = yf.download(str7, start=start1, end=end1)
tech = yf.download(str8, start=start1, end=end1)
util = yf.download(str9, start=start1, end=end1)
misc = yf.download(str10, start=start1, end=end1)


# In[95]:


consumerC = consumer['Close'].to_numpy()
capitalC = capital['Close'].to_numpy()
basicC = basic['Close'].to_numpy()
enrgC = enrg['Close'].to_numpy()
finC = fin['Close'].to_numpy()
healthC = health['Close'].to_numpy()
transC = trans['Close'].to_numpy()
techC = tech['Close'].to_numpy()
utilC = util['Close'].to_numpy()
miscC = misc['Close'].to_numpy()

consumerV = consumer['Volume'].to_numpy()
capitalV = capital['Volume'].to_numpy()
basicV = basic['Volume'].to_numpy()
enrgV = enrg['Volume'].to_numpy()
finV = fin['Volume'].to_numpy()
healthV = health['Volume'].to_numpy()
transV = trans['Volume'].to_numpy()
techV = tech['Volume'].to_numpy()
utilV = util['Volume'].to_numpy()
miscV = misc['Volume'].to_numpy()


# In[92]:



def marketCorr3(data,percent,days,frame,vol,market,ind,change):
    col = frame.columns.values
    dates =list(frame.index)
    dates2 = list(ind.index)
    market = list(market)
    lose_change2=0
    countTotalBigJumps2=0
    countNeg2 = 0
    countNeg = 0
    day =0
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    most_neg = list()
    change2 =0
    

    for i in range(len(dates2)-253,len(dates2)-(1+days)):
        if(dates2[i]!=dates[i] ):
            dates2.insert(i,dates[i])
            market.insert(i,market[i])
            continue
        if(market[i+1]<market[i]*(1-change)):
            day +=1
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j]) or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent) and data[i+1][j] * vol[i+1][j] > 200000 and market[i+1]<market[i]*(1-change)):
                    countTotalBigJumps2+=1
                    val2 = (data[i+1+days][j]-data[i+1][j])/data[i+1][j]
                    lose_change2+=round(val2,4)
                    change2 +=val2
                    
                    
                    if(change2<=0):
                        most_neg.append(change2)
                            
                    if(change2>0):
                        most_neg.append(0)
                        change2 = 0
                    
                    if(val2<0):
                        countNeg += 1
                        countNeg2 += 1
                        list_changes_neg.append(countNeg)
                        neg.append(val2+neg[countNeg2-1])
                    
                    if(val2>=0):
                        list_changes_neg.append(0)
                        countNeg=0
                        neg.append(val2+neg[countNeg2-1])
                    
                        
                   
    print()
    print(day)
    print(countTotalBigJumps2) 
    print("most possible lose: " + str(min(most_neg)))
    print(max(list_changes_neg))
    print(min(neg))
    if(not(countTotalBigJumps2==0)): 
        print(countNeg2/countTotalBigJumps2)
        print(lose_change2/countTotalBigJumps2)
        

        


# In[93]:



def marketCorr4(data,percent,days,frame,vol,market,ind,change):
    col = frame.columns.values
    dates =list(frame.index)
    dates2 = list(ind.index)
    market = list(market)
    lose_change2=0
    countTotalBigJumps2=0
    countNeg2 = 0
    countNeg = 0
    day =0
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    most_neg = list()
    change2 =0
    

    for i in range(len(dates2)-506,len(dates2)-(1+days)):
        if(dates2[i]!=dates[i] ):
            dates2.insert(i,dates[i])
            market.insert(i,market[i])
            continue
        if(market[i+1]<market[i]*(1-change)):
            day +=1
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j]) or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent) and data[i+1][j] * vol[i+1][j] > 200000 and market[i+1]<market[i]*(1-change)):
                    countTotalBigJumps2+=1
                    val2 = (data[i+1+days][j]-data[i+1][j])/data[i+1][j]
                    lose_change2+=round(val2,4)
                    change2 +=val2
                    
                    
                    if(change2<=0):
                        most_neg.append(change2)
                            
                    if(change2>0):
                        most_neg.append(0)
                        change2 = 0
                    
                    if(val2<0):
                        countNeg += 1
                        countNeg2 += 1
                        list_changes_neg.append(countNeg)
                        neg.append(val2+neg[countNeg2-1])
                    
                    if(val2>=0):
                        list_changes_neg.append(0)
                        countNeg=0
                        neg.append(val2+neg[countNeg2-1])
                    
                        
                   
    print()
    print(day)
    print(countTotalBigJumps2) 
    print("most possible lose: " + str(min(most_neg)))
    print(max(list_changes_neg))
    print(min(neg))
    if(not(countTotalBigJumps2==0)): 
        print(countNeg2/countTotalBigJumps2)
        print(lose_change2/countTotalBigJumps2)
        

        


# In[98]:


marketCorr2(consumerC,.15,1,consumer['Volume'],consumerV,spc,sp,0.01)
marketCorr2(capitalC,.15,1,capital['Volume'],capitalV,spc,sp,0.01)
marketCorr2(basicC,.15,1,basic['Volume'],basicV,spc,sp,0.01)
marketCorr2(enrgC,.15,1,enrg['Volume'],enrgV,spc,sp,0.01)
marketCorr2(finC,.15,1,fin['Volume'],finV,spc,sp,0.01)
marketCorr2(techC,.15,1,tech['Volume'],techV,spc,sp,0.01)
marketCorr2(utilC,.15,1,util['Volume'],utilV,spc,sp,0.01)
marketCorr2(miscC,.15,1,misc['Volume'],miscV,spc,sp,0.01)


# In[99]:


marketCorr3(consumerC,.15,1,consumer['Volume'],consumerV,spc,sp,0.01)
marketCorr3(capitalC,.15,1,capital['Volume'],capitalV,spc,sp,0.01)
marketCorr3(basicC,.15,1,basic['Volume'],basicV,spc,sp,0.01)
marketCorr3(enrgC,.15,1,enrg['Volume'],enrgV,spc,sp,0.01)
marketCorr3(finC,.15,1,fin['Volume'],finV,spc,sp,0.01)
marketCorr3(techC,.15,1,tech['Volume'],techV,spc,sp,0.01)
marketCorr3(utilC,.15,1,util['Volume'],utilV,spc,sp,0.01)
marketCorr3(miscC,.15,1,misc['Volume'],miscV,spc,sp,0.01)


# In[100]:


marketCorr4(consumerC,.15,1,consumer['Volume'],consumerV,spc,sp,0.01)
marketCorr4(capitalC,.15,1,capital['Volume'],capitalV,spc,sp,0.01)
marketCorr4(basicC,.15,1,basic['Volume'],basicV,spc,sp,0.01)
marketCorr4(enrgC,.15,1,enrg['Volume'],enrgV,spc,sp,0.01)
marketCorr4(finC,.15,1,fin['Volume'],finV,spc,sp,0.01)
marketCorr4(techC,.15,1,tech['Volume'],techV,spc,sp,0.01)
marketCorr4(utilC,.15,1,util['Volume'],utilV,spc,sp,0.01)
marketCorr4(miscC,.15,1,misc['Volume'],miscV,spc,sp,0.01)


# In[101]:


marketCorr2(healthC,.15,1,health['Volume'],healthV,spc,sp,0.01)
marketCorr2(transC,.15,1,trans['Volume'],transV,spc,sp,0.01)
marketCorr3(healthC,.15,1,health['Volume'],healthV,spc,sp,0.01)
marketCorr3(transC,.15,1,trans['Volume'],transV,spc,sp,0.01)
marketCorr4(healthC,.15,1,health['Volume'],healthV,spc,sp,0.01)
marketCorr4(transC,.15,1,trans['Volume'],transV,spc,sp,0.01)


# # All together

# In[ ]:


def gainloseNew(data,percent,days,frame,vol):
    col = frame.columns.values
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    lose_change=0
    countTotalBigJumps=0
    countNeg = 0
    countNeg2 = 0
    

    for i in range(len(data)-(2+days)):
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j]) or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent) and data[i][j] * vol[i][j] > 200000): #
                    countTotalBigJumps+=1
                    val = (data[i+1+days][j]-data[i+1][j])/data[i+1][j]
                    lose_change+=round(val,4)
                    
                    if(val<0):
                        countNeg += 1
                        countNeg2 += 1
                        list_changes_neg.append(countNeg2)
                        neg.append(val+neg[countNeg-1])
                    
                    if(val>=0):
                        list_changes_neg.append(0)
                        countNeg2=0
                        neg.append(val+neg[countNeg-1])
                    
                        
                    
                    #print(col[j])
                    #print(i)
                    #print(data[i+1][j])
                    #print(data[i+2][j])
                    #print(val)
                    #print()
                

    print(countTotalBigJumps)  
    print(countNeg/countTotalBigJumps)
    print(max(list_changes_neg))
    print(min(neg))
    if(not(countTotalBigJumps==0)):
        print(lose_change/countTotalBigJumps)
        
    print()
    print()


# In[157]:



def gains(data,percent,days,frame,vol):
    col = frame.columns.values
    dates =list(frame.index)
    dates2 = list(ind.index)
    market = list(market)
    lose_change2=0
    countTotalBigJumps2=0
    countNeg2 = 0
    countNeg = 0
    day =0
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    most_neg = list()
    change2 =0
    tradesperday = list()
    

    for i in range(len(dates)-2530,len(dates)-(1+days)-506):
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j]) or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent) and data[i+1][j] * vol[i+1][j] > 200000 ):
                    countTotalBigJumps2+=1
                    val2 = (data[i+1+days][j]-data[i+1][j])/data[i+1][j]
                    lose_change2+=round(val2,4)
                    change2 +=val2
                    
                    
                    if(change2<=0):
                        most_neg.append(change2)
                            
                    if(change2>0):
                        most_neg.append(0)
                        change2 = 0
                    
                    if(val2<0):
                        countNeg += 1
                        countNeg2 += 1
                        list_changes_neg.append(countNeg)
                        neg.append(val2+neg[countNeg2-1])
                    
                    if(val2>=0):
                        list_changes_neg.append(0)
                        countNeg=0
                        neg.append(val2+neg[countNeg2-1])
                    
                        
                    #print(col[j])
                    #print(dates2[i+1])
                    #print(market[i])
                    #print(market[i+1])
                    #print(data[i+1][j])
                    #print(data[i+2][j])
                    #print()
                     
    print()
    print(day)
    print(countTotalBigJumps2) 
    print("most possible lose: " + str(min(most_neg)))
    print(max(list_changes_neg))
    print(min(neg))
    if(not(countTotalBigJumps2==0)): 
        print(countNeg2/countTotalBigJumps2)
        print(lose_change2/countTotalBigJumps2)
        

        


# look at when market is positive and when stocks are up (shorting)
# 
# look at no market correlation

# In[ ]:


print("large")
gains(larC,0.15,1,lar,larV)
print("medium")
gains(medC,0.15,1,med,medV)
print("small")
gains(smaC,0.15,1,sma,smaV)
print("micro")
gains(micC,0.15,1,mic,micV)


# In[ ]:



def marketCorrUp(data,percent,days,frame,vol,market,ind,change):
    col = frame.columns.values
    dates =list(frame.index)
    dates2 = list(ind.index)
    market = list(market)
    lose_change2=0
    countTotalBigJumps2=0
    countNeg2 = 0
    countNeg = 0
    day =0
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    most_neg = list()
    change2 =0
    tradesperday = list()
    

    for i in range(len(dates2)-2530,len(dates2)-(1+days)-506):
        if(dates2[i]!=dates[i] ):
            dates2.insert(i,dates[i])
            market.insert(i,market[i])
            continue
        if(market[i+1]<market[i]*(1-change)):
            day +=1
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j]) or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1+percent) and data[i+1][j] * vol[i+1][j] > 200000 and market[i+1]<market[i]*(1+change)): #
                    countTotalBigJumps2+=1
                    val2 = (data[i+1+days][j]-data[i+1][j])/data[i+1][j]
                    lose_change2+=round(val2,4)
                    change2 +=val2
                    
                    
                    if(change2<=0):
                        most_neg.append(change2)
                            
                    if(change2>0):
                        most_neg.append(0)
                        change2 = 0
                    
                    if(val2<0):
                        countNeg += 1
                        countNeg2 += 1
                        list_changes_neg.append(countNeg)
                        neg.append(val2+neg[countNeg2-1])
                    
                    if(val2>=0):
                        list_changes_neg.append(0)
                        countNeg=0
                        neg.append(val2+neg[countNeg2-1])
                    
                        
                    #print(col[j])
                    #print(dates2[i+1])
                    #print(market[i])
                    #print(market[i+1])
                    #print(data[i+1][j])
                    #print(data[i+2][j])
                    #print()
                     
    print()
    print(day)
    print(countTotalBigJumps2) 
    print("most possible lose: " + str(min(most_neg)))
    print(max(list_changes_neg))
    print(min(neg))
    if(not(countTotalBigJumps2==0)): 
        print(countNeg2/countTotalBigJumps2)
        print(lose_change2/countTotalBigJumps2)
        

        


# In[ ]:


marketCorrUp(larC,0.1,1,lar,larV,spc,sp,0.01)
marketCorrUp(medC,0.1,1,med,medV,spc,sp,0.01)
marketCorrUp(smaC,0.1,1,sma,smaV,spc,sp,0.01)
marketCorrUp(micC,0.1,1,mic,micV,spc,sp,0.01)


# In[ ]:





# Avg Stocks per Day and deviation of stocks per day
# - finding the gains based on how many sotcks went down
# 
# Stocks Up market up
# 
# Group of stocks with something in common
# - geography
# - industry
# 
# Change on the hour day or week scale that stocks have in common
# - market up or down
# - price up or down
# - duration of stock being down or up
# - earnings calls
# - IPOs
# - news 
# - indicators change
#     - moving avg
#     - volume
#     - rsi
#     - pattern trading
# - look at fundamentals
# - trade sets of numbers (options) that historically succeed and make any trade that matches these numbers
#     
# Find the pattern that emerges with multiple industry 
# 
# - machine learning programming
# - evolutionary programming
# 

# In[ ]:




