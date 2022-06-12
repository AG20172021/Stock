
# market movers

# In[1]:


import yfinance as yf
import datetime
import warnings
import pandas_datareader.data as web
warnings.filterwarnings('ignore')
import yahoo_fin.stock_info as ya
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests 
import numpy as np


# In[2]:


import math
def gainlose(data,percent,days,frame,vol):
    col = frame.columns.values
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    lose_change=0
    count1=0
    countNeg = 0
    countNeg2 = 0
    

    for i in range(len(data)-(2+days)):
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j]) ):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent)): #and data[i][j] * vol[i][j] > 10000):
                    count1+=1
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
                        neg.append(0)
                        
                    
                    print(col[j])
                    print(i)
                    print(data[i+1][j])
                    print(data[i+2][j])
                    print()
                

    print()

    print(count1)  
    print(countNeg/count1)
    print(max(list_changes_neg))
    print(min(neg))
    if(not(count1==0)):
        print(lose_change/count1)
        
    


# In[34]:


def gainloseNew(data,percent,days,frame):
    col = frame.columns.values
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    lose_change=0
    countTotalBigJumps=0
    countNeg = 0
    countNeg2 = 0
    countPerDay =0
    

    for i in range(len(data)-(2+days)):
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j])): #or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent)): #and data[i][j] * vol[i+1][j] > 200000): 
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
                        
                    countPerDay +=1
#                     print(col[j])
#                     print(i)
#                     print(data[i+1][j])
#                     print(data[i+2][j])
#                     print(val)
                    
#         if(countPerDay >60):
#             print(frame.index[i])
#             print(i)
#             print("Count Per Day: ")
#             print(countPerDay)
#             print("")
#         countPerDay=0
                

    print(countTotalBigJumps)  
    print(countNeg/countTotalBigJumps)
    print(max(list_changes_neg))
    print(min(neg))
    if(not(countTotalBigJumps==0)):
        print(lose_change/countTotalBigJumps)
        
    print()
    print()


# In[86]:


def r2008(data,percent,days,frame,vol):
    col = frame.columns.values
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    lose_change=0
    countTotalBigJumps=0
    countNeg = 0
    countNeg2 = 0
    countPerDay =0
    money =0

    for i in range(2200,2310):
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j])or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])): #):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent) and data[i][j] * vol[i+1][j] > 1000000): #): 
                    countTotalBigJumps+=1
                    val = (data[i+1+days][j]-data[i+1][j])/data[i+1][j]
                    lose_change+=round(val,4)
                    if(val<0):
                        money += 10000*(1-round(val,4))
                        countNeg += 1
                        countNeg2 += 1
                        list_changes_neg.append(countNeg2)
                        neg.append(val+neg[countNeg-1])
                    
                    if(val>=0):
                        money += 10000*(1+round(val,4))
                        list_changes_neg.append(0)
                        countNeg2=0
                        neg.append(val+neg[countNeg-1])
                        
                    countPerDay +=1
                    #print(col[j])
                    #print(i)
                    #print(data[i+1][j])
                    #print(data[i+2][j])
                    #print(val)
                    #print() "Day: "+ str(data[i]) + 
                    
#         if(countPerDay >60):
#             print(frame.index[i])
#             print(i)
#             print("Count Per Day: ")
#             print(countPerDay)
#             print("")
#         countPerDay=0
                
    if(countTotalBigJumps>0):
        arr = [countTotalBigJumps,money,money/countTotalBigJumps,lose_change/countTotalBigJumps]
        return arr

    else:
        return 0
#     print(countTotalBigJumps)  
#     print(countNeg/countTotalBigJumps)
#     print(max(list_changes_neg))
#     print(min(neg))
#     print(money)
#     print(money/countTotalBigJumps)
#     if(not(countTotalBigJumps==0)):
#         print(lose_change/countTotalBigJumps)
        
#     print()
#     print()


# In[87]:


def r2020(data,percent,days,frame,vol):
    col = frame.columns.values
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    lose_change=0
    countTotalBigJumps=0
    countNeg = 0
    countNeg2 = 0
    countPerDay =0
    money =0
    

    for i in range(5075,5170):
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j])or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])): #):
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent) and data[i][j] * vol[i+1][j] > 1000000): #): 
                    countTotalBigJumps+=1
                    val = (data[i+1+days][j]-data[i+1][j])/data[i+1][j]
                    lose_change+=round(val,4)
                    
                    if(val<0):
                        money += 10000*(1-round(val,4))
                        countNeg += 1
                        countNeg2 += 1
                        list_changes_neg.append(countNeg2)
                        neg.append(val+neg[countNeg-1])
                    
                    if(val>=0):
                        money += 10000*(1+round(val,4))
                        list_changes_neg.append(0)
                        countNeg2=0
                        neg.append(val+neg[countNeg-1])
                        
                    countPerDay +=1
                    #print(col[j])
                    #print(i)
                    #print(data[i+1][j])
                    #print(data[i+2][j])
                    #print(val)
                    #print() "Day: "+ str(data[i]) + 
                    
#         if(countPerDay >60):
#             print(frame.index[i])
#             print(i)
#             print("Count Per Day: ")
#             print(countPerDay)
#             print("")
#         countPerDay=0
    
    if(countTotalBigJumps>0):
        arr = [countTotalBigJumps,money,money/countTotalBigJumps,lose_change/countTotalBigJumps]
        return arr

    else:
        return 0
#     print(countTotalBigJumps)  
#     print(countNeg/countTotalBigJumps)
#     print(max(list_changes_neg))
#     print(min(neg))
#     print(money)
#     print(money/countTotalBigJumps)
#     if(not(countTotalBigJumps==0)):
#         print(lose_change/countTotalBigJumps)
        
#     print()
#     print()


# In[88]:


def All(data,percent,days,frame,vol):
    col = frame.columns.values
    list_changes_neg= list()
    neg= list()
    neg.append(0)
    lose_change=0
    countTotalBigJumps=0
    countNeg = 0
    countNeg2 = 0
    countPerDay =0
    money =0

    for i in range(5600):
        for j in range(len(data[i])):
            if(math.isnan(data[i][j]) or math.isnan(data[i+1][j]) or math.isnan(data[i+1+days][j]) or math.isnan(vol[i+1+days][j]) or math.isnan(vol[i+1][j]) or math.isnan(vol[i][j])): 
                continue
            else:
                if(data[i+1][j] < data[i][j]*(1-percent) and data[i][j] * vol[i+1][j] > 1000000):
                    countTotalBigJumps+=1
                    val = (data[i+1+days][j]-data[i+1][j])/data[i+1][j]
                    lose_change+=round(val,4)
                    if(val<0):
                        money += 10000*(1-round(val,4))
                        countNeg += 1
                        countNeg2 += 1
                        list_changes_neg.append(countNeg2)
                        neg.append(val+neg[countNeg-1])
                    
                    if(val>=0):
                        money += 10000*(1+round(val,4))
                        list_changes_neg.append(0)
                        countNeg2=0
                        neg.append(val+neg[countNeg-1])
                        
                    countPerDay +=1
                    #print(col[j])
                    #print(i)
                    #print(data[i+1][j])
                    #print(data[i+2][j])
                    #print(val)
                    #print() "Day: "+ str(data[i]) + 
                    
#         if(countPerDay >60):
#             print(frame.index[i])
#             print(i)
#             print("Count Per Day: ")
#             print(countPerDay)
#             print("")
#         countPerDay=0
                
    if(countTotalBigJumps>0):
        arr = [countTotalBigJumps,money,money/countTotalBigJumps,lose_change/countTotalBigJumps]
        return arr

    else:
        return 0

#     print(countTotalBigJumps)  
#     print(countNeg/countTotalBigJumps)
#     print(max(list_changes_neg))
#     print(min(neg))
#     print(money)
#     print(money/countTotalBigJumps)
#     if(not(countTotalBigJumps==0)):
#         print(lose_change/countTotalBigJumps)
        
#     print()
#     print()


# In[16]:


# mega
mega = pd.read_csv("/Users/ethan/Desktop/mega.csv")
mega = mega["Symbol"]
large = pd.read_csv("/Users/ethan/Desktop/large.csv")
large = large["Symbol"]
medium = pd.read_csv("/Users/ethan/Desktop/medium.csv")
medium = medium["Symbol"]
small = pd.read_csv("/Users/ethan/Desktop/small.csv")
small = small["Symbol"]
micro = pd.read_csv("/Users/ethan/Desktop/micro.csv")
micro = micro["Symbol"]
nano = pd.read_csv("/Users/ethan/Desktop/nano.csv")
nano = nano["Symbol"]

# start1 = datetime.datetime(2000,1,1)
# end1 = datetime.datetime(2022,5,1)
start1 = datetime.datetime(2022,4,1)
end1 = datetime.datetime(2022,6,12)


# In[7]:


str = ""
for x in mega:
    str += x + " "
   
   
dataMega = yf.download(str, start=start1, end=end1)
arrayMega = dataMega.to_numpy()


# In[17]:


str = ""
for x in large:
    str += x + " "
   
   
dataLarge = yf.download(str, start=start1, end=end1)
arrayLarge = dataLarge.to_numpy()


# In[75]:


str = ""
for x in large:
    str += x + " "
   
   
lv = yf.download(str, start=start1, end=end1)['Volume']
alv = dataLarge.to_numpy()


# In[19]:


str = ""
for x in medium:
    str += x + " "
   
   
dataMedium = yf.download(str, start=start1, end=end1)
arrayMedium = dataMedium.to_numpy()


# In[23]:


str = ""
for x in small:
    str += x + " "
dataSmall = yf.download(str, start=start1, end=end1)

arraySmall = dataSmall.to_numpy()


# In[77]:


str = ""
for x in small:
    str += x + " "
 

sv = yf.download(str, start=start1, end=end1)['Volume']

asv = dataSmall.to_numpy()


# In[24]:


str = ""
for x in micro:
    str += x + " "
   
   
dataMicro = yf.download(str, start=start1, end=end1)
arrayMicro = dataMicro.to_numpy()


# In[78]:


str = ""
for x in micro:
    str += x + " "
   
   
miv = yf.download(str, start=start1, end=end1)['Volume']
amiv = dataMicro.to_numpy()


# In[ ]:


str = ""
for x in nano:
    str += x + " "
   
   
dataNano = yf.download(str, start=start2, end=end2)
arrayNano = dataNano.to_numpy()


# In[28]:





# In[35]:


gainloseNew(dataLarge['Close'].to_numpy(),.15,1,dataLarge['Close'])
gainloseNew(dataMedium['Close'].to_numpy(),.15,1,dataMedium['Close'])
gainloseNew(dataSmall['Close'].to_numpy(),.15,1,dataSmall['Close'])
gainloseNew(dataMicro['Close'].to_numpy(),.15,1,dataMicro['Close'])
# gainloseNew(arrayNano,.15,1,dataNano)


# In[90]:


# mg8 = r2008(arrayMega,.15,1,dataMega)
# mg2=r2020(arrayMega,.15,1,dataMega)
l8=r2008(arrayLarge,.15,1,dataLarge,alv)
l2=r2020(arrayLarge,.15,1,dataLarge,alv)
me8=r2008(arrayMedium,.15,1,dataMedium,amv)
me2=r2020(arrayMedium,.15,1,dataMedium,amv)
s8=r2008(arraySmall,.15,1,dataSmall,asv)
s2=r2020(arraySmall,.15,1,dataSmall,asv)
mi8=r2008(arrayMicro,.15,1,dataMicro,amiv)
mi2=r2020(arrayMicro,.15,1,dataMicro,amiv)
# n8=r2008(arrayNano,.15,1,dataNano)
# n2=r2020(arrayNano,.15,1,dataNano)

arrs = [l8,l2,me8,me2,s8,s2,mi8,mi2]#,n8,n2,mg8,mg2

count =0
money = 0
change = 0
for i in arrs:
    if(i==0):
        continue
    count += i[0]
    money += i[1]
    change += i[0]*i[3]
    
    
print(count)
print(money-count*10000)
print(change/count)


# In[73]:


l=All(arrayLarge,.15,1,dataLarge,alv)
me=All(arrayMedium,.15,1,dataMedium,amv)
s=All(arraySmall,.15,1,dataSmall,asv)
mi=All(arrayMicro,.15,1,dataMicro,amiv)

arrs = [l,me,s,mi]

count =0
money = 0
change = 0
for i in arrs:
    count += i[0]
    money += i[1]
    change += i[0]*i[3]
    
    
print(count)
print(money-count*10000)
print(change/count)

