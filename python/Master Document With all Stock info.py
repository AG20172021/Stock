#!/usr/bin/env python
# coding: utf-8

# ## Main Indicators to for each Stock 
# 
# - RSI
# 
# The Relative Strength Index (RSI) was published by J. Welles Wilder. The current price is normalized as a percentage between 0 and 100. The name of this oscillator is misleading because it does not compare the instrument relative to another instrument or set of instruments, but rather represents the current price relative to other recent pieces within the selected lookback window length.
# Formula
# RSI = 100 - (100 / (1 + RS))
# Where: RS = ratio of smoothed average of n-period gains divided by the absolute value of the smoothed average of n-period losses.
# 
# - OBV
# 
# On Balance Volume (OBV) maintains a cumulative running total of the amount of volume occurring on up periods compared to down periods.
# Formula
# OBV = Cumulative (Up Volume - Down Volume)
# Where:
# Volume = Actual, Tick
# Up Volume = Quantity of volume occurring on up price change
# Down Volume = Quantity of volume occurring on down price change
# 
# - MACD
# 
# The Moving Average Convergence Divergence (MACD) was developed by Gerald Appel, and is based on the differences between two moving averages of different lengths, a Fast and a Slow moving average. A second line, called the Signa” line is plotted as a moving average of the MACD. A third line, called the MACD Histogram is optionally plotted as a histogram of the difference between the MACD and the Signal Line.
# Formula
# MACD = FastMA - SlowMA
# Where:
# FastMA is the shorter moving average and SlowMA is the longer moving average.
# SignalLine = MovAvg (MACD)
# MACD Histogram = MACD - SignalLine
# 
# - simple moving averages
# 
# The Simple Moving Average (SMA) is calculated by adding the price of an instrument over a number of time periods and then dividing the sum by the number of time periods. The SMA is basically the average price of the given time period, with equal weighting given to the price of each period.
# Formula
# SMA = ( Sum ( Price, n ) ) / n    
# Where: n = Time Period
# 
# - exponential moving avereages
# 
# The Exponential Moving Average (EMA) represents an average of prices, but places more weight on recent prices. The weighting applied to the most recent price depends on the selected period of the moving average. The shorter the period for the EMA, the more weight that will be applied to the most recent price.
# Formula
# EMA = ( P - EMAp ) * K + EMAp
# Where:
# P = Price for the current period
# EMAp = the Exponential moving Average for the previous period
# K = the smoothing constant, equal to 2 / (n + 1)
# n = the number of periods in a simple moving average roughly approximated by the EMA
# 
# - stochastic 
# 
# The Stochastic (Stoch) normalizes price as a percentage between 0 and 100. Normally two lines are plotted, the %K line and a moving average of the %K which is called %D. A slow stochastic can be created by initially smoothing the %K line with a moving average before it is displayed. The length of this smoothing is set in the Slow K Period. Without the initial smoothing ( i.e., setting the Slow K Period to a value of 1 ) the %K becomes the ‘Raw %K’ value, and is also known as a fast stochastic.
# Formula
# Fast %K = 100 SMA ( ( ( Close - Low ) / ( High - Low ) ),Time Period )
# Slow %K = SMA ( Fast  %K, Kma )
# Slow %D = SMA ( Slow K%, Dma )
# Where: 
# Close = the current closing price
# Low = the lowest low in the past  n periods
# High = the highest high in the past n periods
# Kma = Period of Moving Average used to smooth the  Fast %K Values
# Dma = Period of Moving Average used to smooth the Slow %K Values
# 
# - Bollinger bands
# 
# The Bollinger Band (BBANDS) study created by John Bollinger plots upper and lower envelope bands around the price of the instrument. The width of the bands is based on the standard deviation of the closing prices from a moving average of price.
# Formula
# Simplified:
# Middle Band = n-period moving average
# Upper Band = Middle Band + ( y * n-period standard deviation)
# Lower Band = Middle Band - ( y * n-period standard deviation)
# Where:
# n = number of periods
# y = factor to apply to the standard deviation value, (typical default for y = 2)
# 
# - fibinocci retracement
# 
# The Fibonacci tool provides a series of levels which measure the percentage a market has reversed between two different points. This means that within an uptrend, traders will typically use the tool to measure the amount of the last rally that has been surrendered, with a view to another leg higher before long.
# The Fibonacci tool is applied by placing the two anchor points onto the prior swing high and swing low, utilising the resulting Fibonacci levels as reference points when the market begins to retrace. It is advised to use the absolute tops and bottoms of the wicks rather than the body.
# 
# - ichimoku cloud
# 
# The Ichimoku study was developed by Goichi Hosoda pre-World War II as a forecasting model for financial markets. The study is a trend following indicator that identifies mid-points of historical highs and lows at different lengths of time and generates trading signals similar to that of moving averages/MACD. A key difference between Ichimoku and moving averages is Ichimoku charts lines are shifted forward in time creating wider support/resistance areas mitigating the risk of false breakouts.
# Formula
# Turning Line = ( Highest High + Lowest Low ) / 2, for the past 9 days
# Standard Line = ( Highest High + Lowest Low ) / 2, for the past 26 days
# Leading Span 1 = ( Standard Line + Turning Line ) / 2, plotted 26 days ahead of today
# Leading Span 2 = ( Highest High + Lowest Low ) / 2, for the past 52 days, plotted 26 days ahead of today
# Cloud = Shaded Area between Span 1 and Span 2
# 
# - Average directional index 
# 
# The Average Directional Movement Index (ADX) is designed to quantify trend strength by measuring the amount of price movement in a single direction. The ADX is part of the Directional Movement system published by J. Welles Wilder, and is the average resulting from the Directional Movement indicators.
# Formula
# Directional Movement (DM) is defined as the largest part of the current period’s price range that lies outside the previous period’s price range. For each period calculate:
# +DM =  positive or plus DM = High - Previous High
#  -DM = negative or minus DM = Previous Low - Low
# The smaller of the two values is reset to zero, i.e., if +DM > -DM, then -DM = 0. On an inside bar (a lower high and higher low), both +DM and -DM are negative values, so both get reset to zero as there was no directional movement for that period.
# The True Range (TR) is calculated for each period, where:
# TR = Max of ( High - Low ), ( High -PreviousClose ), ( PreviousClose - Low )
# The +DM, -DM and TR are each accumulated and smoothed using a custom smoothing method proposed by Wilder. For an n period smoothing, 1/n of each period’s value is added to the total each period, similar to an exponential smoothing:
# +DMt = (+DMt-1 - (+DMt-1 / n))  + (+DMt)
#  -DMt = (-DMt-1 - (-DMt-1 / n)) + (-DMt)
#   TRt = (TRt-1 - (TRt-1 / n)) + (TRt)
# Compute the positive/negative Directional Indexes, +DI and -DI, as a percentage of the True Range:
# +DI = ( +DM / TR ) * 100
#  -DI = ( -DM / TR ) * 100
# Compute the Directional Difference as the absolute value of the differences:   DIdiff = | ((+DI) - (-DI)) |
# Sum the directional indicator values: DIsum = ((+DI) + (-DI)) .
# Calculate the Directional Movement index:  DX = ( DIdiff / DIsum ) * 100 .  The DX is always between 0 and 100.
# Finally, apply Wilder’s smoothing technique to produce the final ADX value:
# ADXt = ( ( ADXt-1 * ( n - 1) ) + DXt ) / n
# 
# ## Other Potential Indicators to Use
# 
# - Acceleration Bands (ABANDS)
# 
# The Acceleration Bands (ABANDS) created by Price Headley plots upper and lower envelope bands around a simple moving average. The width of the bands is based on the formula below.
# Upper Band = Simple Moving Average (High * ( 1 + 4 * (High - Low) / (High + Low)))
# Middle Band = Simple Moving Average
# Lower Band = Simple Moving Average (Low * (1 - 4 * (High - Low)/ (High + Low)))
# 
# - Accumulation/Distribution (AD)
# 
# The Average Directional Movement Index (ADX) is designed to quantify trend strength by measuring the amount of price movement in a single direction. The ADX is part of the Directional Movement system published by J. Welles Wilder, and is the average resulting from the Directional Movement indicators.
# 
# Formula
# Directional Movement (DM) is defined as the largest part of the current period’s price range that lies outside the previous period’s price range. For each period calculate:
# +DM =  positive or plus DM = High - Previous High
#  -DM = negative or minus DM = Previous Low - Low
# The smaller of the two values is reset to zero, i.e., if +DM > -DM, then -DM = 0. On an inside bar (a lower high and higher low), both +DM and -DM are negative values, so both get reset to zero as there was no directional movement for that period.
# The True Range (TR) is calculated for each period, where:
# TR = Max of ( High - Low ), ( High -PreviousClose ), ( PreviousClose - Low )
# The +DM, -DM and TR are each accumulated and smoothed using a custom smoothing method proposed by Wilder. For an n period smoothing, 1/n of each period’s value is added to the total each period, similar to an exponential smoothing:
# +DMt = (+DMt-1 - (+DMt-1 / n))  + (+DMt)
#  -DMt = (-DMt-1 - (-DMt-1 / n)) + (-DMt)
#   TRt = (TRt-1 - (TRt-1 / n)) + (TRt)
# Compute the positive/negative Directional Indexes, +DI and -DI, as a percentage of the True Range
# +DI = ( +DM / TR ) * 100
#  -DI = ( -DM / TR ) * 100
# Compute the Directional Difference as the absolute value of the differences:   DIdiff = | ((+DI) - (-DI)) | 
# Sum the directional indicator values: DIsum = ((+DI) + (-DI)) .
# Calculate the Directional Movement index:  DX = ( DIdiff / DIsum ) * 100 .  The DX is always between 0 and 100.
# Finally, apply Wilder’s smoothing technique to produce the final ADX value:
# ADXt = ( ( ADXt-1 * ( n - 1) ) + DXt ) / n
# 
# - Average Directional Movement (ADX)
# - Absolute Price Oscillator (APO)
# - Aroon (AR)
# - Aroon Oscillator (ARO)
# - Average True Range (ATR)
# - Volume on the Ask (AVOL)
# - Volume on the Bid and Ask (BAVOL)
# - Band Width (BW)
# - Bar Value Area (BVA)
# - Bid Volume (BVOL)
# - Commodity Channel Index (CCI)
# - Chande Momentum Oscillator (CMO)
# - Double Exponential Moving Average (DEMA)
# - Plus DI (DI+)
# - Directional Movement Indicators (DMI)
# - Ichimoku (ICH)
# - Fill Indicator (FILL)
# - Keltner Channel (KC)
# - Linear Regression (LR)
# - Linear Regression Angle (LRA)
# - Linear Regression Intercept (LRI)
# - Linear Regression Slope (LRM)
# - Max (MAX)
# - Money Flow Index (MFI)
# - Midpoint (MIDPNT)
# - Midprice (MIDPRI)
# - Min (MIN)
# - MinMax (MINMAX)
# - Momentum (MOM)
# - Adaptive Moving Average (AMA)
# - Simple Moving Average (SMA)
# - T3 (T3)
# - Triple Exponential Moving Average (TEMA)
# - Triangular Moving Average (TRIMA)
# - Triple Exponential Moving Average Oscillator (TRIX)
# - Weighted Moving Average (WMA)
# - Normalized Average True Range (NATR)
# - Price Channel (PC)
# - PLOT (PLT)
# - Percent Price Oscillator (PPO)
# - Price Volume Trend (PVT)
# - Rate of Change (ROC)
# - Rate of Change (ROC100)
# - Rate of Change (ROCP)
# - Rate of Change (ROCR)
# - Parabolic Sar (SAR)
# - Session Cumulative Ask (SAVOL)
# - Session Cumulative Bid (SBVOL)
# - Standard Deviation (STDDEV)
# - Stochastic (STOCH)
# - Stochastic Fast (StochF)
# - Session Volume (S_VOL)
# - Time Series Forecast (TSF)
# - TT Cumulative Vol Delta (TT CVD)
# - Ultimate Oscillator (ULTOSC)
# - Volume At Price (VAP)
# - Volume Delta (Vol ∆)
# - Volume (VOLUME)
# - Volume Weighted Average Price (VWAP)
# - Williams % R (WillR)
# - Welles Wilder's Smoothing Average (WWS)

# When considering a final project idea, consider the following questions:
# 
# 1) Is there interest and passion for the project?
# - I find stocks and investing quite interesting and an important aspect of finance to learn about.
# 
# 2) Will the project assist you in your future?
# - If I can generate a way to be more successful in the stock market I will have gained a tremendous personal gain.
# 
# 3) Is the project idea and scope appropriate for the timeframe and skillset?
# - We have learned many of the skills required through our data science class and with some additional research it will absolutely be feasible.
# 
# 4) Does the project play to your strengths?
# - It does, I feel like I have taken to investing and data science which are both strengths of mine.
# 
# 5) Does the project help others (e.g. community service, service learning, etc.)?
# - It does not directly help others but could be used to teach other or potentially make money for others.
# 
# 6) Does the project have potential for wide outreach? (individual, very small group, school-wide, state-wide, country-wide, etc.)
# - Realistically, the reach is individual but could be put on the internet for anyone to use.
# 
# 7) Does the project have potential for a long life? (week, month, year, decade, etc.)
# - The project will likely last indefinitely
# 
# 8) Does the project solve a problem or enhance an existing process?
# - This will enhance an existing process

# PROJECT NAME
# Using Technical Indicators and Historical Information to Predict Stock Movement.
# 
# ABSTRACT
#     The goal of the project is to automate the generation of many techhnical indicators for any given stock. I hope to develop these indicators in a readable and compiled list of numbers and charts. Ultimately, I hope to predict future trends on current stocks through a compliation of information on the given stock. I wnat to use this information to make personal financial gain. 
#     The objectives required to complete this are as follows. I need to research and understand a multitude of technical stock indicators, and I need to implement the these indicators into pyhton code and apply them to various stocks. The first objective will be found through a simple google search and the second will be completed through data science guides and python code implementation.
# 
# CONCEPTUAL DESIGN
# 
#     First I will implement exponential moving averages (EMA) that represents an average of prices, but places more weight on recent prices. The weighting applied to the most recent price depends on the selected period of the moving average. The shorter the period for the EMA, the more weight that will be applied to the most recent price.)
#     
#     I will implement Moving Average Convergence Divergence (MACD) which is based on the differences between two moving averages. A second line, called the "Signal" line is plotted as a moving average of the MACD. Formula MACD = FastMA - SlowMA. Signal Line = MovAvg (MACD)
#     
#     I will implement Bolinger Bands (BBANDS) that plot upper and lower envelope bands around the price of the stock. The width of the bands is based on the standard deviation of the closing prices from a moving average of price.
#     I will implement Relative Strength Index (RSI). The current price is normalized as a percentage between 0 and 100. It represents the current price relative to other recent pieces within the selected lookback window length. Formula RSI = 100 - (100 / (1 + RS)).
#     
#     Lastly, I will implement the Stochastic Indicator (STOCH). STOCH normalizes price as a percentage between 0 and 100. Normally two lines are plotted, the %K line and a moving average of the %K which is called %D. Formula Fast %K = 100 SMA ( ( ( Close - Low ) / ( High - Low ) ),Time Period ) Slow %K = SMA ( Fast %K, Kma ) Slow %D = SMA ( Slow K%, Dma ). 
#     
#     I am going to do analysis of all of the stocks in the S&P 500 and the DOW Jones Index. This will be a good indicator of how my trading strategy performs compared to the market. I will use historical data from Yahoo to try these stragies with. I will compare my strategy to a buy and hold strategy and outline results in a table. I will do this comparison with all potentiall moving average lengths and RSI windows. I will compile the data into the best historically backed trading strategies. 
#     
#     After generating lots of comparisons with the historical data, I will create a user interface that will take input from the user of the stock or stocks they wish to analyze. I then I will generate basic data (a graph, current price, volume... etc). After I will have an interface with which they interact and will be able to choose what indicators they plan on getting data from, or if they want all of the indicators that will be a possibility as well. I will generate data analysis for the 5 indicators (potentiall more but this is time dependant). I will use the best possible trading strategies as suggested from the historical data analysis to allow the user to make informed buying and selling choices with the given stock. Thus the code will output its recommendation of a buy or sell on that day.
# 
# REQUIREMENTS
# - Paper trading account
# - TowardsDataScience account
# - Jupyter Notebook
# - Yahoo Finance
# 
# REMAINING QUESTIONS
# - How many technical indicators should I work with?
# - How should I campare differeing results between each indicator (if they don't agree)?
# 

# I plan on using BBANDS, EMA, MACD, RSI, STOCH as the initial indicators that I use

# # Table of Fundamental Analysis

# In[8]:


import yahoo_fin.stock_info as si
import pandas as pd
ticker = input("Ticker: ")
quote = si.get_quote_table(ticker)
print(quote)
val = si.get_stats_valuation(ticker)
print(val)
sheet = si.get_balance_sheet(ticker)
print(sheet)
print(si.get_income_statement(ticker))
flow = si.get_cash_flow(ticker)
print(flow)


# In[1]:


#import yahoo_fin.stock_info as si
import yfinance as yf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import date
import talib
from ta.trend import ADXIndicator


import warnings
import pandas_datareader.data as web

warnings.filterwarnings('ignore')

#company = pd.read_csv("/Users/ethan/Desktop/company.csv")
#tickers = company['Symbol']
#tickers = tickers.tolist()

dow = ['AXP','AMGN','AAPL','BA','CAT','CSCO','CVX','GS','HD','HON','IBM','INTC','JNJ','KO','JPM','MCD','MMM','MRK','MSFT','NKE','PG','TRV','UNH','CRM','VZ','V','WBA','WMT','DIS','DOW']
index = ['^IXIC', '^DJI', '^GSPC']


# # Table of Technical Analysis

# In[2]:


#inputTicker = input("Ticker: ")
#inputTicker

#dataframe = web.DataReader('BA', 'yahoo', start = date(2021,4,1), end = date(2021,8,26))
dataframe = yf.download('ECM.V', start = '2021-07-01', end='2021-08-30')

indicator = ["Value","RSI","ADX","Volume","Avg Volume","%K","%D", "Bollinger Upper", "Bollinger Lower","exp1","exp2","signal line","MACD"]
result = []
#


#Value
values = dataframe["Adj Close"]

result.append(round(values[len(values)-1],2))


#RSI

rsi = talib.RSI(dataframe["Adj Close"])

result.append(round(rsi[len(rsi)-1],2))



#ADX
dataframe['Adj Open'] = dataframe.Open * dataframe['Adj Close']/dataframe['Close']
dataframe['Adj High'] = dataframe.High * dataframe['Adj Close']/dataframe['Close']
dataframe['Adj Low'] = dataframe.Low * dataframe['Adj Close']/dataframe['Close']
dataframe.dropna(inplace=True)

def ADX(data: pd.DataFrame, period: int):
    
    df = data.copy()
    alpha = 1/period

    # TR
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']

    # ATR
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # +-DX
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']

    # +- DMI
    df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()
    df['+DMI'] = (df['S+DM']/df['ATR'])*100
    df['-DMI'] = (df['S-DM']/df['ATR'])*100
    del df['S+DM'], df['S-DM']

    # ADX
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI'])/(df['+DMI'] + df['-DMI']))*100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['DX'], df['ATR'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']

    return df

adx = ADX(dataframe,10)

result.append(round(adx['ADX'][len(adx)-2],2))


#Volume
vol = dataframe['Volume']
length = len(dataframe)
result.append(round(vol[len(vol)-1],2))

#AVg VOlume
total = 0
for i in range(length-1):
    total += vol[i]
avg = total/(length-1)
result.append(round(avg,2))
        
    
#Stochastic %K (20-80)
dataframe['L14'] = dataframe['Low'].rolling(window=14).min()
dataframe['H14'] = dataframe['High'].rolling(window=14).max()
dataframe['%K'] = 100*((dataframe['Close'] - dataframe['L14']) / (dataframe['H14'] - dataframe['L14']) )
k = dataframe['%K']

result.append(round(k[len(k)-1],2))


#Stochastic %D (20-80)
dataframe['%D'] = dataframe['%K'].rolling(window=3).mean()
d = dataframe['%D']

result.append(round(d[len(d)-1],2))


#bollinger upper and lower band values
period = 20
multiplier = 2
dataframe['UpperBand'] = dataframe['Close'].rolling(period).mean() + dataframe['Close'].rolling(period).std() * multiplier
dataframe['LowerBand'] = dataframe['Close'].rolling(period).mean() - dataframe['Close'].rolling(period).std() * multiplier
u = dataframe['UpperBand']
l = dataframe['LowerBand']

result.append(round(u[len(u)-1],2))
result.append(round(l[len(l)-1],2))


#exp1, exp2, signal line, and macd values
exp1 = dataframe['Adj Close'].ewm(span=12,adjust=False).mean()
exp2 = dataframe['Adj Close'].ewm(span=26,adjust=False).mean()
macd = exp2 - exp1
signal = macd.ewm(span=9,adjust=False).mean()

result.append(round(exp1[len(exp1)-1],2))
result.append(round(exp2[len(exp2)-1],2))
result.append(round(macd[len(macd)-1],2))
result.append(round(signal[len(signal)-1],2))

print(result)
d = {'indicators': indicator, 'results': result}
df2 = pd.DataFrame(data=d)
print(df2)


# In[ ]:




#inputTicker = input("Ticker: ")

dow = ['AXP','AMGN','AAPL','BA','CAT','CSCO','CVX','GS','HD','HON','IBM','INTC','JNJ','KO','JPM','MCD','MMM','MRK','MSFT','NKE','PG','TRV','UNH','CRM','VZ','V','WBA','WMT','DIS','DOW']
index = ['^IXIC', '^DJI', '^GSPC']

totalprofit = 0
totalother = 0
count = 0

totals= 0


for n in index:

    dataframe = web.DataReader(n, 'yahoo', start = date(2016,1,1), end = date(2021,3,8))


    #Value
    values = dataframe["Adj Close"]


    #RSI

    rsi = talib.RSI(dataframe["Adj Close"])


    #ADX
    dataframe['Adj Open'] = dataframe.Open * dataframe['Adj Close']/dataframe['Close']
    dataframe['Adj High'] = dataframe.High * dataframe['Adj Close']/dataframe['Close']
    dataframe['Adj Low'] = dataframe.Low * dataframe['Adj Close']/dataframe['Close']
    dataframe.dropna(inplace=True)

    def ADX(data: pd.DataFrame, period: int):

        df = data.copy()
        alpha = 1/period

        # TR
        df['H-L'] = df['High'] - df['Low']
        df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
        df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
        del df['H-L'], df['H-C'], df['L-C']

        # ATR
        df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

        # +-DX
        df['H-pH'] = df['High'] - df['High'].shift(1)
        df['pL-L'] = df['Low'].shift(1) - df['Low']
        df['+DX'] = np.where(
            (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
            df['H-pH'],
            0.0
        )
        df['-DX'] = np.where(
            (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
            df['pL-L'],
            0.0
        )
        del df['H-pH'], df['pL-L']

        # +- DMI
        df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
        df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()
        df['+DMI'] = (df['S+DM']/df['ATR'])*100
        df['-DMI'] = (df['S-DM']/df['ATR'])*100
        del df['S+DM'], df['S-DM']

        # ADX
        df['DX'] = (np.abs(df['+DMI'] - df['-DMI'])/(df['+DMI'] + df['-DMI']))*100
        df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
        del df['DX'], df['ATR'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']

        return df

    adx = ADX(dataframe,10)


    #Volume
    vol = dataframe['Volume']
    length = len(dataframe)

    #Stochastic %K (20-80)
    dataframe['L14'] = dataframe['Low'].rolling(window=14).min()
    dataframe['H14'] = dataframe['High'].rolling(window=14).max()
    dataframe['%K'] = 100*((dataframe['Close'] - dataframe['L14']) / (dataframe['H14'] - dataframe['L14']) )
    k = dataframe['%K']

    #Stochastic %D (20-80)
    dataframe['%D'] = dataframe['%K'].rolling(window=3).mean()
    d = dataframe['%D']

    #bollinger upper and lower band values
    period = 20
    multiplier = 2
    dataframe['UpperBand'] = dataframe['Close'].rolling(period).mean() + dataframe['Close'].rolling(period).std() * multiplier
    dataframe['LowerBand'] = dataframe['Close'].rolling(period).mean() - dataframe['Close'].rolling(period).std() * multiplier
    u = dataframe['UpperBand']
    l = dataframe['LowerBand']


    #exp1, exp2, signal line, and macd values
    exp1 = dataframe['Adj Close'].ewm(span=12,adjust=False).mean()
    exp2 = dataframe['Adj Close'].ewm(span=26,adjust=False).mean()
    macd = exp2 - exp1
    signal = macd.ewm(span=9,adjust=False).mean()



    profit = 0
    buyShares = 0
    sellShares = 0
    totalSpent = 0
    maxShares = 0

    for i in range(30,len(values)-1):
        buyCount= 0
        sellCount =0

        if rsi[i] < 30:
            buyCount +=1
        if rsi[i] > 70:
            sellCount +=1
        if adx['ADX'][i] > 30:
            buyCount +=1
            sellCount +=1
        total = 0
        for j in range(i):
            total += vol[j]
        avg = total/(length-1)
        if vol[i] > avg:
            buyCount += 1
        if vol[i] < avg:
            sellCount += 1
        if values[i] < l[i]:
            buyCount +=1
        if values[i] > u[i]:
            sellCount +=1
        if k[i] < 20:
            buyCount +=1
        if k[i] > 80:
            sellCount +=1
        if exp2[i] <= exp1[i] and exp2[i-1] > exp1[i-1]:
            buyCount +=1
        if exp2[i] >= exp1[i] and exp2[i-1] < exp1[i-1]:
            sellCount +=1
        if macd[i] >= signal[i] and macd[i-1] < signal[i-1]:
            buyCount +=1
        if macd[i] <= signal[i] and macd[i-1] > signal[i-1]:
            sellCount +=1





        if buyCount >=  4 and sellShares ==0:
            profit -= values[i]
            buyShares +=1
            #print("Value: " + str(values[i]))
            #print("Profit: "+str(profit))
        if buyCount >=  4 and sellShares > 0:
            profit -= values[i]*(sellShares+1)
            #print("Value: " + str(values[i]))
            #print("Profit: "+str(profit) + " Shares: " + str(sellShares))
            sellShares = 0
            buyShares +=1
        if sellCount >= 4 and buyShares == 0:
            profit += values[i]
            sellShares +=1
            #print("Value: " + str(values[i]))
            #print("Profit: "+str(profit))
        if sellCount >= 4 and buyShares > 0:
            profit += values[i]*(buyShares+1)
            #print("Value: " + str(values[i]))
            #print("Profit: "+str(profit) + " Shares: " + str(buyShares))
            buyShares = 0
            sellShares +=1

        if buyShares > maxShares:
            maxShares = buyShares
        if sellShares > maxShares:
            maxShares = sellShares


    if buyShares > 0:
        profit += buyShares*values[i]
        #print("Profit: "+str(profit) + " Shares: " + str(buyShares))
    if sellShares > 0:
        profit -= sellShares*values[i]
        #print("Profit: "+str(profit) + " Shares: " + str(sellShares))


    otherProfit = values[len(values)-1]-values[30]
    print(otherProfit)
    print(profit/maxShares)
    print(profit/maxShares/otherProfit)
    print()
    count+=1
    totalprofit += profit/(maxShares)
    totalother += otherProfit

print()
print()
print(totalother)
print(totalprofit)


# In[ ]:


inputTicker = input("Ticker: ")

count = 0

totals= 0


dataframe = web.DataReader(inputTicker, 'yahoo', start = date(2019,1,1), end = date(2021,3,8))


#Value
values = dataframe["Adj Close"]


#RSI

rsi = talib.RSI(dataframe["Adj Close"])


#ADX
dataframe['Adj Open'] = dataframe.Open * dataframe['Adj Close']/dataframe['Close']
dataframe['Adj High'] = dataframe.High * dataframe['Adj Close']/dataframe['Close']
dataframe['Adj Low'] = dataframe.Low * dataframe['Adj Close']/dataframe['Close']
dataframe.dropna(inplace=True)

def ADX(data: pd.DataFrame, period: int):

    df = data.copy()
    alpha = 1/period

    # TR
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']

    # ATR
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # +-DX
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']

    # +- DMI
    df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()
    df['+DMI'] = (df['S+DM']/df['ATR'])*100
    df['-DMI'] = (df['S-DM']/df['ATR'])*100
    del df['S+DM'], df['S-DM']

    # ADX
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI'])/(df['+DMI'] + df['-DMI']))*100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['DX'], df['ATR'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']

    return df

adx = ADX(dataframe,10)


#Volume
vol = dataframe['Volume']
length = len(dataframe)

#Stochastic %K (20-80)
dataframe['L14'] = dataframe['Low'].rolling(window=14).min()
dataframe['H14'] = dataframe['High'].rolling(window=14).max()
dataframe['%K'] = 100*((dataframe['Close'] - dataframe['L14']) / (dataframe['H14'] - dataframe['L14']) )
k = dataframe['%K']

#Stochastic %D (20-80)
dataframe['%D'] = dataframe['%K'].rolling(window=3).mean()
d = dataframe['%D']

#bollinger upper and lower band values
period = 20
multiplier = 2
dataframe['UpperBand'] = dataframe['Close'].rolling(period).mean() + dataframe['Close'].rolling(period).std() * multiplier
dataframe['LowerBand'] = dataframe['Close'].rolling(period).mean() - dataframe['Close'].rolling(period).std() * multiplier
u = dataframe['UpperBand']
l = dataframe['LowerBand']


#exp1, exp2, signal line, and macd values
exp1 = dataframe['Adj Close'].ewm(span=12,adjust=False).mean()
exp2 = dataframe['Adj Close'].ewm(span=26,adjust=False).mean()
macd = exp2 - exp1
signal = macd.ewm(span=9,adjust=False).mean()



profit = 0
buyShares = 0
sellShares = 0
totalSpent = 0
maxShares = 0

for i in range(30,len(values)-1):
    buyCount= 0
    sellCount =0

    if rsi[i] < 30:
        buyCount +=1
    if rsi[i] > 70:
        sellCount +=1
    if adx['ADX'][i] > 30:
        buyCount +=1
        sellCount +=1
    total = 0
    for j in range(i):
        total += vol[j]
    avg = total/(length-1)
    if vol[i] > avg:
        buyCount += 1
    if vol[i] < avg:
        sellCount += 1
    if values[i] < l[i]:
        buyCount +=1
    if values[i] > u[i]:
        sellCount +=1
    if k[i] < 20:
        buyCount +=1
    if k[i] > 80:
        sellCount +=1
    if exp2[i] <= exp1[i] and exp2[i-1] > exp1[i-1]:
        buyCount +=1
    if exp2[i] >= exp1[i] and exp2[i-1] < exp1[i-1]:
        sellCount +=1
    if macd[i] >= signal[i] and macd[i-1] < signal[i-1]:
        buyCount +=1
    if macd[i] <= signal[i] and macd[i-1] > signal[i-1]:
        sellCount +=1


    

    if buyCount >=  4 and sellShares ==0:
        profit -= values[i]
        buyShares +=1
        #print("Value: " + str(values[i]))
        #print("Profit: "+str(profit))
    if buyCount >=  4 and sellShares > 0:
        profit -= values[i]*(sellShares+1)
        #print("Value: " + str(values[i]))
        #print("Profit: "+str(profit) + " Shares: " + str(sellShares))
        sellShares = 0
        buyShares +=1
    if sellCount >= 4 and buyShares == 0:
        profit += values[i]
        sellShares +=1
        #print("Value: " + str(values[i]))
        #print("Profit: "+str(profit))
    if sellCount >= 4 and buyShares > 0:
        profit += values[i]*(buyShares+1)
        #print("Value: " + str(values[i]))
        #print("Profit: "+str(profit) + " Shares: " + str(buyShares))
        buyShares = 0
        sellShares +=1

    if buyShares > maxShares:
        maxShares = buyShares
    if sellShares > maxShares:
        maxShares = sellShares


if buyShares > 0:
    profit += buyShares*values[i]
    #print("Profit: "+str(profit) + " Shares: " + str(buyShares))
if sellShares > 0:
    profit -= sellShares*values[i]
    #print("Profit: "+str(profit) + " Shares: " + str(sellShares))


otherProfit = values[len(values)-1]-values[30]
print(otherProfit)
print(profit/maxShares)
print(profit/maxShares/otherProfit)
print()
count+=1


# # MACD

# In[ ]:


ticker = '^DJI'

df = web.DataReader(ticker, 'yahoo', start = date(2020,8,1), end = date(2021,3,8))
plt.plot(df['Adj Close'], label=ticker)
plt.legend(loc='upper left')
plt.show()

exp1 = df['Adj Close'].ewm(span=12,adjust=False).mean()
exp2 = df['Adj Close'].ewm(span=26,adjust=False).mean()
macd = exp2 - exp1
exp3 = macd.ewm(span=9,adjust=False).mean()

plt.plot( macd, label=(ticker + " MACD"), color = '#EBD2BE')
plt.plot(exp3,label = 'Signal Line', color = '#E5A4CB')
plt.legend(loc='upper left')
plt.show()

exp1 = df['Adj Close'].ewm(span=12,adjust=False).mean()
exp2 = df['Adj Close'].ewm(span=26,adjust=False).mean()
macd = exp2 - exp1
exp3 = df['Adj Close'].ewm(span=9,adjust=False).mean()

#plt.plot( macd, label=(ticker + " MACD"), color = 'orange')
#plt.plot(exp3,label = 'Signal Line', color = 'Magenta')
#plt.legend(loc='upper left')
#plt.show()

#plt.plot( macd, label=(ticker + " MACD"), color = 'orange')
#plt.legend(loc='upper left')
#plt.show()


macd.dropna(inplace = True, axis = 0)
exp3.dropna(inplace = True, axis = 0)

nums = df['Adj Close']
nums = nums.tolist()
values = []
for x in nums:
    values.append(round(x,2))

count = 0
count2 = 0
shares = 0
profit = 0
firstbuy = 0
firstsell = 0

#print(exp1)
#print(exp2)
length = len(exp3)
otherProfit = (values[length-1] - values[0])




for i in range(len(exp2)-1):
    if exp3[i+1] >= macd[i+1] and exp3[i] < macd[i] and shares > 0 :
        if firstsell == 0:
            firstsell += 1
            shares = 2
            profit += shares * values[i+1]
            #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
            shares = 0
            count2 += 1
            count -= 1
        else:
            profit += shares * values[i+1]
            #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
            shares = 0
            count2 += 1
            count -= 1
    elif exp3[i+1] <= macd[i+1] and exp3[i] > macd[i] :
        if firstbuy == 0:
            firstbuy += 1
            shares = 1
            profit = profit - (shares* values[i+1])
            #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
            count2 +=1
            count += 1

        else:
            shares = 2
            profit = profit - (shares* values[i+1])
            #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
            count2 +=1
            count += 1
#print(count)
if count == 0:
    profit -=  values[length-1]
    #print(str(values[length-1])+ " bought 1" + " profit is currently at " + str(profit))
if count >=1:
    profit +=  values[length-1]
    #print(str(values[length-1])+ " sold 1" + " profit is currently at " + str(profit))

#totalProfit += profit
print("buy and hold: " + str(otherProfit))
print("buy and hold % increase:" + str((otherProfit/(values[0]*1))))
print("")
print("final profit "+ str(profit))
print("MACD % increase :" + str((profit/(values[0]*1))))
print("")
print("")


# # EMA

# In[ ]:


df = web.DataReader('^DJI', 'yahoo', start = date(2019,1,1), end = date(2020,12,31))
exp1 = df['Adj Close'].ewm(span=2, adjust=False).mean()
exp2 = df['Adj Close'].ewm(span=20, adjust=False).mean()
plt.plot(df['Adj Close'], label='DJI')
plt.plot(exp1, label='df 2 Day EMA')
plt.plot(exp2, label='df 20 Day EMA')
plt.legend(loc='upper left')
plt.show()

exp1.dropna(inplace = True, axis = 0)
exp2.dropna(inplace = True, axis = 0)

nums = df['Adj Close']
nums = nums.tolist()
values = []
for x in nums:
    values.append(round(x,2))

count = 0
count2 = 0
shares = 0
profit = 0
firstbuy = 0
firstsell = 0

#print(exp1)
#print(exp2)
length = len(exp2)
otherProfit = (values[length-1] - values[0])




for i in range(len(exp2)-1):
    if exp2[i+1] >= exp1[i+1] and exp2[i] < exp1[i] and shares > 0 :
        if firstsell == 0:
            firstsell += 1
            shares = 2
            profit += shares * values[i+1]
            #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
            shares = 0
            count2 += 1
            count -= 1
        else:
            profit += shares * values[i+1]
            #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
            shares = 0
            count2 += 1
            count -= 1
    elif exp2[i+1] <= exp1[i+1] and exp2[i] > exp1[i] :
        if firstbuy == 0:
            firstbuy += 1
            shares = 1
            profit = profit - (shares* values[i+1])
            #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
            count2 +=1
            count += 1

        else:
            shares = 2
            profit = profit - (shares* values[i+1])
            #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
            count2 +=1
            count += 1
#print(count)
if count ==0:
    profit -=  values[length-1]
    #print(str(values[length-1])+ " bought 1" + " profit is currently at " + str(profit))
if count ==1:
    profit +=  values[length-1]
    #print(str(values[length-1])+ " sold 1" + " profit is currently at " + str(profit))


print("buy and hold: " + str(otherProfit))
print("buy and hold % increase:" + str((otherProfit/(values[0]*1))))
print("")
print("final profit "+ str(profit))
print("EMA % increase :" + str((profit/(values[0]*1))))
print("")
print("")


# In[ ]:


df = web.DataReader('^DJI', 'yahoo', start = date(1999,1,1), end = date(2020,12,31))
    
maxes = [] 


nums = df['Adj Close']
nums = nums.tolist()
values = []
for x in nums:
    values.append(round(x,2))
otherProfit = (values[length-1] - values[0])


    
for k in range(10,40):
    bestOfLongs = []
    for j in range(50,80):
        totalProfit = 0
        exp1 = df['Adj Close'].ewm(span=k, adjust=False).mean()
        exp2 = df['Adj Close'].ewm(span=j, adjust=False).mean()
        exp1.dropna(inplace = True, axis = 0)
        exp2.dropna(inplace = True, axis = 0)

        count = 0
        count2 = 0
        shares = 0
        profit = 0
        firstbuy = 0
        firstsell = 0
        length = len(exp2)
        
        for i in range(len(exp2)-1):
            if exp2[i+1] >= exp1[i+1] and exp2[i] < exp1[i] and shares > 0 :
                if firstsell == 0:
                    firstsell += 1
                    shares = 2
                    profit += shares * values[i+1]
                    #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
                    shares = 0
                    count2 += 1
                    count -= 1
                else:
                    profit += shares * values[i+1]
                    #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
                    shares = 0
                    count2 += 1
                    count -= 1
            elif exp2[i+1] <= exp1[i+1] and exp2[i] > exp1[i] :
                if firstbuy == 0:
                    firstbuy += 1
                    shares = 1
                    profit = profit - (shares* values[i+1])
                    #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
                    count2 +=1
                    count += 1
                else:
                    shares = 2
                    profit = profit - (shares* values[i+1])
                    #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
                    count2 +=1
                    count += 1
        if count ==0:
            profit -=  values[length-1]
            #print(str(values[length-1])+ " bought 1" + " profit is currently at " + str(profit))
        if count ==1:
            profit +=  values[length-1]
            #print(str(values[length-1])+ " sold 1" + " profit is currently at " + str(profit))
        totalProfit += profit
        
        bestOfLongs.append(totalProfit)
    print("EMA Short: " + str(k) + " Long: " + str(bestOfLongs.index(max(bestOfLongs))+50) +" 20 total profit: " + str(max(bestOfLongs)))
    maxes.append(max(bestOfLongs))
        
for k in range(2,10):
    bestOfLongs = []
    for j in range(10,30):
        totalProfit = 0
        exp1 = df['Adj Close'].ewm(span=k, adjust=False).mean()
        exp2 = df['Adj Close'].ewm(span=j, adjust=False).mean()
        exp1.dropna(inplace = True, axis = 0)
        exp2.dropna(inplace = True, axis = 0)

        nums = df['Adj Close']
        nums = nums.tolist()
        values = []
        for x in nums:
            values.append(round(x,2))
        count = 0
        count2 = 0
        shares = 0
        profit = 0
        firstbuy = 0
        firstsell = 0
        length = len(exp2)
        otherProfit = (values[length-1] - values[0])
        for i in range(len(exp2)-1):
            if exp2[i+1] >= exp1[i+1] and exp2[i] < exp1[i] and shares > 0 :
                if firstsell == 0:
                    firstsell += 1
                    shares = 2
                    profit += shares * values[i+1]
                    #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
                    shares = 0
                    count2 += 1
                    count -= 1
                else:
                    profit += shares * values[i+1]
                    #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
                    shares = 0
                    count2 += 1
                    count -= 1
            elif exp2[i+1] <= exp1[i+1] and exp2[i] > exp1[i] :
                if firstbuy == 0:
                    firstbuy += 1
                    shares = 1
                    profit = profit - (shares* values[i+1])
                    #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
                    count2 +=1
                    count += 1
                else:
                    shares = 2
                    profit = profit - (shares* values[i+1])
                    #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
                    count2 +=1
                    count += 1
        if count ==0:
            profit -=  values[length-1]
            #print(str(values[length-1])+ " bought 1" + " profit is currently at " + str(profit))
        if count ==1:
            profit +=  values[length-1]
            #print(str(values[length-1])+ " sold 1" + " profit is currently at " + str(profit))
        totalProfit += profit
        bestOfLongs.append(totalProfit)
    print("EMA Short: " + str(k) + " Long: " + str(bestOfLongs.index(max(bestOfLongs))+10) + " total profit: " + str(max(bestOfLongs)))
    maxes.append(max(bestOfLongs))
    
print(max(maxes))
    


# # SMA

# In[ ]:


df = web.DataReader('^DJI', 'yahoo', start = date(2009,1,1), end = date(2020,12,31))
nums = df['Adj Close']
nums = nums.tolist()
values = []
for x in nums:
    values.append(round(x,2))
otherProfit = (values[length-1] - values[0])


    
maxes = []
    
    
for k in range(10,40):
    bestOfLongs = []

    for j in range(50,80):
        totalProfit = 0
        
        exp1 = df['Adj Close'].rolling(window=k).mean()
        exp2 = df['Adj Close'].rolling(window=j).mean()
        exp1.dropna(inplace = True, axis = 0)
        exp2.dropna(inplace = True, axis = 0)

        

        count = 0
        count2 = 0
        shares = 0
        profit = 0
        firstbuy = 0
        firstsell = 0

        length = len(exp2)
        



        for i in range(len(exp2)-1):
            if exp2[i+1] >= exp1[i+1] and exp2[i] < exp1[i] and shares > 0 :
                if firstsell == 0:
                    firstsell += 1
                    shares = 2
                    profit += shares * values[i+1]
                    #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
                    shares = 0
                    count2 += 1
                    count -= 1
                else:
                    profit += shares * values[i+1]
                    #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
                    shares = 0
                    count2 += 1
                    count -= 1
            elif exp2[i+1] <= exp1[i+1] and exp2[i] > exp1[i] :
                if firstbuy == 0:
                    firstbuy += 1
                    shares = 1
                    profit = profit - (shares* values[i+1])
                    #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
                    count2 +=1
                    count += 1

                else:
                    shares = 2
                    profit = profit - (shares* values[i+1])
                    #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
                    count2 +=1
                    count += 1


        #print(count)
        if count ==0:
            profit -=  values[length-1]
            #print(str(values[length-1])+ " bought 1" + " profit is currently at " + str(profit))
        if count ==1:
            profit +=  values[length-1]
            #print(str(values[length-1])+ " sold 1" + " profit is currently at " + str(profit))

        totalProfit += profit
        bestOfLongs.append(totalProfit)
    print("SMA Short: " + str(k) + " Long: " + str(bestOfLongs.index(max(bestOfLongs))+50) +" 20 total profit: " + str(max(bestOfLongs)))
    maxes.append(max(bestOfLongs))
        
    
for k in range(2,10):
    bestOfLongs = []

    for j in range(10,30):
        totalProfit = 0
        exp1 = df['Adj Close'].rolling(window=k).mean()
        exp2 = df['Adj Close'].rolling(window=j).mean()
        exp1.dropna(inplace = True, axis = 0)
        exp2.dropna(inplace = True, axis = 0)

        nums = df['Adj Close']
        nums = nums.tolist()
        values = []
        for x in nums:
            values.append(round(x,2))

        count = 0
        count2 = 0
        shares = 0
        profit = 0
        firstbuy = 0
        firstsell = 0

        length = len(exp2)
        otherProfit = (values[length-1] - values[0])



        for i in range(len(exp2)-1):
            if exp2[i+1] >= exp1[i+1] and exp2[i] < exp1[i] and shares > 0 :
                if firstsell == 0:
                    firstsell += 1
                    shares = 2
                    profit += shares * values[i+1]
                    #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
                    shares = 0
                    count2 += 1
                    count -= 1
                else:
                    profit += shares * values[i+1]
                    #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
                    shares = 0
                    count2 += 1
                    count -= 1
            elif exp2[i+1] <= exp1[i+1] and exp2[i] > exp1[i] :
                if firstbuy == 0:
                    firstbuy += 1
                    shares = 1
                    profit = profit - (shares* values[i+1])
                    #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
                    count2 +=1
                    count += 1

                else:
                    shares = 2
                    profit = profit - (shares* values[i+1])
                    #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
                    count2 +=1
                    count += 1


        #print(count)
        if count ==0:
            profit -=  values[length-1]
            #print(str(values[length-1])+ " bought 1" + " profit is currently at " + str(profit))
        if count ==1:
            profit +=  values[length-1]
            #print(str(values[length-1])+ " sold 1" + " profit is currently at " + str(profit))

        totalProfit += profit
        bestOfLongs.append(totalProfit)
    print("SMA Short: " + str(k) + " Long: " + str(bestOfLongs.index(max(bestOfLongs))+10) + " total profit: " + str(max(bestOfLongs)))
    maxes.append(max(bestOfLongs))
    
print(max(maxes))
print
    
    


# # RSI

# In[13]:


ticker = 'BA'
data = web.DataReader(ticker, 'yahoo', start = date(2021,1,1), end = date(2021,8,25))


rsi = talib.RSI(data["Adj Close"])


fig = plt.figure()
fig.set_size_inches((25, 18))
ax_rsi = fig.add_axes((0, 0.24, 1, 0.2))
ax_rsi.plot(data.index, [70] * len(data.index), label="overbought")
ax_rsi.plot(data.index, [30] * len(data.index), label="oversold")
ax_rsi.plot(data.index, rsi, label="rsi")
ax_rsi.legend()
fig = plt.figure()
fig.set_size_inches((25, 18))
plt.plot(data["Adj Close"])


# In[ ]:


total = 0
totalOther = 0

for x in index:

    df = web.DataReader(x, 'yahoo', start = date(2000,1,1), end = date(2021,3,8))
    rsi = talib.RSI(df["Adj Close"])


    nums = df['Adj Close']
    nums = nums.tolist()
    values = []
    for x in nums:
        values.append(round(x,2))

    count = 0
    shares = 0
    profit = 0
    firstSell = 0


    length = len(rsi)
    length2 = len(values)

    otherProfit = (values[length2-1] - values[length2-length])



    for i in range(length -1):
        if rsi[i] < 30 and shares == 0:
            firstSell = 1
            shares += 1
            profit -= values[i] * 2 
            #print(str(values[i]) + " bought -- profit is currently at " + str(profit))
            
        elif rsi[i] >70 and shares == 1 and firstSell == 1:
            shares -= 1 
            profit += values[i] * 2
            #print(str(values[i]) + " sold -- profit is currently at " + str(profit))


    if shares == 1:
        profit += values[length-1] *2
        #print(str(values[length-1]) + " sold -- profit is currently at " + str(profit))

    profit-= values[0]
    total+= profit
    totalOther += otherProfit

    print(str(profit))
    print(otherProfit)
    print("")

print(total)
print(totalOther)


# # Bollinger Bands

# In[5]:


ticker = 'BA'
df = web.DataReader(ticker, 'yahoo', start = date(2021,4,1), end = date(2021,8,26))

period = 20
multiplier = 2

df['UpperBand'] = df['Close'].rolling(period).mean() + df['Close'].rolling(period).std() * multiplier
df['LowerBand'] = df['Close'].rolling(period).mean() - df['Close'].rolling(period).std() * multiplier


plt.rcParams['figure.figsize'] = [12, 7]

plt.rc('font', size=14)

plt.plot(df['Close'], label = "S&P 500")
plt.plot(df['UpperBand'], label = "Upper Bollinger Band")
plt.plot(df['LowerBand'], label = "Lower Bollinger Band")

plt.legend()

plt.show()


# In[ ]:


total = 0
totalOther = 0

for x in index:

    df = web.DataReader(x, 'yahoo', start = date(2010,1,1), end = date(2020,12,24))
    
    period = 20
    multiplier = 2

    df['UpperBand'] = df['Adj Close'].rolling(period).mean() + df['Adj Close'].rolling(period).std() * multiplier
    df['LowerBand'] = df['Adj Close'].rolling(period).mean() - df['Adj Close'].rolling(period).std() * multiplier

    upper = df['UpperBand']
    lower = df['LowerBand']

    nums = df['Adj Close']
    nums = nums.tolist()
    values = []
    for x in nums:
        values.append(round(x,2))

    count = 0
    profit = 0
    buyShares = 0
    sellShares = 0
    maxShares = 0

    
    length = len(df['UpperBand'])

    length2 = len(values)

    otherProfit = (values[length2-1] - values[length2-length])



    for i in range(length -1):
        if values[i] < lower[i] and sellShares ==0:
            profit -= values[i]
            buyShares +=1
            #print("Value: " + str(values[i]))
            #print("Profit: "+str(profit))
        if values[i] < lower[i] and sellShares > 0:
            profit -= values[i]*(sellShares+1)
            #print("Value: " + str(values[i]))
            #print("Profit: "+str(profit) + " Shares: " + str(sellShares))
            sellShares = 0
            buyShares +=1
        if values[i] > upper[i] and buyShares == 0:
            profit += values[i]
            sellShares +=1
            #print("Value: " + str(values[i]))
            #print("Profit: "+str(profit))
        if values[i] > upper[i] and buyShares > 0:
            profit += values[i]*(buyShares+1)
            #print("Value: " + str(values[i]))
            #print("Profit: "+str(profit) + " Shares: " + str(buyShares))
            buyShares = 0
            sellShares +=1

        if buyShares > maxShares:
            maxShares = buyShares
        if sellShares > maxShares:
            maxShares = sellShares

    if buyShares > 0:
        profit += buyShares*values[i]
        #print("Profit: "+str(profit) + " Shares: " + str(buyShares))
    if sellShares > 0:
        profit -= sellShares*values[i]
        #print("Profit: "+str(profit) + " Shares: " + str(sellShares))


    otherProfit = values[len(values)-1]-values[30]
    print(otherProfit)
    print(profit/maxShares)
    print(profit/maxShares/otherProfit)
    print()
    count+=1
    total += profit
    totalOther += otherProfit

print(totalOther)
print(total)


# # Stochastic Occilator

# In[ ]:


df = web.DataReader('^DJI', 'yahoo', start = date(2020,12,1), end = date(2021,1,28))
#print out first 5 rows of data DataFrame to check in correct format


df['L14'] = df['Low'].rolling(window=14).min()
#Create the "H14" column in the DataFrame
df['H14'] = df['High'].rolling(window=14).max()
#Create the "%K" column in the DataFrame
df['%K'] = 100*((df['Close'] - df['L14']) / (df['H14'] - df['L14']) )
#Create the "%D" column in the DataFrame
df['%D'] = df['%K'].rolling(window=3).mean()

fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(20,10))
df['Close'].plot(ax=axes[0]); axes[0].set_title('Close')
df[['%K','%D']].plot(ax=axes[1]); axes[1].set_title('Oscillator')


# In[ ]:


nums = df['Adj Close']
nums = nums.tolist()
values = []
for x in nums:
    values.append(round(x,2))

count = 0
shares = 0
profit = 0
firstSell = 0

length = len(df['%K'])
length2 = len(values)

k = df['%K']
d = df['%D']

otherProfit = (values[length2-1] - values[length2-length])







for i in range(length -1):
    if k[i] < 20 and shares == 0:
        firstSell = 1
        shares += 1
        profit -= values[i] * 2 
        print(str(values[i]) + " bought -- profit is currently at " + str(profit))

    elif k[i] > 80 and shares == 1 and firstSell == 1:
        shares -= 1 
        profit += values[i] * 2
        print(str(values[i]) + " sold -- profit is currently at " + str(profit))


if shares == 1:
    profit += values[length-1] *2
    print(str(values[length-1]) + " sold -- profit is currently at " + str(profit))


print(profit)
print(otherProfit)
print("")


# In[ ]:


nums = df['Adj Close']
nums = nums.tolist()
values = []
for x in nums:
    values.append(round(x,2))

count = 0
shares = 0
profit = 0
firstSell = 0
count2 = 0
firstbuy = 0

length = len(df['%K'])
length2 = len(values)

k = df['%K']
d = df['%D']

otherProfit = (values[length2-1] - values[length2-length])


if d[i+1] >= k[i+1] and d[i] < k[i] and shares > 0 :
    if firstsell == 0:
        firstsell += 1
        shares = 2
        profit += shares * values[i+1]
        #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
        shares = 0
        count2 += 1
        count -= 1
    else:
        profit += shares * values[i+1]
        #print(str(values[i+1]) + " sold " + str(shares) + " profit is currently at " + str(profit))
        shares = 0
        count2 += 1
        count -= 1
elif d[i+1] <= k[i+1] and d[i] > k[i] :
    if firstbuy == 0:
        firstbuy += 1
        shares = 1
        profit = profit - (shares* values[i+1])
        #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
        count2 +=1
        count += 1

    else:
        shares = 2
        profit = profit - (shares* values[i+1])
        #print(str(values[i+1]) + " bought " + str(shares) + " profit is currently at " + str(profit))
        count2 +=1
        count += 1

#print(count)
if count ==0:
    profit -=  values[length2-1]
    #print(str(values[length-1])+ " bought 1" + " profit is currently at " + str(profit))
if count ==1:
    profit +=  values[length2-1]
    #print(str(values[length-1])+ " sold 1" + " profit is currently at " + str(profit))

print(profit)
print(otherProfit)
print("")


# # On Balance Volume

# In[ ]:


df = web.DataReader('^DJI', 'yahoo', start = date(2020,1,1), end = date(2021,1,14))
values = df['Adj Close']
#print(df.columns)
vol = df[['Volume']]
pos_move = []  # List of days that the stock price increased
neg_move = []  # List of days that the stock price increased
OBV_Value = 0
OBV_List = []
count_list = []
#print(vol.iloc[0])

length = len(df)


for i in range(length-1):
    count_list.append(i)
    if values[i] <= values[i+1]:
        pos_move.append(values[i+1])
        #print(vol[i])
        OBV_Value += int(vol.iloc[i+1])
        OBV_List.append(OBV_Value)
    elif values[i] > values[i+1]:
        neg_move.append(values[i+1])
        #print(vol[i])
        OBV_Value -=int(vol.iloc[i+1])
        OBV_List.append(OBV_Value)
        
plt.plot(count_list,OBV_List)

plt.show()

df['Adj Close'].plot(ax=axes[0]); axes[0].set_title('Adj Close')
plt.show()


# # Average Directional Index

# In[ ]:


total = 0
totalOther = 0

dji = web.DataReader(index[1], 'yahoo', start = date(2000,1,1), end = date(2021,1,1))

dji['Adj Open'] = dji.Open * dji['Adj Close']/dji['Close']
dji['Adj High'] = dji.High * dji['Adj Close']/dji['Close']
dji['Adj Low'] = dji.Low * dji['Adj Close']/dji['Close']
dji.dropna(inplace=True)



# In[ ]:


def ADX(data: pd.DataFrame, period: int):
    
    df = data.copy()
    alpha = 1/period

    # TR
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']

    # ATR
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # +-DX
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']

    # +- DMI
    df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()
    df['+DMI'] = (df['S+DM']/df['ATR'])*100
    df['-DMI'] = (df['S-DM']/df['ATR'])*100
    del df['S+DM'], df['S-DM']

    # ADX
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI'])/(df['+DMI'] + df['-DMI']))*100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['DX'], df['ATR'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']

    return df

df = ADX(dji,16)


# In[ ]:


length = len(df)
profit = 0
num = 0

for i in range(10,length-2):
    if df['ADX'][i] > 25 and df['Adj Close'][i] > df['Adj Open'][i] and num == 1:
        profit -= 2* df['Adj Close'][i]
        num = 0
        
    if df['ADX'][i] > 25 and df['Adj Close'][i] < df['Adj Open'][i] and num == 0:
        profit += 2* df['Adj Close'][i]
        num = 1

print(profit)
print(df['Adj Close'][length-1]-df['Adj Close'][1])
        


# # Fibonacci Retracement

# In[ ]:


df = web.DataReader('AMZN', 'yahoo', start = date(2000,1,1), end = date(2020,1,1))
count = []
for i in range (len(df)):
    count.append(i)
    
Price_Min =df['Low'].min()
print(Price_Min)


Price_Max =df['High'].max()
print(Price_Max)


Diff = Price_Max-Price_Min
level1 = Price_Max - 0.236 * Diff
level2 = Price_Max - 0.382 * Diff
level3 = Price_Max - 0.618 * Diff
print( "Level", "Price")
print( "0 ", Price_Max)
print( "0.236", level1)
print ("0.382", level2)
print ("0.618", level3)
print ("1 ", Price_Min)

plt.axhspan(level1, Price_Min, alpha=0.4, color='lightsalmon')
plt.axhspan(level2, level1, alpha=0.5, color='palegoldenrod')
plt.axhspan(level3, level2, alpha=0.5, color='palegreen')
plt.axhspan(Price_Max, level3, alpha=0.5, color='powderblue')

plt.plot(df.Close, color='black')

plt.ylabel("Price")
plt.xlabel("Date")
plt.show()


# # Mass Coverage

# In[ ]:


allTicks = pd.read_csv("/Users/ethan/Desktop/NYSE.csv")
arr = allTicks['Symbol']
arr = arr.tolist()
arr.remove("AAC")
arr.remove("ABRN")

arr.remove("AAV")
arr.remove("ABX")
arr.remove("ACMP")
arr.remove("ACW")
arr.remove("AEC")


# In[ ]:




for x in index:
    print(x)
    df = web.DataReader(x, 'yahoo', start = date(2000,1,1), end = date(2020,12,31))
    
    
    
    rsi = talib.RSI(df["Adj Close"])


    nums = df['Adj Close']
    nums = nums.tolist()
    values = []
    for x in nums:
        values.append(round(x,2))

    count = 0
    shares = 0
    profit = 0
    firstSell = 0
    anyTrade = 0
    
    initialValue = 0


    length = len(rsi)
    length2 = len(values)

    otherProfit = (values[length2-1] - values[length2-length])
    
    for i in range(length-1):
        if rsi[i] < 25 and shares == 0:
            profit -= values[i]
            initialValue = values[i]
            shares= 1
        elif values[i] >= initialValue*1.25 and shares == 1:
            profit += values[i]
            shares = 0

    if shares == 1:
        profit += values[length-1]
        shares == 0
        

    for i in range(length-1):
        if rsi[i] > 85 and shares == 0:
            profit += values[i]
            initialValue = values[i]
            shares= 1
        elif values[i] <= initialValue*1.25 and shares == 1:
            profit += values[i]
            shares = 0
        

    if shares == 1:
        profit += values[length-1]
        shares == 0
        #print(str(values[length-1]) + " sold -- profit is currently at " + str(profit))

    print(str(profit))
    print(otherProfit)
    print("")
    

