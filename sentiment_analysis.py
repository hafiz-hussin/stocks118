# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:40:49 2019
@author: Keh-Soon.Yong
"""

import pandas as pd
import re

#%%% Extract - Import Data from GBQ

## ensure private key is in the same execution folder

project_id = 'datamining-118118'
pkey = 'datamining-118118-8190b6341891.json'

## StocksPrice
sql = "SELECT * FROM Stocks.StockPrice_"
StocksPrice_ = pd.read_gbq(sql,
         project_id=project_id,
         private_key=pkey,
         dialect='standard',
         verbose=False)

## BusinessNews_
sql = "SELECT * FROM Stocks.BusinessNews_"
BusinessNews_ = pd.read_gbq(sql,
         project_id=project_id,
         private_key=pkey,
         dialect='standard',
         verbose=False)

## BusinessNews_
sql = "SELECT * FROM Stocks.ForumPosts_"
ForumPosts_ = pd.read_gbq(sql,
         project_id=project_id,
         private_key=pkey,
         dialect='standard',
         verbose=False)

## StocksTweetFeed_
sql = "SELECT * FROM Stocks.StocksTweetFeed_"
StocksTweetFeed_ = pd.read_gbq(sql,
         project_id=project_id,
         private_key=pkey,
         dialect='standard',
         verbose=False)

## Commodities_
sql = "SELECT * FROM Stocks.Commodities_"
Commodities_ = pd.read_gbq(sql,
         project_id=project_id,
         private_key=pkey,
         dialect='standard',
         verbose=False)

## MajorForex_
sql = "SELECT * FROM Stocks.MajorForex_"
MajorForex_ = pd.read_gbq(sql,
         project_id=project_id,
         private_key=pkey,
         dialect='standard',
         verbose=False)

## MajorIndices_
sql = "SELECT * FROM Stocks.MajorIndices_"
MajorIndices_ = pd.read_gbq(sql,
         project_id=project_id,
         private_key=pkey,
         dialect='standard',
         verbose=False)

#%% Initial Filter Phase

## Business News : Standardize Key to Stock Name instead of Stock Code
all_quotes = StocksPrice_.groupby(['Code','Quote']).count().reset_index().set_index('Code')['Quote']
all_quotes.drop('7165',inplace=True)   # drop counter 7165, duplicate names
BusinessNews__ = BusinessNews_.join(all_quotes,on='Code',how='left').drop(columns=['Code'])
BusinessNews__.dropna(inplace=True)

## Filter selected Industry
#selected_sectors = ['Banking']
#selected_quotes = StocksPrice_[StocksPrice_['Sector'].isin(selected_sectors)].Quote.unique()
selected_quotes=['AIRASIA','AIRPORT','TOPGLOV','GENTING','IWCITY','EKOVEST','TENAGA','GAMUDA']
## Filter selected symbols
selected_currencies =  ['USD','CNY','SGD']
selected_indices =     ['^IXIC','^DJI']
selected_commodities = ['CL=F','GC=F','ZG=F']
Commodities__  = Commodities_[Commodities_.Symbol.isin(selected_commodities)]
MajorIndices__ = MajorIndices_[MajorIndices_.Symbol.isin(selected_indices)]
MajorForex__   = MajorForex_[MajorForex_.CurrencyCode.isin(selected_currencies)]
BusinessNews__ = BusinessNews__[BusinessNews__.Quote.isin(selected_quotes)]
StocksPrice__  = StocksPrice_[StocksPrice_.Quote.isin(selected_quotes)]


## Filter date range, filter out weekend
from_date = '2019-03-02'   ## start from date
to_date   = '2019-05-11'   ## end date
dates = pd.DataFrame( pd.date_range(start = from_date, end = to_date, freq='D').to_series().dt.dayofweek, columns=['Weekday'])
dates = dates[dates.Weekday<=4]
dates.index.name = 'DT'
StocksPrice__  = StocksPrice__ [(StocksPrice__.Date>=from_date)  & (StocksPrice__.Date<=to_date)].sort_values(by=['Date','Quote'])
BusinessNews__ = BusinessNews__[(BusinessNews__.Date>=from_date) & (BusinessNews__.Date<=to_date)].sort_values(by=['Date','Quote'])
MajorIndices__ = MajorIndices__[(MajorIndices__.Date>=from_date) & (MajorIndices__.Date<=to_date)].sort_values(by=['Date'])
MajorForex__   = MajorForex__  [(MajorForex__.Date>=from_date)   & (MajorForex__.Date<=to_date)].sort_values(by=['Date'])
Commodities__  = Commodities__ [(Commodities__.Date>=from_date)  & (Commodities__.Date<=to_date)].sort_values(by=['Date'])

#%% Sentiment Scoring on Business News

## Translate Chinese to English,  new column : Tranlsated_Title
from py_translator import Translator
translator = Translator()
BusinessNews__['Translated_Title'] = BusinessNews__.Title.apply( 
        lambda x: translator.translate(text=x, dest='en').text if (re.findall('[\u4e00-\u9fff]+', x)) else x)

## Sentiment Scoring
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
BusinessNews__['Polarity'] = BusinessNews__.Translated_Title.apply(lambda x: sid.polarity_scores(x)['compound'])
BusinessNews__[['Quote','Title','Translated_Title','Polarity']].head()

#%% Clean and Join Data For Symbols

### We Want These Columns
indices_cols = ['Date','Symbol','LastPrice','Change','ChgPct','Volume']
forex_cols   = ['Date','CurrencyCode','MYRPerUnit']
commodities_cols = ['Date','Symbol','LastPrice','Change','ChgPct','Volume']
business_cols = ['Date','Quote','Polarity']

MajorIndices = MajorIndices__[indices_cols].rename(columns={'Date':'DT'})
Commodities  = Commodities__[indices_cols].rename(columns={'Date':'DT'})
MajorForex   = MajorForex__[forex_cols].rename(columns={'Date':'DT'})
BusinessNews = BusinessNews__[business_cols].rename(columns={'Date':'DT'})
StocksPrice  = StocksPrice__.drop(columns=['Name','Sector','Code','UpdateDate']).rename(columns={'Date':'DT'})

## Set Index to all Dataframe
dts = pd.DataFrame( dates.index).set_index('DT')
MajorIndices.set_index('DT', inplace=True)
Commodities.set_index('DT', inplace=True)
MajorForex.set_index('DT', inplace=True)
StocksPrice.set_index('DT', inplace=True)
BusinessNews.set_index('DT', inplace=True)

## Major Indices
### Pivot to Date, then join to Dates table
Indices = MajorIndices.pivot(columns='Symbol')      ## transpose selected indices
Indices = dts.join(Indices)                         ## we want series of dates
##### We interpolate, forwardfill and backward fill missing data
Indices.interpolate(method='nearest',inplace=True)  ## for intermediat eNAs
Indices.fillna(method='ffill', inplace=True)        ## for trailing NAs
Indices.fillna(method='bfill', inplace=True)        ## for heading NAs

### Forex
### Pivot to Date, then join to Dates table
Forex = MajorForex.pivot(columns='CurrencyCode')  ## transpose selected indices
Forex = dts.join(Forex)                           ## we want series of dates
#### Fill up missing data
Forex.interpolate(method='nearest',inplace=True)  ## for intermediat eNAs
Forex.fillna(method='ffill', inplace=True)        ## for trailing NAs
Forex.fillna(method='bfill', inplace=True)        ## for heading NAs

### Commodities
### Pivot to Date, then join to Dates table
Commo = Commodities.pivot(columns='Symbol')       ## transpose selected indices
Commo = dts.join(Commo)                           ## we want series of dates
#### Fill up missing data
Commo.interpolate(method='nearest',inplace=True)  ## for intermediat eNAs
Commo.fillna(method='ffill', inplace=True)        ## for trailing NAs
Commo.fillna(method='bfill', inplace=True)        ## for heading NAs

## Stock Price
Price = dts.join(StocksPrice)

#%% Clean and Join Data for Stocks

## News
News = BusinessNews.groupby(['DT','Quote']).mean()

### Initialize Clean Data
clean_data = pd.DataFrame()
scaled_data = pd.DataFrame()

### Loop through every selected quotes
for i in selected_quotes:
    
    Price_ = Price[Price.Quote==i]  # subselect the specific stock quote
    ## fix ChgPct Error, recalculate
    new_pct_chg = Price_.LastPrice.pct_change()*100
    new_pct_chg[0] = Price_.ChgPct[0]  # first_new_pct
    Price_['ChgPct'] = new_pct_chg
    
    n2 = News.reset_index()
    News_ = n2[n2.Quote==i][['DT','Polarity']].set_index('DT')  #subselect specific news
    News_ = dts.join(News_)
    News_.interpolate(method='nearest',inplace=True)  ## for intermediat eNAs
    News_.fillna(method='ffill', inplace=True)        ## for trailing NAs
    News_.fillna(method='bfill', inplace=True)        ## for heading NAs
    clean_data_ = dates.join(Price_).join(Forex).join(Indices).join(Commo).join(News_)
    
    ## new features
    clean_data_['Spread'] = clean_data_.apply(lambda x : x.High-x.Low, axis=1)
    clean_data_['SMA3'] = clean_data_.LastPrice.rolling(window=3).mean()
    clean_data_ = clean_data_.fillna(method='bfill')  #backfill SMA3 for first two rows
    
    ## Binary Encoding for Weekday
    clean_data_['Weekday'] = clean_data_.index.to_series().dt.day_name()
    clean_data_ = pd.get_dummies(clean_data_, columns=["Weekday"], prefix=['WD'])
    
    ## Create label UP (True or False)
    clean_data_['UP'] = clean_data_.ChgPct.apply(lambda x: True if x>0 else False).shift(-1) ## create label
    clean_data_ = clean_data_[:-1]  # drop last row
    
    ## clean up nice column names
    clean_data_.rename(columns={
            ('LastPrice', '^DJI') : 'DJI_LastPrice', 
            ('LastPrice', '^IXIC'): 'IXIC_LastPrice', 
            ('Change', '^DJI')    : 'DJI_Change',
            ('Change', '^IXIC')   : 'IXIC_Change',   
            ('ChgPct', '^DJI')    : 'DJI_ChgPct', 
            ('ChgPct', '^IXIC')   : 'IXIC_ChgPct',
            ('Volume', '^DJI')    : 'DJI_ChgVolume',    
            ('Volume', '^IXIC')   : 'IXIC_ChgVolume',  
            ('MYRPerUnit', 'CNY') : 'MYRCNY_PerUnit',
            ('MYRPerUnit', 'SGD') : 'MYRSGD_PerUnit',  
            ('MYRPerUnit', 'USD') : 'MYRUSD_PerUnit', 
            ('LastPrice', 'CL=F') : 'CLF_LastPrice',
            ('LastPrice', 'GC=F') : 'GCF_LastPrice',   
            ('LastPrice', 'ZG=F') : 'ZGF_LastPrice',        
            ('Change', 'CL=F')    : 'CLF_Change',
            ('Change', 'GC=F')    : 'GCF_Change',  
            ('Change', 'ZG=F')    : 'ZGF_Change',  
            ('ChgPct', 'CL=F')    : 'CLF_ChgPct',
            ('ChgPct', 'GC=F')    : 'GCF_ChgPct',  
            ('ChgPct', 'ZG=F')    : 'ZGF_ChgPct',  
            ('Volume', 'CL=F')    : 'CLF_Volume',
            ('Volume', 'GC=F')    : 'GCF_Volume',  
            ('Volume', 'ZG=F')    : 'ZGF_Volume'}, inplace=True)
    ## append to consolidates data frame
    clean_data = clean_data.append(clean_data_)
    
    ### Scale Data
    features_to_scale = ['DJI_LastPrice', 'IXIC_LastPrice', 'DJI_Change', 'IXIC_Change',
            'DJI_ChgPct', 'IXIC_ChgPct', 'DJI_ChgVolume', 'IXIC_ChgVolume',
            'MYRUSD_PerUnit', 'CLF_LastPrice',
            'GCF_LastPrice', 'ZGF_LastPrice', 'CLF_Change', 'GCF_Change',
            'ZGF_Change', 'CLF_ChgPct', 'GCF_ChgPct', 'ZGF_ChgPct', 'CLF_Volume',
            'GCF_Volume', 'MarketCap', 'LastPrice', 'PE', 'DY', 'ROE', 'Open',
            'Low', 'High', 'ChgPct', 'Volume', 'BidPrice', 'BidVolume',
            'AskPrice', 'AskVolume','Polarity','Spread','SMA3']
    clean_data_  = clean_data_.reset_index().set_index(['DT','Quote'])
    scaled_data_ = clean_data_[features_to_scale].transform( lambda x: (x-x.min()) / (x.max()-x.min()) ).fillna(0)  # minmax scaling
    
    ## Binary Encoding for Weekday
    scaled_data_ = scaled_data_.reset_index().set_index(['DT'])
    scaled_data_['Weekday'] = scaled_data_.index.to_series().dt.day_name()
    scaled_data_ = pd.get_dummies(scaled_data_, columns=["Weekday"], prefix=['WD'])
    ## Create label UP (True or False)
    scaled_data_['UP'] = clean_data_.UP.values
    
    ## append to consolidates data frame
    scaled_data = scaled_data.append(scaled_data_)

clean_data.to_csv('data/clean_data.csv')
scaled_data.to_csv('data/scaled_data.csv')
