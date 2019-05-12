# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:40:49 2019
@author: Keh-Soon.Yong
"""

import pandas as pd
import numpy as np
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
selected_quotes=['AIRASIA','AIRPORT','GENTING','GAMUDA']
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
            'MYRCNY_PerUnit', 'MYRSGD_PerUnit', 'MYRUSD_PerUnit', 'CLF_LastPrice',
            'GCF_LastPrice', 'ZGF_LastPrice', 'CLF_Change', 'GCF_Change',
            'ZGF_Change', 'CLF_ChgPct', 'GCF_ChgPct', 'ZGF_ChgPct', 'CLF_Volume',
            'GCF_Volume', 'MarketCap', 'LastPrice', 'PE', 'DY', 'ROE', 'Open',
            'Low', 'High', 'Chg', 'ChgPct', 'Volume', 'BidPrice', 'BidVolume',
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
    
    #scaled_data_['UP'] = scaled_data_.ChgPct.apply(lambda x: True if x>0 else False).shift(-1) ## create label
    #scaled_data_ = scaled_data_[:-1]  # drop last row
    #scaled_data = scaled_data.append(scaled_data_.reset_index())

clean_data.to_csv('data/clean_data.csv')
scaled_data.to_csv('data/scaled_data.csv')

#%% Fix Correlation

## Build Correlation Matrix
corr_matrix = scaled_data.corr()
## Discover correlation pair above >0.8
sol = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False)
df = sol[sol.abs()>0.9]
print('Colleration')
print( df )

reduced_features = ['Quote','DJI_LastPrice', 'IXIC_LastPrice', 'DJI_Change', 'IXIC_Change',
            'MYRUSD_PerUnit', 'CLF_LastPrice',
            'GCF_LastPrice', 'ZGF_LastPrice', 'CLF_Change', 'GCF_Change',
            'ZGF_Change', 'CLF_Volume',
            'GCF_Volume', 'MarketCap', 'LastPrice', 'PE', 'DY', 'ROE', 'Open',
            'Chg', 'Volume', 'BidVolume',
            'AskVolume','Polarity','UP']

scaled_data = scaled_data[reduced_features]
scaled_data.to_csv('data/scaled_data_reduced_features.csv')
print('Clean data features (row,col): ', clean_data.shape)
print('Reduced features to (row,col): ', scaled_data.shape)

#%% Splitting Data

split_ratio = 0.7
train_x = train_y = test_x = test_y = pd.DataFrame()

for i in selected_quotes:
       
    scaled_data_ = scaled_data[scaled_data.Quote==i]
    
    #### Partiion Train/Test
    train_len = round(len(scaled_data_)*split_ratio)
    train_data_ = scaled_data_[:train_len]
    test_data_  = scaled_data_[train_len:]

    #### Train/Test Dataframe
    train_x_, train_y_ = train_data_.iloc[:,:-1], train_data_.iloc[:,-1].values   # Features columns
    test_x_,  test_y_  = test_data_.iloc[:,:-1], test_data_.iloc[:,-1].values     # Label Column
    train_x_.drop(columns=['Quote'], inplace=True)
    test_x_.drop(columns=['Quote'], inplace=True)
    
    train_y_ = pd.DataFrame(train_y_)
    test_y_  = pd.DataFrame(test_y_)
    
    train_x = train_x.append(train_x_)
    train_y = train_y.append(train_y_)
    test_x  = test_x.append(test_x_)
    test_y  = test_y.append(test_y_)

test_y = test_y[0]# convert from datafraeme to array
train_y = train_y[0]# convert from datafraeme to array

## Baseline Before Modeling
freq_table = pd.crosstab(index=train_y,  columns="count")
print('Training Set Baseline : ')
print(freq_table/freq_table.sum())

## Baseline Before Modeling
freq_table = pd.crosstab(index=test_y,  columns="count")
print('\nTest Set Baseline : ')
print(freq_table/freq_table.sum())

train_y = train_y.tolist()
test_y  = test_y.tolist()

#%% XGBoost Baseline

import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, recall_score, roc_auc_score, roc_curve, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score

xgb_model = xgb.XGBClassifier(random_state=123)
print('XGB 5xCV with Default Params:\n')
print(xgb_model)
# Cross validation scores, default binary CV sampling is stratified
precision_scores = cross_val_score(xgb_model, train_x, train_y, cv=5, scoring='precision')
print("\nPrecision-score = ",precision_scores,"\n\nMean Precision score = ",np.mean(precision_scores))
 
#%% XGBoost Best Param Search Grid

params = {
     'learning_rate': [0.01],
     'n_estimators': [900,1000],
     'max_depth':[7,8],
     'reg_alpha':[0.3, 0.4, 0.5]
    }

print('Grid Parameters Setup, using 5xCV on :\n')
print ( params )
print( '\n')

# Initializing the XGBoost Regressor
xgb_model = xgb.XGBClassifier(random_state=12345)
# Define GirdSearch, default CV is stratified on binary outcome
gsearch = GridSearchCV(xgb_model, params, scoring='precision', verbose=True, cv=5, n_jobs=8)
# Run the Search
gsearch.fit(train_x, train_y)
#Printing the best chosen params
print("\nBest Parameters :",gsearch.best_params_)

#%% XGBoost Model (CV)

best_params = {'objective':'binary:logistic', 'booster':'gbtree', 'random_state': 12345 }
# Updating the parameter as per grid search
best_params.update(gsearch.best_params_)
# Initializing the XGBoost Regressor
xgb_model = xgb.XGBClassifier(**best_params)
print("\nXGB 5xCV with Best Parameters:\n")
print(xgb_model)
 
# Cross validation scores
print("\nFitting XGB on Best Parameters with CV ...\n")
precision_scores = cross_val_score(xgb_model, train_x, train_y, cv=5, scoring='precision', n_jobs=8)
print("\nPrecision_scores per fold : ", precision_scores," \n\nMean Precision_score= ",np.mean(precision_scores ))

#%% XGBoost with 70/30

# Model with best params dsicovered earlier
xgb_model = xgb.XGBClassifier(**best_params)

# train model with best param using training data
xgb_model.fit( train_x, train_y )

# predict on train data
y_pred = xgb_model.predict(test_x)

print(xgb_model)

precision = precision_score(test_y, y_pred)

print('\nPrecision Score Result: ')
print( precision )

#%% Logistic Regression
from sklearn.linear_model import LogisticRegression

print('Training with Logistic Regression')
logreg = LogisticRegression()
logreg.fit(train_x, train_y)
print(logreg)

y_pred = logreg.predict(test_x)
precision = precision_score(test_y, y_pred)
print('\nPrecision Score Result: ')
print( precision )
print('\n\n')

#%%% Progressive LogReg
#### LogReg has better result, let's do prorgressive model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logreg = LogisticRegression()
train_data = scaled_data.reset_index().drop(columns=['Quote'])

def get_lr_precision(days):
    
    predictions = pd.DataFrame()
    #low_bound_date = train_x.index.min() + pd.offsets.Day(+10)
    low_bound_date = train_x.index.max() - pd.offsets.Day(days)
    y_pred =list()
    
    for d in test_x.index.unique():
        #d2 = pd.datetime.strftime(low_bound_date,'%Y-%m-%d')
        train_data_ = train_data[(train_data.DT < d) & (train_data.DT >= low_bound_date)]
        test_data_  = train_data[train_data.DT == d]
        #print(low_bound_date, d)
        #print('Train size: ', train_data_.shape, 'Test size: ', test_data_.shape)
        low_bound_date = low_bound_date + pd.offsets.Day(+1)
    
        #### Train/Test Dataframe    
        train_x_, train_y_ = train_data_.iloc[:,:-1], train_data_.iloc[:,-1].values   # Features columns
        test_x_,  test_y_  = test_data_.iloc[:,:-1], test_data_.iloc[:,-1].values     # Label Column
        test_x_.drop(columns=['DT'], inplace=True)    # DT is not required for modeliing
        train_x_.drop(columns=['DT'], inplace=True)   # DT is not requried for modeling
              
        train_y_ = train_y_.tolist()
        test_y_  = test_y_.tolist()
        
        logreg.fit(train_x_, train_y_)
        y_pred_ = logreg.predict(test_x_)
        #print(test_y_)
        #print(y_pred_)
        df_ =  pd.DataFrame( {'test_y':test_y_, 'y_pred':y_pred_ })
        #print(df_)
        predictions = predictions.append(df_)
        y_pred.append(y_pred_)
    
    precision = precision_score(predictions.test_y, predictions.y_pred)
    predictions.to_csv('data/prediction_'+str(days)+'.csv')
    
    return precision

### Search, how many days for best model
precision_range = pd.DataFrame(columns=['Days','Precision'])
for i in range(6, len(train_data.DT.unique())):
    pre = get_lr_precision(i)
    precision_range.loc[len(precision_range)] = [ i, get_lr_precision(i)]

### plot chart
precision_range.plot(x='Days', y='Precision', title='Precision over Windows Size', fontsize=18)

### get max_precision
max_precision = precision_range.iloc[precision_range.Precision.idxmax()]['Precision']
max_day = precision_range.iloc[precision_range.Precision.idxmax()]['Days']
print('Maximum Logistic Regression Preceision with Windows Size', max_day, ' : ',max_precision)

precision_range.to_csv('./data/precision_range.csv')
