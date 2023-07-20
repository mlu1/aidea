import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
torch.manual_seed(0)
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import reverse_geocoder as rg

from math import sin, cos, sqrt, atan2, radians
import time
import datetime
from scipy.signal import argrelextrema
from xgboost import XGBRegressor
import optuna


def distance_km(lat1,lon1,lat2,lon2):
# approximate radius of earth in km
    R = 6373.0

    lt1 = radians(lat1)
    ln1 = radians(lon1)
    lt2 = radians(lat2)
    ln2 = radians(lon2)

    dlon = ln2 - ln1
    dlat = lt2 - lt1

    a = sin(dlat / 2)**2 + cos(lt1) * cos(lt2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance_km = R * c
    
    return distance_km


def convert_date(s):
    all_times = []
    for date in s:
        timestamp = time.mktime(datetime.datetime.strptime(date, "%Y/%m/%d").timetuple())
        all_times.append(timestamp)
    return all_times


def data_input(file_name):
    train = pd.read_csv('w_data/'+file_name)
    return train


from sklearn.cluster import KMeans
def cluster(data):
  '''
  input: dataframe containing Latitude(x) and Longitude(y) coordinates
  output: series of cluster labels that each row of coordinates belongs to.
  '''
  model = KMeans(n_clusters=50)
  labels = model.fit_predict(data)
  return labels


import matplotlib.pyplot as plt



def find_season(month, hemisphere):
    if hemisphere == 'Southern':
        season_month_south = {
            12:'Summer', 1:'Summer', 2:'Summer',
            3:'Autumn', 4:'Autumn', 5:'Autumn',
            6:'Winter', 7:'Winter', 8:'Winter',
            9:'Spring', 10:'Spring', 11:'Spring'}
        return season_month_south.get(month)
        
    elif hemisphere == 'Northern':
        season_month_north = {
            12:'3', 1:'3', 2:'3',
            3:'4', 4:'4', 5:'4',
            6:'1', 7:'1', 8:'1',
            9:'2', 10:'2', 11:'2'}
        return season_month_north.get(month)
    else:
        print('Invalid selection. Please select a hemisphere and try again')


from sklearn.decomposition import PCA
def pca(data):
  '''
  input: dataframe containing Latitude(x) and Longitude(y)
  '''
  coordinates = data[['Lat','Lon']].values
  pca_obj = PCA().fit(coordinates)
  pca_x = pca_obj.transform(data[['Lat', 'Lon']])[:,0]
  pca_y = pca_obj.transform(data[['Lat', 'Lon']])[:,1]
  return pca_x, pca_y


season_list = []
hemisphere = 'Northern'
df = data_input('super_train1.csv')
print(df.tail(20))
df = df.replace(to_replace ='X',value =0)
df = df.replace(to_replace ='T',value =0)
df= df.replace(to_replace ='/',value =0)
df= df.replace(to_replace ='...',value =0)

pca_x ,pca_y = pca(df) 
df['pca_x'] = pca_x
df['pca_y'] = pca_y
train_times = convert_date(df['Date'])
df['Dates'] = train_times
k_means_cluster = cluster(df[['Lat','Lon']])
df['cluster'] = k_means_cluster 
df['x_data']= pd.to_datetime(df['Date']) 
df['x_month'] = df['x_data'].dt.month


for month in df['x_month']:
    season = find_season(month, hemisphere)
    season_list.append(season)


df['season'] = season_list
df['x_day'] = df['x_data'].dt.day
df['dayofweek'] = df['x_data'].dt.dayofweek
df['dayofweek_name'] = df['x_data'].dt.day_name()
df['is_weekend'] = np.where(df['dayofweek_name'].isin(['Sunday','Saturday']),1,0)
df['Quarter'] = df['x_data'].dt.quarter
df['DayOfYear'] = df['x_data'].dt.dayofyear
df['ismonthstart'] = df['x_data'].dt.is_month_start
df['ismonthend'] = df['x_data'].dt.is_month_end
df['Cos Angle Val'] = np.cos(np.radians(df['Angle']))
df['x_year'] = df['x_data'].dt.year
df['x_week'] = df['x_data'].dt.week
df['day'] = df['x_data'].dt.day

df['Td dew point'] = df['Td dew point'].astype(float).astype(float)
df['Temperature'] = df['Temperature'].astype(float).astype(float)



df['DhPoint / Temperature'] = df.apply(lambda row: 0 if row['Temperature']==0 else row['Td dew point']/row['Temperature'], axis=1)
df['Humidity * Water'] = df['RH'].astype(str).astype(float) * df['Precp'].astype(str).astype(float)
df['minHumidity * Water'] = df['RHMin'].astype(str).astype(float) * df['Precp'].astype(str).astype(float)
df['minHumidity * Wind'] = df['RHMin'].astype(str).astype(float) * df['WS'].astype(str).astype(float)

import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3

print(df.dtypes)

df['StnPresMin'] = df['StnPresMin'].astype(str).astype(float)
df['Td dew point'] = df['Td dew point'].astype(str).astype(float)
df['Airmass'] = pvlib.atmosphere.get_relative_airmass(df['Angle'].values)
df['Cos Angle Val'] = np.cos(np.radians(df['Angle']))
df['Airmass_2'] = pvlib.atmosphere.get_absolute_airmass(df['Airmass'].values, df['StnPresMin'].values*100)
df['am_h2o'] = df['Airmass_2'].astype(str).astype(float) * df['Precp'].astype(str).astype(float)
df['Dew Point Var'] = np.exp(0.07 * df['Td dew point'] - 0.075)

loc_values = [0,1,2,3,4,5,6,7,8,9,10]
Loc_names= ['Xiushui','Lukang','Lukang2','Lukang3','Xiushui2','Xiushui3','Xiushui4','Lukang4','Xinwu','Guanyin','Luzhu']


loc_res = {}
for key in Loc_names:
    for value in loc_values:
        loc_res[key] = value
        loc_values.remove(value)
        break 
print(loc_res)

df = df.replace({"Location" : loc_res})
print(df['Location'])
print(df.head(10))

for col in df.columns:
  df[col] = df[col].fillna(0)


y = df['Generation']
modules = df.groupby(['Module']).size().reset_index().rename(columns={0:'count'})
test_values = [0,1,2,3]
modules_list = list(modules['Module'])

res = {}
for key in modules_list:
    for value in test_values:
        res[key] = value
        test_values.remove(value)
        break 

print(res)
df = df.replace({"Module" : res})
print(df['Module'].head(10))

column_names = ['x_date','Generation','ID','Date','dayofweek_name','x_data','StnPresMaxTime','StnPresMinTime','T Max Time','RHMinTime','T Min Time','WGustTime']
df = df.drop(column_names,axis=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
X_train_std = StandardScaler().fit_transform(df)


rfecv = RFECV(
    estimator=RandomForestRegressor(),
    min_features_to_select=40,
    step=5,
    n_jobs=-1,
    scoring="r2",
    cv=5,
)

test_data = data_input('super_test1.csv')
test_data = test_data.replace(to_replace ="X",value =0)
test_data = test_data.replace(to_replace ="T",value =0)
test_data= test_data.replace(to_replace ='/',value =0)
test_data= test_data.replace(to_replace ='...',value =0)


pca_test_x ,pca_test_y = pca(test_data) 
test_data['pca_x'] = pca_test_x
test_data['pca_y'] = pca_test_y
test_times = convert_date(test_data['Date'])
test_data['Dates'] = test_times

k_means_cluster_test = cluster(test_data[['Lat','Lon']])
test_data['cluster'] = k_means_cluster_test 

y_test = test_data['Generation']
test_data['x_data']= pd.to_datetime(test_data['Date']) 
test_data['x_month'] = test_data['x_data'].dt.month

test_season_list = []

for month in test_data['x_month']:
    season_test = find_season(month, hemisphere)
    test_season_list.append(season_test)
    
test_data['season'] = test_season_list

test_data['x_day'] = test_data['x_data'].dt.day
test_data['dayofweek'] = test_data['x_data'].dt.dayofweek
test_data['dayofweek_name'] = test_data['x_data'].dt.day_name()
test_data['is_weekend'] = np.where(test_data['dayofweek_name'].isin(['Sunday','Saturday']),1,0)
test_data['Quarter'] = test_data['x_data'].dt.quarter
test_data['DayOfYear'] = test_data['x_data'].dt.dayofyear
test_data['ismonthstart'] = test_data['x_data'].dt.is_month_start
test_data['ismonthend'] = test_data['x_data'].dt.is_month_end
test_data['Cos Angle Val'] = np.cos(np.radians(test_data['Angle']))
test_data['x_year'] = test_data['x_data'].dt.year
test_data['x_week'] = test_data['x_data'].dt.week
test_data['day'] = test_data['x_data'].dt.day
test_data['Humidity * Water'] = test_data['RH'].astype(str).astype(float) * test_data['Precp'].astype(str).astype(float)
test_data['minHumidity * Water'] = test_data['RHMin'].astype(str).astype(float) * test_data['Precp'].astype(str).astype(float)
test_data['minHumidity * Wind'] = test_data['RHMin'].astype(str).astype(float) * test_data['WS'].astype(str).astype(float)

test_data['StnPresMin'] = test_data['StnPresMin'].astype(str).astype(float)
test_data['Td dew point'] = test_data['Td dew point'].astype(str).astype(float)
#['Td dew point'] = df['Td dew point'].astype(float).astype(float)
test_data['Temperature'] = test_data['Temperature'].astype(float).astype(float)


test_data['DhPoint / Temperature'] = test_data.apply(lambda row: 0 if row['Temperature']==0 else row['Td dew point']/row['Temperature'], axis=1)
test_data['Airmass'] = pvlib.atmosphere.get_relative_airmass(test_data['Angle'].values)
test_data['Cos Angle Val'] = np.cos(np.radians(test_data['Angle']))
test_data['Airmass_2'] = pvlib.atmosphere.get_absolute_airmass(test_data['Airmass'].values, test_data['StnPresMin'].values*100)
test_data['am_h2o'] = test_data['Airmass_2'].astype(str).astype(float) * test_data['Precp'].astype(str).astype(float)
test_data['Dew Point Var'] = np.exp(0.07 * test_data['Td dew point'] - 0.075)


test_data = test_data.replace({"Location" : loc_res})
print(test_data['Location'])


X_tester = test_data.drop(column_names,axis=1)
X_tests = X_tester.replace({"Module" : res})

print(X_tests.columns)

for col in X_tests.columns:
    X_tests[col] = X_tests[col].fillna(0)

#X_trainval, X_val, y_trainval, y_val = train_test_split(X_totals, y, test_size=0.16, random_state=21)
#X_train, X_cal, y_train, y_cal = train_test_split(X_trainval, y_trainval, test_size=0.1, random_state=69)


train_pct_index = int(0.85 * len(df))
print(train_pct_index)
cal =  int((len(df) - train_pct_index)/2) + train_pct_index
print(cal)


#X_trainval, X_val, y_trainval, y_val = train_test_split(X_t, y, test_size=0.25, random_state=21)
#X_train, X_cal, y_train, y_cal = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=69)

X_train, X_val = df[:train_pct_index], df[train_pct_index:cal]
y_train, y_val = y[:train_pct_index], y[train_pct_index:cal]
X_cal,y_cal = df[cal:], y[cal:]


scaler =  MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_tests = scaler.transform(X_tests)
X_cal = scaler.transform(X_cal)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_tests, y_test = np.array(X_tests), np.array(y_test)
X_cal, y_cal = np.array(X_cal), np.array(y_cal)

print(X_train.shape)
print(y_train.shape)


datasets = {'x_train': X_train,
            'y_train': y_train,
            'x_val': X_val,
            'y_val': y_val,
            'x_test': X_tests,
            'y_test': y_test,
            }


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 8)
    early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 800, 2500)
    n_estimators = trial.suggest_int("n_estimators", 0, 10000)
    
    model = XGBRegressor(
        random_state=42,
        tree_method="gpu_hist",
        gpu_id=0,
        predictor="gpu_predictor",
        #tree_method = "exact",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
    )
    model.fit(
        datasets['x_train'],
        datasets['y_train'],
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(datasets['x_val'], datasets['y_val'])],
        verbose=1000,
    )
    preds_valid = model.predict(datasets['x_val'])
    rmse = mean_squared_error(datasets['y_val'], preds_valid, squared=False)
    r_square = r2_score(y_val, preds_valid)
    
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
optuna_params = study.best_params


model = XGBRegressor(
    random_state=42,
    tree_method="gpu_hist",
    #tree_method = "exact",
    gpu_id=0,
    predictor="gpu_predictor",
    **optuna_params
)

model.fit(
    datasets['x_train'],
    datasets['y_train'],
    early_stopping_rounds=optuna_params['early_stopping_rounds'],
    eval_set=[(datasets['x_val'], datasets['y_val'])],
    verbose=1000,
)

preds_test = model.predict(datasets['x_test'])

submit_data = pd.read_csv('data/submission.csv')
submit_data['Generation'] = preds_test
submit_data.to_csv('submit_ALL.csv',index=None)

