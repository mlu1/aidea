from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,PolynomialFeatures
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering
import gc
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.cluster import AgglomerativeClustering 
from catboost import CatBoostClassifier, Pool, cv
from sklearn.datasets import make_classification
from functools import partial
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import skew, kurtosis
from numpy.fft import fft
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=5)
from sklearn.metrics import log_loss
import hdbscan
neigh = NearestNeighbors(n_neighbors=5)
hdb = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True)
agglo = AgglomerativeClustering(n_clusters=5)
from statsmodels.tsa.seasonal import seasonal_decompose
from model import InceptionTimeClassifier
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
def main():
    

    class IMUDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (B, C, T)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    
    def duplicate_item(item, n):
        return [item] * n
    #    Apply to all swings

    def chunk_xy(X, y, chunk_size):
        """
        Generator yielding batches of (X_chunk, y_chunk) of up to chunk_size rows,
        in order, until both are exhausted. Assumes X and y are NumPy arrays.
        """
        n = len(X)
        assert len(y) == n, "X and y must have the same number of rows"
        for start in range(0, n, chunk_size):
            end = start + chunk_size
            yield X[start:end], y[start:end]
    

    def gaussian_density(n, sigma):
        x = np.arange(n)
        mu = n // 2
        g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
        return g / g.sum()
 
    
    def objective_binary_rgb(trial, X_in, y_in):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'class_weight': 'balanced'
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X_in, y_in, cv=3, scoring='roc_auc')
        return scores.mean()

    def objective_lgb(trial, x_lgb, y_lgb):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 550),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 0.9),
            'max_depth': trial.suggest_int("max_depth", 3, 12),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 0.5, 100.0),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_int("lambda_l1", 0, 100, step=5),
            'lambda_l2': trial.suggest_int("lambda_l2", 0, 100, step=5),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 50),
            'reg_alpha': trial.suggest_uniform('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_uniform('reg_lambda', 1.0, 5.0)
        }
        model = LGBMClassifier(random_state=42, **params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, x_lgb, y_lgb, scoring='roc_auc', cv=skf)
        return np.mean(scores)
 
    def objective_multi_lgb(trial,X_multi_lgb,y_multi_lgb):
        """Optuna objective for LightGBM multiclass with external X, y."""
        n_class = np.unique(y_multi_lgb).size

        params = {
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "n_estimators":      trial.suggest_int("n_estimators", 400, 2000),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "verbosity":         -1,
            "seed":              42,
        }

        model = LGBMClassifier(**params)
        cv    = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        score = cross_val_score(model, X_multi_lgb, y_multi_lgb,cv=cv, scoring="neg_log_loss",n_jobs=-1).mean()
        return -score

    def objective_multi_rgb(trial,X_multi_rgb,y_multi_rgb):
        #    Number of trees in random forest
        n_estimators = trial.suggest_int(name="n_estimators", low=100, high=500, step=100)
        # Maximum number of levels in tree
        max_depth = trial.suggest_int(name="max_depth", low=10, high=110, step=20)

        # Minimum number of samples required to split a node
        min_samples_split = trial.suggest_int(name="min_samples_split", low=2, high=10, step=2)

        # Minimum number of samples required at each leaf node
        min_samples_leaf = trial.suggest_int(name="min_samples_leaf", low=1, high=4, step=1)
    
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
        model = RandomForestClassifier(**params)
    
        cv    = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        score = cross_val_score(model, X_multi_rgb, y_multi_rgb,cv=cv, scoring="neg_log_loss",n_jobs=-1).mean()
        return -score

 

    def objective_cb_multi(trial,X_cb_multi,y_cb_multi,n_class):
        params = {
        "iterations":          trial.suggest_int("iterations", 500, 2000),
        "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth":               trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_seed":         42,
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof = np.zeros((len(X_cb_multi), n_class))

        for tr, vl in cv.split(X_cb_multi, y_cb_multi):
            model = CatBoostClassifier(**params)
            model.fit(X_cb_multi[tr], y_cb_multi[tr],verbose=50)
            oof[vl] = model.predict_proba(X_cb_multi[vl])

        return log_loss(y_cb_multi, oof)



    def objective_cb_binary(trial, X_in, y_in):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 0.0, 20.0),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20)
        }
        
        model = CatBoostClassifier(
            **params,
            loss_function='Logloss',
            eval_metric='AUC',
            verbose=0,
            random_seed=42
        )
        scores = cross_val_score(model, X_in, y_in, cv=3, scoring='roc_auc', n_jobs=-1)
        return scores.mean()


    def gaussian_from_onehot(x, sigma, m=1):
        """
        Calculates the probability density of a Gaussian distribution based on a one-hot encoded array.

        Args:
            x: A 1D numpy array representing a one-hot encoded vector.  Only one element
             should be 1, and the rest should be 0.  The position of the '1'
            indicates the mean of the Gaussian.
            sigma: The standard deviation of the Gaussian distribution.

        Returns:
            A 1D numpy array of the same shape as x, containing the probability density
            values of the Gaussian distribution.  Values beyond 3*sigma from the mean
            are set to 0.
        """

        if not isinstance(x, np.ndarray):
            x = x.values
        if x.ndim != 1:
            raise ValueError("x must be a 1D array")
        if np.sum(x) != 1:
            return x*0
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("x must contain only 0s and 1s")
        if not isinstance(sigma, (int, float)):
            raise TypeError("sigma must be a number")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

           # Find the index of the '1' (the mean)
        mean_index = np.where(x == 1)[0][0]

        #       Create an array of indices corresponding to the positions in x
        indices = np.arange(len(x))

        #    Calculate the Gaussian probability density
        y = norm.pdf(indices, loc=mean_index, scale=sigma)

        #    Set values beyond 3*sigma to 0
        distance_from_mean = np.abs(indices - mean_index)
        y[distance_from_mean > 3 * sigma] = 0

        return y*m


    def reduce_mem_usage(df):
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)                    
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                    elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                        df[col] = df[col].astype(np.uint64)
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
    
        end_mem = df.memory_usage().sum() / 1024**2
        gc.collect()
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        return df
    

    # 若尚未產生特徵，請先執行 data_generate() 生成特徵 CSV 檔案
    # data_generate()
    
    # 讀取訓練資訊，根據 player_id 將資料分成 80% 訓練、20% 測試
    info_test = pd.read_csv('39_Test_Dataset/test_info.csv') 
    unique_test_players = info_test['unique_id'].unique() 
    testpath = './tabular_data_test/' 
    testlist =list(Path(testpath).glob('**/*.csv'))
  
    test_data = pd.DataFrame()
    submission = pd.DataFrame()
    
    

    test_id = []
    for file in tqdm(testlist):
        data1 = pd.read_csv(file, skiprows=1, header=None)
        test_id.append(duplicate_item(str(file)[-8:][0:4],len(data1)))
        test_data = pd.concat([test_data, data1], ignore_index=True)
        
    info = pd.read_csv('39_Training_Dataset/train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
    
    # 讀取特徵 CSV 檔（位於 "./tabular_data_train"）
    datapath = './tabular_data_train/'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']
    
    # 根據 test_players 分組資料
    x_train = pd.DataFrame()
    y_train_all = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test_all = pd.DataFrame(columns=target_mask)
    
    
    #datalist = datalist[0:800]
    
    for file in tqdm(datalist):
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file, skiprows=1, header=None)
            
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train_all = pd.concat([y_train_all, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test_all = pd.concat([y_test_all, target_repeated], ignore_index=True)
    
    
        # 標準化特徵 
     
        
    x_test=x_test.replace([np.inf, -np.inf], np.nan).fillna(0)     
    x_train=x_train.replace([np.inf, -np.inf], np.nan).fillna(0) 
    test_data=test_data.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(x_train)
    print(x_train.shape)
    print(x_test)
    print(x_test.shape)
    print(test_data)
    print(test_data.shape)
   
    x_train[35] = x_train.groupby(0)[4].transform(partial(gaussian_from_onehot, sigma=15, m=100))
    neigh.fit(x_train)
    dist_tr, _ = neigh.kneighbors(x_train)
    x_train[36] = 1 / (dist_tr.mean(axis=1) + 1e-6)
    x_train[37] = gm.fit_predict(x_train)
    
    x_train[38]  = hdb.fit_predict(x_train)
    x_train[39] = hdb.probabilities_
    x_train[40] = agglo.fit_predict(x_train)
  
    lags = [48, 49, 50, 51, 52, 53, 54, 55, 56,57, 60, 62, 68,69]
    windows = [49, 50, 51, 52, 53, 54, 55, 56,57, 60, 61, 67]
    
    p=40
    for window in windows:
        x_train[p+1] = x_train.groupby(37)[5].transform(lambda x: x.shift(1).rolling(window=window, min_periods=0).mean()) 
        x_train[p+1000] = x_train.groupby(37)[5].transform(lambda x: x.shift(1).rolling(window=window, min_periods=0).std())
        p = p+1
    
    gx, gy, gz = x_train[[0,1,2]].values.T
    x_train[70]= np.sqrt(gx**2 + gy**2 + gz**2)
    
    ax, ay, az = x_train[[3,4,5]].values.T
    x_train[100]= np.sqrt(ax**2 + ay**2 + az**2)
    bx, by, bz = x_train[[6,7,8]].values.T
    x_train[101]= np.sqrt(bx**2 + by**2 + bz**2)
    
    cx, cy, cz = x_train[[9,10,11]].values.T
    x_train[102]= np.sqrt(cx**2 + cy**2 + cz**2)
    
    dx, dy, dz = x_train[[12,13,14]].values.T
    x_train[103]= np.sqrt(dx**2 + dy**2 + dz**2)
    x_train[202]= x_train[18] - x_train[20]
    x_train[203]= x_train[21] - x_train[23]
    

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_train = poly.fit_transform(x_train[[4,5,6]])
    poly_cols = [73,74,75,76,77,78]
    poly_df_tr = pd.DataFrame(poly_train, index=x_train.index, columns=poly_cols) 
    x_train = pd.concat([x_train, poly_df_tr], axis=1)
    
    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    lags = [49, 52, 55, 58, 61]
    
    
    print(x_train.head(20))
    x_test[35] = x_test.groupby(0)[4].transform(partial(gaussian_from_onehot, sigma=15, m=100))
    neigh.fit(x_test)
    dist_te, _ = neigh.kneighbors(x_test)
    x_test[36]  = 1 / (dist_te.mean(axis=1)  + 1e-6)
    x_test[37] = gm.fit_predict(x_test)
    lbls_te, probs_te = hdbscan.approximate_predict(hdb, x_test)
    x_test[38]   = lbls_te
    x_test[39]  = probs_te
    x_test[40]  = agglo.fit_predict(x_test)
    
    p=40
    for window in windows:
        x_test[p+1] = x_test.groupby(37)[5].transform(lambda x: x.shift(1).rolling(window=window, min_periods=0).mean()) 
        x_test[p+1000] = x_test.groupby(37)[5].transform(lambda x: x.shift(1).rolling(window=window, min_periods=0).std())
        p=p+1
    
    
    gx, gy, gz = x_test[[0,1,2]].values.T
    x_test[70]= np.sqrt(gx**2 + gy**2 + gz**2)
    

    ax, ay, az = x_test[[3,4,5]].values.T
    x_test[100]= np.sqrt(ax**2 + ay**2 + az**2)
    

    bx, by, bz = x_test[[6,7,8]].values.T
    x_test[101]= np.sqrt(bx**2 + by**2 + bz**2)
    

    cx, cy, cz = x_test[[9,10,11]].values.T
    x_test[102]= np.sqrt(cx**2 + cy**2 + cz**2)
    

    dx, dy, dz = x_test[[12,13,14]].values.T
    x_test[103]= np.sqrt(dx**2 + dy**2 + dz**2)
    x_test[202]= x_test[18] - x_test[20]
    x_test[203]= x_test[21] - x_test[23]
    
  

    poly_test  = poly.transform(x_test[[4,5,6]])
    poly_df_te = pd.DataFrame(poly_test, index=x_test.index, columns=poly_cols)
    x_test  = pd.concat([x_test,  poly_df_te], axis=1) 
    


    test_data[35] =test_data.groupby(0)[4].transform(partial(gaussian_from_onehot, sigma=15, m=100))
    neigh.fit(test_data)
    dist_te, _ = neigh.kneighbors(test_data)
    test_data[36]  = 1 / (dist_te.mean(axis=1)  + 1e-6)
    test_data[37] = gm.fit_predict(test_data)
    lbls_te, probs_te = hdbscan.approximate_predict(hdb, test_data)
    test_data[38]   = lbls_te
    test_data[39]  = probs_te
    test_data[40]  = agglo.fit_predict(test_data)
    
    p=40
    for window in windows:
        test_data[p+1] = test_data.groupby(37)[5].transform(lambda x: x.shift(1).rolling(window=window, min_periods=0).mean()) 
        test_data[p+1000] = test_data.groupby(37)[5].transform(lambda x: x.shift(1).rolling(window=window, min_periods=0).std())
        p=p+1
    
    gx, gy, gz = test_data[[0,1,2]].values.T
    test_data[70]= np.sqrt(gx**2 + gy**2 + gz**2)
    
    ax, ay, az = test_data[[3,4,5]].values.T
    test_data[100]= np.sqrt(ax**2 + ay**2 + az**2)
    

    bx, by, bz = test_data[[6,7,8]].values.T
    test_data[101]= np.sqrt(bx**2 + by**2 + bz**2)
    
    cx, cy, cz = test_data[[9,10,11]].values.T
    test_data[102]= np.sqrt(cx**2 + cy**2 + cz**2)
    

    dx, dy, dz = test_data[[12,13,14]].values.T
    test_data[103]= np.sqrt(dx**2 + dy**2 + dz**2)
    
    test_data[202]= test_data[18] - test_data[20]
    test_data[203]= test_data[21] - test_data[23]
    

    poly_test_data  = poly.transform(test_data[[4,5,6]])
    poly_df_test = pd.DataFrame(poly_test_data, index=test_data.index, columns=poly_cols)
    test_data  = pd.concat([test_data,  poly_df_test], axis=1)
    
    
    x_test=x_test.replace([np.inf, -np.inf], np.nan).fillna(0)     
    x_train=x_train.replace([np.inf, -np.inf], np.nan).fillna(0) 
    test_data=test_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    

    
    scaler = MinMaxScaler()
    le = LabelEncoder()
    X_train_scaled = scaler.fit_transform(x_train)
    
    pca = PCA(n_components=1)
    pc = pca.fit_transform(X_train_scaled)
    b = pc.reshape(-1, 1)      # now shape is (n,1)
    X_train_scaled = np.hstack([X_train_scaled, b])
    kmeans = KMeans(n_clusters=10, random_state=42).fit(X_train_scaled)
    c = kmeans.labels_
    c = c.reshape(-1, 1)
    X_train_scaled = np.hstack([X_train_scaled, c])
   
    x_test=x_test.replace([np.inf, -np.inf], np.nan).fillna(0)     
    x_train=x_train.replace([np.inf, -np.inf], np.nan).fillna(0) 
    test_data=test_data.replace([np.inf, -np.inf], np.nan).fillna(0)



    X_test_scaled = scaler.transform(x_test)
    pc_test = pca.fit_transform(X_test_scaled)
    b_test = pc_test.reshape(-1, 1)      # now shape is (n,1)
    X_test_scaled = np.hstack([X_test_scaled, b_test])
    kmeans = KMeans(n_clusters=10, random_state=42).fit(X_test_scaled)
    c_test = kmeans.labels_
    c_test = c_test.reshape(-1, 1)
    X_test_scaled = np.hstack([X_test_scaled, c_test])
        

    test_scaled = scaler.transform(test_data)
    pc_test = pca.fit_transform(test_scaled)
    b_test = pc_test.reshape(-1, 1)      # now shape is (n,1)
    test_scaled = np.hstack([test_scaled, b_test])
    kmeans = KMeans(n_clusters=10, random_state=42).fit(test_scaled)
    c_test = kmeans.labels_
    c_test = c_test.reshape(-1, 1)
    test_scaled = np.hstack([test_scaled, c_test])
         
              
    group_size = 27
    max_lag=10
    pred_probs=0.5
    def model_binary(X_tr, y_tr, X_test, y_test,test_scaled):
          
        chunk_size = 200000
        
        weights = gaussian_density(len(X_test[0]), sigma=2)
        g_test = X_test @ weights
        g_test = g_test.reshape(-1, 1)
        X_test = np.hstack([X_test, g_test])
   
        row_means = np.mean(X_test, axis=1)   # shape → (n,)
        row_means = row_means.reshape(-1, 1)   # make it (n,1)
        X_test = np.hstack([X_test, row_means])  # now shape → (n, 8)   
        
        lagged_arrays = []
        for lag in range(1, max_lag + 1):
            # roll the data down by `lag` rows
            rolled = np.roll(X_test, shift=lag, axis=0)
            # the first `lag` rows are invalid → set to NaN
            rolled[:lag, :] = np.nan
            lagged_arrays.append(rolled)

    
        lags = np.array(lagged_arrays)
        n_lags, n_samples, n_feats = lags.shape

        # 1️⃣  bring samples to axis 0 → (478905, 10, 11)
        c_swapped = lags.transpose(1, 0, 2)

        # 2️⃣  flatten the last two dims → (478905, 110)
        c2 = c_swapped.reshape(n_samples, n_lags * n_feats)

        # 3️⃣  replace NaNs if you like
        c2 = np.nan_to_num(c2, nan=0.0)

        # 4️⃣  sanity check
        assert X_test.shape[0] == c2.shape[0]
        X_test = np.hstack([X_test, c2])  
        print(X_test.shape) 
               

        for z, (x_t, y_t) in enumerate(chunk_xy(X_tr, y_tr, chunk_size), 1):
            print(z)
            print(len(x_t))
            print(len(y_t))

            weights = gaussian_density(len(x_t[0]), sigma=2)
            g = x_t @ weights
            g = g.reshape(-1, 1)
            x_t = np.hstack([x_t, g])

            
            row_means = np.mean(x_t, axis=1)   # shape → (n,)
            row_means = row_means.reshape(-1, 1)   # make it (n,1)
            x_t = np.hstack([x_t, row_means])  # now shape → (n, 8)   
 
            lagged_arrays = []
            for lag in range(1, max_lag + 1):
                # roll the data down by `lag` rows
                rolled = np.roll(x_t, shift=lag, axis=0)
                # the first `lag` rows are invalid → set to NaN
                rolled[:lag, :] = np.nan
                lagged_arrays.append(rolled)

    
            lags = np.array(lagged_arrays)
            n_lags, n_samples, n_feats = lags.shape

            # 1️⃣  bring samples to axis 0 → (478905, 10, 11)
            c_swapped = lags.transpose(1, 0, 2)

            # 2️⃣  flatten the last two dims → (478905, 110)
            c2 = c_swapped.reshape(n_samples, n_lags * n_feats)

            # 3️⃣  replace NaNs if you like
            c2 = np.nan_to_num(c2, nan=0.0)

            # 4️⃣  sanity check
            assert x_t.shape[0] == c2.shape[0]

            # 5️⃣  hstack!
            x_t = np.hstack([x_t, c2])
             
            
            print('THE SHAPES')
            print(x_t.shape)
            print(X_test.shape)
            #X_reshaped = x_t.reshape((x_t.shape[0], 69, 6))
            X_reshaped = x_t.reshape(-1, 27, 33)  # (num_samples, time_steps, channels)
            X_test_local = X_test.reshape(-1,27,33) 
            X_test_tensor = torch.tensor(X_test_local, dtype=torch.float32).permute(0, 2, 1)

            X_train_re, X_valid_re, y_train_ds, y_valid_ds= train_test_split(X_reshaped,y_t,test_size=0.20)
            train_ds = IMUDataset(X_train_re, y_train_ds)
            val_ds = IMUDataset(X_valid_re, y_valid_ds)

            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            clf_0 = InceptionTimeClassifier(num_classes=2, in_channels=33).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(clf_0.parameters(), lr=1e-3)
            best_auc = 0.0
            epochs = 10

            for epoch in range(epochs):
                clf_0.train()
                running_loss = 0.0

                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = clf_0(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * X_batch.size(0)

                    avg_train_loss = running_loss / len(train_loader.dataset)

                # Validation
                clf_0.eval()
                all_probs = []
                all_targets = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(device)
                        outputs = clf_0(X_batch)
                        probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        all_probs.append(probs)
                        all_targets.append(y_batch.numpy())

                y_true = np.concatenate(all_targets)
                y_pred = np.vstack(all_probs)[:, 1]

                print(y_pred)
                print(y_true)
                val_auc = roc_auc_score(y_true, y_pred)

                print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}")

                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(clf_0.state_dict(), "best_inception_model.pth")
                    print(f"✅ Saved model with AUC: {val_auc:.4f}")
            
            with torch.no_grad():
                clf_0.eval()
                preds = []

                for i in tqdm(range(0, len(X_test_tensor), 128)):  # batch prediction
                    batch = X_test_tensor[i:i+128]
                    batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
                    outputs = clf_0(batch)
                    probs = torch.softmax(outputs, dim=1)
                    preds.append(probs.cpu().numpy())

                # Concatenate results
                probs_all = np.vstack(preds)[:,1]             # shape (N, num_classes)
                #pred_classes = probs_all.argmax(axis=1)  # final class predictions
            
            print(probs_all)
            r1 = roc_auc_score(y_test, probs_all)
            print(f"Test ROC AUC: {r1:.4f}")
            
            # Create loaders
                    
            X_train_cb, X_valid_cb, y_train_cb, y_valid_cb= train_test_split(x_t,y_t,test_size=0.20)


            study_cb_bin = optuna.create_study(direction='maximize')
            study_cb_bin.optimize(lambda t: objective_cb_binary(t,x_t, y_t), n_trials=3)
            best_cb_bin = study_cb_bin.best_params
            print("Best CatBoost params (binary):", best_cb_bin)

            print(X_train_cb.shape)
            print(y_train_cb.shape)
            n_class = len(np.unique(y_t))   
            print(n_class)
            train_pool = Pool(X_train_cb, y_train_cb)
            val_pool   = Pool(X_valid_cb,   y_valid_cb)
    
             
        
            clf = CatBoostClassifier(   
                **best_cb_bin,
                loss_function     = "Logloss",        # use "AUC" if you want it optimised directly
                eval_metric       = "AUC",            # report AUC every 50 trees
                auto_class_weights    = "Balanced", 
                early_stopping_rounds = 100,
                random_state          = 42,
                verbose               = 50            # every n trees
                )
            

            
            
            if z==1:
                clf.fit(train_pool, eval_set=val_pool)
            else:
                clf.fit(train_pool, eval_set=val_pool,init_model='model.cbm')

            clf.save_model('model.cbm')  
            proba = clf.predict_proba(X_test)[:, 1]
            r2 = roc_auc_score(y_test, proba) 
 
            print("Validation CATBOOST  ROC AUC:", roc_auc_score(y_test, proba)) 
            
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda t: objective_lgb(t, X_train_cb, y_train_cb), n_trials=3)
            best = study.best_params
            print('Best LGB multiclass params:', best)
            clf3 = LGBMClassifier(**best, random_state=42) 
            
            if z==1:
                clf3.fit(X_train_cb, y_train_cb,eval_set=[(X_valid_cb, y_valid_cb)])
            else:
                clf3_model = joblib.load('lgbm_model.pkl')
                clf3.fit(X_train, y_train,eval_set=[(X_valid_cb, y_valid_cb)],init_model=clf3_model)
            
            joblib.dump(clf3,'lgbm_model.pkl')  
            y_proba_test = clf3.predict_proba(X_test)[:, 1]
            r3 = roc_auc_score(y_test, y_proba_test)
            print(f"Test ROC AUC: {r3:.4f}")
            
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda t: objective_binary_rgb(t, X_train_cb, y_train_cb), n_trials=3)
            best_random = study.best_params
            print('Best Random binary params:', best)
        

            clf4 = RandomForestClassifier(**best_random,random_state=42,verbose=50)
            clf4.fit(X_train_cb, y_train_cb)
            r4 = clf4.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, r4)
            print(f"Test ROC AUC: {test_auc:.4f}")
            r5 = (r1+r2+r3+test_auc)/4
            print(f"TOTAL ROC AUC: {r5:.4f}")
            
              


        weights = gaussian_density(len(test_scaled[0]), sigma=2)
        g_t = test_scaled @ weights
        g_t = g_t.reshape(-1, 1)
        test_scaled = np.hstack([test_scaled, g_t])
    
        row_means_test = np.mean(test_scaled, axis=1)   # shape → (n,)
        row_means_test = row_means_test.reshape(-1, 1)   # make it (n,1)
        test_scaled = np.hstack([test_scaled, row_means_test])  # now shape → (n, 8)
        print(test_scaled.shape)
        
        lagged_arrays = []
        for lag in range(1, max_lag + 1):
            # roll the data down by `lag` rows
            rolled = np.roll(test_scaled, shift=lag, axis=0)
            # the first `lag` rows are invalid → set to NaN
            rolled[:lag, :] = np.nan
            lagged_arrays.append(rolled)

    
        lags = np.array(lagged_arrays)
        n_lags, n_samples, n_feats = lags.shape

        # 1️⃣  bring samples to axis 0 → (478905, 10, 11)
        c_swapped = lags.transpose(1, 0, 2)

        # 2️⃣  flatten the last two dims → (478905, 110)
        c2 = c_swapped.reshape(n_samples, n_lags * n_feats)

        # 3️⃣  replace NaNs if you like
        c2 = np.nan_to_num(c2, nan=0.0)

        # 4️⃣  sanity check
        assert test_scaled.shape[0] == c2.shape[0]
        test_scaled = np.hstack([test_scaled, c2])   
                
       
        test_scaled_reshape  = test_scaled.reshape(-1, 27, 33)

        # Convert and permute to (B, C, T)
        X_tensor = torch.tensor(test_scaled_reshape, dtype=torch.float32).permute(0, 2, 1)
        with torch.no_grad():
            clf_0.eval()
            preds = []

            for i in tqdm(range(0, len(X_tensor), 128)):  # batch prediction
                batch = X_tensor[i:i+128]
                batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
                outputs = clf_0(batch)
                probs = torch.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())

        # Concatenate results
        probs_all = np.vstack(preds)             # shape (N, num_classes)
        loaded_model_cat = CatBoostClassifier()
        loaded_model_cat.load_model("model.cbm")
        cat_preds = loaded_model_cat.predict_proba(test_scaled)
        
        loaded_lgb_model = joblib.load('lgbm_model.pkl')       
        lgb_preds = loaded_lgb_model.predict_proba(test_scaled) 
        
        rc_preds = clf4.predict_proba(test_scaled)

    
        test_preds = [(a+b+c+d) / 4 for a, b,c,d in zip(cat_preds, rc_preds,lgb_preds,probs_all)]
        test_preds = [test_preds[i][0] for i in range(len(test_preds))]


        return test_preds

    
        # 定義多類別分類評分函數 (例如 play years、level)


    def model_multiary(X_tr, y_tr, X_test, y_test,test_scaled):
        
        chunk_size = 220000
        print(X_test)
        print(X_test.shape)
         
        weights = gaussian_density(len(X_test[0]), sigma=2)
        g_test = X_test @ weights
        g_test = g_test.reshape(-1, 1)
        X_test = np.hstack([X_test, g_test])
   
        row_means = np.mean(X_test, axis=1)   # shape → (n,)
        row_means = row_means.reshape(-1, 1)   # make it (n,1)
        X_test = np.hstack([X_test, row_means])  # now shape → (n, 8)   
        
        lagged_arrays = []
        for lag in range(1, max_lag + 1):
            # roll the data down by `lag` rows
            rolled = np.roll(X_test, shift=lag, axis=0)
            # the first `lag` rows are invalid → set to NaN
            rolled[:lag, :] = np.nan
            lagged_arrays.append(rolled)

    
        lags = np.array(lagged_arrays)
        n_lags, n_samples, n_feats = lags.shape

        # 1️⃣  bring samples to axis 0 → (478905, 10, 11)
        c_swapped = lags.transpose(1, 0, 2)

        # 2️⃣  flatten the last two dims → (478905, 110)
        c2 = c_swapped.reshape(n_samples, n_lags * n_feats)

        # 3️⃣  replace NaNs if you like
        c2 = np.nan_to_num(c2, nan=0.0)

        # 4️⃣  sanity check
        assert X_test.shape[0] == c2.shape[0]
        X_test = np.hstack([X_test, c2])  
        print(X_test.shape) 
         

        for s, (x_t, y_t) in enumerate(chunk_xy(X_tr, y_tr, chunk_size), 1):
            y_pred = []
            
            print(s)
            print(len(x_t))
            print(len(y_t))
            
            weights = gaussian_density(len(x_t[0]), sigma=2)
            g = x_t @ weights
            g = g.reshape(-1, 1)
            x_t = np.hstack([x_t, g])

                
            row_means = np.mean(x_t, axis=1)   # shape → (n,)
            row_means = row_means.reshape(-1, 1)   # make it (n,1)
            x_t = np.hstack([x_t, row_means])  # now shape → (n, 8)   

        
            
            lagged_arrays = []
            for lag in range(1, max_lag + 1):
                # roll the data down by `lag` rows
                rolled = np.roll(x_t, shift=lag, axis=0)
                # the first `lag` rows are invalid → set to NaN
                rolled[:lag, :] = np.nan
                lagged_arrays.append(rolled)

    
            lags = np.array(lagged_arrays)
            n_lags, n_samples, n_feats = lags.shape
        
            # 1️⃣  bring samples to axis 0 → (478905, 10, 11)
            c_swapped = lags.transpose(1, 0, 2)

            # 2️⃣  flatten the last two dims → (478905, 110)
            c2 = c_swapped.reshape(n_samples, n_lags * n_feats)

            # 3️⃣  replace NaNs if you like
            c2 = np.nan_to_num(c2, nan=0.0)

            # 4️⃣  sanity check
            assert x_t.shape[0] == c2.shape[0]

            # 5️⃣  hstack!
            x_t = np.hstack([x_t, c2])
            
                        
            print('The SHAPES')            
            print(x_t.shape)
            print(X_test.shape)

            X_reshaped = x_t.reshape(-1, 27, 33)  # (num_samples, time_steps, channels)
            X_test_local = X_test.reshape(-1, 27, 33) 
            X_test_tensor = torch.tensor(X_test_local, dtype=torch.float32).permute(0, 2, 1)

            X_train_re, X_valid_re, y_train, y_valid= train_test_split(X_reshaped,y_t,test_size=0.20)
            train_ds = IMUDataset(X_train_re, y_train)
            val_ds = IMUDataset(X_valid_re, y_valid)

            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            n_class = len(np.unique(y_t)) 

            clf_0 = InceptionTimeClassifier(num_classes=n_class, in_channels=33).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(clf_0.parameters(), lr=1e-3)
            best_auc = 0.0
            epochs = 10

            for epoch in range(epochs):
                clf_0.train()
                running_loss = 0.0

                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = clf_0(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * X_batch.size(0)

                    avg_train_loss = running_loss / len(train_loader.dataset)

                # Validation
                clf_0.eval()
                all_probs = []
                all_targets = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(device)
                        outputs = clf_0(X_batch)
                        probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        all_probs.append(probs)
                        all_targets.append(y_batch.numpy())

                y_true = np.concatenate(all_targets)
                y_pred = np.vstack(all_probs)

                print(y_pred)
                print(y_true)
                
                val_auc = roc_auc_score(y_true, y_pred,multi_class='ovr', average='macro')

                print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}")

                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(clf_0.state_dict(), "best_inception_model.pth")
                    print(f"✅ Saved model with AUC: {val_auc:.4f}") 

            
            with torch.no_grad():
                clf_0.eval()
                preds = []

                for i in tqdm(range(0, len(X_test_tensor), 128)):  # batch prediction
                    batch = X_test_tensor[i:i+128]
                    batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
                    outputs = clf_0(batch)
                    probs = torch.softmax(outputs, dim=1)
                    preds.append(probs.cpu().numpy())

                # Concatenate results
                probs_all = np.vstack(preds)             # shape (N, num_classes)
                #pred_classes = probs_all.argmax(axis=1)  # final class predictions
            
            print(probs_all)
            r1 = roc_auc_score(y_test, probs_all,multi_class='ovr', average='macro')
            print(f"Test ROC AUC: {r1:.4f}")
            

            X_mul, X_valid_mul, y_mul, y_valid_mul= train_test_split(x_t,y_t,test_size=0.20)
            
        
            train_pool = Pool(X_mul, y_mul)
            val_pool   = Pool(X_valid_mul,y_valid_mul)
            
            n_class = np.unique(y_t).size
            
            study_cb_multi = optuna.create_study(direction='maximize')
            study_cb_multi.optimize(lambda t: objective_cb_multi(t,x_t, y_t,n_class), n_trials=3)
            best_cb_multi = study_cb_multi.best_params
            print("Best CatBoost params (binary):", best_cb_multi)


            
            clf = CatBoostClassifier( 
                **best_cb_multi,
                auto_class_weights    = "Balanced", 
                early_stopping_rounds = 100,
                random_state          = 42,
                verbose               = 50            # every n trees
                )
        
             
            if s==1:
                clf.fit(train_pool, eval_set=val_pool)
            else:
                clf.fit(train_pool, eval_set=val_pool,init_model='model.cbm')

            clf.save_model('model.cbm')  
            predicted_0 =  clf.predict_proba(X_test) 
                        
            n_class = len(np.unique(y_mul))
            
            from sklearn.utils import class_weight
            n_class = len(np.unique(y_mul))
            from sklearn.utils import class_weight


            study = optuna.create_study(direction='maximize')
            study.optimize(lambda t: objective_multi_lgb(t, X_mul, y_mul), n_trials=3)
            lgb_best = study.best_params
            print('Best LGB multiclass params:', lgb_best)


            # compute balanced weights
            classes = np.unique(y_mul)
            weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_mul
            )
            cw = dict(zip(classes, weights))

            
            clf3 = LGBMClassifier(
                **lgb_best,
                num_class=len(classes),
                class_weight=cw,
                random_state=42)
            
            if s==1:
                clf3.fit(X_mul, y_mul,eval_set=[(X_valid_mul, y_valid_mul)])
            else:
                clf3_model = joblib.load('lgbm_model.pkl')
                clf3.fit(X_mul, y_mul,eval_set=[(X_valid_mul, y_valid_mul)],init_model=clf3_model)
            
            
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda t: objective_multi_rgb(t, X_mul, y_mul), n_trials=3)
            rgb_best = study.best_params
            print('Best LGB multiclass params:', rgb_best)


            clf4 = RandomForestClassifier(**rgb_best,random_state=42, class_weight='balanced',verbose=50)
            clf4.fit(X_mul, y_mul)
            
            joblib.dump(clf3,'lgbm_model.pkl')  
            predicted_1 = clf3.predict_proba(X_test)
            predicted_2 = clf4.predict_proba(X_test)
            

            predicted = (predicted_0+predicted_1+predicted_2+probs_all)/4

            num_groups = len(predicted) // group_size
            y_pred = []
 
            for i in tqdm(range(num_groups)):
                group_pred = predicted[i*group_size: (i+1)*group_size]
                num_classes = len(np.unique(y_mul))
                # 對每個類別計算該組內的總機率
                class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
                chosen_class = np.argmax(class_sums)
                candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
                best_instance = np.argmax(candidate_probs)
                y_pred.append(group_pred[best_instance])
        
            y_test_agg = [y_test[l*group_size] for l in range(num_groups)]
            auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
            print('Multiary AUC:', auc_score)
           
        
        weights = gaussian_density(len(test_scaled[0]), sigma=2)
        g_t = test_scaled @ weights
        g_t = g_t.reshape(-1, 1)
        test_scaled = np.hstack([test_scaled, g_t])
        
        
        row_means_test = np.mean(test_scaled, axis=1)   # shape → (n,)
        row_means_test = row_means_test.reshape(-1, 1)   # make it (n,1)
        test_scaled = np.hstack([test_scaled, row_means_test])  # now shape → (n, 8)
        
        
        print(test_scaled.shape)
        
        lagged_arrays = []
        for lag in range(1, max_lag + 1):
            # roll the data down by `lag` rows
            rolled = np.roll(test_scaled, shift=lag, axis=0)
            # the first `lag` rows are invalid → set to NaN
            rolled[:lag, :] = np.nan
            lagged_arrays.append(rolled)

    
        lags = np.array(lagged_arrays)
        n_lags, n_samples, n_feats = lags.shape

        # 1️⃣  bring samples to axis 0 → (478905, 10, 11)
        c_swapped = lags.transpose(1, 0, 2)

        # 2️⃣  flatten the last two dims → (478905, 110)
        c2 = c_swapped.reshape(n_samples, n_lags * n_feats)

        # 3️⃣  replace NaNs if you like
        c2 = np.nan_to_num(c2, nan=0.0)

        # 4️⃣  sanity check
        assert test_scaled.shape[0] == c2.shape[0]
        test_scaled = np.hstack([test_scaled, c2])  
        test_scaled_reshape  = test_scaled.reshape(-1, 27, 33)

        # Convert and permute to (B, C, T)
        X_tensor = torch.tensor(test_scaled_reshape, dtype=torch.float32).permute(0, 2, 1)
        with torch.no_grad():
            clf_0.eval()
            preds = []

            for i in tqdm(range(0, len(X_tensor), 128)):  # batch prediction
                batch = X_tensor[i:i+128]
                batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
                outputs = clf_0(batch)
                probs = torch.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())

        # Concatenate results
                     # shape (N, num_classes)
    
                
        loaded_model_cat = CatBoostClassifier()
        loaded_model_cat.load_model("model.cbm")
        cat_preds = loaded_model_cat.predict_proba(test_scaled)
        lgb_preds = clf3.predict_proba(test_scaled)
        rgb_preds = clf4.predict_proba(test_scaled)
        probs_all = np.vstack(preds)
        print("Model CAT MULTI PREDS!!")
        final_preds = (cat_preds+lgb_preds+rgb_preds+probs_all)/4
        return final_preds


    # 評分：針對各目標進行模型訓練與評分
    gender_preds=[]
    hold_racket=[]
    play_years=[]
    level = []

    
    y_train_le_gender = le.fit_transform(y_train_all['gender'])
    y_test_le_gender = le.transform(y_test_all['gender'])
    gender_preds=model_binary(X_train_scaled, y_train_le_gender, X_test_scaled, y_test_le_gender,test_scaled)
    submission['gender'] = gender_preds
    
    y_train_le_hold = le.fit_transform(y_train_all['hold racket handed'])
    y_test_le_hold = le.transform(y_test_all['hold racket handed'])
    
    hold_racket = model_binary(X_train_scaled, y_train_le_hold, X_test_scaled, y_test_le_hold,test_scaled)
    submission['hold racket handed'] = hold_racket
    
    print(submission.head(10))
    

    y_train_le_years = le.fit_transform(y_train_all['play years'])
    y_test_le_years = le.transform(y_test_all['play years'])
    n_class = np.unique(y_train_le_years).size

    play_years = model_multiary(X_train_scaled, y_train_le_years, X_test_scaled, y_test_le_years,test_scaled)
   
    print(play_years)

    play_years_df = pd.DataFrame(
    play_years,columns=[f'play years_{i}' for i in range(3)]
    )
 
    submission = pd.concat([submission, play_years_df], axis=1)
    flat_list = [x for xs in test_id for x in xs]

    print(submission.head(10))
    
    y_train_le_level = le.fit_transform(y_train_all['level'])
    y_test_le_level = le.transform(y_test_all['level'])
    preds_level=model_multiary(X_train_scaled, y_train_le_level, X_test_scaled, y_test_le_level,test_scaled)
    
    level_df = pd.DataFrame(
    preds_level,                      # shape (N, 4)
    columns=[f'level_{i}' for i in range(2, 6)]
    )
    submission = pd.concat([submission, level_df], axis=1) 
    submission.insert(loc=0,column='unique_id',value=flat_list)
    
    grouped = submission.groupby('unique_id', as_index=False).mean()
    print(grouped)    
    print(grouped.head(10))
    #grouped.to_csv("m42.csv", index=None)
    score_cols = grouped.columns.difference(["unique_id"])
    grouped[score_cols] = grouped[score_cols].round(6)
    grouped.to_csv("m101.csv", index=False)
    

    #AUC SCORE: 0.792(gender) + 0.998(hold) + 0.660(years) + 0.822(levels)

if __name__ == '__main__':
    main()
