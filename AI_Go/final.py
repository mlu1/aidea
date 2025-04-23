import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from sklearn.compose import ColumnTransformer
import optuna  # For hyperparameter optimization
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier,sum_models
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')
import cudf
csv_file='training.csv'
from tqdm import tqdm
from catboost import Pool
import joblib
from sklearn.linear_model import LogisticRegression
from scipy.stats import mode
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, KFold, cross_val_score, train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.cluster import KMeans
from collections import defaultdict
import re
import difflib
from functools import partial
from sklearn.decomposition import PCA
from scipy.stats import norm
seed=42
test=pd.read_csv('38_Public_Test_Set_and_Submmision_Template_V2/public_x.csv')
t1 = pd.read_csv('38_Public_Test_Set_and_Submmision_Template_V2/public_x.csv')

chunk_size = 75000

test_df = test.drop(columns=['ID'])
test_df=test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
classes = [0, 1]


def gaussian_density(n, sigma):
    x = np.arange(n)
    mu = n // 2
    g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    return g / g.sum()


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

    # Create an array of indices corresponding to the positions in x
    indices = np.arange(len(x))

    # Calculate the Gaussian probability density
    y = norm.pdf(indices, loc=mean_index, scale=sigma)

    # Set values beyond 3*sigma to 0
    distance_from_mean = np.abs(indices - mean_index)
    y[distance_from_mean > 3 * sigma] = 0

    return y*m


def preprocess_data(X):
    # Replace infinite values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    # Fill NaN values with column mean
    X = X.fillna(0)
    return X

test_df=preprocess_data(test_df)

# Define the batch size (number of rows per batch)
batch_size = 75000  # adjust this value based on your data and memory constraints

# Read the CSV file in chunks


all_models = []
trees = 5000
epochs = 2
batches = 10


# Define the batch size (number of rows per batch)
# adjust this value based on your data and memory constraints
test_size=0.25
random_state=42


def read_csv_chunks(csv_path, chunk_size):
    """
    Generator that yields chunks of the CSV file until the end is reached.
    
    Parameters:
    - csv_path (str): Path to the CSV file.
    - chunk_size (int): Number of rows per chunk.
    
    Yields:
    - pd.DataFrame: A DataFrame containing a chunk of the CSV file.
    """
    # First, count the total number of rows (excluding header)
    with open(csv_path, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # subtract header
    
    start = 0
    while start < total_rows:
        # Calculate the end row (exclusive)
        end = start + chunk_size
        # Use skiprows to skip rows from 1 to start (header is row 0)
        df_chunk = pd.read_csv(csv_path, skiprows=range(1, start + 1), nrows=chunk_size)
        yield df_chunk
        start += chunk_size

# Example usage:

def extract_metric(col_name):
    # Return ID unchanged.
    if col_name == "ID":
        return col_name

    # Split the column name by the underscore.
    parts = col_name.split("_")
    
    # Check if the structure is as expected.
    if len(parts) < 2:
        return col_name

    # The part after the underscore should include the metric descriptor.
    descriptor = parts[1]

    # If the descriptor includes a time indicator (like "前1天") before the metric,
    # you might want to remove it.
    # For simplicity, let's remove any occurrence of "前" and "天" if they are at the start.
    import re
    descriptor = re.sub(r"^前\d+天", "", descriptor)
    
    # Remove the literal prefix "分點" so that we extract only the core metric.
    if descriptor.startswith("分點"):
        key_metric = descriptor[len("分點"):]
    else:
        key_metric = descriptor

    return key_metric.strip()


def group_similar_columns(columns, threshold=0.8):
    """
    Groups column names together based on a similarity threshold.
    
    Parameters:
      columns (list): A list of column name strings.
      threshold (float): A value between 0 and 1; names with similarity scores
                         equal or above this value are placed in the same group.
                         
    Returns:
      groups (list): A list of groups (each group is a list of similar column names).
    """
    groups = []
    for col in columns:
        added = False
        # Try to add the current column to an existing group
        for group in groups:
            # Compare with every column already in the group.
            if any(difflib.SequenceMatcher(None, col, existing).ratio() >= threshold
                   for existing in group):
                group.append(col)
                added = True
                break
        # If no group was similar enough, start a new group.
        if not added:
            groups.append([col])
    return groups

for i, batch in enumerate(read_csv_chunks(csv_file, batch_size)):   
    print(f"Processing batch {i}")
    # Here you can process each batch as needed
    # For example, splitting features and target:
    X = batch.drop(columns=['ID','飆股'])
    y = batch['飆股'] 
    X=preprocess_data(X)
    X=X.replace([np.inf, -np.inf], np.nan).fillna(0)
    columns_with_percent = X.columns[X.columns.str.contains("%")].tolist()
    cluster_features = X[columns_with_percent]  
    kmeans = KMeans(n_clusters=10, random_state=42).fit(cluster_features)
    X['customer_behavior_cluster'] = kmeans.labels_
    
    X['外資_買賣力_momentum'] = X['外資券商_前1天分點買賣力'] - X['外資券商_前3天分點買賣力']
    X['外資_買賣力_acceleration'] = (X['外資券商_前1天分點買賣力'] - 2 * X['外資券商_前2天分點買賣力'] + X['外資券商_前3天分點買賣力'])
    X['主力_買賣力_flip'] = ((X['主力券商_前1天分點買賣力'] > 0) & (X['主力券商_前2天分點買賣力'] < 0)
    ) | (
    (X['主力券商_前1天分點買賣力'] < 0) & (X['主力券商_前2天分點買賣力'] > 0)
    )
    cols = [f"外資券商_前{i}天分點買賣力" for i in range(1, 11)]
    weights = gaussian_density(len(cols), sigma=2)
    X['外資_買賣力_weighted'] = X[cols].values @ weights
    
    grouped_columns = group_similar_columns(list(X.columns), threshold=0.75)
    X['主力_買賣方向改變'] = np.sign(X['主力券商_前1天分點買賣力']) != np.sign(X['主力券商_前2天分點買賣力'])
 
   
    cols = [f"外資券商_前{i}天分點吃貨比(%)" for i in range(1, 21)]
    X_cluster = X[cols].copy()
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)    
    from sklearn.cluster import AgglomerativeClustering

    cluster_model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
    X['外資_吃貨比_cluster'] = cluster_model.fit_predict(X_scaled)   
    # Print the grouped columns.
    
    cols = [f"主力券商_前{i}天分點成交力(%)" for i in range(1, 6)]
    X['主力_成交力_volatility'] = X[cols].std(axis=1)
    
    X['外資_成交力_jump_ratio'] = (
    X['外資券商_前1天分點成交力(%)'] / (X[[f"外資券商_前{i}天分點成交力(%)" for i in range(2, 6)]].mean(axis=1) + 1e-6)
    )
    X['主力_吃貨比_rank'] = X[[f"主力券商_前{i}天分點吃貨比(%)" for i in range(1, 6)]].rank(axis=1, pct=True).mean(axis=1)
    X['吃貨買賣交互'] = (X['外資券商_前1天分點吃貨比(%)'] *X['外資券商_前1天分點買賣力'])

    cols = [f'外資券商_前{i}天分點買賣力' for i in [3, 2, 1]]
    X['外資_買賣力_bull_pattern'] = (
    (X[cols[0]] < X[cols[1]]) & (X[cols[1]] < X[cols[2]])
    ).astype(int)
   
    X['外資_連續吃貨'] = (
    (X['外資券商_前1天分點吃貨比(%)'] > 60) &
    (X['外資券商_前2天分點吃貨比(%)'] > 60) &
    (X['外資券商_前3天分點吃貨比(%)'] > 60)
     ).astype(int)
    
    buy_strength = [f'外資券商_前{i}天分點買賣力' for i in range(1, 6)]
    X['外資_5日_買超天數'] = X[buy_strength].gt(0).sum(axis=1)
    
    for lag in range(1, 6):
        X[f'外資_買賣力_corr_lag{lag}'] = X['外資券商_分點買賣力'] * X[f'外資券商_前{lag}天分點買賣力']
    
    X['外資_買賣力_delta_ratio'] = (
    (X['外資券商_前1天分點買賣力'] - X['外資券商_前2天分點買賣力']) /
    (abs(X['外資券商_前2天分點買賣力']) + 1e-6)
    ) 

    groups_columns = []
    for k, group in enumerate(grouped_columns, 1):
        print(f"Group {k}:")
        if (len(group) >1):
            group_data = X[group].values
            pca = PCA(n_components=1)
            pc = pca.fit_transform(group_data)
            X['group_'+str(k)] = pc
            X["group_mean_"+str(k)] = X[group].mean(axis=1) 
            X["group_sum_"+str(k)] = X[group].sum(axis=1)  
            X["group_median_"+str(k)] = X[group].median(axis=1)
            X["group_max_"+str(k)] = X[group].max(axis=1)
            X["group_min_"+str(k)] = X[group].min(axis=1)
            X["group_std_"+str(k)] = X[group].std(axis=1)
            cluster_features_group = X[group]  
            kmeans = KMeans(n_clusters=10, random_state=42).fit(cluster_features_group)
            X['group_means_cluster'+str(k)] = kmeans.labels_   
            X['gaussian'+str(k)] = X.groupby('group_'+str(k))['group_means_cluster'+str(k)].transform(partial(gaussian_from_onehot, sigma=15, m=100)  )
            X['group_means_cluster_logs'+str(k)] = np.log1p(X['group_means_cluster'+str(k)])  
            X['lagged_loan_amount'+str(k)] = X.groupby('group_means_cluster'+str(k))['group_'+str(k)].shift(1).fillna(0)
            X['lagged_loan_amount_2'+str(k)] = X.groupby('group_means_cluster'+str(k))['group_'+str(k)].shift(2).fillna(0)
            X['lagged_loan_amount_3'+str(k)] = X.groupby('group_means_cluster'+str(k))['group_'+str(k)].shift(3).fillna(0)
             
            window_size=10
            col = f'rolling_mean_{window_size}'+str(k)
            col_1 = f'rolling_sum_{window_size}'+str(k)
            
            X[col] = X.groupby('group_means_cluster'+str(k))['group_'+str(k)].transform(lambda x: x.rolling(window_size).mean())  
            X[col_1] = X.groupby('group_means_cluster'+str(k))['group_'+str(k)].transform(lambda x: x.rolling(window_size).sum())  
             
 
        else:
            continue
  
        
    X_train, X_valid, y_train, y_valid= train_test_split(X,y,test_size=test_size,random_state=random_state) 
    X_t, X_test, y_t, y_test= train_test_split(X_train,y_train,test_size=test_size,random_state=random_state) 
    
    train_pool = Pool(data=X_t, label=y_t)
    test_pool = Pool(data=X_test, label=y_test)
    
    cat_model = CatBoostClassifier() 
    
    if i==0:
        cat_model.fit(train_pool, eval_set=test_pool)
    else:
        cat_model.fit(train_pool, eval_set=test_pool,init_model='model.cbm')

    
    cat_model.save_model('model.cbm') 

        # Use the model for prediction (example on X_test)
    loaded_model = CatBoostClassifier()

    loaded_model.load_model("model.cbm")
    print("Model loaded successfully!")

    predictions = loaded_model.predict(X_valid)
    
    # Calculate and store the f1 score for this fold
    f1 = f1_score(predictions, y_valid)
    print(classification_report(y_valid, predictions))
    print(f1)

columns_with_percent = test_df.columns[test_df.columns.str.contains("%")].tolist()
cluster_features = test_df[columns_with_percent]  
kmeans = KMeans(n_clusters=10, random_state=42).fit(cluster_features)
test_df['customer_behavior_cluster'] = kmeans.labels_
test_df['外資_買賣力_momentum'] = test_df['外資券商_前1天分點買賣力'] - test_df['外資券商_前3天分點買賣力']
test_df['外資_買賣力_acceleration'] = (test_df['外資券商_前1天分點買賣力'] - 2 * test_df['外資券商_前2天分點買賣力'] + test_df['外資券商_前3天分點買賣力'])
test_df['主力_買賣力_flip'] = ((test_df['主力券商_前1天分點買賣力'] > 0) & (test_df['主力券商_前2天分點買賣力'] < 0)
    ) | ((test_df['主力券商_前1天分點買賣力'] < 0) & (test_df['主力券商_前2天分點買賣力'] > 0))
cols = [f"外資券商_前{i}天分點買賣力" for i in range(1, 11)]
weights = gaussian_density(len(cols), sigma=2)
test_df['外資_買賣力_weighted'] = test_df[cols].values @ weights
    
grouped_columns = group_similar_columns(list(test_df.columns), threshold=0.75)
test_df['主力_買賣方向改變'] = np.sign(X['主力券商_前1天分點買賣力']) != np.sign(X['主力券商_前2天分點買賣力'])
cols = [f"外資券商_前{i}天分點吃貨比(%)" for i in range(1, 21)]
X_cluster = test_df[cols].copy()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)    
from sklearn.cluster import AgglomerativeClustering
cluster_model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
test_df['外資_吃貨比_cluster'] = cluster_model.fit_predict(X_scaled)   
    # Print the grouped columns.
    
cols = [f"主力券商_前{i}天分點成交力(%)" for i in range(1, 6)]
test_df['主力_成交力_volatility'] = test_df[cols].std(axis=1)
    
test_df['外資_成交力_jump_ratio'] = (
test_df['外資券商_前1天分點成交力(%)'] / (test_df[[f"外資券商_前{i}天分點成交力(%)" for i in range(2, 6)]].mean(axis=1) + 1e-6))
test_df['主力_吃貨比_rank'] = test_df[[f"主力券商_前{i}天分點吃貨比(%)" for i in range(1, 6)]].rank(axis=1, pct=True).mean(axis=1)
test_df['吃貨買賣交互'] = (test_df['外資券商_前1天分點吃貨比(%)'] *test_df['外資券商_前1天分點買賣力'])
cols = [f'外資券商_前{i}天分點買賣力' for i in [3, 2, 1]]
test_df['外資_買賣力_bull_pattern'] = ((test_df[cols[0]] < test_df[cols[1]]) & (test_df[cols[1]] < test_df[cols[2]])).astype(int)
test_df['外資_連續吃貨'] = ((test_df['外資券商_前1天分點吃貨比(%)'] > 60) &
(test_df['外資券商_前2天分點吃貨比(%)'] > 60) &(test_df['外資券商_前3天分點吃貨比(%)'] > 60)).astype(int)
    
buy_strength = [f'外資券商_前{i}天分點買賣力' for i in range(1, 6)]
test_df['外資_5日_買超天數'] = test_df[buy_strength].gt(0).sum(axis=1)
for lag in range(1, 6):
    test_df[f'外資_買賣力_corr_lag{lag}'] = test_df['外資券商_分點買賣力'] * test_df[f'外資券商_前{lag}天分點買賣力']

test_df['外資_買賣力_delta_ratio'] = (
    (test_df['外資券商_前1天分點買賣力'] - test_df['外資券商_前2天分點買賣力']) /
    (abs(test_df['外資券商_前2天分點買賣力']) + 1e-6)
)

for k, group in enumerate(grouped_columns, 1):
    print(f"Group {k}:")
    if (len(group) >1):
        group_data = test_df[group].values
        pca = PCA(n_components=1)
        pc = pca.fit_transform(group_data)
        test_df['group_'+str(k)] = pc
        test_df["group_mean_"+str(k)] = test_df[group].mean(axis=1) 
        test_df["group_sum_"+str(k)] = test_df[group].sum(axis=1)
        test_df["group_median_"+str(k)] = test_df[group].median(axis=1)
        test_df["group_max_"+str(k)] = test_df[group].max(axis=1)
        test_df["group_min_"+str(k)] = test_df[group].min(axis=1)
        test_df["group_std_"+str(k)] = test_df[group].std(axis=1)
         

        cluster_features_group = test_df[group]  
        kmeans = KMeans(n_clusters=10, random_state=42).fit(cluster_features_group)
        test_df['group_means_cluster'+str(k)] = kmeans.labels_
        test_df['gaussian'+str(k)] = test_df.groupby('group_'+str(k))['group_means_cluster'+str(k)].transform(partial(gaussian_from_onehot, sigma=15, m=100)  )

        test_df['group_means_cluster_logs'+str(k)] = np.log1p(test_df['group_means_cluster'+str(k)])
        test_df['lagged_loan_amount'+str(k)] = test_df.groupby('group_means_cluster'+str(k))['group_'+str(k)].shift(1).fillna(0)
        test_df['lagged_loan_amount_2'+str(k)] = test_df.groupby('group_means_cluster'+str(k))['group_'+str(k)].shift(2).fillna(0)
        test_df['lagged_loan_amount_3'+str(k)] = test_df.groupby('group_means_cluster'+str(k))['group_'+str(k)].shift(3).fillna(0)

        window_size=10
        col = f'rolling_mean_{window_size}'+str(k)
        col_1 = f'rolling_sum_{window_size}'+str(k)
        test_df[col] = test_df.groupby('group_means_cluster'+str(k))['group_'+str(k)].transform(lambda x: x.rolling(window_size).mean())
        test_df[col_1] = test_df.groupby('group_means_cluster'+str(k))['group_'+str(k)].transform(lambda x: x.rolling(window_size).sum())


         # Load the saved model from the file
loaded_model = CatBoostClassifier()
loaded_model.load_model("model.cbm")
print("Model loaded successfully!")

# Use the model for prediction (example on X_test)

pred_1 = loaded_model.predict(test_df)
# Step 7: Make predictions on the test set using the (loaded) model

all_preds = pred_1

t1['飆股'] = all_preds
#test['credit_score'] = test_predictions_proba  # Add probability column

# Select required columns
sub2 = t1[['ID', '飆股']]
sub2.to_csv('sub53.csv',index=None)

