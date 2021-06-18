#libararies for evaluating
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
#import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pm4py
import seaborn as sns
import math
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from pandas.plotting import parallel_coordinates
import datetime as dt
from datetime import datetime
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Path to relevant directories
path_save_process_discovery_base = os.path.abspath('./figures/process_discovery')
path_error_analysis = os.path.abspath('./figures/error_analysis')
path_prediction_results = os.path.abspath('./figures/prediction_results')
path_results = os.path.abspath('./results')
path_pred_res = "../PycharmProjects/time-prediction-benchmark/"
path_preprocessed_data = "../PycharmProjects/time-prediction-benchmark/experiments/logdata"
path_pred_data_save = "../PycharmProjects/ProcessSequencePrediction/data"
path_pred_activity_res = "../PycharmProjects/ProcessSequencePrediction/code/output_files/results"
path_new_rtp_data = "../PycharmProjects/time-prediction-benchmark/experiments/feature_enriched_data"
path_model = os.path.abspath('./models')


# load and preprocess data
def get_segment_data(df,a1,a2,cols):
    #shift the datafarme
    df_1 = df.shift(-1)
    rename_columns = {x:x+"_1" for x in df.columns}
    df_1.rename(columns=rename_columns, inplace = True)
    data = pd.concat([df,df_1],axis=1)
    #filter directly follow activities
    data = data[data["Case ID"]==data["Case ID_1"]]
    #filter for activities a1,a2  
    data["merged_activity"] = data.apply(lambda x: x["Activity"]+"_"+x["Activity_1"], axis=1)
    dfg_merged = a1+"_"+a2 
    data = data[data["merged_activity"]==dfg_merged]
    #sort on timestamps of activity a1
#     data = data[["Case ID", "Complete Timestamp", "Complete Timestamp_1"]]
    data.sort_values(["Complete Timestamp"], axis=0, ascending=True, inplace=True, kind='quicksort')
#     cases = data["Case ID"].values
    filter_df = data[cols+["Case ID"]]
    filter_df.rename(columns={"Complete Timestamp":"ta", "Complete Timestamp_1":"tb"},inplace=True)
    filter_df["ta"] = pd.to_datetime(filter_df["ta"])
    filter_df["tb"] = pd.to_datetime(filter_df["tb"])
    return filter_df

def get_eventually_follows_data(df,a1,a2,cols):
    #shift the datafarme

    df_1 = df.shift(-1)
    rename_columns = {x:x+"_1" for x in df.columns}
    df_1.rename(columns=rename_columns, inplace = True)
    data = pd.concat([df,df_1],axis=1)


    #filter directly follow activities
    data = data[data["Case ID"]==data["Case ID_1"]]

    #filter for activities a1,a2
    data["merged_activity"] = data.apply(lambda x: x["Activity"]+"_"+x["Activity_1"], axis=1)

    dfg_merged = a1+"_"+a2
    data = data[data["Activity"]==a1]
    print("data here:", data.head())
#     data = data[data["merged_activity"]==dfg_merged]
    #sort on timestamps of activity a1
    #     data = data[["Case ID", "Complete Timestamp", "Complete Timestamp_1"]]
    data.sort_values(["Complete Timestamp"], axis=0, ascending=True, inplace=True, kind='quicksort')
    #     cases = data["Case ID"].values
    filter_df = data[cols+["Case ID"]]
    filter_df.rename(columns={"Complete Timestamp":"ta", "Complete Timestamp_1":"tb"},inplace=True)
    filter_df["ta"] = pd.to_datetime(filter_df["ta"])
    filter_df["tb"] = pd.to_datetime(filter_df["tb"])
    print(filter_df.head())
    return filter_df

def return_ending_cases(filter_df, ts, te):
    temp = filter_df[(filter_df["tb"]>=ts) & (filter_df["tb"]<te)] 
    if temp.empty:
        return 0
    return temp.shape[0]
def return_starting_cases(filter_df, ts, te):
    temp = filter_df[(filter_df["ta"]>=ts) & (filter_df["ta"]<te)] 
    if temp.empty:
        return 0
    return temp.shape[0]
def return_non_pending_cases(filter_df, ts, te):
    temp = filter_df[(filter_df["ta"]>=ts) & (filter_df["tb"]<te)] 
    if temp.empty:
        return 0
    return temp.shape[0]
#     else:
#         return 0

def assign_perf_class(x,q_25,q_50,q_75):
    if(x<=q_25):
        return "0"
    elif(x>q_25 and x<=q_50):
        return "1"
    else:
        return "2"

def add_workload_features(filter_df):
    x = filter_df.groupby("ta").size().reset_index(name="size")
    x["prev_ta"] = x["ta"].shift(1)
    x["starting cases"] = x.apply(lambda x: return_starting_cases(filter_df, x["prev_ta"], x["ta"]), axis=1)
    x["ending cases"] = x.apply(lambda x: return_ending_cases(filter_df, x["prev_ta"], x["ta"]), axis=1)
    x["pending cases"] = x.apply(lambda x: return_non_pending_cases(filter_df, x["prev_ta"], x["ta"]), axis=1)
    x["pending cases"] = x["starting cases"] - x["pending cases"]
    x= x.drop(columns=["prev_ta","size"],axis=1).set_index("ta")
    return x    

def split_test_train_data(data, train_ratio):
    # split into train and test using temporal split
    grouped = data.groupby("Case ID")
    start_timestamps = grouped["Complete Timestamp"].min().reset_index()
    start_timestamps = start_timestamps.sort_values("Complete Timestamp", ascending=True, kind='mergesort')
    train_ids = list(start_timestamps["Case ID"])[:int(train_ratio*len(start_timestamps))]
#     train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
#     test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
    return train_ids
    

def split_data(X,y):
    split_index = int(0.20*len(X))
#     split_index=1
    #split in test and train
    X_train = X[X.columns][:-split_index]
    Y_train = y[:-split_index]
    X_test = X[X.columns][-split_index:]
    Y_test = y[-split_index:]
    return X_train,X_test,Y_train,Y_test

params_classifier = {
#     "colsample_bytree": 0.5,
               "learning_rate": 0.03,
               "max_depth": 10,
#                "n_clusters": 1,
#                "n_estimators": 250,
#                "subsample": 0.8,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss', # this is abs(a-e)/max(1,a)
            'num_classes': 3
}

def train_model(X_train,y_train,X_test,y_test,categorical_features, params, segment_name, mode, file):
    lgb_train = lgb.Dataset(X_train,y_train,categorical_feature= categorical_features)
    lgb_valid = lgb.Dataset(X_test,y_test,categorical_feature = categorical_features)
    model = lgb.train(params, lgb_train, 3000, valid_sets=[lgb_train, lgb_valid],early_stopping_rounds=100, verbose_eval=50)
    directory = path_model+"/"+file
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save_model(directory+"/"+segment_name+"_"+mode+".txt")    

# step 1: Feature Engineering
def generate_features(filter_df):
    segment_data = filter_df.drop(["tb","Case ID"],axis=1)
#     segment_data["Case_inter_arrival_time"] = segment_data["ta"].diff().dt.total_seconds().fillna(0).values
#     segment_data["observations"] = np.repeat(1,segment_data.shape[0])
#     segment_data["observations"] = segment_data["observations"].cumsum()
    segment_data["LES"] = segment_data["waiting_time"].shift(1).fillna(method='bfill')
    segment_data = segment_data.set_index("ta")
    # Adding the lag of the target variable from 6 steps back up to 24
#     for i in range(1,3):
#         segment_data["lag_{}".format(i)] = segment_data["waiting_time"].shift(i)
    y = segment_data["waiting_time"]
    X = segment_data.dropna().drop(["waiting_time"], axis = 1)
#     print(segment_data.head())
    return X,y    

def prepare_data_for_variant_classification(filter_df):
    filter_df["waiting_time"]= (filter_df["tb"] - filter_df["ta"]).dt.total_seconds()
    q_25 = np.quantile(filter_df["waiting_time"].unique(),0.25)
    q_50 = np.quantile(filter_df["waiting_time"].unique(),0.50)
    q_75 = np.quantile(filter_df["waiting_time"].unique(),0.75)
    # print(q_25,q_50,q_75)
    filter_df["performance_class"] = filter_df["waiting_time"].apply(lambda x: assign_perf_class(x,q_25,q_50,q_75))
#     #take previous context
#     q_25 = np.quantile(filter_df["duration"].unique(),0.25)
#     q_50 = np.quantile(filter_df["duration"].unique(),0.50)
#     q_75 = np.quantile(filter_df["duration"].unique(),0.75)
#     # print(q_25,q_50,q_75)
#     filter_df["context_variant_class"] = filter_df["duration"].apply(lambda x: assign_perf_class(x,q_25,q_50,q_75))
    return filter_df

def get_inter_batch_completion_times(x):
    x=sorted(x)
    if(len(x)==1):
        return x
    IBCR = list(map(lambda x: x[0]-x[1], zip(x[1:],x[:-1]) ))
    return IBCR

#make batch stats in a datframe
def get_batch_info(filter_df):
    tmp_batch = filter_df.groupby(["ta"]).size().reset_index(name="observations")
    batching_indices = tmp_batch["ta"].values
    batched_cases = filter_df.groupby("ta")["waiting_time"].apply(list).values
    tmp_batch["mean_IBCR"]= list(map(lambda x: np.mean(get_inter_batch_completion_times(x)), batched_cases))
    tmp_batch["std_IBCR"]= list(map(lambda x: np.std(get_inter_batch_completion_times(x)), batched_cases))
    tmp_batch["min_delay"] = list(map(lambda x: min(x), batched_cases))
    tmp_batch["max_delay"] = list(map(lambda x: max(x), batched_cases))
    tmp_batch["batch_day"] = tmp_batch["ta"].dt.weekday
    tmp_batch["batch_month"] = tmp_batch["ta"].dt.month
    tmp_batch["batch_hour"] = tmp_batch["ta"].dt.hour
    tmp_batch["is_weekend"] = tmp_batch["batch_day"].apply(lambda x: 1 if(x==5 or x==6) else 0 )
    # tmp_batch["is_Christmas"] = np.where(tmp_batch["batch_month"], 1) 
    return tmp_batch

#make batch stats in a datframe
def get_end_batch_info(filter_df):
    tmp_batch = filter_df.groupby(["tb"]).size().reset_index(name="observations")
    batching_indices = tmp_batch["tb"].values
    batched_cases = filter_df.groupby("tb")["waiting_time"].apply(list).values
    tmp_batch["mean_IBCR"]= list(map(lambda x: np.mean(get_inter_batch_completion_times(x)), batched_cases))
    tmp_batch["std_IBCR"]= list(map(lambda x: np.std(get_inter_batch_completion_times(x)), batched_cases))
    tmp_batch["min_delay"] = list(map(lambda x: min(x), batched_cases))
    tmp_batch["max_delay"] = list(map(lambda x: max(x), batched_cases))
    tmp_batch["batch_day"] = tmp_batch["tb"].dt.weekday
    tmp_batch["batch_month"] = tmp_batch["tb"].dt.month
    tmp_batch["batch_hour"] = tmp_batch["tb"].dt.hour
    tmp_batch["is_weekend"] = tmp_batch["batch_day"].apply(lambda x: 1 if(x==5 or x==6) else 0 )
    # tmp_batch["is_Christmas"] = np.where(tmp_batch["batch_month"], 1) 
    return tmp_batch

def train_model_gt(X_train,Y_train,X_test,Y_test,categorical_cols,y,case_values,merged_activity,file, segparams):

    lgb_train = lgb.Dataset(X_train,Y_train,categorical_feature= categorical_cols)
    lgb_valid = lgb.Dataset(X_test,Y_test,categorical_feature = categorical_cols)
    model = lgb.train(segparams, lgb_train, 3000, valid_sets=[lgb_train, lgb_valid],early_stopping_rounds=100, verbose_eval=3)
    prediction = model.predict(X_test)
    plt.figure(figsize=(15, 7))
    #plt.figure(figsize=(12, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(Y_test.values, label="actual", linewidth=2.0,alpha=0.5)
    error = mean_absolute_error(Y_test, prediction)/(3600*24)
    plt.title("Mean absolute error in days {0:.2f}".format(error), fontsize=16)
    plt.legend(loc="best")
    #plt.tight_layout()
    plt.grid(True)
    plt.xlabel("$t_a$",fontsize=16)
    plt.ylabel("waiting time in $S=(a,b)$",fontsize=16)
    plt.legend()
    plt.xticks([], [])
    directory = path_error_analysis+"/"+file
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory+"/"+merged_activity+"nb_sl2p.pdf",
                        format='pdf')
    #plt.show()
    
    #make dataframe
    true_distance = list(Y_train.values) + list(Y_test.values)
    predicted_distance = list(Y_train.values) + list(prediction)
    batching_partition = np.zeros(len(Y_train)+len(Y_test))
    return model
#     return mae_days,y_pred,model

def train_non_batching_dynamics_model(df,a1,a2,cols,file, segparams):
    filter_df = get_eventually_follows_data(df, a1, a2, cols)
    filter_df["waiting_time"] = (filter_df["tb"] - filter_df["ta"]).dt.total_seconds()

    print(filter_df.head())

    # prepare data
    workload_df = add_workload_features(filter_df)
    filter_df = pd.concat([filter_df.set_index("ta"), workload_df], axis=1).reset_index()
    filter_df["waiting_time_last"] = filter_df["waiting_time"].shift(1).fillna(method='bfill')
    filter_df["time_since_last_event"] = filter_df["ta"].diff().dt.total_seconds().fillna(0)

    X = filter_df[["starting cases", "ending cases", "pending cases", "waiting_time_last"]]
    y = filter_df["waiting_time"]
    numerical_features = ["starting cases", "ending cases", "pending cases", "waiting_time_last"]
    categorical_features = []
    for feature in categorical_features:
        X[feature] = X[feature].astype('category')
    X_train, X_test, y_train, y_test = split_data(X, y)


    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    case_values = filter_df["Case ID"].values
    merged_activity = a1 + "_" + a2
    #print(X_train.dtypes)
    model = train_model_gt(X_train, y_train, X_test, y_test, categorical_features, y, case_values, merged_activity, file, segparams)
    segment_name = a1 + "_" + a2
    mode = "regression"
    directory = path_model + "/" + file
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save_model(directory + "/" + segment_name + "_" + mode + ".txt")

def train_starting_batch_dynamics_model(df,a1,a2,cols,file, segparams):
    filter_df = get_segment_data(df, a1, a2, cols)
    filter_df["waiting_time"] = (filter_df["tb"] - filter_df["ta"]).dt.total_seconds()

    # generate workload info ( these features should be the same, std features...)
    # prepare data
    workload_df = add_workload_features(filter_df)
    filter_df = pd.concat([filter_df.set_index("ta"), workload_df], axis=1).reset_index()
    filter_df["LES"] = filter_df["waiting_time"].shift(1).fillna(method='bfill')
    filter_df["time_since_last_event"] = filter_df["ta"].diff().dt.total_seconds().fillna(0)
    filter_df.head()

    # get batch info and add those features
    tmp_batch = get_batch_info(filter_df)
    x = filter_df.groupby("ta").size().reset_index()[1:]
    batch_df = x["ta"].apply(lambda x: tmp_batch[tmp_batch["ta"] < x].tail(1).drop(columns=["ta"], axis=1).squeeze())
    batch_df["ta"] = x["ta"]
    batch_df.head()
    filter_df = pd.concat([filter_df.set_index("ta"), batch_df.set_index("ta")], axis=1).reset_index().fillna(0)

    # add no. of observations in current batch
    batch_size = filter_df.groupby("ta").size().reset_index(name="batch_size")
    filter_df = pd.concat([filter_df.set_index("ta"), batch_size.set_index("ta")], axis=1).reset_index().fillna(0)
    filter_df["behaviour class"] = np.where(filter_df["batch_size"] <= 2, "non-batching", "batching-at-start")
    q_25 = np.quantile(filter_df["batch_size"].unique(), 0.25)
    q_50 = np.quantile(filter_df["batch_size"].unique(), 0.50)
    q_75 = np.quantile(filter_df["batch_size"].unique(), 0.75)
    # print(q_25,q_50,q_75)
    filter_df["size class"] = filter_df["batch_size"].apply(lambda x: assign_perf_class(x, q_25, q_50, q_75))

    # ##add variant class to features
    # filter_df["variant_class"] = list(y_train.values) + list(y_pred)
    # X,y = generate_features(filter_df[["starting cases", "ending cases", "pending cases", "variant_class","waiting_time","ta","tb","Case ID"]])
    X = filter_df[["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event", "observations"
        , "mean_IBCR", "std_IBCR", "min_delay", "max_delay", "batch_day"
        , "batch_month", "batch_hour", "is_weekend", "weekday", "month"
        , "duration", "expense", "points", "Resource", "behaviour class", "size class"]]
    y = filter_df["waiting_time"]
    numerical_features = ["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event",
                          "observations",
                          "mean_IBCR", "std_IBCR", "min_delay", "max_delay", "duration", "expense", "points"]
    categorical_features = ["batch_day", "batch_month", "batch_hour", "is_weekend", "weekday", "month", "Resource",
                            "behaviour class", "size class"]
    for feature in categorical_features:
        X[feature] = X[feature].astype('category')
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    case_values = filter_df["Case ID"].values
    merged_activity = a1 + "_" + a2
    #print(X_train.dtypes)
    model = train_model_gt(X_train, y_train, X_test, y_test, categorical_features, y, case_values, merged_activity, file, segparams)
    segment_name = a1 + "_" + a2
    mode = "regression"
    directory = path_model + "/" + file
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save_model(directory + "/" + segment_name + "_" + mode + ".txt")

def train_starting_batch_dynamics_average_model(df,a1,a2,cols,file, segparams):
    # load dataset for segment
    filter_df = get_segment_data(df, a1, a2, cols)
    filter_df["waiting_time"] = (filter_df["tb"] - filter_df["ta"]).dt.total_seconds()

    # generate workload info ( these features should be the same, std features...)
    # prepare data
    workload_df = add_workload_features(filter_df)
    filter_df = pd.concat([filter_df.set_index("ta"), workload_df], axis=1).reset_index()
    filter_df["LES"] = filter_df["waiting_time"].shift(1).fillna(method='bfill')
    filter_df["time_since_last_event"] = filter_df["ta"].diff().dt.total_seconds().fillna(0)
    filter_df.head()

    # get batch info and add those features
    tmp_batch = get_batch_info(filter_df)
    x = filter_df.groupby("ta").size().reset_index()[1:]
    batch_df = x["ta"].apply(lambda x: tmp_batch[tmp_batch["ta"] < x].tail(1).drop(columns=["ta"], axis=1).squeeze())
    batch_df["ta"] = x["ta"]
    batch_df.head()
    filter_df = pd.concat([filter_df.set_index("ta"), batch_df.set_index("ta")], axis=1).reset_index().fillna(0)

    # add no. of observations in current batch
    batch_size = filter_df.groupby("ta").size().reset_index(name="batch_size")
    filter_df = pd.concat([filter_df.set_index("ta"), batch_size.set_index("ta")], axis=1).reset_index().fillna(0)
    filter_df["behaviour class"] = np.where(filter_df["batch_size"] <= 2, "non-batching", "batching-at-start")
    q_25 = np.quantile(filter_df["batch_size"].unique(), 0.25)
    q_50 = np.quantile(filter_df["batch_size"].unique(), 0.50)
    q_75 = np.quantile(filter_df["batch_size"].unique(), 0.75)
    # print(q_25,q_50,q_75)
    filter_df["size class"] = filter_df["batch_size"].apply(lambda x: assign_perf_class(x, q_25, q_50, q_75))

    # if agg is needed
    # aggregate and add new features
    data_pred = filter_df.groupby("ta").mean()
    data_pred["current_batch_observations"] = filter_df.groupby("ta").size()
    data_pred["avg_waiting_time_lag_1"] = data_pred["waiting_time"].shift(1).fillna(0)
    # print(data_pred.head())

    ##add variant class to features
    # X,y = generate_features(filter_df[["starting cases", "ending cases", "pending cases", "variant_class","waiting_time","ta","tb","Case ID"]])
    X = data_pred[["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event", "observations"
        , "mean_IBCR", "std_IBCR", "min_delay", "max_delay", "batch_day", "batch_month", "batch_hour", "duration",
                   "is_weekend", "expense", "points", "weekday", "month"
        , "current_batch_observations", "avg_waiting_time_lag_1"]]
    y = data_pred["waiting_time"]
    numerical_features = ["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event",
                          "observations",
                          "mean_IBCR", "std_IBCR", "min_delay", "max_delay", "duration", "expense", "points",
                          "current_batch_observations", "avg_waiting_time_lag_1"]
    categorical_features = ["batch_day", "batch_month", "batch_hour", "is_weekend", "weekday", "month"]
    for feature in categorical_features:
        X[feature] = X[feature].astype('category')
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    case_values = filter_df["Case ID"].values
    merged_activity = a1 + "_" + a2
    #print(X_train.dtypes)
    model = train_model_gt(X_train, y_train, X_test, y_test, categorical_features, y, case_values, merged_activity, file, segparams)
    segment_name = a1 + "_" + a2
    mode = "regression"
    directory = path_model+"/"+file
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save_model(directory+"/"+segment_name+"_"+mode+".txt")




def train_end_batch_dynamics_model(df,a1,a2,cols,file, segparams):
    # load dataset for segment
    filter_df = get_segment_data(df, a1, a2, cols)
    filter_df["waiting_time"] = (filter_df["tb"] - filter_df["ta"]).dt.total_seconds()

    # generate workload info ( these features should be the same, std features...)
    # prepare data
    workload_df = add_workload_features(filter_df)
    filter_df = pd.concat([filter_df.set_index("ta"), workload_df], axis=1).reset_index()
    filter_df["LES"] = filter_df["waiting_time"].shift(1).fillna(method='bfill')
    filter_df["time_since_last_event"] = filter_df["ta"].diff().dt.total_seconds().fillna(0)
    filter_df.head()

    # get batch info and add those features
    tmp_batch = get_end_batch_info(filter_df)
    x = filter_df.groupby("tb").size().reset_index()[1:]
    batch_df = x["tb"].apply(lambda x: tmp_batch[tmp_batch["tb"] < x].tail(1).squeeze())
    batch_df.rename(columns={"tb": "prev_batch_BM"}, inplace=True)
    batch_df["tb"] = x["tb"]
    filter_df = filter_df.merge(batch_df, on='tb', how='left', suffixes=('_1', '_2')).fillna(0)

    # time from last batching moment
    # batching_moments = filter_df.groupby("tb").size().reset_index(name="batch_size")
    # print(filter_df.dtypes)
    obs_first_batch = filter_df.groupby("tb")["Case ID"].apply(list)[0]
    filter_df["prev_batch_BM"] = pd.to_datetime(np.where(filter_df["prev_batch_BM"] == 0, pd.to_datetime("1990-01-01 00:00:00"), filter_df["prev_batch_BM"]))
    # filter_df["prev_batch_BM"] = pd.to_datetime(filter_df["prev_batch_BM"])
    filter_df["prev_batch_BM"] = np.where(filter_df["Case ID"].isin(obs_first_batch), filter_df["ta"],
                                          filter_df["prev_batch_BM"])
    filter_df["time_since_last_batch"] = (filter_df["ta"] - filter_df["prev_batch_BM"]).dt.total_seconds()
    filter_df["pos_within_batch"] = filter_df.apply(
        lambda x: filter_df[(filter_df["ta"] > x["prev_batch_BM"]) & (filter_df["ta"] <= x["ta"])].shape[0], axis=1)

    # filter_df["variant_class"] = list(y_train.values) + list(y_pred)
    #  "observations", "mean_IBCR", "std_IBCR", "batch_day", "batch_month",
    sample_batche_log = filter_df
    # X,y = generate_features(filter_df[["starting cases", "ending cases", "pending cases", "variant_class","waiting_time","ta","tb","Case ID"]])
    X = filter_df[
        ["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event", "time_since_last_batch"
            , "min_delay", "max_delay"
            , "month", "weekday"]]
    y = filter_df["waiting_time"]
    numerical_features = ["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event",
                          "time_since_last_batch"
        , "min_delay", "max_delay"]
    categorical_features = ["month", "weekday"]
    for feature in categorical_features:
        X[feature] = X[feature].astype('category')
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    case_values = filter_df["Case ID"].values
    merged_activity = a1 + "_" + a2
    #print(X_train.dtypes)
    model = train_model_gt(X_train, y_train, X_test, y_test, categorical_features, y, case_values, merged_activity,file,segparams)
    segment_name = a1 + "_" + a2
    mode = "regression"
    directory = path_model + "/" + file
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save_model(directory + "/" + segment_name + "_" + mode + ".txt")

def get_segment_prediction(data_file,activity_indice,uncertain_segments,file):
    # predict next activity for different prefix lengths and results
    next_activity_preds = pd.read_csv(path_pred_activity_res + "/next_activity_and_time_" + file + '.csv')
    next_activity_preds.rename(columns={"CaseID": "Case ID"}, inplace=True)
    # add prefix lengths to join data for remaintining time prediction to new features needed for experiments
    data_file["Prefix length"] = data_file.groupby("Case ID")["Activity"].transform('cumcount').transform(
        lambda x: x + 1)
    # make sure joining keys have same datatypes
    df_1 = data_file.copy()
    df_2 = next_activity_preds[["Case ID", "Prefix length", "Groud truth", "Predicted"]].copy()
    df_1["Case ID"] = df_1["Case ID"].astype(str)
    df_2["Case ID"] = df_2["Case ID"].astype(str)
    # join on Case ID and Prefix Length
    merge_df = pd.merge(df_1, df_2, how='left', left_on=['Case ID', 'Prefix length'],
                        right_on=['Case ID', 'Prefix length']).fillna(0)
    train_ids = split_test_train_data(merge_df, 0.8)
    merge_df["next_activity_prediction"] = np.where(merge_df["Case ID"].isin(train_ids), merge_df["Groud truth"],
                                                    merge_df["Predicted"])
    merge_df["next_activity_prediction"] = merge_df["next_activity_prediction"].apply(lambda x: activity_indice.get(x))
    merge_df.fillna("none", inplace=True)
    # merge_df["merged_activity"] = merge_df.apply(lambda x: x["Activity"]+"_"+x["Activity_1"], axis=1)
    merge_df["merged_activity"] = merge_df.apply(lambda x: x["Activity"] + "_" + x["next_activity_prediction"], axis=1)
    #print(merge_df["merged_activity"].value_counts())
    uncertain_seg_dict = dict({str(v[1:-1].split(",")[0]) + "_" + str(v[1:-1].split(",")[1]): k + 1 for k, v in enumerate(uncertain_segments)})
   # print(uncertain_seg_dict)
    merge_df["Segment_Prediction"] = merge_df["merged_activity"].apply(lambda x: uncertain_seg_dict.get(x, 0))
    return merge_df


# get event data related to a predicted segment for waiting time inclusion
def get_predicted_segment_data(df, a1, a2, cols):
    # shift the datafarme
    df_1 = df.shift(-1)
    rename_columns = {x: x + "_1" for x in df.columns}
    df_1.rename(columns=rename_columns, inplace=True)
    data = pd.concat([df, df_1], axis=1)
    # filter directly follow activities
    data = data[data["Case ID"] == data["Case ID_1"]]

    # replace all activities next to a1 as a2 base on next activity prediction
    data["Activity_1"] = np.where(data["Activity"] == a1, a2, 0)

    # filter for activities a1,a2
    data["merged_activity"] = data.apply(lambda x: x["Activity"] + "_" + x["Activity_1"], axis=1)

    dfg_merged = a1 + "_" + a2
    data = data[data["merged_activity"] == dfg_merged]
    # sort on timestamps of activity a1
    #     data = data[["Case ID", "Complete Timestamp", "Complete Timestamp_1"]]
    data.sort_values(["Complete Timestamp"], axis=0, ascending=True, inplace=True, kind='quicksort')
    #     cases = data["Case ID"].values
    filter_df = data[cols + ["Case ID"]]
    filter_df.rename(columns={"Complete Timestamp": "ta", "Complete Timestamp_1": "tb"}, inplace=True)
    filter_df["ta"] = pd.to_datetime(filter_df["ta"])
    filter_df["tb"] = pd.to_datetime(filter_df["tb"])
    return filter_df

def test_non_batching_dynamics_model(df,a1,a2,cols,file, segClass, merge_df, train_ids):
    # retain without classifier
    # time prediction
    directory = directory = path_model + "/" + file
    segment_name = a1 + "_" + a2

    # load data
    data_seg = df[df["Case ID"].isin(merge_df[merge_df["Segment_Prediction"] == segClass]["Case ID"].values)]
    filter_df = get_predicted_segment_data(data_seg, a1, a2, cols)
    # filter_df= get_eventually_follows_data(df,a1,a2,cols)
    filter_df["waiting_time"] = (filter_df["tb"] - filter_df["ta"]).dt.total_seconds()

    # prepare data
    workload_df = add_workload_features(filter_df)
    filter_df = pd.concat([filter_df.set_index("ta"), workload_df], axis=1).reset_index()
    filter_df["LES"] = filter_df["waiting_time"].shift(1).fillna(method='bfill')
    filter_df["time_since_last_event"] = filter_df["ta"].diff().dt.total_seconds().fillna(0)
    filter_df.head()

    # X,y = generate_features(filter_df[["starting cases", "ending cases", "pending cases", "variant_class","waiting_time","ta","tb","Case ID"]])
    X = filter_df[["starting cases", "ending cases", "pending cases", "LES"]]
    y = filter_df["waiting_time"]
    numerical_features = ["starting cases", "ending cases", "pending cases", "LES"]
    categorical_features = []
    for feature in categorical_features:
        X[feature] = X[feature].astype('category')
    # X_train, X_test, y_train, y_test = split_data(X, y)
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    mode = "regression"
    model = lgb.Booster(model_file=directory + "/" + segment_name + "_" + mode + ".txt")
    filter_df["predicted_waiting_time"] = model.predict(X)

    # set predicted distance as per train and test case IDs
    filter_df["Predicted_Distance"] = np.where(filter_df["Case ID"].isin(train_ids), filter_df["waiting_time"],
                                               filter_df["predicted_waiting_time"])

    # #make dataframe
    case_values = filter_df["Case ID"].values
    predicted_distance = filter_df["Predicted_Distance"].values
    batching_partition = np.zeros(len(y))
    distance_df = pd.DataFrame(
        {"Case ID": case_values, "Predicted_Distance": predicted_distance, "Batching_Partition": batching_partition,
         "merged_activity": np.repeat(segment_name, len(case_values))})
    # save results of prediction
    distance_df.to_csv(path_results + '/' + file + '_' + segment_name + '.csv', index=False)

def test_starting_batch_dynamics_model(df,a1,a2,cols,file,segClass, merge_df, train_ids):
    directory = directory = path_model + "/" + file
    segment_name = a1 + "_" + a2

    # load data
    data_seg = df[df["Case ID"].isin(merge_df[merge_df["Segment_Prediction"] == segClass]["Case ID"].values)]
    filter_df = get_predicted_segment_data(data_seg, a1, a2, cols)
    filter_df["waiting_time"] = (filter_df["tb"] - filter_df["ta"]).dt.total_seconds()

    # prepare data
    workload_df = add_workload_features(filter_df)
    filter_df = pd.concat([filter_df.set_index("ta"), workload_df], axis=1).reset_index()
    filter_df["LES"] = filter_df["waiting_time"].shift(1).fillna(method='bfill')
    filter_df["time_since_last_event"] = filter_df["ta"].diff().dt.total_seconds().fillna(0)
    filter_df.head()

    # get batch info and add those features
    tmp_batch = get_batch_info(filter_df)
    x = filter_df.groupby("ta").size().reset_index()[1:]
    batch_df = x["ta"].apply(lambda x: tmp_batch[tmp_batch["ta"] < x].tail(1).drop(columns=["ta"], axis=1).squeeze())
    batch_df["ta"] = x["ta"]
    batch_df.head()
    filter_df = pd.concat([filter_df.set_index("ta"), batch_df.set_index("ta")], axis=1).reset_index().fillna(0)

    # add no. of observations in current batch
    batch_size = filter_df.groupby("ta").size().reset_index(name="batch_size")
    filter_df = pd.concat([filter_df.set_index("ta"), batch_size.set_index("ta")], axis=1).reset_index().fillna(0)
    filter_df["behaviour class"] = np.where(filter_df["batch_size"] <= 2, "non-batching", "batching-at-start")
    q_25 = np.quantile(filter_df["batch_size"].unique(), 0.25)
    q_50 = np.quantile(filter_df["batch_size"].unique(), 0.50)
    q_75 = np.quantile(filter_df["batch_size"].unique(), 0.75)
    # print(q_25,q_50,q_75)
    filter_df["size class"] = filter_df["batch_size"].apply(lambda x: assign_perf_class(x, q_25, q_50, q_75))


    X = filter_df[["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event", "observations"
        , "mean_IBCR", "std_IBCR", "min_delay", "max_delay", "batch_day"
        , "batch_month", "batch_hour", "is_weekend", "weekday", "month"
        , "duration", "expense", "points", "Resource", "behaviour class", "size class"]]
    y = filter_df["waiting_time"]
    numerical_features = ["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event",
                          "observations",
                          "mean_IBCR", "std_IBCR", "min_delay", "max_delay", "duration", "expense", "points"]
    categorical_features = ["batch_day", "batch_month", "batch_hour", "is_weekend", "weekday", "month", "Resource",
                            "behaviour class", "size class"]

    for feature in categorical_features:
        X[feature] = X[feature].astype('category')

    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    mode = "regression"
    model = lgb.Booster(model_file=directory + "/" + segment_name + "_wt" + mode + ".txt")
    filter_df["predicted_waiting_time"] = model.predict(X)

    # set predicted distance as per train and test case IDs
    filter_df["Predicted_Distance"] = np.where(filter_df["Case ID"].isin(train_ids), filter_df["waiting_time"],
                                               filter_df["predicted_waiting_time"])

    # #make dataframe
    case_values = filter_df["Case ID"].values
    predicted_distance = filter_df["Predicted_Distance"].values
    batching_partition = np.zeros(len(y))
    distance_df = pd.DataFrame(
        {"Case ID": case_values, "Predicted_Distance": predicted_distance, "Batching_Partition": batching_partition,
         "merged_activity": np.repeat(segment_name, len(case_values))})
    # save results of prediction
    distance_df.to_csv(path_results + '/' + file + '_' + segment_name + '.csv', index=False)

def get_true_avg(x,tmp):
    return tmp[tmp["ta"]==x]["true_avg_wt"].values[0]
def get_predicted_avg(x,tmp):
    return tmp[tmp["ta"]==x]["predicted_avg_wt"].values[0]

def test_starting_batch_dynamics_average_model(df,a1,a2,cols,file,segClass, merge_df, train_ids):
    # time prediction
    directory = directory = path_model + "/" + file
    segment_name = a1 + "_" + a2

    # load data
    data_seg = df[df["Case ID"].isin(merge_df[merge_df["Segment_Prediction"] == segClass]["Case ID"].values)]
    filter_df = get_predicted_segment_data(data_seg, a1, a2, cols)
    filter_df["waiting_time"] = (filter_df["tb"] - filter_df["ta"]).dt.total_seconds()

    # prepare data
    workload_df = add_workload_features(filter_df)
    filter_df = pd.concat([filter_df.set_index("ta"), workload_df], axis=1).reset_index()
    filter_df["LES"] = filter_df["waiting_time"].shift(1).fillna(method='bfill')
    filter_df["time_since_last_event"] = filter_df["ta"].diff().dt.total_seconds().fillna(0)
    filter_df.head()

    # get batch info and add those features
    tmp_batch = get_batch_info(filter_df)
    x = filter_df.groupby("ta").size().reset_index()[1:]
    batch_df = x["ta"].apply(lambda x: tmp_batch[tmp_batch["ta"] < x].tail(1).drop(columns=["ta"], axis=1).squeeze())
    batch_df["ta"] = x["ta"]
    batch_df.head()
    filter_df = pd.concat([filter_df.set_index("ta"), batch_df.set_index("ta")], axis=1).reset_index().fillna(0)

    # add no. of observations in current batch
    batch_size = filter_df.groupby("ta").size().reset_index(name="batch_size")
    filter_df = pd.concat([filter_df.set_index("ta"), batch_size.set_index("ta")], axis=1).reset_index().fillna(0)
    filter_df["behaviour class"] = np.where(filter_df["batch_size"] <= 2, "non-batching", "batching-at-start")
    q_25 = np.quantile(filter_df["batch_size"].unique(), 0.25)
    q_50 = np.quantile(filter_df["batch_size"].unique(), 0.50)
    q_75 = np.quantile(filter_df["batch_size"].unique(), 0.75)
    # print(q_25,q_50,q_75)
    filter_df["size class"] = filter_df["batch_size"].apply(lambda x: assign_perf_class(x, q_25, q_50, q_75))

    # if agg is needed
    # aggregate and add new features
    data_pred = filter_df.groupby("ta").mean()
    data_pred["current_batch_observations"] = filter_df.groupby("ta").size().values
    data_pred["avg_waiting_time_lag_1"] = data_pred["waiting_time"].shift(1).fillna(0)
    data_pred.head()

    ##add variant class to features

    # X,y = generate_features(filter_df[["starting cases", "ending cases", "pending cases", "variant_class","waiting_time","ta","tb","Case ID"]])
    X = data_pred[["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event", "observations"
        , "mean_IBCR", "std_IBCR", "min_delay", "max_delay", "batch_day"
        , "batch_month", "batch_hour", "is_weekend", "weekday", "month"
        , "duration", "expense", "points", "current_batch_observations", "avg_waiting_time_lag_1"]]
    y = data_pred["waiting_time"]
    numerical_features = ["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event",
                          "observations",
                          "mean_IBCR", "std_IBCR", "min_delay", "max_delay", "duration", "expense", "points",
                          "current_batch_observations", "avg_waiting_time_lag_1"]
    categorical_features = ["batch_day", "batch_month", "batch_hour", "is_weekend", "weekday", "month"]

    for feature in categorical_features:
        X[feature] = X[feature].astype('category')

    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    mode = "regression"
    model = lgb.Booster(model_file=directory + "/" + segment_name + "_" + mode + ".txt")

    # create a dataframe that saves avg time as per ta
    tmp = pd.DataFrame(
        {"ta": filter_df.groupby("ta").size().index, "true_avg_wt": filter_df.groupby("ta")["waiting_time"].mean(),
         "predicted_avg_wt": model.predict(X)})

    filter_df["true_waiting_time"] = filter_df["ta"].apply(lambda x: get_true_avg(x, tmp))
    filter_df["predicted_waiting_time"] = filter_df["ta"].apply(lambda x: get_predicted_avg(x, tmp))

    # set predicted distance as per train and test case IDs
    filter_df["Predicted_Distance"] = np.where(filter_df["Case ID"].isin(train_ids), filter_df["true_waiting_time"],
                                               filter_df["predicted_waiting_time"])

    # #make dataframe
    case_values = filter_df["Case ID"].values
    predicted_distance = filter_df["Predicted_Distance"].values
    batching_partition = np.zeros(filter_df.shape[0])
    distance_df = pd.DataFrame(
        {"Case ID": case_values, "Predicted_Distance": predicted_distance, "Batching_Partition": batching_partition,
         "merged_activity": np.repeat(segment_name, len(case_values))})
    # save results of prediction
    distance_df.to_csv(path_results + '/' + file + '_' + segment_name + '.csv', index=False)

def test_end_batch_dynamics_model(df, a1, a2, cols, file, segClass, merge_df, train_ids):
    # tree based
    directory = directory = path_model + "/" + file
    segment_name = a1 + "_" + a2

    # load data
    data_seg = df[df["Case ID"].isin(merge_df[merge_df["Segment_Prediction"] == segClass]["Case ID"].values)]
    filter_df = get_predicted_segment_data(data_seg, a1, a2, cols)
    filter_df["waiting_time"] = (filter_df["tb"] - filter_df["ta"]).dt.total_seconds()

    # prepare data
    workload_df = add_workload_features(filter_df)
    filter_df = pd.concat([filter_df.set_index("ta"), workload_df], axis=1).reset_index()
    filter_df["LES"] = filter_df["waiting_time"].shift(1).fillna(method='bfill')
    filter_df["time_since_last_event"] = filter_df["ta"].diff().dt.total_seconds().fillna(0)
    filter_df.head()

    # get batch info and add those features
    tmp_batch = get_end_batch_info(filter_df)
    x = filter_df.groupby("tb").size().reset_index()[1:]
    batch_df = x["tb"].apply(lambda x: tmp_batch[tmp_batch["tb"] < x].tail(1).squeeze())
    batch_df.rename(columns={"tb": "prev_batch_BM"}, inplace=True)
    batch_df["tb"] = x["tb"]
    filter_df = filter_df.merge(batch_df, on='tb', how='left', suffixes=('_1', '_2')).fillna(0)

    # time from last batching moment
    obs_first_batch = filter_df.groupby("tb")["Case ID"].apply(list)[0]
    filter_df["prev_batch_BM"] = pd.to_datetime(
        np.where(filter_df["prev_batch_BM"] == 0, pd.to_datetime("1990-01-01 00:00:00"), filter_df["prev_batch_BM"]))
    # filter_df["prev_batch_BM"] = pd.to_datetime(filter_df["prev_batch_BM"])
    filter_df["prev_batch_BM"] = np.where(filter_df["Case ID"].isin(obs_first_batch), filter_df["ta"],
                                          filter_df["prev_batch_BM"])
    filter_df["time_since_last_batch"] = (filter_df["ta"] - filter_df["prev_batch_BM"]).dt.total_seconds()
    filter_df["pos_within_batch"] = filter_df.apply(
        lambda x: filter_df[(filter_df["ta"] > x["prev_batch_BM"]) & (filter_df["ta"] <= x["ta"])].shape[0], axis=1)

    X = filter_df[
        ["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event", "time_since_last_batch"
            , "min_delay", "max_delay"
            , "month", "weekday"]]
    y = filter_df["waiting_time"]
    numerical_features = ["starting cases", "ending cases", "pending cases", "LES", "time_since_last_event",
                          "time_since_last_batch"
        , "min_delay", "max_delay"]
    categorical_features = ["month", "weekday"]

    for feature in categorical_features:
        X[feature] = X[feature].astype('category')
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    mode = "regression"
    model = lgb.Booster(model_file=directory + "/" + segment_name + "_" + mode + ".txt")
    filter_df["predicted_waiting_time"] = model.predict(X)

    # set predicted distance as per train and test case IDs
    filter_df["Predicted_Distance"] = np.where(filter_df["Case ID"].isin(train_ids), filter_df["waiting_time"],
                                               filter_df["predicted_waiting_time"])

    # #make dataframe
    case_values = filter_df["Case ID"].values
    predicted_distance = filter_df["Predicted_Distance"].values
    batching_partition = np.zeros(len(y))
    distance_df = pd.DataFrame(
        {"Case ID": case_values, "Predicted_Distance": predicted_distance, "Batching_Partition": batching_partition,
         "merged_activity": np.repeat(segment_name, len(case_values))})
    # save results of prediction
    distance_df.to_csv(path_results + '/' + file + '_' + segment_name + '.csv', index=False)