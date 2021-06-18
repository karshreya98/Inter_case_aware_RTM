import helper
import helper_prediction_models
import json
import dataset_confs
import os
import pandas as pd
import numpy as np


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


file = "traffic_fines"
use_average = True

#load event log as dataframe
df = helper.load_data(file)

# fetch uncertain segments and respective inter-case patterns
with open("segments.json") as segmentsJSON:
    segment_params = json.load(segmentsJSON)
    segmentsJSON.close()

uncertain_segments = list(segment_params[file].keys())
patterns = list(segment_params[file].values())
print(segment_params)

# fetch numerical and categorical features for the event log to use and encode them while training prediction models
dataset = file
dynamic_cat_cols = list(dataset_confs.dynamic_cat_cols[dataset])
static_cat_cols = dataset_confs.static_cat_cols[dataset]
dynamic_num_cols =dataset_confs.dynamic_num_cols[dataset]
static_num_cols = dataset_confs.static_num_cols[dataset]
cat_cols = dynamic_cat_cols + static_cat_cols
num_cols = dynamic_num_cols + static_num_cols

print(cat_cols,num_cols)

#load json object for hyperparameters
with open("parameters_waiting_time_models.json") as parametersJSON:
    hyperparams = json.load(parametersJSON)
    parametersJSON.close()

#enrich event log with segment predictions
data_file = df
#map activity label to a classes 1,2,...
activity_indice = dict((i,a) for i,a in enumerate (data_file["Activity"].unique()))
merge_df = helper_prediction_models.get_segment_prediction(data_file,activity_indice,uncertain_segments,file)

# get case ids for training set
train_ids = helper_prediction_models.split_test_train_data(merge_df,0.8)


#enrich event log with waiting time predictions
for i in range(len(uncertain_segments)):
    segment = (uncertain_segments[i][1:-1])
    a1 = str(segment.split(",")[0])
    a2 = str(segment.split(",")[1])
    print(a1, a2)
    pattern = patterns[i]

    segparams = hyperparams[file][uncertain_segments[i]]["params"]

    print(segparams)
    cols = num_cols + cat_cols + ["Complete Timestamp", "Complete Timestamp_1"]
    if (pattern == "non-batching"):
        print(df.head())
        helper_prediction_models.test_non_batching_dynamics_model(df,a1,a2,cols,file, int(i+1), merge_df, train_ids)
    elif (pattern == "batching-at-start"):
        if (use_average == False):
            helper_prediction_models.test_starting_batch_dynamics_model(df,a1,a2,cols,file, int(i+1), merge_df, train_ids)
        else:
            helper_prediction_models.test_starting_batch_dynamics_average_model(df,a1,a2,cols,file, int(i+1), merge_df, train_ids)
    elif (pattern == "batching-at-end"):
        helper_prediction_models.test_end_batch_dynamics_model(df,a1,a2,cols,file, int(i+1), merge_df, train_ids)
    else:
        print("waiting time model for inter-case pattern not found...")

# add created inter-case features to event log
distance_df = pd.DataFrame()
for i in range(len(uncertain_segments)):
    segment = (uncertain_segments[i][1:-1])
    a1 = str(segment.split(",")[0])
    a2 = str(segment.split(",")[1])
    print(a1, a2)
    merged_activity= a1+"_"+a2
    seg_df = pd.read_csv(path_results+'/'+file+'_'+ merged_activity+'.csv')
    distance_df = pd.concat([distance_df,seg_df],axis=0).fillna(0)


#enrich event log with waiting time predictions
inter_case_feature_names = ["Segment_Prediction", "Predicted_Distance"]
for i in range(len(uncertain_segments)):
    segment = (uncertain_segments[i][1:-1])
    a1 = str(segment.split(",")[0])
    a2 = str(segment.split(",")[1])
    print(a1, a2)
    pattern = patterns[i]
    dfg_merged = a1+"_"+a2
    distance_df["C_S"+str(i+1)] = np.where(distance_df["merged_activity"] == dfg_merged, 1, 0)
    inter_case_feature_names.append("C_S"+str(i+1))

#add distances within segment
df_1 = merge_df
df_2 = distance_df
df_1["Case ID"] = df_1["Case ID"].astype(str)
df_2["Case ID"] = df_2["Case ID"].astype(str)
data_rtp = pd.merge(df_1, df_2,  how='left', left_on=['Case ID', 'merged_activity'], right_on = ['Case ID', 'merged_activity']).fillna(0)
data_rtp = data_rtp[list(data_file.columns) + inter_case_feature_names]
data_rtp = data_rtp.drop(columns=["Prefix length"])

data_rtp["Segment_Prediction"] = np.where(data_rtp["Segment_Prediction"]>0, 1, 0)
print(data_rtp.sample(5))
data_rtp.to_csv(path_new_rtp_data+'/'+file+'_inter_case_date.csv',sep=";",index=False)
