import helper_prediction_models
import helper
import json
import dataset_confs

file = "traffic_fines"

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


#for each uncertain segment train a waiting-time prediction model for the respective inter-case pattern(s) found in it
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
        helper_prediction_models.train_non_batching_dynamics_model(df, a1, a2, cols, file, segparams)
    elif (pattern == "batching-at-start"):
        helper_prediction_models.train_starting_batch_dynamics_model(df, a1, a2, cols, file, segparams)
        helper_prediction_models.train_starting_batch_dynamics_average_model(df, a1, a2, cols, file, segparams)
    elif (pattern == "batching-at-end"):
        helper_prediction_models.train_end_batch_dynamics_model(df, a1, a2, cols, file, segparams)
    else:
        print("waiting time model for inter-case pattern not found...")

