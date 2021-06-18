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
# for process exploration
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.util import constants
from pm4py.util.business_hours import BusinessHours
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import interval_lifecycle
from pm4py.algo.filtering.pandas.variants import variants_filter as variants_filter_df
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.log import case_statistics
from pm4py.statistics.performance_spectrum import algorithm as performance_spectrum
# for prediction task
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

###### GLOBAL VARIABLES #######
max_prefix_len = 20

#### PARAMETER KEYS FOR EVENT LOG #########
param_keys={constants.PARAMETER_CONSTANT_CASEID_KEY: 'Case ID',
        constants.PARAMETER_CONSTANT_RESOURCE_KEY: 'Resource', 
        constants.PARAMETER_CONSTANT_ACTIVITY_KEY: 'Activity',
#         constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY: 'Start Timestamp',
        constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: 'Complete Timestamp'}

###### LOADING PREPROCESSED EVENT LOG ######

def sort_df(df):
    #sort event by case ID and event timestamp
    df.sort_values(["Case ID","Complete Timestamp"], axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last')
    return df

def load_data(file):
    #load event data from csv file into a datframe
    df = pd.read_csv(path_preprocessed_data+'/'+file+'.csv', delimiter=';')
    df = sort_df(df)
    df["Case ID"] = df["Case ID"].astype(str)
    return df

def convert_df_to_log(df):
    #convert df to log
    log = log_converter.apply(df, parameters=param_keys)
    return log 

#### CALCULATE THRESHOLD PREFIX LENGTH FOR EVENT LOG #####
def get_variants_dict(log):
    variants = variants_filter.get_variants(log, parameters = param_keys)
    variants_count = case_statistics.get_variant_statistics(log, parameters = param_keys)
    variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
    return variants, variants_count

def plot_variants_distribution(variants,variants_count):
    # plot distribution of variants and its variants count
    counts = list(map(lambda x: x['count'], variants_count))
    variants = list(map(lambda x:"variant_"+str(x[0]),enumerate(counts)))
    f, ax = plt.subplots(figsize=(15, 6))
    plt.plot(variants,counts)
    start = 0
    end = len(counts)
    #5
    ax.xaxis.set_ticks(np.arange(start, end,10))
    start = 0
    end = max(counts)
    # ax.yaxis.set_ticks(np.arange(start, end, 5000))
    plt.title("variants vs variant_counts")
    plt.xticks(rotation='vertical')
    plt.show()
    
def get_threshold_prefix(df,variants_count):
    # take only most frequent variants by calculating 90th percentile
    counts  = list(map(lambda x: x['count'], variants_count))
    threshold_prefix = min(max_prefix_len,int(np.floor(df.groupby("Case ID").size().quantile(0.90))))
    threshold_count = int(np.floor(np.quantile(counts,0.95)))
    print("Threshold Prefix Length:",threshold_prefix)
    freq_variants = list(filter(lambda x: (len(x['variant'].split(","))<=threshold_prefix) & (x['count']>0),variants_count))
    return threshold_prefix,freq_variants

##### HELPER FUNCTIONS FOR FINE-GRAINED ERROR ANALYSIS ########

def get_performance_spectrum_data(df_filtered,a1,a2):
    #shift the datafarme
    df_1 = df_filtered.shift(-1)
    rename_columns = {x:x+"_1" for x in df_filtered.columns}
    df_1.rename(columns=rename_columns, inplace = True)
    data = pd.concat([df_filtered,df_1],axis=1)
    #filter directly follow activities
    data = data[data["Case ID"]==data["Case ID_1"]]
    #filter for activities a1,a2  
    data["merged_activity"] = data.apply(lambda x: x["Activity"]+"_"+x["Activity_1"], axis=1)
    dfg_merged = a1+"_"+a2 
    data = data[data["merged_activity"]==dfg_merged]
    #sort on timestamps of activity a1
    data = data[["Case ID", "Complete Timestamp", "Complete Timestamp_1"]]
    data.sort_values(["Complete Timestamp"], axis=0, ascending=True, inplace=True, kind='quicksort')
    cases = data["Case ID"].values.astype(str)
    filter_df = data[["Complete Timestamp","Complete Timestamp_1"]]
    filter_df.rename(columns={"Complete Timestamp":"ta", "Complete Timestamp_1":"tb"},inplace=True)
    return cases,filter_df

def plot_error_progression(data,a1,a2,valid_cases,colormap,file):
    cases, filter_df= get_performance_spectrum_data(data,a1,a2)
    print("Number of events in Segment:", filter_df.shape)
    
    # get a datetime that is equal to epoch
    epoch = dt.datetime(1970, 1, 1)
    epoch = pd.to_datetime(epoch, utc=False)
    filter_df["ta"] = (pd.to_datetime(filter_df["ta"],utc=True).dt.tz_convert(None)- epoch).dt.total_seconds()
    filter_df["tb"] = (pd.to_datetime(filter_df["tb"],utc=True).dt.tz_convert(None)- epoch).dt.total_seconds()
        
    if(not filter_df.empty):
        times = filter_df.values
#         x_values = np.array(sorted(times, key = lambda x: (x[0],x[1])))
        x_values = np.array(times)
        n = filter_df.shape[0]
        y_values = np.repeat(np.array([[1,0]], dtype=float),n,axis=0)
#         plt.set_title(a1+":"+a2)
        
        # Mask some values to test masked array support:
        x = np.array(x_values, dtype= 'int64')
        ys = np.array(y_values, dtype = 'int64')
        segs = np.stack((x,ys),axis=-1)
        
        # We need to set the plot limits.
        fig, ax = plt.subplots(figsize=(30,10))
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(ys.min(), ys.max())

#         # We need to set the plot limits.
#         plt.set_xlim(x.min(), x.max())
#         plt.set_ylim(ys.min(), ys.max())

       
        colors = [colormap.get(r, "black") for r in cases]

        line_segments = LineCollection(segs, linewidths=(0.5, 1, 1.5, 2),
                                       colors=colors, linestyle='solid')
        ax.add_collection(line_segments)
                #ax.set_title("Error Progression for"+ a1+"-"+a2)
        #ax.yaxis.set_ticks(np.arange(0, 1, 1))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title("Error Progression for "+ a1+"-"+a2, fontsize=20)
        fig.tight_layout()
#         plt.savefig(path_error_analysis+"/"+file+"_"+a1+"_"+a2+"_error_prog.pdf", dpi=1200)
        directory = path_error_analysis+"/"+file
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory+"/"+a1+"_"+a2+"_error_prog.pdf",dpi=1200,bbox_inches="tight")
        #plt.show()
        return ax
    

def get_relevant_variants(a1,a2,freq_variants):
    # check variants which have a1 and a2 and as directly follows activities
    variants = list(map(lambda x:x['variant'], freq_variants))
    result_variants = []
    for trace in variants:
        trace_dict = {k: v for v, k in enumerate(trace.split(","))}
        if a1 in trace_dict.keys():
            if a2 in trace_dict.keys():
                p2 = trace_dict.get(a2)+1
                p1 = trace_dict.get(a1)+1
                if((p2-p1)==1):
#                     print("Positions:", p1, p2)
                    result_variants.append((trace,p1,p2))
    return result_variants  


def get_relevant_cases(df, variant_list):
    # get all case Ids that exhibit the variants
    filtered_df1 = variants_filter_df.apply(df, variant_list,parameters= param_keys)
    cases = pd.DataFrame({"Case ID":filtered_df1["Case ID"].unique()})
    return cases


def get_relative_error(relevant_cases,p1,p2,file):
    # calculate relative error for relevant cases based on prediction results
    #read prediction result files to calculaute relative errors
    df_a1 = pd.read_csv(path_pred_res+"results/results_train_"+file+"_"+str(p1)+".csv")
#     df_a1 = df_a1[df_a1["true_value"]!=0]
    df_a2 = pd.read_csv(path_pred_res+"results/results_train_"+file+"_"+str(p2)+".csv")
#     df_a2 = df_a2[df_a2["true_value"]!=0]
    if(p1!=1):
        cases_p1 = relevant_cases["Case ID"].apply(lambda x: "%s_%s"%(x, p1))
    else:
        cases_p1 = relevant_cases["Case ID"]
    cases_p2 = relevant_cases["Case ID"].apply(lambda x: "%s_%s"%(x, p2))
#     x_vals = df_a1[df_a1["Case ID"].isin(cases_p1)]["Case ID"].values
    e1 = df_a1[df_a1["Case ID"].isin(cases_p1)]
    e1 = df_a1[["Case ID", "relative_error"]]
    #+"_"+str(x).split("_")[1]
    e1["Case ID"] = e1["Case ID"].apply(lambda x: str(x).rsplit("_",1)[0])
    e2 = df_a2[df_a2["Case ID"].isin(cases_p2)]
    e2 = df_a2[["Case ID", "relative_error"]]
    e2["Case ID"] = e2["Case ID"].apply(lambda x: str(x).rsplit("_",1)[0])
    data = e1.merge(e2, on="Case ID", how="left")
    data = data.dropna()
#     print(data.head())
    error_a1 = data["relative_error_x"].values
    error_a2 = data["relative_error_y"].values
    rae = list(map(lambda x: 1 if(x[0]>x[1]) else 0,  zip(error_a1,error_a2)))
    return df_a1, cases_p1,rae

def set_rae(x,y):
    # classify error progress of occurence as increase case, decrease case or same
    x=round(x,2)
    y =round(y,2)
    if (x==y):
        return 999
    elif (x>y):
        return 1
    else:
        return 0
    
def get_relative_error_test(relevant_cases,p1,p2,file):
    # calculate relative error for relevant cases based on prediction results
    #read prediction result files to calculaute relative errors
    df_a1 = pd.read_csv(path_pred_res+"results/results_test_"+file+"_"+str(p1)+".csv")
#     df_a1 = df_a1[df_a1["true_value"]!=0]
    df_a2 = pd.read_csv(path_pred_res+"results/results_test_"+file+"_"+str(p2)+".csv")
#     df_a2 = df_a2[df_a2["true_value"]!=0]
    if(p1!=1):
        cases_p1 = relevant_cases["Case ID"].apply(lambda x: "%s_%s"%(x, p1))
    else:
        cases_p1 = relevant_cases["Case ID"]
    cases_p2 = relevant_cases["Case ID"].apply(lambda x: "%s_%s"%(x, p2))
#     x_vals = df_a1[df_a1["Case ID"].isin(cases_p1)]["Case ID"].values
    e1 = df_a1[df_a1["Case ID"].isin(cases_p1)]
    e1 = df_a1[["Case ID", "relative_error"]]
#     +"_"+str(x).split("_")[1]
    e1["Case ID"] = e1["Case ID"].apply(lambda x: str(x).rsplit("_",1)[0])
    e2 = df_a2[df_a2["Case ID"].isin(cases_p2)]
    e2 = df_a2[["Case ID", "relative_error"]]
    e2["Case ID"] = e2["Case ID"].apply(lambda x: str(x).rsplit("_",1)[0])
    data = e1.merge(e2, on="Case ID", how="left")
    data = data.dropna()
    print(data.shape)
    error_a1 = data["relative_error_x"].values
    error_a2 = data["relative_error_y"].values
#      1 if(x[0]>x[1]) else 0
    rae = list(map(lambda x: set_rae(x[0],x[1]),  zip(error_a1,error_a2)))
    return df_a1, cases_p1,rae

# build colour map that assigns each case a color based on relative error
def build_colormap(df_a1, cases_p1,rae):
    x_vals = list(df_a1[df_a1["Case ID"].isin(cases_p1)]["Case ID"].values)
    cases = list(map(lambda x: str(x).rsplit("_",1)[0], x_vals))
    colormap = dict()
    for key,val in zip(cases, rae):
        if(val==1 or val == 999):
            color = "red"
#         elif(val==999):
#             color="green"
        else:
            color = "blue"
        colormap[key] = color
    return cases, colormap 

# Find activities at prefix p1 and p2 for all cases
def calculate_rae_for_prefixes(df,df_a1,df_a2,p1,p2):
    case_ids =  list(df_a2["Case ID"].values)
    cases = list(map(lambda x: x.rsplit("_",1)[0], case_ids))
    temp = df[df["Case ID"].isin(cases)]
    t1 = temp.groupby("Case ID")["Activity"].apply(list).apply(lambda x: x[p1 -1]).reset_index(name="Activity_x")
    t2 = temp.groupby("Case ID")["Activity"].apply(list).apply(lambda x: x[p2-1]).reset_index(name="Activity_y")
    t = t1.merge(t2, on="Case ID", how="left")
    e1 = df_a1[["Case ID", "relative_error"]]
    if(p1>1):
#         +"_"+str(x).split("_")[1]
        e1["Case ID"] = e1["Case ID"].apply(lambda x: str(x).rsplit("_",1)[0])
    else:
        e1["Case ID"] = e1["Case ID"].apply(lambda x: str(x))
    e2 = df_a2[["Case ID", "relative_error"]]
    e2["Case ID"] = e2["Case ID"].apply(lambda x: str(x).rsplit("_",1)[0])
    tmp = e1.merge(e2, on="Case ID", how="left")
    tmp["Case ID"] = tmp["Case ID"].astype(str)
    data = t.merge(tmp, on="Case ID", how="left")
    #data.rename(columns={data.columns[1]:"Activity_3", data.columns[2]:"Activity_4", data.columns[3]:"e_3", data.columns[4]:"e_4"}, inplace=True)
    data["rae"] = data["relative_error_x"] - data["relative_error_y"]
    data["rae"] = np.sign(data["rae"])*np.log(abs(data["rae"])+1)
    return data[["Case ID","Activity_x", "Activity_y","relative_error_x", "relative_error_y", "rae"]]

def get_train_data_segment_analysis(threshold_prefix,file,df):
    data = pd.DataFrame()
    for n in range(2,threshold_prefix):
        p1=n
        p2=n+1
        print("Calculating Relative Error for prefixes of length", p1, p2)
        df_a1 = pd.read_csv(path_pred_res+"results/results_train_"+file+"_"+str(p1)+".csv")
        df_a2 = pd.read_csv(path_pred_res+"results/results_train_"+file+"_"+str(p2)+".csv")
        data =pd.concat([data,calculate_rae_for_prefixes(df,df_a1,df_a2,p1,p2)],axis=0)
    return data
def get_test_data_segment_analysis(threshold_prefix,file,df):
    data = pd.DataFrame()
    for n in range(2,threshold_prefix):
        p1=n
        p2=n+1
        print("Calculating Relative Error for prefixes of length", p1, p2)
        df_a1 = pd.read_csv(path_pred_res+"results/results_test_"+file+"_"+str(p1)+".csv")
        df_a2 = pd.read_csv(path_pred_res+"results/results_test_"+file+"_"+str(p2)+".csv")
        data =pd.concat([data,calculate_rae_for_prefixes(df,df_a1,df_a2,p1,p2)],axis=0)
    return data

def create_segment_uncertainity_plot(data,file,type_error):
    #create scatter plot for overall uncertainity 
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(15, 13))
    ax = sns.stripplot(x="Activity_x", y="rae", hue="Activity_y",data=data,dodge=True, jitter=0.5, alpha=0.25)
#     ,bbox_to_anchor=(0.5, 1.05)
    legend = ax.legend(shadow=True, fancybox=True, title="Activity b", bbox_to_anchor=(0., 1.02, 1., .102),loc=3, mode="expand", fontsize=16, ncol=3)
#     ax.set_title("Magnitude of Relative Error Change for each Case Passing through a Segment",fontsize=18,weight='bold')
    
    ax.text(x=0.5, y=1.5, s='Magnitude of Difference between Relative Prediction Errors for each Case Passing through a Segment S=(a,b)', fontsize=18, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    plt.xticks(rotation='45',fontsize=16)
    plt.xlabel("Activity a",fontsize=16)
    plt.ylabel("Difference in Relative Prediction Error, DRAE (Log Scale)",fontsize=16)
    legend.get_title().set_fontsize('16')
    plt.tight_layout()
    directory = path_error_analysis+"/"+file
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory+"/"+type_error+"_error_prog.png",
                        format='png')
    
def get_uncertain_most_segments(data):
    data["rae"]=data["rae"].round(2)
    temp = data.dropna()
    temp['rae_red'] = np.where(temp['rae']> 0,1,0)
    temp['rae_green'] = np.where(temp['rae']==0,1,0)
    temp['rae_blue'] = np.where(temp['rae']<0,1,0)
    average_observations_test_seg = np.mean(temp.groupby(["Activity_x", "Activity_y"]).size().values)
    std_observations = np.std(temp.groupby(["Activity_x", "Activity_y"]).size().values)
    size_df = temp.groupby(["Activity_x", "Activity_y"]).agg({"rae": 'count', "rae_red":'sum', "rae_green":'sum',"rae_blue":'sum'})
    seg_df = size_df
#     print("grouped segment info:", size_df)
# (x/average_observations_test_seg)>=threshold)==True
    size_df = size_df[(size_df['rae']>=2*std_observations) & (round(size_df['rae_red']/size_df['rae_blue'])>=1)]
    #add condition for dealing with end activities if needed; can be also done by sepcifiying prefix length upto which calculating rae is significant
    uncertain_segments = list(size_df.index)
    return uncertain_segments,size_df,seg_df




    
