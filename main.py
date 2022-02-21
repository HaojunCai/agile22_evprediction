# -*- coding: utf-8 -*-
"""
Created on Jun 2021
@author: Haojun Cai  
""" 

import os
import geopandas as gpd
import numpy as np 
import pandas as pd
from sqlalchemy import create_engine

# import local scripts
codepath = "E:/Haojun/code"
os.chdir(codepath)
import db_login # credential info to access the database
import extract_mobility, extract_evfeatures
import extract_soc, extract_arrival, extract_depart
import predict_probablistic_results, calculate_under_overestimation, calculate_feature_importance, compare_probablistic_results
import evaluate_unidirectional_smartcharging as uni_smart
import evaluate_bidirectional_smartcharging as bi_smart
import evaluate_uncontrolled_charging as base_charge
import compare_baseline_unismart as base_uni
import compare_baseline_bismart as base_bi
import compare_three_charging_onpeakdef2 as plot_threecharging

# set working directory
path = "E:/Haojun"
os.chdir(path)
print("Current working directory:",os.getcwd())

#%% Connect Database and Access Datasets
from db_login import DSN
engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(**DSN))

pandas_query = """SELECT vin,zustand,timestamp_start_utc,timestamp_end_utc,soc_customer_start,soc_customer_end,
                id,consumed_electric_energy_total,user_id,a_temp_start,a_temp_end 
                FROM version_20181213.bmw""" 
bmwdf = pd.read_sql(pandas_query, engine)
pandas_query = """SELECT * FROM version_20181213.staypoints""" 
stpgdf = gpd.read_postgis(pandas_query, engine, geom_col='geometry_raw')
pandas_query = """SELECT id, user_id, trip_id, started_at, finished_at, mode_validated, validated
                FROM version_20181213.triplegs""" 
triplegdf = pd.read_sql(pandas_query, engine)

# keep 113 users with valid home labels
userlist = bmwdf['user_id'].value_counts()[:].index.tolist()
filter_list = [1600, 1729, 1730, 1668, 1605, 1734, 1737, 1738, 1746, 1686, 1815, 1690, 1627, 1629, 1757, 1694, 1761, 1826, 1772, 1645, 1712, 1598]
filtered_userlist = list(set(userlist)-set(filter_list))

#%% Extract Mobility Features
# preprocess staypoints data
mob_folder = path + '/data/preprocess/mob_features'
extract_mobility.preprocess_staypoints(bmwdf, stpgdf, mob_folder)

# extract daily mobility features for each user
stp_path = mob_folder + '/stp_cls.csv'
stp_cls = pd.read_csv(stp_path)
userlist = list(set(stp_cls['user_id']))
PREPROCESS_PATH = mob_folder + '/daily'
extract_mobility.extract_mobility_daily(stp_cls, userlist, PREPROCESS_PATH)

#%% Extract E-Car Related Features
# extract hhiindex features
userlist = bmwdf['user_id'].value_counts()[:].index.tolist()
RESULT_PATH = path + '/data/preprocess/evrelated_features'            
extract_evfeatures.extract_hhindex_daily(triplegdf, userlist, RESULT_PATH)

# extract ecar duration and distance features
process_folder = path + '/data/preprocess'
extract_evfeatures.preprocess_bmw(process_folder, engine)
bmw_process = pd.read_csv(process_folder+'/bmw_process.csv')
extract_evfeatures.extract_evstat_daily(bmw_process, userlist, RESULT_PATH)

#%% Prepare SoC Inputs and Targets
# preprocess bmw data
bmwdf_negsoc = extract_soc.preprocess_bmw(bmwdf)

# extract soc features
saveflag = True
SOCTARGET_PATH = path + '/data/preprocess/soc_prediction/soc'
userlist = list(set(bmwdf_negsoc['user_id']))
soc_above100_stat = extract_soc.extract_soc_target(userlist, bmwdf_negsoc, saveflag, SOCTARGET_PATH)

# add ecar hhiindex features 
HHINDEX_PATH = path + '/data/preprocess/evrelated_features/hhindex' 
SOCHHINDEX_PATH = path + '/data/preprocess/soc_prediction/soc_hhindex'
extract_soc.add_soc_hhindex(userlist, SOCTARGET_PATH, HHINDEX_PATH, SOCHHINDEX_PATH) 

# add ecar duration and distance features 
EVSTAT_PATH = path + '/data/preprocess/evrelated_features/evstat' 
SOCHHINDEXEVSTAT_PATH =  path + '/data/preprocess/soc_prediction/soc_hhindex_evstat'
extract_soc.add_sochhindex_evstat(userlist, SOCHHINDEX_PATH, EVSTAT_PATH, SOCHHINDEXEVSTAT_PATH)

# add mobility features
MOB_PATH = path + '/data/preprocess/mob_features'
SOCINPUT_PATH = path + '/data/inputs/soc_prediction'
extract_soc.add_soc_mob(filtered_userlist, SOCHHINDEXEVSTAT_PATH, MOB_PATH, SOCINPUT_PATH) 

#%% Prepare Arrival Time Inputs and Targets
# extract arrival targets
savefile_flag = True
ARRIVALTARGET_PATH = path + '/data/preprocess/arrival_prediction/arrival'
if not os.path.exists(ARRIVALTARGET_PATH):
    os.makedirs(ARRIVALTARGET_PATH)
arrival_stat = extract_arrival.extract_arrival_target(engine, filtered_userlist, savefile_flag, ARRIVALTARGET_PATH)

# add mobility features
ARRIVALMOB_PATH = path + '/data/preprocess/arrival_prediction/arrival_mob'
extract_arrival.add_arrival_mob(filtered_userlist, ARRIVALTARGET_PATH, MOB_PATH, ARRIVALMOB_PATH)

# convert arrival features to float numbers in [0, 24]
ARRIVAL_PATH = path + '/data/inputs/arrival_prediction'
extract_arrival.construct_arrival_input(filtered_userlist, ARRIVALMOB_PATH, ARRIVAL_PATH)  

#%% Prepare Departure Time Inputs and Targets
# extract departure features
savefile_flag = True
DEPARTTARGET_PATH = path + '/data/preprocess/depart_prediction/depart'
if not os.path.exists(DEPARTTARGET_PATH):
    os.makedirs(DEPARTTARGET_PATH)
depart_stat = extract_depart.extract_depart_target(engine, filtered_userlist, savefile_flag, DEPARTTARGET_PATH, ARRIVALTARGET_PATH)

# add mobility features
DEPARTMOB_PATH = path + '/data/preprocess/depart_prediction/depart_mob'
extract_depart.add_depart_mob(filtered_userlist, DEPARTTARGET_PATH, MOB_PATH, DEPARTMOB_PATH)  

# convert departure features to float numbers in [0, 24]
DEPART_PATH = path + '/data/inputs/depart_prediction'
extract_depart.construct_depart_input(filtered_userlist, DEPARTMOB_PATH, DEPART_PATH) 

#%% Run Quantile Regression Predictions
data_types = ['soc', 'depart', 'arrival']
models = ['lqr', 'qrf', 'gbqr']
mob_flags = [True, False]
quan_list = np.round(np.linspace(0,1,41),3).tolist()[1:-1]
save_flag = True

all_test_feat = [['radgyr_3dayavr', 'ecar_hhindex_3dayavr',              
                  'top10locfre_3dayavr', 'avrjumplen_3dayavr', 
                  'realentro_3dayavr',     
                  'ev_duration_3dayavr', 'ev_dist_3dayavr']]

ev_feats_base = ['ev_duration', 'ev_dist']

for test_feat in all_test_feat:
    attrs = test_feat[0].split("_")
    temporal_res = attrs[-1]
    print(temporal_res)
    print('START------------------------------')
    ev_feats = [ev_feat+'_'+temporal_res for ev_feat in ev_feats_base]
    
    for data_type in data_types:
        print(data_type)
        for model_type in models:
            for mob_flag in mob_flags:            
                INPUT_PATH = path + '/data/inputs/'+data_type+'_prediction'
                PREDICTION_PATH = path + '/data/results/predictions/'+data_type+'_prediction'+'_'+temporal_res
                if not os.path.exists(PREDICTION_PATH):
                    os.makedirs(PREDICTION_PATH) 
                    
                predict_probablistic_results.predict_interval(test_feat, ev_feats, data_type, model_type, mob_flag, quan_list, filtered_userlist, temporal_res, save_flag, INPUT_PATH, PREDICTION_PATH)

#%% Evaluate Quantile Regression Predictions
for test_feat in all_test_feat:
    attrs = test_feat[0].split("_")
    temporal_res = attrs[-1]
    
    # calculate evaluation metrics
    for data_type in data_types:
        print(data_type)
        for model_type in models:
            for mob_flag in mob_flags:
                
                if mob_flag == False:
                    model_name = model_type
                else:
                    model_name = model_type + '_mob'
                    
                print(model_name)
                RESULT_PATH = path + '/data/results/predictions/'+data_type+'_prediction'+'_'+temporal_res
                inbound_stat = calculate_under_overestimation.cal_inbound_plus(quan_list, filtered_userlist, model_name, RESULT_PATH)
                eval_folder = RESULT_PATH + '/evaluation/' + model_name   
                inbound_stat.to_csv(eval_folder+'/'+'prob_inbound_underover.csv', index=False)
    
    # calculate feature importance for QRF model
    qrfmodel = 'qrf'
    for data_type in data_types:
        EVAL_PATH = path + '/data/results_0208/predictions/'+data_type+'_prediction'+'_'+temporal_res
        calculate_feature_importance.compare_feat_importance(qrfmodel, mob_flags, EVAL_PATH)
    
    # output direct comparison for evaluation metrics across different models 
    for data_type in data_types:
        EVAL_PATH = path + '/data/results/predictions/'+data_type+'_prediction'+'_'+temporal_res
        [deter_eval, quanloss_quan, quanloss_level, inbound_stat] = compare_probablistic_results.compare_deter_eval(models, mob_flags, quan_list, EVAL_PATH)
        deter_eval.to_csv(EVAL_PATH+'/evaluation/'+'deter_eval.csv', index=True)
        quanloss_quan.to_csv(EVAL_PATH+'/evaluation/'+'quanloss_quan.csv', index=True)
        quanloss_level.to_csv(EVAL_PATH+'/evaluation/'+'quanloss_level.csv', index=True)
        inbound_stat.to_csv(EVAL_PATH+'/evaluation/'+'inbound_stat.csv', index=True)

#%% Simulate Charging Strategies
# preprocess price data
PRICE_PATH = path + '/data/inputs/auction_spot_prices_switzerland_2017.csv'
price = pd.read_csv(PRICE_PATH)
price = price[price.columns.tolist()[0:26]]
price = price.drop(columns=['Hour 3B'])
price.columns = ['date'] + [str(n) for n in range(0,24)]

# set parameters for smart charging strategies
model_type = 'qrf'
save_flag = True
soc_quan_list = np.round(np.linspace(0.45,1,12),3).tolist()[1:-1]
depart_quan = 0.10
arrival_quan = 0.90
soc_start_thres = 0
soc_end_penalty = [0,-20]
mob_flags = [True]

ARRIVAL_PATH = path + '/data/results/predictions/arrival_prediction_3dayavr/' 
DEPART_PATH =  path + '/data/results/predictions/depart_prediction_3dayavr/' 
SOC_PATH =  path + '/data/results/predictions/soc_prediction_3dayavr/'
if not os.path.exists(ARRIVAL_PATH):
    os.makedirs(ARRIVAL_PATH)
if not os.path.exists(DEPART_PATH):
    os.makedirs(DEPART_PATH)
if not os.path.exists(SOC_PATH):
    os.makedirs(SOC_PATH)

#%% Simulate unidirectional smart charging
UNISMARTCHARGE_PATH = path + '/data/results/charging_strategies/unidirectional_smart_charging'
if not os.path.exists(UNISMARTCHARGE_PATH):
    os.makedirs(UNISMARTCHARGE_PATH)

soc_end_neg_quan = {}
for mob_flag in mob_flags:
    for quan in soc_quan_list: 
        if mob_flag == False:
            model_type_name = model_type
        else:
            model_type_name = model_type + '_mob'
                   
        RESULT_PATH = UNISMARTCHARGE_PATH + '/' + model_type_name + '_soc' + str(quan)
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        uni_smart.evaluate_uni_smartcharging(price, soc_start_thres, soc_end_penalty, filtered_userlist, quan, depart_quan, arrival_quan, model_type, mob_flag, save_flag, ARRIVAL_PATH, DEPART_PATH, SOC_PATH, RESULT_PATH)

#%% Simulate bidirectional smart charging
onpeak_defs = ['def2']
BISMARTCHARGE_PATH = path + '/data/results/charging_strategies/bidirectional_smart_charging'
if not os.path.exists(BISMARTCHARGE_PATH):
    os.makedirs(BISMARTCHARGE_PATH)

for mob_flag in mob_flags:
    for quan in soc_quan_list:        
        for onpeak_def in onpeak_defs:
            if mob_flag == False:
                model_type_name = model_type
            else:
                model_type_name = model_type + '_mob'
            
            RESULT_PATH = BISMARTCHARGE_PATH + '/' + model_type_name + '_soc' + str(quan)+ '_onpeak' + onpeak_def
            if not os.path.exists(RESULT_PATH):
                os.makedirs(RESULT_PATH)
            bi_smart.evaluate_bi_smartcharging(onpeak_def, price, soc_start_thres, soc_end_penalty, filtered_userlist, 0.95, depart_quan, arrival_quan, model_type, mob_flag, save_flag, ARRIVAL_PATH, DEPART_PATH, SOC_PATH, RESULT_PATH)

#%% Simulate uncontrolled charging as baseline
BASELINE_PATH = path + '/data/results/charging_strategies/uncontrolled_charging'
if not os.path.exists(BASELINE_PATH):
    os.makedirs(BASELINE_PATH)
base_charge.evaluate_baseline(price, filtered_userlist, save_flag, UNISMARTCHARGE_PATH, ARRIVAL_PATH, DEPART_PATH, SOC_PATH, BASELINE_PATH)

#%% Evaluate unidirectional smart charging compared with baseline
RESULT_PATH = path + '/data/results/charging_strategies/comparison/baseline_unidirectional'
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)
    
# calculate financial measures
for mob_flag in mob_flags:
    for quan in soc_quan_list:
        
        if mob_flag == False:
            model_type_name = model_type
        else:
            model_type_name = model_type + '_mob'
            
        SMARTCHARGE_PATH = UNISMARTCHARGE_PATH + '/' + model_type_name + '_soc' + str(quan)
        base_uni.calculate_cost_user(filtered_userlist, save_flag, model_type_name, quan, BASELINE_PATH, SMARTCHARGE_PATH, RESULT_PATH)

# compare financial costs
base_uni.evaluate_cost_model(model_type, mob_flags, soc_quan_list, save_flag, UNISMARTCHARGE_PATH, RESULT_PATH)

# compare peak-shaving effects
LOADPROFILE_PATH = path + '/data/inputs/electricity_load_profile_day_Jun2nd.csv'
base_uni.evaluate_peakshaving_way2(model_type, mob_flags, soc_quan_list, LOADPROFILE_PATH, BASELINE_PATH, UNISMARTCHARGE_PATH, RESULT_PATH)

#%% Evaluate bidirectional smart charging compared with baseline
RESULT_PATH = path + '/data/results/charging_strategies/comparison/baseline_bidirectional'
onpeak_def = 'def2'
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

# calculate financial measures
for mob_flag in mob_flags:
    for quan in soc_quan_list:
        if mob_flag == False:
            model_type_name = model_type
        else:
            model_type_name = model_type + '_mob'
            
        SMARTCHARGE_PATH = BISMARTCHARGE_PATH + '/' + model_type_name + '_soc' + str(quan) + '_onpeak' + onpeak_def
        base_bi.calculate_cost_user(filtered_userlist, save_flag, model_type_name, quan, BASELINE_PATH, SMARTCHARGE_PATH, RESULT_PATH)

# compare financial costs
base_bi.evaluate_cost_model(model_type, mob_flags, soc_quan_list, onpeak_def, save_flag, BISMARTCHARGE_PATH, RESULT_PATH)

# compare peak-shaving effects
LOADPROFILE_PATH = 'E:/Haojun/data/inputs/electricity_load_profile_day_Jun2nd.csv'    
base_bi.evaluate_peakshaving_way2(model_type, mob_flags, soc_quan_list, LOADPROFILE_PATH, BASELINE_PATH, BISMARTCHARGE_PATH, RESULT_PATH)

#%% Plot load profiles of three charging strategies
RESULT_PATH = path + '/data/results/charging_strategies/comparison'  

for quan in soc_quan_list: 
    print(quan)
    plot_threecharging.compare_three_charging(model_type, [True], [quan], LOADPROFILE_PATH, BASELINE_PATH, UNISMARTCHARGE_PATH, BISMARTCHARGE_PATH, RESULT_PATH)








