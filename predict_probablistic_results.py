# -*- coding: utf-8 -*-
"""
Created on Jun 2021
@author: Haojun Cai
""" 

import pandas as pd
import numpy as np
import os
from statistics import mean
import matplotlib.pyplot as plt
from time import time

import statsmodels.regression.quantile_regression as sm
import skgarden
from skgarden import RandomForestQuantileRegressor
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
from sklearn.linear_model import QuantileRegressor
from scipy.stats import norm
import statsmodels.formula.api as smf

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
def quantile_loss(quantile, y_true, y_pred):
    """
    Caculate quantile loss for quantile regression model.
    
    Paramaters
    ----------
    quantile : float, input quantile to be evaluated, e.g., 0.5 for median.
    y_true : dataframe, true values
    y_pred : dataframe, predicted values
    
    Returns
    ----------
    quan_loss : dataframe, quantile loss
    """
    
    error = y_true - y_pred
    quan_loss = np.mean(np.maximum(quantile*error, (quantile-1)*error))
    
    return quan_loss

def cal_inbound(y_pred, quan_list, inbound_stat):
    """
    Caculate outbound ratio and avarage inbound range.
    
    Paramaters
    ----------
    y_true : dataframe, true values
    quan_list : list, given quantile lists
    inbound_stat : list, initialized empty list to be appended
    
    Returns
    ----------
    inbound_stat : dataframe, outbound ratio and avarage inbound range
    """
    
    inbound = pd.DataFrame()
    y_pred_range = pd.DataFrame()
    
    # calculate significance levels
    sig_levels = []
    for i in range(0,int(len(quan_list)/2)): 
        sig_levels.append(round(quan_list[-(i+1)] - quan_list[i],3))
    
    column_names = [str(sig_level)+'_outbound' for sig_level in sig_levels] + [str(sig_level)+'_inbound_range' for sig_level in sig_levels]
    inbound_stat_user = pd.DataFrame(columns = column_names)
    
    # calcualte outbound ratio and avarage inbound range for given significance levels
    for i in range(0,int(len(quan_list)/2)):
        lower_quan = quan_list[i]
        upper_quan = quan_list[-(i+1)]
        if lower_quan + upper_quan != 1:
            print('Error: wrong match of upper and lower quantile.')
        sig_level = round((upper_quan-lower_quan), 3)
        
        inbound[str(sig_level)+'_inbound'] = y_pred['true'].between(left=y_pred[lower_quan], right=y_pred[upper_quan])
        y_pred_range[str(sig_level)+'_range'] = y_pred[upper_quan] - y_pred[lower_quan]
        
        outbound_stat = 1 - inbound.loc[inbound[str(sig_level)+'_inbound']==True, str(sig_level)+'_inbound'].sum()/len(inbound)
        inbound_range = y_pred_range.loc[inbound[str(sig_level)+'_inbound']==True, str(sig_level)+'_range'].mean()
        inbound_stat_user.loc[0,str(sig_level)+'_outbound'] = outbound_stat
        inbound_stat_user.loc[0,str(sig_level)+'_inbound_range'] = inbound_range        
    
    inbound_stat = inbound_stat.append(inbound_stat_user,ignore_index=True)

    return inbound_stat

def cal_feat_importance(model, X, user, mob_flag, plotfig_flag, savefig_flag):
    """
    Calculate feature importance.
    
    Paramaters
    ----------
    model : str, model type
    X : dataframe, input columns
    user : float, user to be valuated
    plotfig_flag : boolean, flag indicating whether to plot results
    savefig_flag : boolean, flag indicating whether to save results
    
    Returns
    ----------
    importance : dataframe, importance matrix
    """
    
    feats = {} 
    for feature, importance in zip(X.columns, model.feature_importances_):
        feats[feature] = importance # add the name/value pair 
    
    importance = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
    importance = importance.sort_values(by='Importance')
    
    if plotfig_flag == True:
        fig = plt.figure(figsize=(15,7))
        plt.barh(importance.index, importance['Importance'])
        plt.ylabel('Feature')
        plt.xlabel('Importance')

    if savefig_flag == True:
        if mob_flag == True:
            fig_name = 'F:/0_Thesis_2021/3_prediction/graphs/'+str(user)+'_soc_qrf_features_mob.png'
        else:
            fig_name = 'F:/0_Thesis_2021/3_prediction/graphs/'+str(user)+'_soc_qrf_features.png'
        plt.savefig(fig_name, dpi=100)
    
    if plotfig_flag == True:   
        plt.show()
    
    return importance

def predict_interval(test_feat, ev_feats, data_type, model_type, mob_flag, quan_list, userlist, save_flag, INPUT_PATH, PREDICTION_PATH):
    """
    Predict quantile regression for three targets.
    
    Paramaters
    ----------
    test_feat : list, mobility features to be evaluated
    ev_feats : list, ev-related features (only by used in soc prediction)
    data_type : list, three types of targets
    model_type : list, three types of models
    mob_flag : boolean, flag indicating whether use mobility features as input or not
    quan_list : list, given input quantiles
    userlist : list, users to be evaluated
    save_flag : boolean, flag indicating whether to save results or not
    INPUT_PATH : str, path of inputs
    PREDICTION_PATH : str, path to save results
    
    Returns
    ----------
    N/A
    """
    
    # initializations
    if mob_flag == False:
        model_type_name = model_type
    else:
        model_type_name = model_type + '_mob'
    
    print(model_type_name)
    print('---------------------START--------------------------')
    
    mean_models = ['qrf']    
    if model_type in mean_models:
        deter_eval = {'user_id':[], 'r2_meanpred':[], 'mae_meanpred':[], 'rmse_meanpred':[], 'r2_medpred':[], 'mae_medpred':[], 'rmse_medpred':[]}
    else:
        deter_eval = {'user_id':[], 'r2_medpred':[], 'mae_medpred':[], 'rmse_medpred':[]}

    quan_loss = {'user_id':[]}
    for quan in quan_list:
        quan_loss[quan] = []
    
    inbound_stat = pd.DataFrame()
    
    all_best_model = [] # store best cross validation model
    all_cv_time = []
    
    if model_type=='qrf' or model_type=='gbqr':
        importances = pd.DataFrame()
    
    if data_type == 'soc':
        truecol_name = 'soc'
    if data_type == 'depart':
        truecol_name = 'depart_float'
    if data_type == 'arrival':
        truecol_name = 'arrival_float'
    
    # iterate over all users: train each user an independent model
    for user in userlist:
        print(user)
        quan_loss['user_id'].append(user)
        deter_eval['user_id'].append(user)
        
        # load inputs
        input_path = INPUT_PATH + '/' + str(int(user)) + '_input.csv'
        data = pd.read_csv(input_path)
        
        if data_type == 'soc':
            all_feats =['day_of_year','last_time_of_day','first_time_of_day', 'mean_time_of_day','out_temp',
                         
                        'top10locfre_1day', 'radgyr_1day', 
                        'avrjumplen_1day', 'realentro_1day', 'uncorentro_1day',
                        'ecar_hhindex_1day', 
                        'ev_dist_1day', 'ev_duration_1day',
                                    
                        'top10locfre_2day', 'radgyr_2day', 
                        'avrjumplen_2day', 'realentro_2day', 'uncorentro_2day',
                        'ecar_hhindex_2day', 
                        'ev_duration_2day', 'ev_dist_2day',
                        
                        'top10locfre_3day', 'radgyr_3day', 
                        'avrjumplen_3day', 'realentro_3day', 'uncorentro_3day',
                        'ecar_hhindex_3day', 
                        'ev_duration_3day', 'ev_dist_3day',
              
                        'top10locfre_1weekday', 'radgyr_1weekday', 
                        'avrjumplen_1weekday', 'uncorentro_1weekday', 'realentro_1weekday',
                        'ecar_hhindex_1weekday', 
                        'ev_duration_1weekday', 'ev_dist_1weekday',
                                                                             
                        'top10locfre_2weekday', 'radgyr_2weekday', 'avrjumplen_2weekday', 
                        'uncorentro_2weekday', 'realentro_2weekday', 
                        'ecar_hhindex_2weekday', 
                        'ev_duration_2weekday', 'ev_dist_2weekday',
                  
                        'top10locfre_3weekday', 'radgyr_3weekday', 'avrjumplen_3weekday', 
                        'uncorentro_3weekday', 'realentro_3weekday',                                          
                        'ecar_hhindex_3weekday', 
                        'ev_duration_3weekday', 'ev_dist_3weekday',  
                        
                        'top10locfre_3dayavr', 'radgyr_3dayavr',
                        'avrjumplen_3dayavr', 'uncorentro_3dayavr', 'realentro_3dayavr',
                        'ecar_hhindex_3dayavr', 
                        'ev_duration_3dayavr', 'ev_dist_3dayavr',
    
                        'top10locfre_7day', 'radgyr_7day', 
                        'avrjumplen_7day', 'realentro_7day', 'uncorentro_7day',
                        'ev_duration_7day', 
                        'ecar_hhindex_7day', 'ev_dist_7day',  
                                          
                        'top10locfre_4weekday', 'radgyr_4weekday', 
                        'avrjumplen_4weekday', 'uncorentro_4weekday', 'realentro_4weekday',                                           
                        'ecar_hhindex_4weekday', 
                        'ev_duration_4weekday', 'ev_dist_4weekday'
                        ]
            
            # only keep test features
            removal_feats = list(set(all_feats)-set(test_feat))
            data = data.drop(columns=removal_feats)

            # drop soc na values
            data = data[data['soc'].notnull()]
            
            # fill mob feats as 0
            data = data.fillna(0)
        
            # drop mob features
            if mob_flag == False:
                attrs = test_feat[0].split("_")
                temporal_res = attrs[-1]
                evdata_feats = [feat+'_'+temporal_res for feat in ['ev_dist', 'ev_duration']]
                mob_feat = list(set(test_feat)-set(evdata_feats))
                data = data.drop(columns=mob_feat)
                
            # set target feature
            target_cols = list(set(data.columns)-set(['soc', 'date']))
            target_limit = [0, 100]
        
        if data_type == 'depart':
            data = data.rename(columns={'finish_ymd': 'date'})
            
            all_feat_depart = ['day_of_year',
                                   
                             'top10locfre_1day', 'radgyr_1day', 
                             'avrjumplen_1day', 'realentro_1day', 'uncorentro_1day',
                             
                             'top10locfre_2day', 'radgyr_2day', 
                             'avrjumplen_2day', 'realentro_2day', 'uncorentro_2day',
                             
                             'top10locfre_3day', 'radgyr_3day', 
                             'avrjumplen_3day', 'realentro_3day', 'uncorentro_3day',
                             
                             'top10locfre_1weekday', 'radgyr_1weekday', 
                             'avrjumplen_1weekday', 'uncorentro_1weekday', 'realentro_1weekday',
                                                                   
                             'top10locfre_2weekday', 'radgyr_2weekday', 'avrjumplen_2weekday', 
                             'uncorentro_2weekday', 'realentro_2weekday', 
                       
                             'top10locfre_3weekday', 'radgyr_3weekday', 'avrjumplen_3weekday', 
                             'uncorentro_3weekday', 'realentro_3weekday',

                              'top10locfre_4weekday', 'radgyr_4weekday', 
                             'avrjumplen_4weekday', 'uncorentro_4weekday', 'realentro_4weekday', 
                                       
                             'top10locfre_3dayavr', 'radgyr_3dayavr',
                             'avrjumplen_3dayavr', 'uncorentro_3dayavr', 'realentro_3dayavr',
                             
                             'top10locfre_7day', 'radgyr_7day', 
                             'avrjumplen_7day', 'realentro_7day', 'uncorentro_7day']
            
            # only keep test features
            test_feat = list(set(test_feat)-set(ev_feats))
            
            # remove ecar_hhindex feats for departure prediction
            attrs = test_feat[0].split("_")
            temporal_res = attrs[-1]
            ecar_feat = ['ecar_hhindex'+'_'+temporal_res]
            test_feat = list(set(test_feat)-set(ecar_feat))
            
            removal_feats = list(set(all_feat_depart)-set(test_feat))
            data = data.drop(columns=removal_feats)
                 
            # drop depart na values
            data = data[data['depart_float'].notnull()]
            
            # fill empty mobility features 
            data = data.fillna(0)
            
            # drop mob features
            if mob_flag == False:
                data = data.drop(columns=test_feat)
            
            # set target feature
            target_cols = list(set(data.columns)-set(['depart_float','date_id', 'depart', 'user_id', 'date']))
            target_limit = [0, 24]
            
        if data_type == 'arrival':
            data = data.rename(columns={'start_ymd': 'date'})

            all_feat_arrival = ['day_of_year',
                                      
                                'top10locfre_1day', 'radgyr_1day', 
                                'avrjumplen_1day', 'realentro_1day', 'uncorentro_1day',
                                
                                'top10locfre_2day', 'radgyr_2day', 
                                'avrjumplen_2day', 'realentro_2day', 'uncorentro_2day',
                                
                                'top10locfre_3day', 'radgyr_3day', 
                                'avrjumplen_3day', 'realentro_3day', 'uncorentro_3day',
                                
                                
                                'top10locfre_1weekday', 'radgyr_1weekday', 
                                'avrjumplen_1weekday', 'uncorentro_1weekday', 'realentro_1weekday',
                                                                      
                                'top10locfre_2weekday', 'radgyr_2weekday', 'avrjumplen_2weekday', 
                                'uncorentro_2weekday', 'realentro_2weekday', 
                          
                                'top10locfre_3weekday', 'radgyr_3weekday', 'avrjumplen_3weekday', 
                                'uncorentro_3weekday', 'realentro_3weekday',
                            
                                'top10locfre_3dayavr', 'radgyr_3dayavr',
                                'avrjumplen_3dayavr', 'uncorentro_3dayavr', 'realentro_3dayavr',
                                
                                'top10locfre_4weekday', 'radgyr_4weekday', 
                                 'avrjumplen_4weekday', 'uncorentro_4weekday', 'realentro_4weekday', 
                                          
                                'top10locfre_7day', 'radgyr_7day', 
                                'avrjumplen_7day', 'realentro_7day', 'uncorentro_7day']
            
            # only keep test features
            test_feat = list(set(test_feat)-set(ev_feats))
            removal_feats = list(set(all_feat_arrival)-set(test_feat))
            data = data.drop(columns=removal_feats)
            
            # remove ecar_hhindex feats for departure prediction
            attrs = test_feat[0].split("_")
            temporal_res = attrs[-1]
            ecar_feat = ['ecar_hhindex'+'_'+temporal_res]
            test_feat = list(set(test_feat)-set(ecar_feat))
            
            # drop depart na values
            data = data[data['arrival_float'].notnull()]
            
            # fill empty mobility features
            data = data.fillna(0)
            
            # drop mobility features
            if mob_flag == False:
                data = data.drop(columns=test_feat)
                
            target_cols = list(set(data.columns)-set(['arrival_float','date_id', 'arrival', 'user_id', 'date']))
            target_limit = [0, 24]
        
        # split training and test datasets
        data.index = range(0,len(data))
        
        if data_type == 'soc':
            data['soc_p1'] = data['soc'].shift(1)
            data['soc_p2'] = data['soc'].shift(2)
            data['soc_p3'] = data['soc'].shift(3)
            data[['soc_p1', 'soc_p2', 'soc_p3']] = data[['soc_p1', 'soc_p2', 'soc_p3']].fillna(0)
            target_cols.append('soc_p1')
            target_cols.append('soc_p2')
            target_cols.append('soc_p3')

        if data_type == 'depart':
            data['depart_p1'] = data['depart_float'].shift(1)
            data['depart_p2'] = data['depart_float'].shift(2)
            data['depart_p3'] = data['depart_float'].shift(3)
            data[['depart_p1', 'depart_p2', 'depart_p3']] = data[['depart_p1', 'depart_p2', 'depart_p3']].fillna(0)
            target_cols.append('depart_p1')
            target_cols.append('depart_p2')
            target_cols.append('depart_p3')
 
        if data_type == 'arrival':
            data['arrival_p1'] = data['arrival_float'].shift(1)
            data['arrival_p2'] = data['arrival_float'].shift(2)
            data['arrival_p3'] = data['arrival_float'].shift(3)
            data[['arrival_p1', 'arrival_p2', 'arrival_p3']] = data[['arrival_p1', 'arrival_p2', 'arrival_p3']].fillna(24)
            target_cols.append('arrival_p1')
            target_cols.append('arrival_p2')
            target_cols.append('arrival_p3')
            
        split_len = len(data)
        split_interval = np.ceil(split_len*0.75)
           
        X_train = data.loc[0:split_interval-1, target_cols].copy()
        y_train = data.loc[0:split_interval-1,truecol_name].copy()
        
        X_test = data.loc[split_interval:, target_cols].copy()
        y_test = data.loc[split_interval:,truecol_name].copy()
            
        assert X_train.index.max() < X_test.index.min()    
            
        X = data.loc[:, target_cols].copy()
        y = data[truecol_name].copy() 
            
        print(X.columns)
        print(len(X.columns))
        print(y)        

        # normalize data for LQR models
        models_wostandard = ['qrf', 'gbqr']
        if model_type not in models_wostandard:
            minmax_scaler = preprocessing.MinMaxScaler().fit(X_train)
            X_train_scaled = minmax_scaler.transform(X_train)
            X_test_scaled = minmax_scaler.transform(X_test)
            X_scaled = minmax_scaler.transform(X)
            
        ## MODEL 1: LQG MODEL
        if model_type == 'lqr':    
            model_cv = []
            
            # interate over all model hyperparameters
            alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
            for alphas_cv in alphas:
                model = QuantileRegressor(alpha=alphas_cv, solver='highs')
                model_cv.append(model)
                                    
            quanloss_val = [] 
            quanloss_val_cv = []            
            user_cv_time = []
            cv_split_len = len(X_train)
            
            # start cross validation
            for model_cv_loop in model_cv:
                start_cv = time()
                
                for kfold in range(1,6):
    
                    val_split_lower = int(np.ceil(cv_split_len*(1-kfold/5)))
                    val_split_upper = int(np.ceil(cv_split_len*(1-(kfold-1)/5)))
                    train_idx = list(set(range(0,cv_split_len))-set(range(val_split_lower,val_split_upper)))
                    
                    X_train_cv = X_train.loc[train_idx, ].copy()
                    y_train_cv = y_train.loc[train_idx, ].copy()
        
                    X_val = X_train.loc[val_split_lower:val_split_upper, ].copy()
                    y_val = y_train.loc[val_split_lower:val_split_upper, ].copy()
                  
                    # validate on validatoin dataset
                    y_val_pred = pd.DataFrame(y_val)
                    y_val_pred  = y_val_pred.rename(columns={truecol_name:'true'}) 
                    
                    for quan in quan_list:
                        model_cv_loop = model_cv_loop.set_params(**{'quantile': quan})
                        model_cv_loop = model_cv_loop.fit(X_train_cv, y_train_cv)
                        y_val_pred[quan] = model_cv_loop.predict(X_val)
                        quanloss_val.append(quantile_loss(quan, y_val, y_val_pred[quan]))
                
                print(model_cv_loop)
                print(mean(quanloss_val))
                quanloss_val_cv.append(mean(quanloss_val))
            
                end_cv = time()
                per_cv_time = end_cv - start_cv
                user_cv_time.append(per_cv_time)
            
            print('Total time for the user %.3f seconds' % sum(user_cv_time))
            all_cv_time.append(sum(user_cv_time))
            
            # find best model with smallest mean quantile loss
            best_model_idx = quanloss_val_cv.index(min(quanloss_val_cv))
            best_model = model_cv[best_model_idx]
            print(best_model)            
            
            # return deterministic evaluation metrics using Q=0.5 as expectation value
            best_model = best_model.set_params(**{'quantile': 0.5})
            best_model = best_model.fit(X_train_scaled, y_train)
            y_test_medpred = best_model.predict(X_test_scaled)

            r2_medpred = metrics.r2_score(y_test,y_test_medpred)
            mae_medpred = metrics.mean_absolute_error(y_test, y_test_medpred)
            rmse_medpred = np.sqrt(metrics.mean_squared_error(y_test, y_test_medpred))

            deter_eval['r2_medpred'].append(r2_medpred)
            deter_eval['mae_medpred'].append(mae_medpred)
            deter_eval['rmse_medpred'].append(rmse_medpred)
            
            # return quantile loss of probablistic prediction on test dataset 
            best_model = model_cv[best_model_idx] 
            y_test_pred = pd.DataFrame(y_test)
            y_test_pred= y_test_pred.rename(columns={truecol_name:'true'})
            
            for quan in quan_list:  
                best_model = best_model.set_params(**{'quantile': quan})
                best_model = best_model.fit(X_train_scaled, y_train)
                y_test_pred[quan] = best_model.predict(X_test_scaled)

                quan_loss_user = quantile_loss(quan, y_test, y_test_pred[quan])
                quan_loss[quan].append(quan_loss_user)
    
            # fit over complete dataset for given quantiles
            y_pred = pd.DataFrame(y)
            y_pred = y_pred.rename(columns={truecol_name:'true'}) 
                        
            for quan in quan_list:
                best_model = model_cv[best_model_idx] 
                best_model = best_model.set_params(**{'quantile': quan})
                best_model = best_model.fit(X_train_scaled, y_train)
                y_pred[quan] = best_model.predict(X_scaled)
                
            # limit the range of prediction
            upper_limit = target_limit[1]
            lower_limit = target_limit[0]
            if (y_pred>upper_limit).sum().sum() != 0:
                y_pred[y_pred>upper_limit] = upper_limit
            if (y_pred<lower_limit).sum().sum() != 0:
                y_pred[y_pred<lower_limit] = lower_limit
            
            # return outbound ratio and average inbound range of probablistic prediction on whole dataset 
            inbound_stat = cal_inbound(y_pred, quan_list, inbound_stat)
            
            # save results
            y_pred['date'] = data['date']
            if save_flag == True:
                y_pred_folder = PREDICTION_PATH + '/prediction/' + model_type_name
                if not os.path.exists(y_pred_folder):
                    os.makedirs(y_pred_folder)            
                y_pred_path = y_pred_folder + '/' + str(int(user)) + '_result.csv'
                y_pred.to_csv(y_pred_path, index=False)
        
        ## MODEL 2: QRF MODEL
        if model_type=='qrf':
            
            model_cv = []
            
            # interate over all model hyperparameters
            n_estimators = [100, 200, 300] # number of trees
            max_depth = [1, 3, 5, 7]
            max_depth.append(None) # maximum number of levels in tree
            min_samples_split = [2, 5, 10] # minimum number of samples required to split a node
            min_samples_leaf = [1, 2, 4] # minimum number of samples required at each leaf node
    
            for n_estimators_cv in n_estimators:
                for max_depth_cv in max_depth:
                    for min_samples_split_cv in min_samples_split:
                        for min_samples_leaf_cv in min_samples_leaf:
                            model = RandomForestQuantileRegressor(random_state=0, n_jobs=-1, 
                                                                  n_estimators=n_estimators_cv, 
                                                                  max_depth=max_depth_cv, 
                                                                   min_samples_split=min_samples_split_cv,
                                                                   min_samples_leaf=min_samples_leaf_cv
                                                                  )
                            model_cv.append(model)
                                    
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            quanloss_val_cv = []
            from statistics import mean
            
            user_cv_time = []
            
            # start cross validation
            for model_test in model_cv:
                quanloss_val = [] 
                
                start_cv = time()
                
                for train_index, val_index in kf.split(X_train):
                    # split into training dataset and validation dataset
                    X_train_cv, X_val, y_train_cv, y_val = (X_train.loc[train_index], X_train.loc[val_index], y_train.loc[train_index], y_train.loc[val_index])
                    
                    # fit on training dataset
                    model_test.fit(X_train_cv, y_train_cv)
                    
                    # validate on validatoin dataset
                    y_val_pred = pd.DataFrame(y_val)
                    y_val_pred = y_val_pred.rename(columns={truecol_name:'true'}) 
                    
                    for quan in quan_list:
                        y_val_pred[quan] = model_test.predict(X_val, quantile=quan*100)
                        quanloss_val.append(quantile_loss(quan, y_val, y_val_pred[quan]))
                    
                quanloss_val_cv.append(mean(quanloss_val))
                print(model_test)
                print(mean(quanloss_val))
  
                end_cv = time()
                per_cv_time = end_cv - start_cv
                print('%.3f seconds' % per_cv_time)
                user_cv_time.append(per_cv_time)
            
            print('Total time for the user %.3f seconds' % sum(user_cv_time))
            all_cv_time.append(sum(user_cv_time))
            
            # find best model with smallest mean quantile loss
            print('Cross Validation Finished for One Round')
            best_model_idx = quanloss_val_cv.index(min(quanloss_val_cv))
            best_model = model_cv[best_model_idx]
            print(best_model)
            all_best_model.append(best_model)
        
            # return deterministic evaluation metrics using Q=0.5 as expectation value
            best_model = best_model.fit(X_train, y_train)

            y_test_meanpred = best_model.predict(X_test)
            y_test_medpred = best_model.predict(X_test, quantile=50)
            
            r2_meanpred = metrics.r2_score(y_test,y_test_meanpred)
            mae_meanpred = metrics.mean_absolute_error(y_test, y_test_meanpred)
            rmse_meanpred = np.sqrt(metrics.mean_squared_error(y_test, y_test_meanpred))   
            r2_medpred = metrics.r2_score(y_test,y_test_medpred)
            mae_medpred = metrics.mean_absolute_error(y_test, y_test_medpred)
            rmse_medpred = np.sqrt(metrics.mean_squared_error(y_test, y_test_medpred))              
  
            deter_eval['r2_meanpred'].append(r2_meanpred)
            deter_eval['mae_meanpred'].append(mae_meanpred)
            deter_eval['rmse_meanpred'].append(rmse_meanpred)
            deter_eval['r2_medpred'].append(r2_medpred)
            deter_eval['mae_medpred'].append(mae_medpred)
            deter_eval['rmse_medpred'].append(rmse_medpred)
            
            # return quantile loss of probablistic prediction on test dataset 
            y_test_pred = pd.DataFrame(y_test)
            y_test_pred = y_test_pred.rename(columns={truecol_name:'true'}) 

            for quan in quan_list:
                best_model = model_cv[best_model_idx]
                y_test_pred[quan] = best_model.predict(X_test, quantile=quan*100)
                quan_loss_user = quantile_loss(quan, y_test, y_test_pred[quan])
                quan_loss[quan].append(quan_loss_user)
            
            # return feature importance
            savefig_flag = False
            plotfig_flag = False
            importance = cal_feat_importance(best_model, X_train, user, mob_flag, plotfig_flag, savefig_flag)
            importances = pd.concat([importances, importance.T], axis=0)
            
            # fit over complete dataset for given quantiles
            y_pred = pd.DataFrame(y)
            y_pred = y_pred.rename(columns={truecol_name:'true'}) 
            
            for quan in quan_list:
                best_model = model_cv[best_model_idx]
                y_pred[quan] = best_model.predict(X, quantile=quan*100)
                
            # limit the range of prediction
            upper_limit = target_limit[1]
            lower_limit = target_limit[0]
            if (y_pred>upper_limit).sum().sum() != 0:
                y_pred[y_pred>upper_limit] = upper_limit
            if (y_pred<lower_limit).sum().sum() != 0:
                y_pred[y_pred<lower_limit] = lower_limit
            
            # return outbound ratio and average inbound range of probablistic prediction on whole dataset 
            inbound_stat = cal_inbound(y_pred, quan_list, inbound_stat)
            
            # save results
            y_pred['date'] = data['date']
            if save_flag == True:
                y_pred_folder = PREDICTION_PATH + '/prediction/' + model_type_name
                if not os.path.exists(y_pred_folder):
                    os.makedirs(y_pred_folder)            
                y_pred_path = y_pred_folder + '/' + str(int(user)) + '_result.csv'
                y_pred.to_csv(y_pred_path, index=False)
                                
        ## MODEL 3: GBQR MODEL
        if model_type == 'gbqr':
            
            model_cv = []
            
            # interate over all model hyperparameters
            n_estimators = [100, 200, 300] # number of trees
            max_depth = [1, 3, 5, 7]
            learning_rate = [0.1, 0.06, 0.02]
            subsample = [1.0, 0.8, 0.6]      
            min_samples_split = [2, 5, 10] # minimum number of samples required to split a node
            min_samples_leaf = [1, 2, 4] # minimum number of samples required at each leaf node
    
            for n_estimators_cv in n_estimators:
                for max_depth_cv in max_depth:
                    for learning_rate_cv in learning_rate:
                        for subsample_cv in subsample:
                            for min_samples_split_cv in min_samples_split:
                                for min_samples_leaf_cv in min_samples_leaf:
                            
                                    model = GradientBoostingRegressor(loss='quantile', 
                                                                      n_estimators=n_estimators_cv,
                                                                      max_depth=max_depth_cv,
                                                                      learning_rate=learning_rate_cv,
                                                                      min_samples_split=min_samples_split_cv,
                                                                      min_samples_leaf=min_samples_leaf_cv
                                                                      )
                                    model_cv.append(model)
                         
            cv_split_len = len(X_train)
                    
            quanloss_val_cv = []
            user_cv_time = []
            
            # start cross validation
            for model_test in model_cv:
                print('Model started', model_cv.index(model_test))
                quanloss_val = [] 
                
                start_cv = time()
                
                for kfold in range(1,6):
                    val_split_lower = int(np.ceil(cv_split_len*(1-kfold/5)))
                    val_split_upper = int(np.ceil(cv_split_len*(1-(kfold-1)/5)))
                    train_idx = list(set(range(0,cv_split_len))-set(range(val_split_lower,val_split_upper)))
                    
                    X_train_cv = X_train.loc[train_idx, ].copy()
                    y_train_cv = y_train.loc[train_idx, ].copy()
        
                    X_val = X_train.loc[val_split_lower:val_split_upper, ].copy()
                    y_val = y_train.loc[val_split_lower:val_split_upper, ].copy()
                    
                    y_val_pred = pd.DataFrame(y_val)
                    y_val_pred = y_val_pred.rename(columns={truecol_name:'true'}) 
                    
                    for quan in quan_list:
                        # fit on training dataset
                        model_test = model_test.set_params(**{'alpha': quan})
                        model_test.fit(X_train_cv, y_train_cv)
                        
                        # validate on validatoin dataset
                        y_val_pred[quan] = model_test.predict(X_val)
                        quanloss_val.append(quantile_loss(quan, y_val, y_val_pred[quan]))
                    
                quanloss_val_cv.append(mean(quanloss_val))
                print(model_test)
                print(mean(quanloss_val))
                end_cv = time()
                per_cv_time = end_cv - start_cv
                print('%.3f seconds' % per_cv_time)
                user_cv_time.append(per_cv_time)
            
            print('Total time for the user %.3f seconds' % sum(user_cv_time))
            all_cv_time.append(sum(user_cv_time))
            
            # find best model with smallest mean quantile loss
            print('Cross Validation Finished for One Round')
            best_model_idx = quanloss_val_cv.index(min(quanloss_val_cv))
            best_model = model_cv[best_model_idx]
            print(best_model)
            all_best_model.append(best_model)
            
            # return deterministic evaluation metrics using Q=0.5 as expectation value
            best_model = best_model.fit(X_train, y_train)
            y_test_pred = best_model.predict(X_test)
            r2_medpred = metrics.r2_score(y_test,y_test_pred)
            mae_medpred = metrics.mean_absolute_error(y_test, y_test_pred)
            rmse_medpred = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
            deter_eval['r2_medpred'].append(r2_medpred)
            deter_eval['mae_medpred'].append(mae_medpred)
            deter_eval['rmse_medpred'].append(rmse_medpred)

            # return quantile loss of probablistic prediction on test dataset
            y_test_pred = pd.DataFrame(y_test)
            y_test_pred = y_test_pred.rename(columns={truecol_name:'true'}) 
            
            for quan in quan_list:
                best_model = model_cv[best_model_idx]
                best_model = best_model.fit(X_train, y_train)
                y_test_pred[quan] = best_model.predict(X_test)
                quan_loss_user = quantile_loss(quan, y_test, y_test_pred[quan])
                quan_loss[quan].append(quan_loss_user)
            
            # fit over complete dataset for given quantiles            
            y_pred = pd.DataFrame(y)
            y_pred = y_pred.rename(columns={truecol_name:'true'}) 
   
            for quan in quan_list:
                best_model = model_cv[best_model_idx]
                best_model = best_model.fit(X_train, y_train)
                y_pred[quan] = best_model.predict(X)
                
            # limit the range of prediction
            upper_limit = target_limit[1]
            lower_limit = target_limit[0]
            if (y_pred>upper_limit).sum().sum() != 0:
                y_pred[y_pred>upper_limit] = upper_limit
            if (y_pred<lower_limit).sum().sum() != 0:
                y_pred[y_pred<lower_limit] = lower_limit
            
            # return outbound ratio and average inbound range of probablistic prediction on whole dataset 
            inbound_stat = cal_inbound(y_pred, quan_list, inbound_stat)
            
            # save results
            y_pred['date'] = data['date']
            if save_flag == True:
                y_pred_folder = PREDICTION_PATH + '/prediction/' + model_type_name
                if not os.path.exists(y_pred_folder):
                    os.makedirs(y_pred_folder)             
                y_pred_path = y_pred_folder + '/' + str(int(user)) + '_result.csv'
                y_pred.to_csv(y_pred_path, index=False)

    # save results
    deter_eval = pd.DataFrame(deter_eval)
    quan_loss = pd.DataFrame(quan_loss)
    inbound_stat['user_id'] = userlist
    if model_type=='qrf': 
        importances['user_id'] = userlist
        
    if save_flag == True:
        eval_folder = PREDICTION_PATH + '/evaluation/' + model_type_name
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)     
        deter_eval.to_csv(eval_folder+'/'+'deter_eval.csv', index=False)
        quan_loss.to_csv(eval_folder+'/'+'prob_quanloss.csv', index=False)
        inbound_stat.to_csv(eval_folder+'/'+'prob_inbound.csv', index=False)
        if model_type=='qrf':
            importances.to_csv(eval_folder+'/'+'importances.csv', index=False)










