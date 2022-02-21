# -*- coding: utf-8 -*-
"""
Created on Jun 2021 
@author: Haojun Cai 
"""

import pandas as pd
import numpy as np

def compare_feat_importance(model, mob_flags, EVAL_PATH):
    """
    Return mean feature importance and the ranks of the importance for QRF model.
    
    Paramaters
    ----------
    model : str, model type (QRF)
    mob_flags : boolean, flag indicating whether to consider mobility features
    EVAL_PATH : str, path to save results
    
    Returns
    ----------
    N/A
    """
    
    for mob_flag in mob_flags:
        if mob_flag == False:
            model_name = model
        else:
            model_name = model + '_mob'
            
        print(model_name)
        eval_folder = EVAL_PATH + '/evaluation/' + model_name
        
        feat_importance_model = pd.read_csv(eval_folder+'/'+'importances.csv')
        if feat_importance_model[feat_importance_model<0].sum().sum() > 0:
            print('Feature importance smaller than 0:\n', feat_importance_model[feat_importance_model<0].sum())
            
        # calculate mean importance for each feature across all users
        feat_mean_importance_model = feat_importance_model.drop(columns=['user_id']).mean(axis=0)
        feat_mean_importance_model = pd.DataFrame(feat_mean_importance_model)
        feat_mean_importance_model.columns = ['mean_importance']
        feat_mean_importance_model = feat_mean_importance_model.sort_values(by=['mean_importance'], ascending=False)
        
        # calculate when it ranks in the first third of all features
        feats = list(set(feat_importance_model.columns))
        feats.remove('user_id')
        
        timefeat = ['1day', '2day', '3day', '3dayavr', '7day', '1weekday', '2weekday', '3weekday', '4weekday']
        mean_importance_attr = feat_mean_importance_model.copy()
        mean_importance_attr['time'] = np.nan
        mean_importance_attr['feat'] = np.nan
        for idx in feat_mean_importance_model.index:
            attrs = idx.split("_")
            mean_importance_attr.loc[idx,'time'] = attrs[-1]
            mean_importance_attr.loc[idx,'feat'] = idx.replace('_'+attrs[-1],'')
        
        mean_importance_attr_part = mean_importance_attr[mean_importance_attr['time'].isin(timefeat)]
        mean_importance_attr_left = mean_importance_attr.drop(mean_importance_attr_part.index)
        mean_importance_attr_left = pd.DataFrame(mean_importance_attr_left['mean_importance'])
        mean_importance_attr_save = {}
        for time in mean_importance_attr_part['time']:
            mean_importance_attr_save[time] = mean_importance_attr_part.loc[mean_importance_attr_part['time']==time,'mean_importance'].sum()
        for feat in mean_importance_attr_part['feat']:
            mean_importance_attr_save[feat] = mean_importance_attr_part.loc[mean_importance_attr_part['feat']==feat,'mean_importance'].sum()            
        mean_importance_attr_save = pd.DataFrame(mean_importance_attr_save, index=[0]).T
        mean_importance_attr_save.columns = ['mean_importance']

        mean_importance_attr_save_new1 = mean_importance_attr_save.loc[mean_importance_attr_part['time'].unique()].sort_values(by='mean_importance',ascending=False)
        mean_importance_attr_save_new2 = mean_importance_attr_save.loc[mean_importance_attr_part['feat'].unique()].sort_values(by='mean_importance',ascending=False)
        mean_importance_attr_save = pd.concat([mean_importance_attr_save_new1,mean_importance_attr_save_new2,mean_importance_attr_left])
                
        feat_mean_importance_model.to_csv(eval_folder+'/'+'mean_importance.csv', index=True)
        mean_importance_attr_save.to_csv(eval_folder+'/'+'mean_importance_byattr.csv', index=True)

        


    
    
    