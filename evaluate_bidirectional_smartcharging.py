# -*- coding: utf-8 -*-
"""
Created on Jun 2021
@author: Haojun Cai  
"""

import pandas as pd
import numpy as np
import os
import math
import datetime

def calculate_soc_parkingduration(soc_charge, parking_duration, soc_end_prev_temp):
    """
    Compare if parking duration is enough to charge EVs to predicted enenrgy consumption
    
    Paramaters
    ----------
    soc_charge : float, initialized soc charge 
    parking_duration : float, predicted parking duration
    soc_end_prev_temp : float, soc_end on previous day
    
    Returns
    ----------
    soc_charge : float, corrected soc charge 
    charge_hrs : float, needed charging time 
    charge_energy : float, needed charging energy
    charge_hrs_ceil : float, ceiled total needed charging hours
    """
    
    charge_energy = 27.2 * (soc_charge/100) # in the unit of kWh
    charge_hrs = charge_energy / 11 # 11kW is the rated power of charging voltbox installed at home
    charge_hrs_ceil = int(np.ceil(charge_hrs))
    
    if soc_end_prev_temp + soc_charge >= 100:
        soc_charge = 100 - soc_end_prev_temp
        charge_energy = 27.2 * (soc_charge/100)
        charge_hrs = charge_energy / 11
        charge_hrs_ceil = int(np.ceil(charge_hrs))
    
    # for the case that parking duration is not enough to charge to 100, charge as much as parking duration could provide
    if parking_duration < charge_hrs:
        charge_hrs = parking_duration
        charge_energy = charge_hrs * 11
        soc_charge = charge_energy / 27.2 * 100
        charge_hrs_ceil = int(np.ceil(charge_hrs))
    
    return soc_charge, charge_hrs, charge_energy, charge_hrs_ceil
            
def charge_price_lowest(charge_hr_temp, price_sort_temp, price_avail_temp, charge_time_temp, charge_time_avail_temp, finan_cost_temp):
    """
    Simulate charging processes during off-peak to record charging periods and corresponding charging prices.
    
    Paramaters
    ----------
    charge_hr_temp : float, needed charging time 
    price_sort_temp : dict, price sorted from lowest to highest by hour
    price_avail_temp : dict, prices during the period that is available for charging by hour
    charge_time_temp : dict, initialzied charged time by hour
    charge_time_avail_temp : dict, available charged time by hour
    finan_cost_temp : float, initialzied financial cost
    
    Returns
    ----------
    finan_cost_temp : float, total financial costs
    charge_time_temp : dict, actual charged time by hour
    charge_time_avail_temp : dict, available charged time by hour
    """

    if charge_hr_temp != 0:    
        for price_lowest_temp in price_sort_temp:
            # find date for price_lowest
            for key, value in price_avail_temp.items():
                if price_lowest_temp in value:
                    price_lowest_hr = key
                    break
            
            # check if there are left available charging time
            if charge_time_avail_temp[price_lowest_hr] >= charge_hr_temp:
                charge_energy_temp = charge_hr_temp * 11
                finan_cost_temp += charge_energy_temp/1000 * price_lowest_temp # price data is in the unit of EUR/mWh
                charge_time_temp[price_lowest_hr] += charge_hr_temp
                charge_time_avail_temp[price_lowest_hr] += -charge_hr_temp
                charge_hr_temp = 0
                break
            
            # in case that parking duration within price_lowest hour is not enough for charging
            if charge_time_avail_temp[price_lowest_hr] < charge_hr_temp and charge_time_avail_temp[price_lowest_hr] != 0: 
                charge_hr_part = charge_time_avail_temp[price_lowest_hr]
                charge_energy_part = charge_hr_part * 11
                finan_cost_temp += charge_energy_part/1000 * price_lowest_temp
                charge_time_temp[price_lowest_hr] += charge_hr_part
                charge_time_avail_temp[price_lowest_hr] += -charge_hr_part
                charge_hr_temp += -charge_hr_part
    
    return finan_cost_temp, charge_time_temp, charge_time_avail_temp

def calculate_finan_cost(charge_hrs_ceil_temp, charge_hrs_temp, price_sort_temp, price_avail_temp, charge_time_temp, charge_time_avail_temp, finan_cost_temp):
    """
    Calculate total prices.
    
    Paramaters
    ----------
    charge_hrs_ceil_temp : float, ceiled total needed charging hours 
    charge_hrs_temp : float, total needed charging hours 
    price_sort_temp : dict, price sorted from lowest to highest by hour
    price_avail_temp : dict, prices during the period that is available for charging by hour
    charge_time_temp : dict, initialzied actual charged time by hour
    charge_time_avail_temp : dict, available charged time by hour
    finan_cost_temp : float, initialzied financial cost
    
    Returns
    ----------
    finan_cost_temp : float, total financial costs
    charge_time_temp : dict, actual charged time by hour
    charge_time_avail_temp : dict, available charged time by hour
    """
    
    ## caculate financial cost: in the unit of EUR
    if charge_hrs_ceil_temp <=1 and charge_hrs_ceil_temp!=0:
        [finan_cost_temp, charge_time_temp, charge_time_avail_temp] = charge_price_lowest(charge_hrs_temp, price_sort_temp, price_avail_temp, charge_time_temp, charge_time_avail_temp, finan_cost_temp)
        
    elif charge_hrs_ceil_temp > 1:
        [charge_hrs_frac, charge_hrs_whole] = math.modf(charge_hrs_temp)
        # calculate for the whole part of charging hours
        for hour in range(0,int(charge_hrs_whole)):
            charge_hr_temp = 1
            [finan_cost_temp, charge_time_temp, charge_time_avail_temp] = charge_price_lowest(charge_hr_temp, price_sort_temp, price_avail_temp, charge_time_temp, charge_time_avail_temp, finan_cost_temp)                                                       
        # calculate for the fraction part of charging hours
        charge_hr_temp = charge_hrs_frac
        [finan_cost_temp, charge_time_temp, charge_time_avail_temp] = charge_price_lowest(charge_hr_temp, price_sort_temp, price_avail_temp, charge_time_temp, charge_time_avail_temp, finan_cost_temp)
    
    # when the parking duration is 0, the car cannot be charged, thus no energy was traded from the market
    elif charge_hrs_ceil_temp == 0:
        finan_cost_temp = 0
    
    else:
        print('Error: charge_hrs_int CANNOT be negative.')
             
    return finan_cost_temp, charge_time_temp, charge_time_avail_temp

def find_soc_sold_real(onpeak_def, soc_pred_date_temp, soc_end_prev_temp, arr_pred_date_temp, soc_start_thres_temp, parking_duration_temp, price_sort_temp, price_avail_temp, charge_time_temp, charge_time_avail_temp):
    """
    Calculate traded enerngy from vehicle to grid during peaks and before charging process starts.
    
    Paramaters
    ----------
    onpeak_def : str, definition of on-peak hours
    soc_pred_date_temp : float, predicted soc charge 
    soc_end_prev_temp : float, soc_end on previous day
    arr_pred_date_temp : float, predicted arrival time
    soc_start_thres_temp : float, starting threshold of soc
    parking_duration_temp : float, predicted parking duration
    price_sort_temp : dict, price sorted from lowest to highest by hour
    price_avail_temp : dict, prices during the period that is available for charging by hour
    charge_time_temp : dict, initialzied actual charged time by hour
    charge_time_avail_temp : dict, available charged time by hour
    
    Returns
    ----------
    soc_real_sold : float, traded energy from vehicle to grid
    """

    soc_charge_temp = max(soc_pred_date_temp, soc_start_thres_temp)
    [soc_charge_temp, charge_hrs_temp, charge_energy_temp, charge_hrs_ceil_temp] = calculate_soc_parkingduration(soc_charge_temp, parking_duration_temp, 0)
    [finan_cost_temp, charge_time_temp, charge_time_avail_temp] = calculate_finan_cost(charge_hrs_ceil_temp, charge_hrs_temp, price_sort_temp, price_avail_temp, charge_time_temp, charge_time_avail_temp, 0)          
    
    charge_time_above0 = dict((int(hour),time_used) for hour, time_used in charge_time_temp.items() if time_used > 0)
    charge_hour_temp = [*charge_time_above0.keys()]
    charge_hour_temp_mod = []
    for hour in charge_hour_temp:
        if hour < 12: 
            charge_hour_temp_mod.append(hour+24)
        else: 
            charge_hour_temp_mod.append(hour)
    
    charge_end_time = min(charge_hour_temp_mod)
    
    soc_sold_temp = soc_end_prev_temp
    if arr_pred_date_temp < charge_end_time:
        soc_sold_max = 11*(charge_end_time-arr_pred_date_temp)/27.2*100
        if soc_sold_max >= soc_sold_temp:
            soc_real_sold = soc_sold_temp
        else:
            soc_real_sold = soc_sold_max
    else:
        soc_real_sold = 0

    return soc_real_sold, charge_end_time
               
def evaluate_bi_smartcharging(onpeak_def, price, soc_start_thres, soc_end_penalty, userlist, quan, depart_quan, arrival_quan, model_type, mob_flag, save_flag, ARRIVAL_PATH, DEPART_PATH, SOC_PATH, RESULT_PATH):
    """
    Simluate the bidirectional smart charging.
    
    Paramaters
    ----------
    onpeak_def : str, the type of on-peak definition 
    price : dataframe, price data over all days
    soc_start_thres : float, buffer soc_start
    soc_end_penalty : list, threshold to penalize negative soc_end
    userlist : list, users to be evaluated
    quan : list, soc prediction at different quantiles
    depart_quan : float, departure prediction at chosen quantile
    arrival_quan : float, arrival prediction at chosen quantile
    model_type : str, model to be evaluated (QRF)
    mob_flag : list, flags indicating with mobility features or not
    save_flag : boolean, flag indicating whether to save results
    ARRIVAL_PATH : str, path of arrival prediction results
    DEPART_PATH : str, path of departure prediction results
    SOC_PATH : str, path of soc prediction results
    RESULT_PATH : str, path to save smart charging results
    
    Returns
    ----------
    N/A
    """

    if mob_flag == False:
        model_type_name = model_type
    else:
        model_type_name = model_type + '_mob'
    
    print(model_type_name)
    print(quan)
    print('----------------------------------')
    soc_end_neg = {'user_id':[], 'below20_num':[], 'below0_num':[]}
    soc_end_neg_items = pd.DataFrame()
     
    # create ditionary to store charging behavior by hour
    charge_time = {}
    time_24hr = [str(n) for n in range(0,24)]
    for time in time_24hr:
        charge_time[time] = 0
         
    risk_out_all = {'day_sum':[], 'energy_charge_out':[], 'time_charge_out':[]}
        
    for user in userlist:
  
        energy_charge_out = 0
        time_charge_out = 0
        
        soc_end_below20 = 0
        soc_end_below0 = 0
        soc_end_neg['user_id'].append(user)
        soc_end_neg_idxs = []
        soc_end_below20_flag = False
        
        # read data
        arrival_path = ARRIVAL_PATH + 'prediction/' + model_type_name + '/' + str(int(user)) + '_result.csv'
        arrival_pred = pd.read_csv(arrival_path)
        depart_path = DEPART_PATH + 'prediction/' + model_type_name + '/' + str(int(user)) + '_result.csv'
        depart_pred = pd.read_csv(depart_path)
        soc_path = SOC_PATH + 'prediction/' + model_type_name + '/' + str(int(user)) + '_result.csv'
        soc_pred = pd.read_csv(soc_path)
    
        # filter out daily soc consumption that is over 100
        soc_pred = soc_pred[soc_pred['true']>0]
        
        soc_dates = list(soc_pred['date'].unique()[:])
        arr_dates = list(arrival_pred['date'].unique()[:])
        dep_dates = list(depart_pred['date'].unique()[:])
        delta = datetime.timedelta(days=1)
        
        soc_end_prev = 0 # on day 1, starting soc is set as 0
        cost_user = {'date':[], 'money_cost':[], 'money_gain_v2g':[], 'tech_cost':[]}
        soc_user = {'user_id':[],'date':[], 'soc_start':[], 'soc_start_cor':[], 'soc_end':[], 'soc_end_cor':[], 'soc_charge':[], 'soc_diff_true':[], 'soc_diff_pred':[], 'parking_duration_pred':[]}
        dep_pred_date = -99
        arr_pred_date = -99   
        dep_true_date = -99
        arr_true_date = -99
        
        # for soc on certain day
        for date in soc_dates: 
            print(date)
            parking_flag = False
            date_prev = str((pd.to_datetime(date)-delta).date())
            
            # extract soc prediction and true value on that day          
            soc_pred_date = list(soc_pred.loc[soc_pred['date']==date, str(quan)])[0]
            soc_true_date = list(soc_pred.loc[soc_pred['date']==date,'true'])[0]           
            
            # extract departure on the same day
            if date in dep_dates:
                dep_true_date = list(depart_pred.loc[depart_pred['date']==date,'true'])[0]          
                dep_pred_date = list(depart_pred.loc[depart_pred['date']==date, str(depart_quan)])[0]          
            
            # extract arrival on the last day
            if date_prev in arr_dates:
                arr_true_date = list(arrival_pred.loc[arrival_pred['date']==date_prev,'true'])[0]
                arr_pred_date = list(arrival_pred.loc[arrival_pred['date']==date_prev, str(arrival_quan)])[0]
            
            # parking_duration = dep_pred_date + 24 - arr_pred_date
            invalid_values = [-99]
            if dep_true_date not in invalid_values and arr_true_date not in invalid_values:
                parking_flag = True
            if dep_true_date == -1 or arr_true_date == -1:
                print('Error: invalid values for arrival or departure time')
                
            if parking_flag == True:                  
                parking_duration = dep_pred_date + 24 - arr_pred_date
                if parking_duration > 48 or parking_duration < 0:
                    print('Error: invalid predicted paring duration')

                cost_user['date'].append(date)
                soc_user['user_id'].append(user)
                soc_user['date'].append(date)
                soc_user['soc_diff_true'].append(soc_true_date)
                soc_user['soc_diff_pred'].append(soc_pred_date)
                soc_user['parking_duration_pred'].append(parking_duration)

                # read price data on that day and on previous day
                price_prev_date = price.loc[price['date']==date_prev]
                price_prev_date.index = range(0,len(price_prev_date))
                price_date = price.loc[price['date']==date]
                price_date.index = range(0,len(price_date))
     
                # find prices for available charging slots and prices from arrival time (previous day) to depature time
                charge_time_avail = {}
                time_24hr = [str(n) for n in range(0,24)]
                for time in time_24hr:
                    charge_time_avail[time] = 0
                
                price_avail = {}
                
                prev_hour = max(13, int(np.floor(arr_pred_date))) 
                period_prev_date = list(range(prev_hour,24))
                period_prev_date = [str(hour) for hour in period_prev_date]
                for time in period_prev_date:
                    price_avail[time] = [price_prev_date.loc[0,time]]
                    if time != period_prev_date[0]:
                        charge_time_avail[time] += 1
                    else:
                        charge_time_avail[time] += 1 - (arr_pred_date%1)
                
                current_hour = min(12, int(np.ceil(dep_pred_date)))
                period_date = list(range(0,current_hour))
                period_date = [str(hour) for hour in period_date]
                for time in period_date:
                    if time in list(price_avail.keys()):
                        price_avail[time].append(price_date.loc[0,time])
                    else:
                        price_avail[time] = [price_date.loc[0,time]]
                    if time != period_date[-1]:
                        charge_time_avail[time] += 1
                    else:
                        charge_time_avail[time] += (dep_pred_date%1)
                        
                # sort prices from lowest to highest
                price_sort = []
                for key in price_avail.keys():
                    price_sort += price_avail[key]               
                price_sort.sort()
                
                ## Step 1: calculate traded enerngy
                # calculate charging hours needed 
                soc_pred_date_copy = soc_pred_date
                soc_end_prev_copy = soc_end_prev
                arr_pred_date_copy = arr_pred_date
                price_sort_copy = price_sort.copy()
                price_avail_copy = price_avail.copy()
                charge_time_avail_copy = charge_time_avail.copy()
                
                # create ditionary to store temp charging behavior by hour to find soc_sold
                charge_time_copy = {}
                time_24hr = [str(n) for n in range(0,24)]
                for time in time_24hr:
                    charge_time_copy[time] = 0
        
                # find the traded enerngy in percentage (soc) before charging process starts
                [soc_real_sold, charge_end_time] = find_soc_sold_real(onpeak_def, soc_pred_date_copy, soc_end_prev_copy, arr_pred_date_copy, 
                                                   soc_start_thres, parking_duration, price_sort_copy, 
                                                   price_avail_copy, charge_time_copy, charge_time_avail_copy)
                
                soc_end_prev = soc_end_prev - soc_real_sold
                soc_charge = soc_pred_date
                
                [soc_charge, charge_hrs, charge_energy, charge_hrs_ceil] = calculate_soc_parkingduration(soc_charge, parking_duration, soc_end_prev)                
                
                ## Step 2: calculate soc_start
                soc_start = soc_end_prev + soc_charge 
                soc_user['soc_charge'].append(soc_charge)
                soc_user['soc_start'].append(soc_start) 
                
                finan_cost = 0
                [finan_cost, charge_time, charge_time_avail] = calculate_finan_cost(charge_hrs_ceil, charge_hrs, price_sort, price_avail, charge_time, charge_time_avail, finan_cost)          
                tech_cost = charge_energy

                # calculate the sold energy from the systen when the price is low
                energy_sold = soc_real_sold/100 * 27.2
                time_sold = energy_sold / 11
                v2g_finan_gain = 0
                
                if soc_real_sold != 0:
                    new_hours_24 = []
                    for hour in price_avail.keys():
                        if int(hour)<=12:
                            new_hours_24.append(str(int(hour)+24))
                        else:
                            new_hours_24.append(hour)
                            
                    price_avail_sold = dict(zip(new_hours_24, list(price_avail.values())))
                    
                    all_hours_sold = []
                    real_hours_sold = []
                    all_hours_sold = [int(i) for i in price_avail_sold.keys()]
                    real_hours_sold = range(min(all_hours_sold),charge_end_time)
                    
                    price_avail_sold = dict((hour,price) for hour,price in price_avail_sold.items() if int(hour) in real_hours_sold)
                    
                    prices_sold = []
                    for i in real_hours_sold:
                        prices_sold.append(price_avail_sold[str(i)])
                    prices_sold.sort(reverse=True)
                    
                    time_sold_whole = int(np.floor(time_sold))
                    time_sold_float = time_sold - time_sold_whole
                    
                    list_of_key = list(price_avail_sold.keys())
                    list_of_value = list(price_avail_sold.values())
                    
                    v2g_finan_gain = 0
                    if len(prices_sold) != 0:
                        if time_sold_whole > len(real_hours_sold):
                            time_sold_whole = len(real_hours_sold)
                            
                        if time_sold_whole != 0:
                            for i in range(0, time_sold_whole):
                                v2g_finan_gain += 1*11/1000 * prices_sold[i][0]
                                position = list_of_value.index(prices_sold[i][0])
                                time_position = list_of_key[position]
                                if int(time_position)>=24: time_position = int(time_position) - 24
                                charge_time[str(time_position)] -= 1
                            
                        if time_sold_float!=0:
                            v2g_finan_gain += time_sold_float*11/1000 * prices_sold[0][0]
                            position = list_of_value.index(prices_sold[0][0])
                            time_position = list_of_key[position]
                            if int(time_position)>=24: time_position = int(time_position) - 24
                            charge_time[str(time_position)] -= time_sold_float
                
                cost_user['money_gain_v2g'].append(v2g_finan_gain)

                # check soc_start range 
                # if it is lower than the threshold, then charge the car (a real charging behavior)
                if soc_start > 100:
                    print('Error: soc_start should not be over 100')
                    soc_start = 100

                soc_user['soc_start_cor'].append(soc_start)
                
                ## Step 3: calculate soc_end
                soc_end = soc_start - soc_true_date
                soc_user['soc_end'].append(soc_end)
 
                # check if soc_end is smaller than 0, if so, record it as a risk
                # which needs to be charged outside/at workplace instead of at home
                if soc_end < 0:
                    print('soc_end is negative')
                    soc_charge_out = 0 - soc_end
                    energy_charge_out += soc_charge_out/100 * 27.2
                    time_charge_out += (soc_charge_out/100 * 27.2) / 11
                    soc_end = 0              
                    
                soc_user['soc_end_cor'].append(soc_end)
                
                soc_end_prev = soc_end
                cost_user['money_cost'].append(finan_cost)
                cost_user['tech_cost'].append(tech_cost)
                
                if soc_end_below20_flag == True:
                    soc_end_neg_idxs.append(len(soc_user['date'])-1)
                    soc_end_below20_flag = False
                    
        soc_end_neg['below20_num'].append(soc_end_below20)
        soc_end_neg['below0_num'].append(soc_end_below0)
        cost_user = pd.DataFrame(cost_user) 
        soc_user = pd.DataFrame(soc_user)
        soc_end_neg_items_user = soc_user.loc[soc_end_neg_idxs]
        soc_end_neg_items = pd.concat([soc_end_neg_items, soc_end_neg_items_user], axis=0)
        
        # save results
        if save_flag == True:
            cost_res_folder = RESULT_PATH + '/cost'
            if not os.path.exists(cost_res_folder):
                os.makedirs(cost_res_folder)
            cost_res_path = cost_res_folder + '/' + str(int(user)) + '_result.csv'
            cost_user.to_csv(cost_res_path, index=False)        

            soc_res_folder = RESULT_PATH + '/soc_state'
            if not os.path.exists(soc_res_folder):
                os.makedirs(soc_res_folder)            
            soc_res_path = soc_res_folder + '/' + str(int(user)) + '_result.csv'
            soc_user.to_csv(soc_res_path, index=False)  

        risk_out_all['energy_charge_out'].append(energy_charge_out)
        risk_out_all['time_charge_out'].append(time_charge_out)
        risk_out_all['day_sum'].append(len(soc_user))
        
    # save results
    soc_end_neg = pd.DataFrame(soc_end_neg)
    charge_time = pd.DataFrame(charge_time, index=[1])
    risk_out_all = pd.DataFrame(risk_out_all)
    
    if save_flag == True:
        chagre_time_path = RESULT_PATH + '/' + 'hourly_charge_profile.csv'
        charge_time.to_csv(chagre_time_path, index=False)   
        risk_out_path = RESULT_PATH + '/' + 'risk_outside_charging.csv'
        risk_out_all.to_csv(risk_out_path, index=False) 



