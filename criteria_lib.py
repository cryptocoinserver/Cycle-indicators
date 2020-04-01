# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numba import jit
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def contract_combine(data_list: list, contract_span: pd.DataFrame):
    """
    根据主力列表，将数据列表进行拼接
    参数：
        data_list——原始数据列表（包含与合约数目相同个数的dataframe)
        contract_span——合约日期表
    """
    month_change = []
    combine_data = pd.DataFrame(None)
    for i in range(len(contract_span)):
        if i==0:
            data = data_list[i]
            month_change.append(datetime.datetime.strptime(str(contract_span["end_date"][i]), "%Y%m%d") + datetime.timedelta(hours=17, minutes=00))
            in_data = data[data.index<=month_change[i]]
            combine_data = pd.concat([combine_data, in_data], axis=0)
        else:
            data = data_list[i]
            month_change.append(datetime.datetime.strptime(str(contract_span["end_date"][i]), "%Y%m%d") + datetime.timedelta(hours=17, minutes=00))
            in_data = data[(data.index>month_change[i-1]) & (data.index<=month_change[i])]
            combine_data = pd.concat([combine_data, in_data], axis=0)
    return combine_data


def ann_return(daily_ret):
    return np.mean(daily_ret)*365

def sharpe_ratio(daily_ret):
    sharpe_ratio = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(250)
    return sharpe_ratio
   
def MAR(daily_ret):
    ar = ann_return(daily_ret)
    maxdd,_,_ = max_drawdown(daily_ret)
    return ar/abs(maxdd)


def max_drawdown(daily_cum_ret):
    balance = pd.Series(daily_cum_ret.flatten())
    highlevel =  balance.rolling(min_periods=1, window=len(balance), center=False).max()
    # abs drawdown
    drawdown = balance - highlevel
    maxdd = np.min(drawdown)
    maxdd10 = drawdown.sort_values(ascending = True).iloc[:10].mean()
    # relative drawdown
    dd_ratio = drawdown / (highlevel + 1)
    maxdd_ratio = np.min(dd_ratio)
    return maxdd,maxdd10,maxdd_ratio

def pnl(signal,price,dayclose,slip,contract,tick_size):
    
    cur_holding = np.zeros_like(signal)# num of contract
    asset = np.zeros_like(signal)# current asset
    asset[0] = 100000000
    ret = np.zeros_like(signal)
    cost = np.zeros_like(signal)
    n = signal.shape[0]
    for i in range(n):
        if i>0:
            ret[i] = cur_holding[i-1]*contract*(price[i]-price[i-1])
                                                # ret = change of price x cur_holding - cost
            asset[i] = asset[i-1]+ret[i]
        money = signal[i]*asset[i] - cur_holding[i-1]*price[i]*contract # money to invest
        det_holding = (money/(price[i])/contract).astype(np.int) # unit: num of contract
        cost[i] = abs(det_holding)*contract*slip*tick_size # transaction cost incurred
        ret[i] = ret[i] - cost[i]
        cur_holding[i] = cur_holding[i-1] + det_holding # n of holding contract
        asset[i] = asset[i] - cost[i] # change of asset = ret
    
    cum_ret = asset/asset[0]-1
    daily_cum_ret = cum_ret[dayclose.astype(bool)]
    daily_ret = daily_cum_ret - np.roll(daily_cum_ret,1)
    daily_ret[0] = daily_cum_ret[0]
    dayopen = np.roll(dayclose.astype(bool),1)
    dayopen[0] = True
    open_asset = (asset+cost)[dayopen]
    if dayclose[-1] != 1:
        open_asset = open_asset[:-1]
    daily_ret_pct = daily_ret/open_asset # percentage daily return
    
    det_holding = cur_holding - np.roll(cur_holding,1) # compute turnover
    det_holding[0] = cur_holding[0]
    cum_det_holding = abs(det_holding).cumsum()
    daily_cum_det_holding = cum_det_holding[dayclose.astype(bool)]
    turnover = daily_cum_det_holding - np.roll(daily_cum_det_holding, 1)
    turnover[0] = daily_cum_det_holding[0]
    avg_turnover = np.mean(turnover)

    return daily_cum_ret, daily_ret_pct, avg_turnover, turnover


def performance(paras_tuple, signal_input, signal_func, contract_span, delete,\
                price, dayclose, pnl_paras, mode = 'both'):
    
    # get signal sequence for each bar
    signals = []
    for i in range(len(contract_span)):
        cur_input = signal_input[i]
        signal = signal_func(*cur_input, *paras_tuple)[0]
        signal[-1] = 0
        signal = pd.DataFrame(signal, index = cur_input[0].index)
        signals.append(signal)
    # concatenate signal sequence and price sequence
    combined_signal = contract_combine(signals, contract_span)
    combined_signal = combined_signal[delete:]

    signal = combined_signal.to_numpy()
    price = price.to_numpy()
    slip, contract, tick_size = pnl_paras['slip'], pnl_paras['contract'], pnl_paras['tick_size']

    cur_holding = np.zeros_like(signal)
    asset = np.zeros_like(signal)
    asset[0] = 100000000
    ret = np.zeros_like(signal)
    cost = np.zeros_like(signal)
    n = signal.shape[0]
    for i in range(n):
        if i > 0:
            ret[i] = cur_holding[i - 1] * contract * (price[i] - price[i - 1])
            asset[i] = asset[i - 1] + ret[i]
        money = signal[i] * asset[i] - cur_holding[i - 1] * price[i] * contract
        det_holding = (money / (price[i]) / contract).astype(np.int)
        cost[i] = abs(det_holding) * contract * slip * tick_size
        ret[i] = ret[i] - cost[i]
        cur_holding[i] = cur_holding[i - 1] + det_holding
        asset[i] = asset[i] - cost[i]

    cum_ret = asset / asset[0] - 1
    daily_cum_ret = cum_ret[dayclose.astype(bool)]

    balance = pd.Series(daily_cum_ret.flatten())
    highlevel = balance.rolling(min_periods=1, window=len(balance), center=False).max()
    drawdown = balance - highlevel
    df = pd.DataFrame({'ret': daily_cum_ret.flatten(), 'dd': drawdown.values, 'price': price[dayclose.astype(bool)].flatten()}, 
                        index = combined_signal.loc[dayclose.astype(bool)].index)
   
    plt.subplot(211)
# =============================================================================
#     plt.title('bandpass'+str(paras_tuple))
# =============================================================================
    sns.lineplot(data=df['price'], label='price')
    plt.subplot(212)
    sns.lineplot(data=df['ret'], label='return')
    sns.lineplot(data=df['dd'], label='drawdown')
    plt.legend(loc='center right')
    

def select_paras(paras_tuple, signal_input, signal_func, contract_span, delete,\
                 price, dayclose, pnl_paras, mode = 'train'):
    # get signal sequence for each bar
    signals = []
    for i in range(len(contract_span)):
        cur_input = signal_input[i]
        signal = signal_func(*cur_input, *paras_tuple)[0]
        signal[-1] = 0
        signal = pd.DataFrame(signal, index=cur_input[0].index)
        signals.append(signal)

    # concatenate signal sequence and price sequence
    combined_signal = contract_combine(signals, contract_span)
    combined_signal = combined_signal[delete:]

    # train test split
    signal_train, signal_test = train_test_split(combined_signal, test_size=0.3, shuffle=False)

    if mode == 'train':
        signal_used = signal_train
    else:
        signal_used = signal_test

    # output criteria
    daily_cum_pct, daily_ret_pct, avg_turnover, turnover = pnl(signal_used.to_numpy(), price.to_numpy(), dayclose,
                                                               **pnl_paras)
    sharpe = sharpe_ratio(daily_ret_pct)
    mar = MAR(daily_ret_pct)
    annprofit = ann_return(daily_ret_pct)
    maxdd, maxdd10, maxdd_ratio = max_drawdown(daily_cum_pct)

    return sharpe, mar, annprofit, maxdd, maxdd10, maxdd_ratio, avg_turnover, turnover




def save_result(result, para_list, para_name):
    
    sharpe_list = [res[0] for res in result]
    mar_list = [res[1] for res in result]
    annprofit_list = [res[2] for res in result]
    maxdd_list = [res[3] for res in result]
    maxdd10_list = [res[4] for res in result]
    maxdd_ratio_list = [res[5] for res in result]
    avg_turnover_list = [res[6] for res in result]
    turnover_list = [res[7] for res in result]
    
    result = pd.DataFrame({'sharpe': sharpe_list, 'mar': mar_list,\
                           'annprofit': annprofit_list, 'maxdd': maxdd_list, 'maxdd10':maxdd10_list, \
                           'maxddratio': maxdd_ratio_list, 'avg_turnover': avg_turnover_list, \
                           'turnover': turnover_list})
    paras = pd.DataFrame(para_list, columns = para_name)
    result = pd.concat([paras,result], axis = 1)
    return result




def show_heatmap(result, para1, para2, values, aggfunc):
    report_result = pd.pivot_table(result, index = [para1], columns = [para2], values = values, aggfunc = aggfunc)
    sns.heatmap(report_result, cmap = 'YlGnBu')
    plt.savefig(para1+'&'+para2+'_'+values+'_'+'.png')
    
    
    
    
# =============================================================================
# 
# def max_drawdown(daily_ret):
#     balance = pd.Series(daily_ret.cumsum())
#     highlevel =  balance.rolling(min_periods=1, window=len(balance), center=False).max()
#     # abs drawdown
#     drawdown = balance - highlevel
#     maxdd = np.min(drawdown)
#     maxdd10 = drawdown.sort_values(ascending = True).iloc[:10].mean()
#     # relative drawdown
#     dd_ratio = drawdown/highlevel
#     maxdd_ratio = np.min(dd_ratio)
#     return maxdd,maxdd10,maxdd_ratio
# 
# def pnl(signal,price,dayclose,slip,contract,tick_size):
# 
#     money = 100000000
#     money_alloc = money*signal
#     tmp = (money_alloc/price)/contract
#     holdings = tmp.astype(np.int)*contract
#     det_holdings = holdings-np.roll(holdings,1)
#     det_holdings[0] = holdings[0]
#     cost = slip*abs(det_holdings)*tick_size
#     
#     holdings_1 = np.roll(holdings,1)
#     holdings_1[0] = 0
#     price_1 = np.roll(price,1)
#     price_1[0] = price[0]
#     ret = (price-price_1)*holdings_1
#     
#     ret = ret - cost
#     ret[np.isnan(ret)] = 0
#     cum_ret = ret.cumsum()
#     daily_cum_ret = cum_ret[dayclose]
#     daily_ret = daily_cum_ret - np.roll(daily_cum_ret,1)
#     daily_ret[0] = daily_cum_ret[0]
#     daily_ret_pct = daily_ret/money
#     
#     cum_det_holdings = abs(det_holdings/contract).cumsum()
#     daily_cum_det_holdings = cum_det_holdings[dayclose]
#     turnover = daily_cum_det_holdings - np.roll(daily_cum_det_holdings, 1)
#     turnover[0] = daily_cum_det_holdings[0]
#     avg_turnover = np.mean(turnover)
# 
#     return daily_ret_pct, avg_turnover, turnover
#   
# =============================================================================  