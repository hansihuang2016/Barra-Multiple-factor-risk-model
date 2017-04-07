# -*- coding: utf-8 -*-
"""
Created on Sat Apr 02 09:45:19 2016

@author: William
"""

from __future__ import print_function
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
np.random.seed(1024)

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import scale
'''
question:
1.sell_signal 就用bool值去回归？
2.the rm sheet has only one column
3.目前非0数目达到阈值的行数太少，暂时将volativity的interval从26改为4
'''

'''
module description: this module provides methods to load data from file,calculate factor value
from them,and format data so that they have proper shape.
'''


def load_file(file_name, sheet_name_list):
    '''
    load xlsx file into a dictionary indexed by sheet names
    :param string file_name:name of file
    :param [string] sheet_name_list: name of selected sheets in the xlsx file
    :return: {string:DataFrame} raw_data: {name of sheet:pure data retrieved from xlsx
    with column and index 0,1,2,...}
    '''
    print 'loading file...'
    cut_head = 2
    file = pd.ExcelFile(file_name)
    raw_data = {}
    # iterate over every sheet and retrieve useful data into raw_data
    for i in range(len(sheet_name_list)):
        print 'parsing sheet', sheet_name_list[i]
        # parse a sheet from the whole file into a DataFrame with headers cut off
        temp = file.parse(sheet_name_list[i]).iloc[cut_head:, :]
        # now temp.dtype = object,because the data read in contains string.Here convert it to float
        temp = temp.astype(np.float)
        # reset index and column with 0,1,2,...,
        temp.columns = range(temp.shape[1])
        temp.index = range(temp.shape[0])
        temp.fillna(0, inplace=True)
        raw_data[sheet_name_list[i]] = temp
    return raw_data


def pre_processing(close, raw_data, ret):
    '''
    清理数据，找出在close中2000年第一个月为0的列，在所有数据中删除这一列
    并标准化数据
    :param DataFrame close :保存收盘价，收盘价将用作去掉2000年第一个月数据为0的列的模板
    :param {string:DataFrame} raw_data: 存放各个指标数据
    :param DataFrame ret :收益率
    :return: {string:DataFrame}data: 经过上述清理的数据
    '''
    # use data from A.D.2000
    reserve_row = 192
    data = dict()
    # 以close在倒数第192行非0的列为模板，在所有的指标矩阵中都只保留这些列
    template = np.where(close.values[-reserve_row] != 0)[0]
    # 删除index为1的，也就是第2只股票，因为这只股票在行业信息中是缺失的，并不知道它属于哪个行业，因此无法计算其行业暴露因子
    template = np.delete(template, 1)
    print 'stocks left', len(template)
    for i in raw_data.items():
        temp = pd.DataFrame(i[1].values[-reserve_row:, template])
        data[i[0]] = temp
        data[i[0]].columns = range(data[i[0]].shape[1])
    ret = pd.DataFrame(ret.values[-reserve_row:, template])
    close = pd.DataFrame(close.values[-reserve_row:, template])
    return [close, data, ret]


def getReturn(close):
    '''
    calculate log return ratio with close price
    :param DataFrame close:close price
    :return: DataFrame ret:log return ratio
    '''
    # get numerator
    up = close.iloc[1:, :]
    up.index = up.index - 1
    # get denominator
    down = close.iloc[:-1, :]
    daily_return = up / down
    ret = daily_return
    # replace null,inf values with 0
    ret.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return ret


def getVol(ret):
    '''
    calculate volatility value of log return ratio
    :param DataFrame ret: return value
    :param int interval: interval over which volatility is calculated
    :return: DataFrame standard_error: volatility value
    '''
    print '''*************************************************************************************
    a kind WARNING from the programmer(not the evil interpreter) function getVol:
    we have different values for interval in test code and real code,because the sample file
    may not have sufficient rows for real interval,leading to empty matrix.So be careful of
    the value you choose
    **************************************************************************************
          '''
    # real value
    # interval = 26
    # test value
    interval = 4
    standard_error = pd.rolling_std(ret, interval)
    standard_error.dropna(inplace=True)
    standard_error.index = range(standard_error.shape[0])
    return standard_error


def getKDJ(close, high, low):
    '''
    calculate KDJ value
    :param DataFrame close:close price
    :param DataFrame high:highest price of a day
    :param DataFrame low: lowest price of a day
    :return: [DataFrame,DataFrame,DataFrame,DataFrame] [RSV, K, D, KDJ]:KDJ value and some subproducts
    '''
    # interval over which KDJ is calculated
    kdj_interval = 9
    N = 3
    # calculate RSV
    # get the close value to be used
    close = pd.DataFrame(close.iloc[(kdj_interval - 1):, :].values)
    # calculate maximum in (kdj_interval) days in high value
    high_max_in_interval = pd.rolling_max(high, kdj_interval)
    # rolling_sum function will set the first (kdj_interval-1) days as np.nan,drop them
    high_max_in_interval.dropna(inplace=True)
    # set index with 0,1,2...,otherwise it will be kdj_interval,kdj_interval+1,...(may not be explicit but fuck the index)
    high_max_in_interval.index = range(high_max_in_interval.shape[0])
    low_min_in_interval = pd.rolling_min(low, kdj_interval)
    low_min_in_interval.dropna(inplace=True)
    low_min_in_interval.index = range(low_min_in_interval.shape[0])
    # calculate RSV
    RSV = 100 * (close - low_min_in_interval) / (high_max_in_interval - low_min_in_interval)
    # replace np.nan and np.inf in RSV because there might be 0 in the denominator of the last formula
    RSV.replace([np.nan, np.inf,-np.inf], 0, inplace=True)
    # get matrix shape
    [row, col] = RSV.shape
    # calculate K
    # assuming N equals n in the formula
    # initialize both N and K with 50
    K = pd.DataFrame(np.zeros([row, col]))
    D = pd.DataFrame(np.zeros([row, col]))
    K.iloc[0, :] = 50 * np.ones([1, col])
    D.iloc[0, :] = 50 * np.ones([1, col])
    # calculate K and D iteratively
    for i in range(1, row):
        K.iloc[i, :] = (RSV.iloc[i, :] + K.iloc[(i - 1), :]) / N
        D.iloc[i, :] = (K.iloc[i, :] - D.iloc[(i - 1), :]) / N
    KDJ = 3 * K - 2 * D
    return [RSV, K, D, KDJ]


def getEMA(close):
    '''
    calculate EMA value
    :param DataFrame close: close price
    :return: DataFrame EMA: EMA value
    '''
    print '''*************************************************************************************
    a kind WARNING from the programmer(not the evil interpreter) function getEMA:
    we have different values for n1,n2,n3 in test code and real code,because the sample file
    may not have sufficient rows for real n1,n2,n3,leading to empty matrix.So be careful of
    the value you choose
    **************************************************************************************
          '''
    # real n1,n2,n3
    n1 = 12
    n2 = 26
    n3 = 9
    # n1,n2,n3 for test
    # n1 = 3
    # n2 = 6
    # n3 = 5
    # calculate MA12
    MA12 = pd.rolling_mean(close, n1)
    # drop np.nan in the first (n1-1) rows
    MA12.dropna(inplace=True)
    # set index with 0,1,2...
    MA12.index = range(MA12.shape[0])
    MA26 = pd.rolling_mean(close, n2)
    MA26.dropna(inplace=True)
    MA26.index = range(MA26.shape[0])
    [row, col] = MA26.shape
    DIF = pd.DataFrame(MA12.iloc[(-row):, :].values) - MA26
    tmp = pd.rolling_mean(DIF, n3)
    tmp.dropna(inplace=True)
    tmp.index = range(tmp.shape[0])
    [row, col] = tmp.shape
    DIF = pd.DataFrame(DIF.iloc[(-row):, :].values)
    EMA = DIF - tmp
    return EMA


def getBuySignal(EMA, trade):
    '''
    calculate buy signal
    :param DataFrame EMA: EMA value
    :param DataFrame trade:trade value
    :return: DataFrame(bool) signal:buy or not
    '''
    [row, col] = EMA.shape
    # here trade_copy has one more row than EMA,so when the .diff() function is applied
    # and the first row full of null is dropped,they have the same shape
    trade_copy = trade.iloc[(-(row + 1)):, :]
    trade_increment = trade_copy.diff()
    trade_increment.dropna(inplace=True)
    trade_increment.index = range(trade_increment.shape[0])
    signal_EMA = EMA > 0
    signal_trade = trade_increment > 0
    signal = signal_EMA * signal_trade
    return signal.astype(np.bool)


def getSellSignal(EMA, trade):
    '''
    calculate buy signal
    :param DataFrame EMA: EMA value
    :param DataFrame trade:trade value
    :return: DataFrame(bool) signal:buy or not
    '''
    [row, col] = EMA.shape
    # here trade_copy has one more row than EMA,so when the .diff() function is applied
    # and the first row full of null is dropped,they have the same shape
    trade_copy = trade.iloc[(-(row + 1)):, :]
    trade_increment = trade_copy.diff()
    trade_increment.dropna(inplace=True)
    trade_increment.index = range(trade_increment.shape[0])
    signal_EMA = EMA < 0
    signal_trade = trade_increment < 0
    signal = signal_EMA * signal_trade
    return signal.astype(np.bool)


def getRSI(close):
    '''
    calculate RSI value
    :param DataFrame close: close price
    :return: DataFrame RSI: RSI value
    '''
    n = 3
    # calculate increment of close price of two succeeding days
    close_increment = close.diff()
    close_increment.dropna(inplace=True)
    close_increment.index = range(close_increment.shape[0])
    close_pos = close_increment.copy()
    close_pos[close_pos < 0] = 0
    close_abs = np.abs(close_increment)
    sum_pos = pd.rolling_sum(close_pos, n)
    sum_pos.dropna(inplace=True)
    sum_pos.index = range(sum_pos.shape[0])
    sum_abs = pd.rolling_sum(close_abs, n)
    sum_abs.dropna(inplace=True)
    sum_abs.index = range(sum_abs.shape[0])
    RSI = sum_pos / sum_abs
    RSI.replace([np.nan, np.inf,-np.inf], 0, inplace=True)
    return RSI


def getMTM(close):
    '''
    calculate MTM value
    :param DataFrame close: close price
    :return: DataFrame MTM: MTM value
    '''
    print '''*************************************************************************************
    a kind WARNING from the programmer(not the evil interpreter) function getEMA:
    we have different values for interval in test code and real code,because the sample file
    may not have sufficient rows for real interval leading to empty matrix.So be careful of
    the value you choose
    **************************************************************************************
    '''
    # real value
    interval = 9
    # test value
    # interval=3
    MTM = close.diff(interval)
    MTM.dropna(inplace=True)
    MTM.index = range(MTM.shape[0])
    return MTM


def getWilliam(close, high, low):
    '''
    计算威廉指数
    :param DataFrame close: 收盘价
    :param DataFrame high: 当日最高价
    :param DataFrame low: 当日最低价
    :return: DataFrame w: 威廉指数
    '''
    # 取14日来算
    n = 14
    high = pd.rolling_max(high, n)
    high.index = range(high.shape[0])
    low = pd.rolling_min(low, n)
    low.index = range(low.shape[0])
    w = 100 - 100 * (close - low) / (high - low)
    w.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
    return w


def clean_data(file_name, index_list):
    '''
    从文件读取数据并清理
    :param string file_name: xlsx文件路径
    :param [string] index_list: 原始指标名列表
    :return: [{string:DataFrame},DataFrame] [factor_data,ret]: 所用的每个指标的数据，各自放在一个DataFrame中，
    每个DataFrame的[i,j]元素是在第(i+1)天第(j+1)只股票在这个指标上的值.并且用相同的方法对ret进行裁剪，以便回归
    '''
    # data:all pure data from file
    data = load_file(file_name, index_list)
    # close:close value as factor
    close = data['close']
    # trade:trade value as factor
    trade = data['trade']
    # ret:return value as factor
    ret = getReturn(data['close'])
    # vol:return volatility as factor
    vol = getVol(ret)
    # KDJ:KDJ value as factor
    [RSV, K, D, KDJ] = getKDJ(close, data['high'], data['low'])
    # ema:EMA value as factor
    EMA = getEMA(close)
    # buy_signal:buy or not?It's a signal,as factor
    buy_signal = getBuySignal(EMA, trade)
    # sell_signal:another signal,as factor
    sell_signal = getSellSignal(EMA, trade)
    # rsi:RSI value as factor
    RSI = getRSI(close)
    # mtm:mtm value as factor
    MTM = getMTM(close)
    ev = data['ev']
    # w William index
    w = getWilliam(close, data['high'], data['low'])
    # 将计算出来的指标存入字典，并找出其最小行数
    unpruned_factor_data = {'KDJ': KDJ, 'EMA': EMA, 'vol': vol, 'MTM': MTM, 'buy_signal': buy_signal,
                            'sell_signal': sell_signal, 'trade': trade, 'RSI': RSI, 'ev': ev, 'William': w}
    [close, data, ret] = pre_processing(close, unpruned_factor_data, ret)
    for i in data.items():
        data[i[0]]=pd.DataFrame(scale(i[1]))
    return [close, data, ret]


def fb_reg_over_time(ret, data,interval):
    '''
    用每只股票在一段时间的收益率与这只股票某个因子在这段时间的值做回归，将回归系数作为每只股票收益率在每个因子上的暴露
    :param DataFrame ret: 收益率
    :param {string:DataFrame} data: 每个因子相关的数据
    :param DataFrame ind_ret: 每个月每个行业的收益率
    :param [int] interval：在滚动中所取的回归区间
    :return: DataFrame X: 每个因子在几个股票上显著？;因子暴露矩阵
    '''
    # X用于记录因子暴露（以回归斜率来刻画），X[i,j]是股票(i+1)的收益率在因子(j+1)上的暴露(row个股票，col个因子)
    X = np.zeros([ret.shape[1], len(data)])
    # num_of_factor是当前正在研究的factor的序号，每个大循环结束之后加1
    num_of_factor = 0
    # name of factors,prepared for converting X to DataFrame,with columns=factor_name
    factor_name = []
    # 对每个因子进行研究,i是一个tuple,i[0]是指标名，i[1]是一个DataFrame，存有[某一月,某个股票]这个因子的值
    for i in data.items():
        factor_name = factor_name + [i[0]]
        interval_data=i[1].ix[interval]
        # 将这个因子显著的股票数目初始化为0
        for j in range(i[1].shape[1]):
            # 取第j个股票在所有时间的收益率与它的因子值进行回归
            model = sm.OLS(ret[j].values, interval_data[j].values).fit()
            # 用回归的斜率来表征因子暴露
            X[j, num_of_factor] = model.params[0]
            # 如果在这个股票上显著，就加1
        num_of_factor += 1
    # 把X转为DataFrame方便处理
    X = pd.DataFrame(X)
    X.fillna(0, inplace=True)
    X.columns = factor_name
    #把显著股票数列表变成DataFrame
    return X
def fb_reg_over_all_time(ret, data):
    '''
    用每只股票在一段时间的收益率与这只股票某个因子在这段时间的值做回归，将回归系数每只股票收益率在每个因子上的暴露
    :param DataFrame ret: 收益率
    :param {string:DataFrame} data: 每个因子相关的数据
    :param DataFrame ind_ret: 每个月每个行业的收益率
    :param [int] interval：在滚动中所取的回归区间
    :return: DataFrame X: 每个因子在几个股票上显著？;因子暴露矩阵
    '''
    # X用于记录因子暴露（以回归斜率来刻画），X[i,j]是股票(i+1)的收益率在因子(j+1)上的暴露(row个股票，col个因子)
    X = np.zeros([ret.shape[1], len(data)])
    # num_of_factor是当前正在研究的factor的序号，每个大循环结束之后加1
    num_of_factor = 0
    # name of factors,prepared for converting X to DataFrame,with columns=factor_name
    factor_name = []
    # 对每个因子进行研究,i是一个tuple,i[0]是指标名，i[1]是一个DataFrame，存有[某一月,某个股票]这个因子的值
    for i in data.items():
        factor_name = factor_name + [i[0]]
        # 将这个因子显著的股票数目初始化为0
        for j in range(i[1].shape[1]):
            # 取第j个股票在所有时间的收益率与它的因子值进行回归
            model = sm.OLS(ret[j].values, i[1][j].values).fit()
            # 用回归的斜率来表征因子暴露
            X[j, num_of_factor] = model.params[0]
            # 如果在这个股票上显著，就加1
        num_of_factor += 1
    # 把X转为DataFrame方便处理
    X = pd.DataFrame(X)
    X.fillna(0, inplace=True)
    X.columns = factor_name
    #把显著股票数列表变成DataFrame
    return X

def ret_reg_loading(tech_loading,ret,dummy):
    '''
    取每月每个指标111个股票的Loading去回归当月这111个股票的收益率，判断是否显著。根据判断结果来筛选变量
    :param tech_loading:
    :param ret:
    :return:
    '''
    # 初始化显著列表
    significant_days=dict()
    for tech in tech_loading.columns:
        significant_days[tech]=0
    # 取每个指标在111只股票上的loading做自变量
    for tech in tech_loading.columns:
        # 取某一个月111只股票的收益率做因变量
        for i in range(ret.shape[0]):
            model = sm.OLS(ret.iloc[i,:].values, pd.concat([tech_loading[tech],dummy],axis=1).values).fit()
            pvalue=model.pvalues[0]
            if pvalue<0.1:
                significant_days[tech]+=1
    return significant_days

if __name__ == '__main__':
    fname = 'E:\\QuantProject2\\temp_data\\hushen_tech.xlsx'
    tecnical_index_list = ['close', 'high', 'low', 'trade', 'growth', 'ev']
    # 指标值和收盘价
    [close, data, ret] = clean_data(fname, tecnical_index_list)
    print (data['EMA'].values).std(axis=0)
