# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:40:45 2016

@author: Administrator
"""

# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import statsmodels.api as sm
import industry_return
import macro
import tech
import widgets
import os
import Get_flow_ev
import total_regression
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
import statsmodels.stats.diagnostic as ss

def Get_total_loading(ret,tech_data,rm,macro_data,finance_loading,dummy):
    interval = np.arange(192)
    #技术指标
    tech_loading = tech.fb_reg_over_time(ret.ix[interval], tech_data,interval).iloc[:,:-1]
    #标准化
    tech_temp = pd.DataFrame(scale(tech_loading,axis=0)[:-3,:])
        
    #市场指标
    mkt_loading = widgets.rm_reg_ri(rm.ix[interval], ret.ix[interval])
    #标准化
    mkt_temp = pd.DataFrame(scale(mkt_loading,axis=0)[:-3,:])
        
    #宏观指标
    macro_loading = macro.macro_reg_ret(ret.ix[interval], macro_data.ix[interval])
    #标准化
    macro_temp = pd.DataFrame(scale(macro_loading,axis=0)[:-3,:])
        
    #财务因子指标
    finance_loading_temp = finance_loading.iloc[:-3,:]
    #标准化
    finance_temp = pd.DataFrame(scale(finance_loading_temp,axis=0))  

    #行业因子指标
    ind_loading = dummy
    #不标准化
    ind_temp = pd.concat([ind_loading.iloc[:-3,:-2],ind_loading.iloc[:-3,-1]],axis=1)
    #拼接数据，得到总的loading
    total_loading = pd.concat([tech_temp,mkt_temp,macro_temp,finance_temp,ind_temp],axis=1)        # 初始化，并且根据考察这两个生成的数据里面有大量0，所以扔了
    return total_loading
if __name__ == '__main__':
    print ('Fine')
