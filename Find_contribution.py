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




if __name__ == '__main__':
    '''
    程序运行各项参数的设定
    '''
    config = {'mkt_file': 'E:\\QuantProject2\\raw_data\\market.xlsx',
              'rf_file': 'E:\\QuantProject2\\raw_data\\TRD_Nrrate.xls',
              'tech_fname': 'E:\\QuantProject2\\temp_data\\hushen_tech.xlsx',
              'tecnical_index_list': ['close', 'high', 'low', 'trade', 'growth', 'ev'],
              'macro_file': 'E:\\QuantProject2\\raw_data\\macro.xlsx'
              }
    '''
    读入市场指标
    '''
    mkt = widgets.load_market(config['mkt_file'])
    '''
    读入无风险利率
    '''
    rf = widgets.load_rf(config['rf_file'])/100
    '''
    计算行业收益率
   '''
    # 产生文件名
    ind_path = 'E:\\QuantProject2\\raw_data\\'
    ind_fname_lst = ['E:\\QuantProject2\\raw_data\\TRD_Mnth.xls']
    for i in range(6):
        ind_fname_lst = ind_fname_lst + ['E:\\QuantProject2\\raw_data\\TRD_Mnth' + str(i + 1) + '.xls']
    # 将所有交易数据文件按列拼好
    [ind_data, ev] = industry_return.load_industry_data(ind_fname_lst)
    ind_ret = industry_return.get_ind_return(ind_data)
    
    #读入市值，未标准化的市值
    [trash,flow_ev] = Get_flow_ev.load_industry_data(ind_fname_lst)
    flow_ev = flow_ev.iloc[:,:-3]

    '''
    技术指标
    '''
    # 指标值和收盘价
    [close, tech_data, ret] = tech.clean_data(config['tech_fname'], config['tecnical_index_list'])
    tech_data['ev'] = ev
    ev = ev**0.5
    # 用股票收益率减去无风险收益率作为新的收益率
    ret = pd.DataFrame(ret.values - rf.values-1)
    rm = pd.DataFrame(mkt.values - rf.values, columns=['rm'])

    '''
    宏观因子
    '''
    macro_data = macro.load_macro_data(config['macro_file'])
    macro_data = pd.DataFrame(macro_data.drop(["Deficit"],axis=1))
#    macro_pca = PCA(n_components=5)
#    macro_data = pd.DataFrame(macro_pca.fit_transform(macro_data))
    # 将2000-2015每一年这五个财务指标存入以int年份为key的字典
    '''
    财务因子
    '''
    finance_loading=widgets.load_finance_data('E:\\financial_data')
    

    # 通过整个192个月范围内的数据两部回归，通过显著性扔掉了trade和EMA
    # 读入所选用的111只股票的代码
    stkcd=pd.read_excel('E:\\QuantProject2\\temp_data\\selected_codes.xlsx')
    for i,code in enumerate(stkcd['stkcd']):
        stkcd['stkcd'].ix[i]=code+1000000
    stkcd['stkcd']=stkcd['stkcd'].apply(lambda x:str(x)[1:])
    # 读入dummy 矩阵并选取其中我们所需的股票
    dummy=pd.read_excel('E:\\QuantProject2\\temp_data\\all_stocks_dummy.xlsx')
    dummy[0]=dummy[0].apply(lambda x:x[:-3])
    dummy=dummy.set_index([0])
    dummy=dummy.ix[stkcd['stkcd'].values]
    dummy=pd.DataFrame(dummy.values)
    # 代码为000009的股票没有行业信息，将其dummy令为0
    dummy.fillna(0,inplace=True)
    # 用全部时间的数据来计算一个loading,然后用这个loading去回归收益率，通过显著月数来判断决定保留哪些技术指标
    all_tech_loading = tech.fb_reg_over_all_time(ret, tech_data)
    significant_days_tech=tech.ret_reg_loading(all_tech_loading,ret,dummy)
    print ("Significant_days_tech:")
    print (significant_days_tech)

    interval=np.arange(192)
    significant_days_mkt = tech.ret_reg_loading(widgets.rm_reg_ri(rm.ix[interval], ret.ix[interval]),ret,dummy)
    print ("Significant_days_market:")    
    print (significant_days_mkt)
    significant_days_macro = tech.ret_reg_loading(macro.macro_reg_ret(ret.ix[interval], macro_data.ix[interval]),ret,dummy)
    print ("Significant_days_macro:")
    print (significant_days_macro)
    # 根据上面注释掉的那段程序的结果，删掉了EMA,trade这两个技术指标,以及最后两个宏观指标
    tech_data.pop('EMA')
    tech_data.pop('trade')
    # 计算loading
    loading=dict()
    for i in range(73):
        interval = range(i,i+119)
        #技术指标（删去了最后一个市值自变量，由于下面要用它来做WLS）
        tech_loading = tech.fb_reg_over_time(ret.ix[interval], tech_data,interval).iloc[:-3,:-1]
        #tech_loading = tech_loading.drop([5],axis=0)
        #标准化
        tech_temp = pd.DataFrame(scale(tech_loading,axis=0))
        
        #市场指标
        mkt_loading = widgets.rm_reg_ri(rm.ix[interval], ret.ix[interval]).iloc[:-3,:]
        #mkt_loading = mkt_loading.drop([5],axis=0)
        #标准化
        mkt_temp = pd.DataFrame(scale(mkt_loading,axis=0))
        
        #宏观指标
        macro_loading = macro.macro_reg_ret(ret.ix[interval], macro_data.ix[interval])
        #macro_loading = macro_loading.drop([5],axis=0)
        #标准化
        macro_temp = pd.DataFrame(scale(macro_loading,axis=0)[:-3,:])
        
        #财务因子指标
        if i%12==0:
            j = i/12-1
        else:
            j = i/12
        if j < 0:
            j = 0
        finance_loading_temp = finance_loading[2000+j].iloc[:-3,:]
        #finance_loading_temp = finance_loading_temp.drop([5],axis=0)
        #标准化
        finance_temp = pd.DataFrame(scale(finance_loading_temp,axis=0))  

        #行业因子指标
        ind_loading = dummy
        #不标准化
        ind_temp = pd.concat([ind_loading.iloc[:-3,:-2],ind_loading.iloc[:-3,-1]],axis=1)
        #ind_temp = ind_temp.drop([5],axis=0)
        ind_temp = pd.DataFrame(ind_temp)
        #拼接数据，得到总的在第i个回合的loading
        current_loading = pd.concat([tech_temp,mkt_temp,macro_temp,finance_temp,ind_temp],axis=1)  
        current_loading = current_loading.drop([5],axis=0)
        loading[i] = current_loading
    print ('Loadings ready\n')
    print ('Now, regression is on ')
    #横截面回归
    #首先将收益率矩阵加以修剪
    del all_tech_loading,code,config,current_loading,ev,dummy,finance_loading,finance_loading_temp,finance_temp,i,j,ind_data,ind_loading,interval,macro_data,macro_loading
    del mkt,mkt_loading,mkt_temp,rm,stkcd,tech_loading,tech_data,tech_temp,ind_temp,ind_path,macro_temp
    newret = ret.iloc[:,:-3]
    newret = newret.drop([5],axis=1)
    residual = dict()
    factor_return = dict()
    Hp = dict()
    total_variance = np.zeros([73,1])
    total_risk = np.zeros([73,1])
    aMCTR = np.zeros([73,107])
    WLS_weight = np.zeros([73,107])
    for i in range(73):
        interval = range(i,i+119)
        #构建回归自变量和因变量
        temp_X = pd.DataFrame(loading[i])
        temp_X = sm.add_constant(temp_X)
        #构建回归结果接收矩阵，因子收益率的和残差的
        temp_residual = np.zeros([120,107])
        temp_factor_return = np.zeros([120,48])
        #输入成分股的权重
        Wchengfen = pd.DataFrame(flow_ev.iloc[i,:].copy())
        Wchengfen = Wchengfen.drop(['000061'],axis=0)
        Wchengfen = pd.DataFrame(Wchengfen.replace(0,1000))
        summ = np.array(Wchengfen)
        temp_Hp = Wchengfen/sum(summ)
        temp_HpT = temp_Hp.transpose()
        Hp[i] = temp_Hp
        for j in range(120):
            row = i + j
            temp_Y = pd.DataFrame(newret.iloc[row,:])
            #WLS
            #将剩下的用于回归，因为前面扔掉了最后三个股票的财务因子，以流通市值开根号为权重，WLS回归
            temp_W = pd.DataFrame((flow_ev.iloc[row,:].copy())**0.5)
            temp_W = temp_W.drop(['000063'],axis=0)
            temp_W = pd.DataFrame(temp_W.replace(0,1000))
            mod_wls = sm.WLS(temp_Y, temp_X, weights = 1./temp_W)
            res_wls = mod_wls.fit()
            residual_here = pd.DataFrame(res_wls.resid)
            temp_residual[j,:] = residual_here.transpose() 
            temp_factor_return[j,:] = res_wls.params
        #记录WLS的回归权重    
        WLS_weight[i,:] = temp_W.transpose() 
        #收录残差和因子收益率进入字典    
        residual[i] = temp_residual
        factor_return[i] = temp_factor_return
        #计算组合总方差
        temp_residual_cov = pd.DataFrame(np.cov(temp_residual.transpose()))
        X = temp_X
        XT = X.transpose()
        #因子收益率方差协方差矩阵
        temp_factor_return_cov = pd.DataFrame(np.cov(temp_factor_return.transpose()))
        F = np.array(temp_factor_return_cov)
        #组合因子暴露及其转置
        Xp = np.dot(XT,temp_Hp)
        XpT = Xp.transpose()
        #残差方差协方差矩阵
        delta = np.array(temp_residual_cov)
        #总方差阵
        V = np.dot(np.dot(X,F),XT) + delta
        #总风险
        total_variance[i] = np.dot(np.dot(XpT,F),Xp) + np.dot(np.dot(temp_HpT,delta),temp_Hp)   
        total_risk[i] = total_variance[i]**0.5
        
        #风险归因
        temp_volatility = pd.DataFrame(total_risk[i])
        temp_volatility = temp_volatility.values
        temp_MCTR = np.array(np.dot(V,Hp[i])/temp_volatility)
        temp_MCTR = pd.DataFrame(temp_MCTR)
        aMCTR[i,:] = temp_MCTR.iloc[:,0]
    #总方差    
    total_variance = pd.DataFrame(total_variance)
    total_variance.to_excel('E:\\QuantProject2\\result_demo\\total_variance.xlsx')
    #年化总风险
    total_risk = pd.DataFrame(12*total_risk)
    total_risk.to_excel('E:\\QuantProject2\\result_demo\\total_risk.xlsx')
    #WLS权重
    WLS_weight = pd.DataFrame(WLS_weight)
    WLS_weight.to_excel('E:\\QuantProject2\\result_demo\\WLS_weight.xlsx')
    #风险贡献向量
    aMCTR = pd.DataFrame(aMCTR)
    aMCTR.to_excel('E:\\QuantProject2\\result_demo\\MCTR.xlsx')
    print ('Mission of model construction completed!!!')
    