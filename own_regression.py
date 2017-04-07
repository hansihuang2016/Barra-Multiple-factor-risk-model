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
    del stkcd
    significant_days_macro = tech.ret_reg_loading(macro.macro_reg_ret(ret.ix[interval], macro_data.ix[interval]),ret,dummy)
    print ("Significant_days_macro:")
    print (significant_days_macro)
    # 根据上面注释掉的那段程序的结果，删掉了EMA,trade这两个技术指标,以及最后两个宏观指标
    tech_data.pop('EMA')
    tech_data.pop('trade')
    del config,close,code,all_tech_loading,default_txt_fmt,ind_ret
    # 生成用于计算loading的时间数据
    time = np.zeros([11,60])
    for i in range(time.shape[0]):
        time[i]=np.arange(60)+12*i
    # 对每个五年计算loading   并用每五年的loading去回归接下来一年每个月的收益率
    loading=dict()
    for i in range(time.shape[0]):
        print ('Calculating loading for Year',i+2004)
        interval=time[i]
        
        #技术指标（删去了最后一个市值自变量，由于下面要用它来做WLS）
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
        finance_loading_temp = finance_loading[2004+i].iloc[:-3,:]
        #标准化
        finance_temp = pd.DataFrame(scale(finance_loading_temp,axis=0))  

        #行业因子指标
        ind_loading = dummy
        #不标准化
        ind_temp = pd.concat([ind_loading.iloc[:-3,:-2],ind_loading.iloc[:-3,-1]],axis=1)
        #拼接数据，得到总的loading
        current_loading=pd.concat([tech_temp,mkt_temp,macro_temp,finance_temp,ind_temp],axis=1)        # 初始化，并且根据考察这两个生成的数据里面有大量0，所以扔了
        loading[2004+i]=current_loading
        loading[2004+i].to_excel('E:\\QuantProject2\\result_demo\\loading'+str(2004+i)+'.xlsx')
    # 对下一年每个月的收益率做回归
    # 总共132个月，每个月要统计108个股票的残差,显著的因子的数量，几十个model param
    residue = np.zeros([132,108])
    factor_return = np.zeros([132,loading[2014].shape[1]])
    #统计132个回归中有多少个存在异方差——LM检验
    Y = pd.DataFrame(np.zeros([108,132]))
    count = 0
    for i in range(time.shape[0]):#从2004到2014中第i年的数据
        for j in range(12):#第j个月的数据
            row = 60+i*12+j 
            Y.iloc[:,count] = ret.iloc[row,:-3]
            count+=1
    total_loading = total_regression.Get_total_loading(ret,tech_data,rm,macro_data,finance_loading[2015],dummy)
    total_loading = pd.DataFrame(total_loading)
    X = total_loading.iloc[:,0]
    X = sm.add_constant(X)
    count = 0
    number_of_het = 0
    het_pvalue = np.zeros([132,1])
    for i in range(132):
        temp_Y = Y.iloc[:,i]
        temp_X = X.iloc[:,(0,1)]
        model = sm.OLS(temp_Y,temp_X).fit()
        temp_resid = model.resid
        test_temp = ss.het_white(temp_resid,temp_X)
        het_pvalue[count,:] = test_temp[1]
        if test_temp[1]>0.1:
            number_of_het+=1
        count+=1 
    print ("Number of heteroskedasticity is: ",number_of_het)
    del het_pvalue,temp_X,temp_Y,model,temp_resid,count
    del finance_temp,finance_loading_temp,interval,macro_data,mkt,mkt_temp    
    #发现异方差问题较为严重，故下面所有回归采用加权最小二乘进行
    # 记录当前所在的是第多少个月
    count = 0
    WLS_Weight = dict()
    for i in range(time.shape[0]):#从2004到2014中第i年的数据
        X = pd.DataFrame(loading[2004+i])
        Xtemp = sm.add_constant(X.iloc[:,0])
        print ('Regression on Year',2000+i)
        for j in range(12):#第j个月的数据
            # 取第2004+i年截止的loading值为自变量，2004+i+1年的第j个月的股票收益率为因变量进行回归
            # 计算这个月在ret中对应的行数
            row = 60+i*12+j #会得到从60到192的row
            Y = ret.iloc[row,:-3]
            
#            #OLS
#            model = sm.OLS(Y,Xtemp).fit()
#            #得到残差
#            temp_resid = model.resid
#            factor_return[count,:]=model.params
#            residue[count,:]=model.resid
#            count+=1
            
            #WLS
            # 将剩下的用于回归，因为前面扔掉了最后三个股票的财务因子，以流通市值开根号为权重，WLS回归
            W = pd.DataFrame((flow_ev.iloc[row,:].copy())**0.5)
            W = pd.DataFrame(W.replace(0,1000))
            mod_wls = sm.WLS(Y, X, weights = 1./W)
            res_wls = mod_wls.fit()
            # 得到残差
            residue[count,:] = res_wls.resid
            factor_return[count,:]=res_wls.params  
            WLS_Weight[i] = W
            count+=1
    del X,Y,trash,W,row,i,j,time
    #减掉第6个股票    
    residue = pd.DataFrame(residue)
    residual = residue.drop([5],axis=1)
    #输出几个基础表格
    pd.DataFrame(residual).to_excel('E:\\QuantProject2\\result_demo\\residual.xlsx')
    #pd.DataFrame(significant).to_excel('E:\\QuantProject2\\result_demo\\significant.xlsx')
    pd.DataFrame(factor_return).to_excel('E:\\QuantProject2\\result_demo\\factor_return.xlsx')
    #输出本身收益率估计出的方差协方差矩阵    
    return_corr = pd.DataFrame(np.corrcoef((ret.iloc[:,:-2]).transpose()))
    #输出因子收益率的相关系数矩阵
    factor_return_corr = pd.DataFrame(np.corrcoef(factor_return.transpose()))
    factor_return_corr.to_excel('E:\\QuantProject2\\result_demo\\factor_return_correlation_matrix.xlsx')
    #输出因子收益率的方差协方差矩阵
    factor_return_cov = pd.DataFrame(np.cov(factor_return.transpose()))
    factor_return_cov.to_excel('E:\\QuantProject2\\result_demo\\factor_return_cov_matrix.xlsx')
    
    #输出特异收益率的相关系数矩阵
    residual_corr=pd.DataFrame(np.corrcoef(residual.transpose()))
    residual_corr.to_excel('E:\\QuantProject2\\result_demo\\residual_correlation_matrix.xlsx')
    #输出特异收益率的方差协方差矩阵
    residual_cov = pd.DataFrame(np.cov(residual.transpose()))
    residual_cov.to_excel('E:\\QuantProject2\\result_demo\\residual_cov_matrix.xlsx')
    
    #统计残差的相关系数矩阵中绝对值大于0.2的有多少占比
    small_count = 0.0
    for i in range(residual_corr.shape[0]):
        for j in range(residual_corr.shape[1]):
            if abs(residual_corr.iloc[i,j])>0.2:
                small_count+=1.0
    print ('Absolute values in corr larger than 0.2: ',small_count/(107**2))
    del tech_data,rm,small_count,i,j,test_temp,tech_loading,rf,ind_data,count,flow_ev,ind_path,Xtemp
    
    #下面计算总风险
    #总因子暴露为X及其转置 
    X = pd.DataFrame(np.array(total_loading))
    X = X.drop([5],axis=0)
    XT = X.transpose()
    #输入成分股的权重
    Hp = np.ones([107,1])/107
    HpT = Hp.transpose()
    #因子收益率方差协方差矩阵
    F = np.array(factor_return_cov)
    #组合因子暴露及其转置
    Xp = np.dot(XT,Hp)
    XpT = Xp.transpose()
    #残差方差协方差矩阵
    delta = np.array(residual_cov)
    #总风险
    total_variance = np.dot(np.dot(XpT,F),Xp) + np.dot(np.dot(HpT,delta),Hp)
    print ('Total variance of this portfolio is: ',total_variance[0,0])
        
    print ('Mission of model construction completed!!!')
    