# -*- coding: utf-8 -*-
'''
此文件存放功能各异、无法被归类到其他文件中或者不值得为其建立文件的函数
'''
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import scale
import os
#从excel中读入市场数据的函数
def load_market(fname):
    mkt=pd.read_excel(fname)
    # mkt=pd.DataFrame(scale(mkt),columns=['market'])
    return mkt
def load_rf(fname):
    rf=pd.read_excel('E:\\QuantProject2\\raw_data\\TRD_Nrrate.xls')
    rf=rf[rf.Clsdt>='2000-01']
    rf=rf[rf.Clsdt<'2016-01']
    rf.Clsdt=rf.Clsdt.apply(lambda x:x[:7])
    #求平均的，将日度数据转换成月度数据
    rf=pd.DataFrame(rf.groupby('Clsdt')['Nrrmtdt'].sum()/rf.groupby('Clsdt')['Nrrmtdt'].count(),columns=['Nrrmtdt'])
    rf.index=range(rf.shape[0])
    return rf
def rm_reg_ri(rm,ret):
    '''
    用市场收益率来回归每一只股票的收益率，将其回归系数作为
    '''
    #创建负载矩阵
    X=np.array(np.zeros([ret.shape[1],1]))
    #对每一只股票的数据进行回归
    for i in range(ret.shape[1]):
        model=sm.OLS(ret.values[:,i],rm.values).fit()
        X[i,:]=model.params[0]
    X=pd.DataFrame(X)
    return X
def get_selected_Stkcd():
    # 读取选出的111只成分股的股票代码并返回
    #从文件读取txt到流
    stock_index=open('E:\\QuantProject2\\temp_data\\filtered_stocks.txt')
    #按照\n来split
    stock_index=stock_index.read().split('\n')
    #去掉最后一个''
    stock_index=stock_index[:-1]
    #转换成int
    stock_index=map(int,stock_index)
    #读入全部成分股代码
    stkcd=pd.read_excel('E:\\QuantProject2\\temp_data\\hushen_codes.xlsx')
    #选择111个过滤剩下的成分股代码
    stkcd=stkcd.ix[stock_index]
    #把代码名字前的标记c去掉
    stkcd=stkcd.Stkcd.apply(lambda x:x[2:])
    stkcd.index=range(stkcd.shape[0])
    stkcd.to_csv('E:\\QuantProject2\\temp_data\\selected_codes.csv')
    return stkcd

def load_raw_finance(path):
    fnames=os.listdir(path)
    for i in fnames:
        tmp=pd.read_excel(path+'\\'+i)
        tmp.fillna(0,inplace=True)
        stkcd=pd.read_csv('E:\\QuantProject2\\temp_data\\selected_codes.csv')['codes'].values
        tmp['Accper']=tmp['Accper'].apply(lambda x:x[:4])
        tmp=tmp[tmp['Accper']>='2000']
        tmp=tmp[tmp.Accper<'2016']
        tmp=tmp.set_index(['Stkcd','Accper'])
        for file_name in tmp.columns:
            print 'file:',i,'col:',file_name
            group_mean=(tmp[file_name].groupby(level=['Stkcd','Accper']).sum())/(tmp[file_name].groupby(level=['Stkcd','Accper']).count())
            group_mean.unstack().ix[stkcd].fillna(0).transpose().to_excel('E:\\financial_data\\'+file_name+'.xlsx')
def load_finance_data(path):
    fnames=os.listdir(path)
    for i in range(len(fnames)):
        fnames[i]=path+'\\'+fnames[i]
    finance_data=[]
    for i in fnames:
        print 'loading',i
        tmp=pd.read_excel(i)
        tmp=tmp.set_index('Accper').transpose()
        finance_data+=[tmp]
    time=range(2000,2016)
    data=dict()
    for i in time:
        tmp=pd.DataFrame()
        for j in finance_data:
            tmp=pd.concat([tmp,j[i]],axis=1)
        data[i]=pd.DataFrame(tmp.values)
    return data

if __name__ == '__main__':
    print 'Fine'