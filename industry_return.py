# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import statsmodels.api as sm
import widgets
def load_industry_data(fname_list):
    '''
    因为万德写入xlsx的数据的大小限制，数据分散在各个文件中，这里把数据纵向拼接起来并选取所需的192个月
    :param fname_list:
    :return:
    '''
    # 将所有表示交易数据的财务数据表格读入
    data = pd.DataFrame()
    for i in fname_list:
        print 'loading', i
        temp = pd.read_excel(i)
        # 剪掉前两行中文
        temp = temp.iloc[2:, :]
        # 从2000年第一个月开始截取
        temp = temp[temp.Trdmnt >= '2000-01']
        temp = temp[temp.Trdmnt < '2016-01']
        # 拼接到data
        data = pd.concat([data, temp], axis=0)
        # 计算每只股票的市值*收益率以便稍后处理
        data['total_Mr']=data.Msmvttl*data.Mretwd
        # 把月个股流通市值提取出来，用以替代tech里面的ev
        stkcd=widgets.get_selected_Stkcd()
        ev=data[['Stkcd','Trdmnt','Msmvosd']]
        ev=ev.set_index('Stkcd',drop=False)
        ev=ev.ix[stkcd.values]
        ev=ev.set_index(['Trdmnt'],append=True)
        ev=ev.unstack()
        ev=ev['Msmvosd']
        ev=ev.transpose()
        ev.fillna(0,inplace=True)
        # 将ev标准化
        ev=pd.DataFrame(scale(ev))
    # 规整index
    data.index = range(data.shape[0])
    # 将股票代码和时间作为Multi_index
    # data=data._index(['Trdmnt','Stkcd'])
    return [data,ev]

def get_ind_return(data):
    '''
    将从xlsx中读取出来按列拼接好的数据进行重组，计算出每个行业每个月的收益率
    :param [DataFrame] data: 从xlsx文件中读取的月份-交易数据
    :return: [DataFrame] ind_ret: 月份*行业 每个行业每个月的收益率
    '''
    # 读入stk_ind_pair.xlsx，用作股票和其所属行业的对照表
    stk_ind = pd.read_excel('E:\\QuantProject2\\temp_data\\stk_ind_pair.xlsx')
    # 把stk_ind里面股票代码数字部分后面的字母去掉
    stk_ind.Stkcd = stk_ind.Stkcd.apply(lambda x: x[:6])
    # 对stk_ind和data进行merge操作，将行业信息插入data
    data = pd.merge(data, stk_ind, on='Stkcd')
    # 按照月份和行业分组
    groups = data.groupby(['Trdmnt', 'ind'])
    # 分组计算每个月每个行业的总市值
    total_Ms = groups['Msmvttl'].sum()
    # 分组计算每个月每个行业按照市值加权的收益率
    total_Mr=groups['total_Mr'].sum()
    # 相除得到每个月每个行业的平均收益率
    ind_ret=total_Mr/total_Ms
    # 将ind_ret的内层level转换为列
    ind_ret=ind_ret.unstack()
    #将ind_ret标准化
    ind_ret=pd.DataFrame(scale(ind_ret),columns=ind_ret.columns)
    return ind_ret

def ind_reg_ret(ind_ret,ret):
    '''
    用行业收益率去回归股票收益率，得到股票在行业因子上的暴露。我们这里只算个暴露，就不关心鲜猪肚了
    :param DataFrame ind_ret: 行业收益率（月份*行业数）
    :param DataFrame ret: 股票收益率（月份*股票数）
    :return:
    '''
    #显著的pvalue阈值定在0.1
    pvalue_threshold=0.1
    ind_loading=np.zeros([ret.shape[1],ind_ret.shape[1]])
    # 依次取出每一只股票的收益率，与行业收益率做多元回归
    for i in range(ret.shape[1]):
        # 取出一个行业来与股票做回归
        for j in range(ind_ret.shape[1]):
            model=sm.OLS(ret.values[:,i],ind_ret.values[:,j]).fit()
            #将回归系数写入暴露矩阵
            ind_loading[i,j]=model.params[0]
        #对于pvalue小于阈值（显著）的因子，将其对应的significant_stocks_list加1
    ind_loading=pd.DataFrame(ind_loading,columns=ind_ret.columns)
    return ind_loading
if __name__ == '__main__':
    # 产生文件名
    path = 'E:\\QuantProject2\\raw_data\\'
    fname_lst = ['E:\\QuantProject2\\raw_data\\TRD_Mnth.xls']
    for i in range(6):
        fname_lst = fname_lst + ['E:\\QuantProject2\\raw_data\\TRD_Mnth' + str(i + 1) + '.xls']
    # 将所有交易数据文件按列拼好
    data = load_industry_data(fname_lst)
    ind_ret=get_ind_return(data)
    ind_ret.to_excel('E:\\QuantProject2\\temp_data\\industry_return.xlsx')