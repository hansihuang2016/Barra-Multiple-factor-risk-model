# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import scale
def load_macro_data(fname):
    '''
    从xlsx文档中读取原始数据并存于一个DataFrame
    :param string fname: 文件路径
    :param [string]factor_list: 宏观因子的名字
    :return: DataFrame: 返回宏观指标和市场指标
    '''
    macro_data= pd.read_excel(fname)
    macro_data=pd.DataFrame(scale(macro_data),columns=macro_data.columns)
    return macro_data


def macro_reg_ret(ret, macro_data):
    '''
    抽取每一只股票的收益率时间序列，与每个宏观因子在这段时间的收益率做单元回归，回归系数就是这只股票在
    这个宏观因子上的暴露。将所得结果排成一个大小为股票数*宏观因子数的矩阵，横向拼入股票-因子暴露矩阵
    :param DataFrame ret: 每只股票每天的收益率
    :param DataFrame macro_index: 每天每个宏观因子的值
    :return: [DataFrame,dict] [loading,significant_list]: 每只股票在每个宏观因子上的暴露;每个因子显著的股票数
    '''
    macro_loading = pd.DataFrame(np.zeros([ret.shape[1], macro_data.shape[1]]))
    # 选取一只股票用于回归
    for i in range(ret.shape[1]):
        y = ret.values[:, i]
        # 用每一个因子去回归股票收益率
        for j in range(macro_data.shape[1]):
            x = macro_data.values[:, j]
            model = sm.OLS(y, x).fit()
            macro_loading.iloc[i, j] = model.params[0]
    return macro_loading
if __name__ == '__main__':
    fname='E:\\QuantProject\\raw_data\\macro.xlsx'
    index_list=['Ind_growth','CPI','Ex_inport','Deficit','M2','Conf','USD']
    macro_index=load_macro_data(fname,index_list)
    print np.std(macro_index)