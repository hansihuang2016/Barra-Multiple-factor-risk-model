# -*- coding: utf-8 -*-
import pandas as pd


def load_finance_data():
    # 所有文件名共有的前缀
    prefix = 'E:\\QuantProject\\raw_data\\finance\\'
    # 文件名字典
    file_struct = {'Cash_flow': ['FI_T6', 'FI_T61', 'FI_T62', 'FI_T63'],
                   'Dividend': ['CD_Dividend', 'CD_Dividend1'],
                   'EPS': ['FI_T9', 'FI_T91', 'FI_T92', 'FI_T93'],
                   'Operating': ['FI_T4', 'FI_T41', 'FI_T42', 'FI_T43'],
                   'Pay_debt': ['FI_T1', 'FI_T11', 'FI_T12', 'FI_T13'],
                   'Profit': ['FI_T5', 'FI_T51', 'FI_T52', 'FI_T53'],
                   'Risk_free': ['TRD_Nrrate'],
                   'Risk_level': ['FI_T7', 'FI_T71', 'FI_T72', 'FI_T73'],
                   'Trade_data': ['TRD_Mnth', 'TRD_Mnth1', 'TRD_Mnth2', 'TRD_Mnth3', 'TRD_Mnth4', 'TRD_Mnth5',
                                  'TRD_Mnth6']}
    # 所有文件名共有的后缀
    suffix = '.xls'
    finance_data = dict()
    # 对每个文件夹循环
    for i in file_struct.items():
        print 'concatenating folder', i[0]
        temp = pd.DataFrame()
        # 对文件夹里的每个文件循环
        for j in i[1]:
            fname = prefix + i[0] + '\\' + j + suffix
            temp = pd.concat([temp, pd.read_excel(fname).iloc[2:, :]], axis=0)
            temp.fillna(0, inplace=True)
        finance_data[i[0]] = temp
    print 'done'
    return finance_data


def clean_finance(finance_data):
    '''
    清洗finance_data中的每一个矩阵，整理成32行（即每半年采集一次数据），然后横向合成一个矩阵
    :param finance_data:
    :return:
    '''
    # 目前先用一个矩阵Risk_level来试一试
    f=dict()
    for tb in finance_data.items():
        if tb[0]=='Trade_data' or tb[0]=='Risk_free':
            continue
        print 'filling ', tb[0]
        a = tb[1]
        # 首先去掉重复的行
        a = a.drop_duplicates(subset=['Stkcd', 'Accper'])
        # 产生半年时间序列
        hfyr = pd.date_range('6/1/2000', '1/1/2016', freq='6M')
        # 将半年时间序列转换成字符串
        hfyr = pd.Series(hfyr.values).apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')[:10])
        a = a.set_index('Accper',drop=False)
        a=a.ix[hfyr]
        a=a.set_index(['Stkcd','Accper'])
        for col in a.columns:
            a[col]=a[col].unstack().fillna(method='pad',axis=1)
            f[col]=a[col].fillna(0)


def output_finance_data(data):
    for i in data.items():
        print 'writing', i[0]
        pd.DataFrame.to_excel(i[1], 'E:\\QuantProject\\temp_data\\finance\\' + i[0] + '.xlsx')


if __name__ == '__main__':
    # finance_data = load_finance_data()
    # 用人工操作过的数据来做finance_data
    finance_data=pd.read_excel()
    output_finance_data(finance_data)


    stkcd=pd.read_csv('E:\\QuantProject\\temp_data\\selected_codes.csv')
    data=pd.read_excel('E:\\QuantProject\\temp_data\\finance_reduced\\Risk_level.xlsx')
    a=data.drop_duplicates(subset=['Stkcd','Accper'])
    a=a[['Stkcd','Accper','F070201B']]
    a['Accper']=a['Accper'].apply(lambda x:x[:4])
    a=a[a['Accper']>='2000']
    a=a[a['Accper']<'2016']
    a.fillna(0,inplace=True)
    grouped=a.groupby(['Stkcd','Accper'])
    a=grouped.sum()/grouped.count()
    a=a.unstack()
    a=a.ix[stkcd['stkcd']]
    a.fillna(0,inplace=True)
    a=a.transpose()
    a.to_excel('E:\\F070201B.xlsx')