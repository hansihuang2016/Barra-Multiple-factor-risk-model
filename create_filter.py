# -*- coding: utf-8 -*-
# 这个程序读入原始的技术指标数据，过滤出收盘价数据在2000年（也就是倒数第192行）非0的列数，用filtered_stocks.txt文件输出，供其他程序使用
import numpy as np
import pandas as pd
import tech
if __name__=='__main__':
    tech_fname = 'E:\\QuantProject\\temp_data\\hushen_tech.xlsx'
    tecnical_index_list = ['close', 'high', 'low', 'trade', 'growth', 'ev']
    data=tech.load_file(tech_fname,tecnical_index_list)
    # 取出收盘价矩阵（月数*股票数）
    close=data['close']
    reserve_row = 192
    template = np.where(close.values[-reserve_row] != 0)[0]
    #删除index为1的，也就是第2只股票，因为这只股票在行业信息中是缺失的，并不知道它属于哪个行业，因此无法计算其行业暴露因子
    template=np.delete(template,1)
    #写入到文件以供matlab使用
    np.savetxt("E:\\QuantProject\\temp_data\\filtered_stocks.txt", template+1, fmt="%d", delimiter=",")
