'''统计每个城市的销售数量和整个净收入总数'''
import pandas as pd
import numpy as np

def urban_counts(filename = u'清洗后的数据'):
    analysis_data = pd.read_csv(filename, encoding="gb18030")
    address_dic = analysis_data["县(市)"].value_counts().index       ###统计自动排序
    address_data = analysis_data["县(市)"][:]
    sales_data = analysis_data["销量"][:]
    net_income = analysis_data["净收入"][:]
    count_data = {}                  #汇总数据
    for name in address_dic:
        count_data[name] = [0, 0]
    for i in range(0, len(address_data)):
        for name in address_dic:
            if name == address_data[i]:
                count_data[name][0] += net_income[i]
                count_data[name][1] += sales_data[i]
                continue
    count_data = pd.DataFrame.from_dict(count_data,
                                        orient = "index")
    count_data.to_csv(u"各县(市)统计汇总.csv")



urban_counts()