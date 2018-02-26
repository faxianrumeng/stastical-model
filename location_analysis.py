'''删除空值处理'''
import numpy as np
import pandas as pd

###获得一个精简的csv文件，以便删除缺失值'''
def series_to_DataFrame(key_word, csvfile):              #key_word 是需要检查的属性对应的关键字
     temp_dic = {}
     for i in key_word:
         temp_dic[i] = csvfile[i][:]
     df = pd.DataFrame(temp_dic, index=csvfile.index, columns=key_word)
     return df

data = pd.read_csv(u"整合的数据.csv", encoding = "gb18030")
print(data.isnull(). any())                                  ###检查是否有空白值

'''删除关键列的缺失值位置'''
key_columns = ['经营部', '客户级别', '省(直辖市)', '地(市)', '县(市)', '档次',
             '门框', '销量', '销售成本', '销售收入', '调拨费用']
analysis_data = series_to_DataFrame(key_columns,data).dropna()

###保存文件
analysis_data.to_csv(u"E://pyfile//locationprediction//清洗后的数据.csv", columns=analysis_data.columns)
