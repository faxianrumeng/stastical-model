
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sl
from  Mahalanobisdistance import mah_distance as md

count_data = pd.read_csv(u"各县(市）统计汇总 .csv", encoding = "gb18030");
name = list(count_data["县名"][:])
inter_array = np.array(count_data)
inter_array = np.delete(inter_array, 0, axis = 1)
new_distance = md(inter_array)[:, None]
new_distance = np.sqrt(new_distance)
inter_array = np.zeros([len(inter_array), 2])
inter_array[:, 0] = new_distance[:, 0]
classify_record = []

##聚类分析kNN算法
if __name__ == '__main__':
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters = 3, n_jobs = 5,       #n_clusters 是需要分成的类别的个数，n_jobs 是并行数
                   max_iter = 500)                   #max_iter 最大迭代次数。
    model.fit(new_distance)
    classify_record = np.array(model.predict(new_distance))
    inter_array[:, 1] = classify_record
    inter_array = pd.DataFrame(inter_array, index = name)
    inter_array.to_csv("聚类分析结果.csv")


