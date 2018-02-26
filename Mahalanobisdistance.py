'''马氏距离计算方法。马氏距离可以在多属性情况下构建度量
   距离，消除了属性之间的量纲性（建议大于1以上的属性个数，
   都使用）'''
import numpy as np

def mah_distance(narray):
    '''narray是numpy.array数组对象'''
    var_narray = narray
    segma = []
    distance=np.zeros(len(var_narray))
    cov = np.mat(np.cov(narray.reshape(len(narray[0,:]),
                                       len(narray))))
    cov =np.linalg.pinv(cov.astype(float))            ###求广义逆矩阵'''
    for i in range(0, len(narray[0, :])):
         segma.append(np.var(narray[:, i]))
         var_narray[:,i]=var_narray[:,i]-np.sqrt(segma[i])
    for i in range(0,len(var_narray)):
       distance[i] = np.mat(var_narray[i,:]) * cov * np.mat(var_narray[i,:]).T
    print(distance)
    return  distance