import numpy as np
import pandas as pd
import pymc as pm
import os
import matplotlib.pyplot as plt
from pylab import  *
from scipy.stats.mstats import mquantiles
from matplotlib.font_manager import FontManager as FM
from matplotlib.font_manager import FontProperties as FP
myfont = FP(fname='D://Lib//site-packages//matplotlib//'
                            'mpl-data//fonts//ttf//vera.ttf')
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 逻辑斯蒂函数
def logistic(x, beta, alpha = 0):
    return 1.0 / (1.0+np.exp((beta * x) + alpha))

# 散点图
def scatter_figure(data,index):

    plt.scatter(data[:,0], data[:,2], color='black', alpha=0.5)
    plt.yticks([0, 1])
    plt.ylabel(u"销售点", fontproperties=myfont)
    plt.xlabel("属性距离", fontproperties=myfont)
    plt.title(u'销售点与每个县城相关属性距离散点图',
                                fontproperties=myfont)
    plt.xticks(data[:,0], index, size='small', fontproperties=myfont, rotation=90)
    plt.legend()
    plt.show()

# 参数beta,alpha的后验分布
def show_figure(beta_sample, alpha_sample):

    plt.figure(1)
    plt.title(u'模型参数的后验分布', fontproperties=myfont)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    plt.hist(beta_sample, histtype='stepfilled', color="r", normed=True)

    plt.xlabel(" beta")
    plt.ylabel(u"密度", fontproperties=myfont)
    plt.sca(ax1)
    plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
             label=r"posterior of $\slpha$", color='b')
    plt.xlabel("alpha")
    plt.ylabel(u"密度", fontproperties=myfont)
    plt.sca(ax2)
    plt.legend()
    plt.show()

address = u"E:/pyfile/locationprediction/聚类分析结果.csv"
csvfile = pd.read_csv(os.path.basename(address),encoding = "gb18030")          #读取数据

'''建立贝叶斯统计模型'''
data = np.array(csvfile)[:, 1:]
new_data = data[:, 0]
distance = np.array(new_data)
distance_plot = pd.Series(distance, index = list(csvfile['名称'][:]))
beta = pm.Normal("beta", 0.0, 0.001, value = 0)
alpha = pm.Normal("alpha", 0.0, 0.001, value = 0)

@pm.deterministic()
def p_value(x = distance, beta = beta, alpha = alpha):
    return 1.0 / (1.0 + np.exp(np.array(np. dot(beta, x) + alpha,
                                                 dtype = float)))

observed = pm.Bernoulli("bernouli_obs", p_value, value = data[:,2],
                                                 observed = True)
model = pm.Model([observed, beta, alpha])
map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
mcmc.sample(360000, 100000, 2)
alpha_samples = mcmc.trace('alpha')[:, None]
beta_samples = mcmc.trace('beta')[:, None]
show_figure(beta_samples,alpha_samples)
distance_ = np.linspace(min(distance) - 0.5, max(distance) + 0.5,
                                                 len(distance))
p_t = logistic(distance_, beta_samples, alpha_samples)
mean_prob_t = np.array(p_t).mean(axis = 0)
plt.plot(distance_, mean_prob_t,lw = 3,
label = u"发生缺陷概率的后验期望(平均)", )
plt.plot(distance_, p_t[0,:], ls = "--", label = u"来自后验的实现",)
plt.plot(distance_, p_t[-2,:],ls ="--", label = u"来自后验的实现")
plt.scatter(distance, data[:, 2], color = 'k', s=len(data[:, 2]),
                                                            alpha = 0.5)
plt.xlabel(u"距离", fontproperties=myfont)
plt.ylabel(u"概率", fontproperties=myfont)
plt.legend(loc = "low left")
plt.ylim(-0.1, 1.1)
plt.xlim(distance_.min(), distance_.max())
plt.show()

qs = mquantiles(p_t,[0.025, 0.975], axis = 0)
plt.fill_between(distance_, *qs, alpha = 0.7, color = '#7A68A6')
plt.plot(distance_, mean_prob_t, lw = 1, ls = "--",color = "k")
plt.scatter(distance_plot, data[:,2], color = "k", s=len(data[:,1]),alpha = 0.5)
plt.xlabel(u"距离", fontproperties = myfont)
plt.ylabel(u"概率", fontproperties = myfont)
plt.xticks(distance_plot, distance_plot.index, size = 'small',fontproperties=myfont, rotation = 90)
plt.xlim(min(distance_), max(distance_))
plt.ylim(-0.02, 1.02)
plt.show()

distance = distance.reshape(len(distance))
prob_ = []
for i in range(0,len(distance)):
    prob_.append(logistic(distance[i], beta_samples, alpha_samples).mean(axis=0))
prob_ = pd.DataFrame(prob_)
prob_.to_csv(u"平均概率. csv")   #####保存每个区域的建立销售点的平均概率数据