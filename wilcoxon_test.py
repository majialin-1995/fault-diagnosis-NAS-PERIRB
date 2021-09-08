import pandas as pd 
import scipy.stats
import numpy as np
np.set_printoptions(suppress=True)

for i in range(1,9,1):
    df = pd.read_csv("plot{}.csv".format(i))

    t1 = scipy.stats.wilcoxon(df.data0.values[-20:], df.data3.values[-20:], zero_method='wilcox', correction=False)
    t2 = scipy.stats.wilcoxon(df.data1.values[-20:], df.data3.values[-20:], zero_method='wilcox', correction=False)
    t3 = scipy.stats.wilcoxon(df.data2.values[-20:], df.data3.values[-20:], zero_method='wilcox', correction=False)
    t4 = scipy.stats.wilcoxon(df.data0.values[-20:], df.data1.values[-20:], zero_method='wilcox', correction=False)
    t5 = scipy.stats.wilcoxon(df.data0.values[-20:], df.data2.values[-20:], zero_method='wilcox', correction=False)

    print(np.array([t1.pvalue, t2.pvalue, t3.pvalue, t4.pvalue, t5.pvalue]))