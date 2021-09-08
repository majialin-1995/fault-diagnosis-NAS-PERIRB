import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np

window = 50
def plot_train_curve():
    df = pd.read_csv("plot1.csv")
    ymin = []
    ymax = []
    x_mean = df.data0.rolling(window).mean().values
    x_std = df.data0.rolling(window).std().values
    plt.plot(df.index0.values, x_mean, linewidth=3, label="NASNet with PE Reward and IRB")
    ymin.append(np.nanmin(x_mean - 0.5 * x_std))
    ymax.append(np.nanmax(x_mean + 0.5 * x_std))
    # facecolor=COLORS[name],
    plt.fill_between(df.index0.values, x_mean - x_std, x_mean + x_std, edgecolor='none', alpha=0.1)

    ymin = []
    ymax = []
    x_mean = df.data1.rolling(window).mean().values
    x_std = df.data1.rolling(window).std().values
    plt.plot(df.index1.values, x_mean, linewidth=3, label="NASNet with PE Reward ")
    ymin.append(np.nanmin(x_mean - 0.5 * x_std))
    ymax.append(np.nanmax(x_mean + 0.5 * x_std))
    # facecolor=COLORS[name],
    plt.fill_between(df.index1.values, x_mean - x_std, x_mean + x_std, edgecolor='none', alpha=0.1)

    ymin = []
    ymax = []
    x_mean = df.data2.rolling(window).mean().values
    x_std = df.data2.rolling(window).std().values
    plt.plot(df.index2.values, x_mean, linewidth=3, label="NASNet with IRB")
    ymin.append(np.nanmin(x_mean - 0.5 * x_std))
    ymax.append(np.nanmax(x_mean + 0.5 * x_std))
    # facecolor=COLORS[name],
    plt.fill_between(df.index2.values, x_mean - x_std, x_mean + x_std, edgecolor='none', alpha=0.1)

    ymin = []
    ymax = []
    x_mean = df.data3.rolling(window).mean().values
    x_std = df.data3.rolling(window).std().values
    plt.plot(df.index3.values, x_mean, linewidth=3, label="NASNet")
    ymin.append(np.nanmin(x_mean - 0.5 * x_std))
    ymax.append(np.nanmax(x_mean + 0.5 * x_std))
    # facecolor=COLORS[name],
    plt.fill_between(df.index3.values, x_mean - x_std, x_mean + x_std, edgecolor='none', alpha=0.1)


plot_train_curve()
# plt.xlim([0, 480])

plt.tick_params(labelsize=23)
plt.xlabel('Explore step', fontsize=24)
plt.ylabel('Final Accuracy', fontsize=24)
plt.legend(loc='lower right',fontsize=16)

plt.show()