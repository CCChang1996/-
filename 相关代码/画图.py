# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:26:06 2019

@author: 13486
"""

from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('1.csv', header=0)
series.astype(float)
series.plot()
ax = pyplot.gca()
ax.set_xlabel(u'时间/天', fontproperties='SimHei',fontsize=14)
ax.set_ylabel(u'金额/元', fontproperties='SimHei',fontsize=14)
pyplot.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
pyplot.rcParams['ytick.direction'] = 'in'
pyplot.show()


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter  

df = pd.read_csv('2.csv',  parse_dates=['dtime'])

plt.plot_date(df.dtime, df.speed, fmt='-')

ax = plt.gca()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))  #设置时间显示格式
ax.xaxis.set_major_locator(AutoDateLocator(maxticks=24))       #设置时间间隔  

plt.xticks(rotation=90, ha='center')
#label = ['speedpoint']
#plt.legend(label, loc='upper right')

#plt.grid()

#ax.set_title(u'传输速度', fontproperties='SimHei',fontsize=14)  
ax.set_xlabel(u'时间/天', fontproperties='SimHei',fontsize=14)
ax.set_ylabel(u'金额/元', fontproperties='SimHei',fontsize=14)
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'
plt.show()