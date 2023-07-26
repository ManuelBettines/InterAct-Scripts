# -*- coding: utf-8 -*-
"""
Created on Thu Jun 8 09:07:29 2023

@author: bettines
"""
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import datetime
from pandas.tseries.offsets import DateOffset
from metpy.plots import add_metpy_logo, add_timestamp

#%% extract measurements
df_nais = pd.read_csv('nais_neutral_mean.csv',sep=",")
df_nais['datetime'] = pd.to_datetime(df_nais['date.utc'])
df_nais['datetime'] = df_nais['datetime'] + DateOffset(hours=6)
df_nais = df_nais.set_index('datetime')
df_nais = df_nais.drop(['date.utc'], axis = 1)

psp_i = '2014-12-01 06:00:00'
psp_f = '2014-12-25 13:00:00'
df_nais = df_nais.loc[(df_nais.index >= psp_i)& (df_nais.index < psp_f)]

# import diameters
df_nais_diam = pd.read_csv('nais_neutral_diameters_nanometer.csv',sep=",", header=None)
nais_diam_centers = (np.array(df_nais_diam) *1e-9).flatten()
log_nais_diam_centers = np.log10(np.array(nais_diam_centers)).flatten()

# select range inside 3nm-300nm
idx_3nm_nais = np.where((nais_diam_centers > 2.95e-9) & (nais_diam_centers < 50e-9))
nais_diam_centers_3nm = nais_diam_centers[idx_3nm_nais]
df_nais_3nm = df_nais#.iloc[:,idx_3nm_nais[0][0]:idx_3nm_nais[0][-1]+1]

nais_h_D = df_nais_3nm#.groupby(df_nais_3nm.index.hour).median()

# compute minmax of the bins in logspace
log_nais_diam_minmax = np.zeros((1,30)).flatten()
for i in range(len(log_nais_diam_centers)+1):
    # print(i)
    if (i==0):
        log_midpoint = (log_nais_diam_centers[i+1]+log_nais_diam_centers[i])/2
        log_diff = log_midpoint - log_nais_diam_centers[i]
        log_nais_diam_minmax[i] = log_nais_diam_centers[i] - log_diff
    elif i==29:
        log_midpoint = (log_nais_diam_centers[i-1]+log_nais_diam_centers[i-2])/2
        log_diff = log_nais_diam_centers[i-1] - log_midpoint
        log_nais_diam_minmax[i] = log_nais_diam_centers[i-1] + log_diff
    else:
        log_midpoint = (log_nais_diam_centers[i-1]+log_nais_diam_centers[i])/2
        log_nais_diam_minmax[i] = log_midpoint

nais_diam_minmax = np.power(10,log_nais_diam_minmax)
idx_3nm_minmax_nais = np.where((nais_diam_minmax > 2.8e-9) & (nais_diam_minmax < 4e-7))
nais_diam_minmax_3nm = nais_diam_minmax[idx_3nm_minmax_nais]

# compute dlogDp
nais_dlogDp = np.zeros((29))
for i in range(len(df_nais_diam)):
        nais_dlogDp[i] = np.log10(nais_diam_minmax[i+1]) - np.log10(nais_diam_minmax[i])
# compute dN/dlog(Dp)
nais_dNdlogDp_mean = np.array(nais_h_D) / np.array(nais_dlogDp)
# cut NAIS below 3 nm
#nais_dNdlogDp_mean_3nm = nais_dNdlogDp_mean[idx_3nm_nais]

mmd = []
for i in range(len(nais_diam_centers)):
        mmd.append(int(nais_diam_centers[i]*10e8))

time = np.arange(1,25,1)
sub = nais_dNdlogDp_mean.T

fig = plt.figure(figsize=(31,9))
ax = fig.add_subplot()
#levels = np.arange(1,15000, 1)
#levels = np.geomspace(1,15000,num=10000)
levels = np.logspace(np.log10(100), np.log10(20000), num=500)
#levels = [1,10,20,50,70,100,200,500,700,1000,2000,5000,7000,10000,20000]
a = ax.contourf(df_nais_3nm.index,nais_diam_centers,sub,levels=levels,locator=ticker.LogLocator(),cmap='YlGnBu', extend='both')
ax.yaxis.set_ticks(nais_diam_centers, mmd)
#ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
plt.yscale("log")
ax.set_ylabel('Diameter (m)', fontsize=18)
#ax.set_xlabel('Datetime [Local Time]', fontsize=18)
#ax.xaxis.set_ticks(df_nais_3nm.index, time)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
fig.autofmt_xdate(rotation=45)
ax.set_ylim([3e-9,3.8e-8])
cbar = fig.colorbar(a, extend='both')
cbar.set_label('dN/d(logD$_p$) (cm$^-3$)',size=18)
cbar.ax.tick_params(labelsize=15)
#cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
#cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
tick_locations = [100, 1000,10000]
tick_labels = ['$10^{{{}}}$'.format(int(np.log10(t))) for t in tick_locations]
cbar.ax.set_yticks(tick_locations)
cbar.ax.set_yticklabels(tick_labels)
ax.set_title('NCO-P - Observation', fontsize=21)
plt.show()
fig.savefig('output_figures/Banana_PYR_NAIS.png', dpi=500)

