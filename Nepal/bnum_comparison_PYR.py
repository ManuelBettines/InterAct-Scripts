#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:59:49 2022

@author: bvitali
"""

import pandas as pd
from pandas.tseries.offsets import DateOffset
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import matplotlib.dates as mdates 
from matplotlib import cm
from metpy.calc import wind_direction
from metpy.units import units
from matplotlib.colors import LinearSegmentedColormap

#%% extract measurements
df_nais = pd.read_csv('nais_neutral_mean.csv',sep=",")
df_nais['datetime'] = pd.to_datetime(df_nais['date.utc'])
df_nais['datetime'] = df_nais['datetime'] + DateOffset(hours=6)
df_nais = df_nais.set_index('datetime')
df_nais = df_nais.drop(['date.utc'], axis = 1)

psp_i = '2014-12-03 00:00:00'
psp_f = '2014-12-25 23:00:00'
df_nais = df_nais.loc[(df_nais.index >= psp_i)& (df_nais.index < psp_f)]

# import diameters
df_nais_diam = pd.read_csv('nais_neutral_diameters_nanometer.csv',sep=",", header=None)
nais_diam_centers = (np.array(df_nais_diam) *1e-9).flatten()
log_nais_diam_centers = np.log10(np.array(nais_diam_centers)).flatten()

# select range inside 3nm-300nm
idx_3nm_nais = np.where((nais_diam_centers > 2.95e-9) & (nais_diam_centers < 3e-7))
nais_diam_centers_3nm = nais_diam_centers[idx_3nm_nais]
df_nais_3nm = df_nais.iloc[:,idx_3nm_nais[0][0]:idx_3nm_nais[0][-1]+1]

# sum-up and average total concentration
nais_tot = df_nais.sum(axis = 1)
nais_mean = df_nais.mean(axis=0)
nais_3nm_tot = df_nais_3nm.sum(axis = 1)
nais_3nm_mean = df_nais_3nm.mean(axis=0)

#%% extract model

ds_name = 'bnum-NEPALR4-NEW.nc'
ds = xr.open_dataset(ds_name)  

minmax_diameters = ds.cut_off_diameters
center_diameters = ds.mmd

PYR_loc = np.array([86.81322, 27.95903]) # Pyramid ori
# PYR_loc = np.array([86.76, 27.85]) # Pyramid valley P2
#PYR_loc = np.array([86.71456, 27.80239]) # Namche
# PYR_loc = np.array([86.731, 27.88]) # 3rd point (adjacent valley)
#PYR_loc = np.array([86.71, 27.88]) # 3rd point (adjacent valley) P1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon_PYR = find_nearest(ds.lon[10,:], PYR_loc[0]) 
idx_lat_PYR = find_nearest(ds.lat[:,10], PYR_loc[1]) 
# chi_lon_PYR = ds.lon[10,idx_lon_PYR]
# chi_lat_PYR = ds.lat[idx_lat_PYR,10]

times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
times = pd.to_datetime(times)
local_times = times + DateOffset(hours=6) # adjust to LST

idxTimei = list(ds.Times.values).index(b'2014-12-03_00:00:00') - 6
idxTimef = list(ds.Times.values).index(b'2014-12-25_23:00:00') - 6

local_times = local_times[idxTimei:idxTimef]

MOD_bnum = ds.bnum[idxTimei:idxTimef,:,0,idx_lat_PYR,idx_lon_PYR]

# cut bnum below 3 nm
idx_3nm = np.where((center_diameters > 2.95e-9) & (center_diameters < 3e-7))
center_diameters_3nm = center_diameters[idx_3nm]
idx_3nm_minmax = np.where((center_diameters > 2.95e-9) & (center_diameters < 4e-7))
minmax_diameters_3nm = minmax_diameters[idx_3nm_minmax]
MOD_bnum_3nm = MOD_bnum[:,idx_3nm[0][0]:idx_3nm[0][-1]+1]

MOD_bnum_tot = MOD_bnum.sum(axis = 1)
# create model series
MOD_bnum_tot = pd.Series(MOD_bnum_tot, index=local_times)
MOD_bnum_mean = MOD_bnum.mean(axis=0)

MOD_bnum_3nm_tot = MOD_bnum_3nm.sum(axis = 1)
# create model series
MOD_bnum_3nm_tot = pd.Series(MOD_bnum_3nm_tot, index=local_times)
MOD_bnum_3nm_mean = MOD_bnum_3nm.mean(axis=0)


#%% Plot timeseries
loc = mdates.DayLocator(interval=1)
fmt = mdates.DateFormatter('%d')

fig = plt.figure(figsize=(18,5))
ax1 = fig.add_subplot()
#fig,ax1 = plt.subplot(figsize=(18,5))
p_mod = ax1.plot(MOD_bnum_3nm_tot.index,MOD_bnum_3nm_tot,label='WRF-CHIMERE')

ax2 = plt.twinx(ax1)
p_obs = ax2.plot(nais_3nm_tot.index, nais_3nm_tot,'r',label='Observations (NAIS)')

ax1.legend(loc='upper left',fontsize=16)
ax2.legend(loc='upper right',fontsize=16)

ax1.tick_params(axis='y', colors='k')
ax1.set(ylim=(0,25000))
ax2.set(ylim=(0,25000))
ax1.set(xlim=(local_times[0],local_times[-1]))
ax1.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)

ax1.set_ylabel('Particles total number',color='k',fontsize=16)
    
ax2.set_ylabel('NAIS particles total number',color='k',fontsize=16)
ax1.xaxis.set_major_locator(loc)
ax1.xaxis.set_major_formatter(fmt)
ax1.grid(b = True, which = 'major', axis='x', color = '#666666', linestyle = '-', alpha = 0.2)
fig.savefig('output_figures/bnum_timeseries_NAM.png')
#plt.show()


#%% Diurnals (median)

nais_h_D = nais_3nm_tot.groupby(nais_3nm_tot.index.hour).median()
MOD_bnum_tot_D = MOD_bnum_3nm_tot.groupby(MOD_bnum_3nm_tot.index.hour).median()

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot()
#fig, ax1 = plt.subplot(figsize=(6,4))
ax1.plot(MOD_bnum_tot_D.index,MOD_bnum_tot_D,label='WRF-CHIMERE')

ax2 = plt.twinx(ax1)

ax2.plot(nais_h_D.index,nais_h_D,'r',label='Observations (NAIS)')

ax1.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set(ylim=(0,25000))
ax2.set(ylim=(0,25000))
ax1.set(xlim=(0,23))
ax1.set_xlabel('Datetime [Local Time]', fontsize = 16)
ax1.set_ylabel('Mean total particles number', fontsize = 16)
fig.savefig('output_figures/bnum_diurnal_NAM.png')
# plt.show()

#%% compute dN/dlog(Dp)

# MODEL
# compute dlogDp
dlogDp = np.zeros((1,30))
for i in range(len(center_diameters)):
	dlogDp[0,i] = np.log10(minmax_diameters[i+1]) - np.log10(minmax_diameters[i])
# compute dN/dlog(Dp)
PYR_dNdlogDp_mean = (np.array(MOD_bnum_mean) / np.array(dlogDp)).flatten()
# cut bnum below 3 nm
PYR_dNdlogDp_mean_3nm = PYR_dNdlogDp_mean[idx_3nm]


# NAIS data
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
nais_dNdlogDp_mean = np.array(nais_mean) / np.array(nais_dlogDp)
# cut NAIS below 3 nm
nais_dNdlogDp_mean_3nm = nais_dNdlogDp_mean[idx_3nm_nais]


#%% plot size distribution

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot()
#fig,ax1 = plt.subplot(figsize=(7,4))
ax1.loglog(center_diameters_3nm*1e9,1*PYR_dNdlogDp_mean_3nm.flatten(), 'bo',label='WRF-CHIMERE') # pyr
ax1.loglog(nais_diam_centers_3nm*1e9,nais_dNdlogDp_mean_3nm, 'ko',label='Observations (NAIS)')
ax1.set(ylim=(1e1,1e5))
ax1.legend()
ax1.set(xlim=(3,40))
# ax1.fill_between([0,3],0,1000000,color='grey')
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.set_xlabel('Particle diameter (nm)', fontsize = 16)
ax1.set_ylabel('dN/dlog(Dp) (particle cm$^{-3}$)', fontsize = 16)
fig.savefig('output_figures/bnum_sizedist_NAM_test.png')
# plt.show()

#%% scatter plot size distribution


#fig = plt.figure(figsize=(11,6))
#ax = fig.add_subplot()
#ax.grid()
#a = ax.scatter(nais_dNdlogDp_mean_3nm, 1*PYR_dNdlogDp_mean_3nm.flatten(),s=25, c=nais_diam_centers_3nm*1e9, cmap='autumn')
#t = np.linspace(0,10e5)
#ax.plot(t,t, color='black')
#r0 = np.linspace(0,10e5)
#y0 = 2*r0
#y1 = 0.5*r0
#ax.plot(r0,y0,'k--')
#ax.plot(r0,y1,'k--')
#ax.set_ylabel("Model dN/dlog(Dp) (particle cm$^{-3}$)", fontsize=18)
#ax.set_xlabel("Measured dN/dlog(Dp) (particle cm$^{-3}$))", fontsize=18)
#cbar = fig.colorbar(a, extend='both')
#cbar.set_label('Particle diameter (nm)',size=18)
#cbar.ax.tick_params(labelsize=15)
#plt.xlim(0, 10e5)
#plt.ylim(0, 10e5)
#ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
#fig.savefig('output_figures/size_distribution_scatter.png',dpi=500)

