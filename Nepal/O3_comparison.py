#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:59:49 2022

@author: bvitali
"""

# print('starting...')
# print('importing packages')

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

df_metcom = pd.read_csv('DATA_NCOP.csv',sep=",")
df_metcom['datetime'] = pd.to_datetime(df_metcom['Date'])
df_metcom = df_metcom.set_index('datetime')
df_metcom = df_metcom.drop(['Date'], axis = 1)

psp_i = '2014-12-02 00:00:00'
psp_f = '2014-12-25 00:00:00'
df_metcom = df_metcom.loc[(df_metcom.index >= psp_i)& (df_metcom.index < psp_f)]

OBS_O3 = pd.Series(df_metcom['ozone.ppb'])

#%% extract model
# print('importing dataset')
ds_name = 'O3.nest-NEPALR4.nc'
ds= xr.open_dataset(ds_name)

# get GEOG file
geog = xr.open_dataset('geog_NEPALR4.nc')
ds_mask = geog.LANDUSEF[:,20,:,:]
# sea 
ds_sea = geog.LANDUSEF[:,16,:,:]
# elevation
ds_mask = geog.HGT_M[0,:,:]

PYR_loc = np.array([86.81322, 27.95903]) # Pyramid
NAM_loc = np.array([86.71456, 27.80239]) # Namche

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def handle_chi_file(ds,times_subset,hour_offset=0):
    times = ds.Times.astype(str)
    times = np.core.defchararray.replace(times,'_',' ')
    times = pd.to_datetime(times)
    local_times = times + DateOffset(hours=hour_offset)
    ds['local_times'] = pd.DatetimeIndex(local_times)
    for var in list(ds.variables):
        if var not in ['Times','local_times','lon','lat']:           
            if 'bottom_top' in ds[var].dims: # 4D variable
                ds[var] = ds[var].swap_dims({'Time':'local_times'}).sel(bottom_top=0)
            else: # 3D variable
                ds[var] = ds[var].swap_dims({'Time':'local_times'})
    ds = ds.drop_dims('Time')
    ds = ds.sel(local_times=times_subset)  
    print(list(ds.variables))
    return ds

def extract_chi_loc(ds,location):
    idx_lon = find_nearest(ds.lon[10,:], location[0])
    idx_lat = find_nearest(ds.lat[:,10], location[1])
    chi_lon = ds.lon[10,idx_lon]
    chi_lat = ds.lat[idx_lat,10]
    return(chi_lat,chi_lon,idx_lat,idx_lon)

chi_lat_PYR,chi_lon_PYR, idx_PYR_lat,idx_PYR_lon = extract_chi_loc(ds,PYR_loc)
chi_lat_NAM,chi_lon_NAM, idx_NAM_lat,idx_NAM_lon = extract_chi_loc(ds,NAM_loc)


def extract_var(ds,variable,location):
    ts = pd.Series(ds[variable].sel(
        south_north=find_nearest(ds.lat[:,10],location[1])).sel(
        west_east=find_nearest(ds.lon[10,:],location[0])),
        index=ds.local_times)
    return(ts)

times_subset1 = pd.date_range(start='2014-12-02 00:00:00', end='2014-12-25 23:00:00', freq='1h') # converted to local time
# times_subset1 = pd.date_range(start='2014-12-05 00:00:00', end='2014-12-12 23:00:00', freq='1h') # converted to local time
times_subset = np.concatenate((times_subset1), axis=None)

ds = handle_chi_file(ds,times_subset,hour_offset=6)


# create model series
MOD_O3 = extract_var(ds,'O3',PYR_loc)
MOD_OH = extract_var(ds,'OH',PYR_loc)

#%% Plot 
loc = mdates.DayLocator(interval=1)
fmt = mdates.DateFormatter('%d')
fig,(ax1) = plt.subplots(1,1, figsize=(12,3))

p_mod = ax1.plot(MOD_O3.index,MOD_O3,'b',label='WRF-CHIMERE')
p_obs = ax1.plot(OBS_O3.index,OBS_O3,'r',label='Observations')

ax1.legend(loc='upper left')
ax1.set_ylabel('O$_3$ (ppbv)',color='k',fontsize=14)
ax1.tick_params(axis='y', colors='k')
ax1.set(ylim=(0,90))
ax1.set(xlim=(OBS_O3.index[0],OBS_O3.index[-1]))
ax1.xaxis.set_major_locator(loc)
ax1.xaxis.set_major_formatter(fmt)
ax1.grid(b = True, which = 'major', axis='x', color = '#666666', linestyle = '-', alpha = 0.2)
plt.savefig('output_figures/O3_timeseries.png')
# plt.show()


#%% Diurnal

OBS_O3_D = OBS_O3.groupby(OBS_O3.index.hour).mean()
OBS_O3_STD = OBS_O3.groupby(OBS_O3.index.hour).std()
MOD_O3_D = MOD_O3.groupby(MOD_O3.index.hour).mean()
MOD_OH_D = MOD_OH.groupby(MOD_OH.index.hour).mean()
MOD_OH_D = MOD_OH_D * 2.46*10e10

fig = plt.figure(figsize=(12,9))
ax1 = plt.subplot()
lns1 = ax1.plot(OBS_O3_D.index,OBS_O3_D,'ro', markersize=8,label="Observations")
lns2 = ax1.plot(MOD_O3_D.index,MOD_O3_D,'b', linewidth=3,label="WRF-CHIMERE (O$_3$)")
ax2 = ax1.twinx()
lns3 = ax2.plot(MOD_OH_D.index,MOD_OH_D,'k', linewidth=3,label="WRF-CHIMERE (OH)")
ax1.fill_between(OBS_O3_D.index, OBS_O3_D-OBS_O3_STD, OBS_O3_D+OBS_O3_STD, color='r', alpha=0.2)
# ax1.set(ylim=(0,90))
lns = lns1 + lns2 +lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs,loc='center left',prop={'size': 13})
ax1.set(ylim=(0,60))
ax1.set_xlabel('Datetime [Local Time]', fontsize = 21)
ax1.set_ylabel('O$_3$ concentration (ppbv)', fontsize = 21)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax2.set_ylabel('OH concentration (molecules cm$^{-3}$)', fontsize = 21)
ax2.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('output_figures/O3_diurnal_mean.png',dpi=500)
# plt.show()


#OBS_O3_D = OBS_O3.groupby(OBS_O3.index.hour).median()
#MOD_O3_D = MOD_O3.groupby(MOD_O3.index.hour).median()

#ig, ax1 = plt.subplots(1, figsize=(7,5))
#ax1.plot(OBS_O3_D.index,OBS_O3_D,'r')
#ax1.plot(MOD_O3_D.index,MOD_O3_D,'b')
# ax1.set(ylim=(0,90))
#ax1.set(xlim=(0,23))
#ax1.set_xlabel('Datetime [Local Time]', fontsize = 14)
#ax1.set_ylabel('Median O$_3$ (ppbv)', fontsize = 14)
#plt.savefig('output_figures/O3_diurnal_median.png')
# plt.show()





