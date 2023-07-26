#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:41:31 2023

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

# Namche and Lukla
df_obs = pd.read_csv('AWS_SNP_december2014_Namche.csv')
df_obs['datetime'] = pd.to_datetime(df_obs[['year', 'month', 'day', 'hour']])
df_obs = df_obs.set_index('datetime')
df_obs = df_obs.drop(['year','month', 'day', 'hour','minute'],axis=1)
# Pyramid
df_metcom = pd.read_csv('DATA_NCOP.csv',sep=",")
df_metcom['datetime'] = pd.to_datetime(df_metcom['Date'])
df_metcom = df_metcom.set_index('datetime')
df_metcom = df_metcom.drop(['Date','Unnamed: 14','Unnamed: 15','17-24 dic','O3','PM10','PM1','BC'], axis = 1)
# Select time range
psp_i = '2014-12-01 06:00:00'
# psp_i = '2014-12-05 06:00:00'
# psp_f = '2014-12-11 06:00:00'
psp_f = '2014-12-25 06:00:00'
df_obs = df_obs.loc[(df_obs.index >= psp_i)& (df_obs.index < psp_f)]
df_metcom = df_metcom.loc[(df_metcom.index >= psp_i)& (df_metcom.index < psp_f)]

#%% extract model
# print('importing dataset')
ds_name = 'SOA.out.201412-NEPALR4.nc'
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
# times_subset1 = pd.date_range(start='2014-12-05 00:00:00', end='2014-12-11 23:00:00', freq='1h') # converted to local time
times_subset = np.concatenate((times_subset1), axis=None)

ds = handle_chi_file(ds,times_subset,hour_offset=6)

OBS_PM1 = pd.Series(df_metcom['PM_1 (mg/m3)_NO_STP'],index=df_metcom.index)
MOD_SOA = extract_var(ds,'SOA',PYR_loc)

#%%
loc = mdates.DayLocator(interval=1)
fmt = mdates.DateFormatter('%d')

fig,(ax1) = plt.subplots(1,1, figsize=(22,4))

tshift = 0.0 * 9.8
p_mod = ax1.plot(MOD_SOA.index,MOD_SOA,'r',label='SOA MOD')
ax2 = ax1.twinx()
p_obs = ax2.plot(OBS_PM1.index,OBS_PM1,'k',label='PM1 OBS')

ax1.legend(loc='upper left',fontsize=16,ncol=2)
ax2.legend(loc='upper right',fontsize=16,ncol=2)

ax1.set_ylabel('SOA (ug/m3)',color='k',fontsize=18)
ax2.set_ylabel('PM1 (ug/m3?)',color='k',fontsize=18)

ax1.tick_params(axis='y', colors='k')
# ax1.set(ylim=(-30,23))
# ax2.set(ylim=(4-tshift,25-tshift))
ax1.set(xlim=(times_subset[0],times_subset[-1]))
ax1.xaxis.set_major_locator(loc)
ax1.xaxis.set_major_formatter(fmt)
ax1.grid(b = True, which = 'major', axis='x', color = '#666666', linestyle = '-', alpha = 0.2)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
# for i in range(len(MOD_SOA.index[18::24])-1):
#     print(i)
#     ax1.fill_between([MOD_SOA.index[18::24][i],MOD_SOA.index[30::24][i]],-200,1000,color='grey',alpha=0.15)
# ax1.fill_between([MOD_SOA.index[0],MOD_SOA.index[6]],-200,1000,color='grey',alpha=0.15)
# ax1.fill_between([MOD_SOA.index[-6],MOD_SOA.index[-1]],-200,1000,color='grey',alpha=0.15)
plt.savefig('output_figures/SOA_PM1_timeseries.png')
# plt.show()


#%% diurnal - MEDIAN !!!
MOD_SOA_dc = MOD_SOA.groupby(MOD_SOA.index.hour).median()
MOD_SOA_dc_st = MOD_SOA.groupby(MOD_SOA.index.hour).std()

OBS_PM1_dc = OBS_PM1.groupby(OBS_PM1.index.hour).median()
OBS_PM1_dc_st = OBS_PM1.groupby(OBS_PM1.index.hour).std()

fig,(ax1) = plt.subplots(1,1, figsize=(6,4))
p_mod = ax1.plot(MOD_SOA_dc.index,MOD_SOA_dc,'r',label='SOA MOD')
ax2 = ax1.twinx()
p_obs = ax2.plot(OBS_PM1_dc.index,OBS_PM1_dc,'k',label='PM1 OBS')
ax1.legend(fontsize=16,ncol=2, bbox_to_anchor=(0.96, 1.2))
ax2.legend(fontsize=16,ncol=2, bbox_to_anchor=(0.52, 1.2))

# ax1.set(ylim=(-30,20))
ax1.set(xlim=(0,23))
ax1.set_ylabel('SOA (ug/m3)',color='k',fontsize=18)
ax2.set_ylabel('PM1 (ug/m3?)',color='k',fontsize=18)
ax1.set_xlabel('hour of the day',color='k',fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

# ax1.fill_between(MOD_t2m_dc.index,
#                   MOD_t2m_dc-273.15-tshift-MOD_t2m_dc_st,
#                   MOD_t2m_dc-273.15-tshift+MOD_t2m_dc_st,
#                   color='red',alpha=0.1,label='std MOD')
# ax1.fill_between(OBS_t2m_dc.index,
#                   OBS_t2m_dc-OBS_t2m_dc_st,
#                   OBS_t2m_dc+OBS_t2m_dc_st,
#                   color='grey',alpha=0.1,label='std OBS')
# ax1.legend(loc='upper left',fontsize=16,ncol=2)
plt.savefig('output_figures/SOA_PM1_diurnal_median.png')
# plt.show()

