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
#psp_i = '2014-12-05 06:00:00'
#psp_f = '2014-12-12 06:00:00'
psp_f = '2014-12-25 06:00:00'
df_obs = df_obs.loc[(df_obs.index >= psp_i)& (df_obs.index < psp_f)]
df_metcom = df_metcom.loc[(df_metcom.index >= psp_i)& (df_metcom.index < psp_f)]

#%% extract model
# print('importing dataset')
#ds_name = 'METEO.out.201412-NEPALR4.nc'
ds_name = 'METEO-NEPALR4-NEW.nc'
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
#times_subset1 = pd.date_range(start='2014-12-05 00:00:00', end='2014-12-12 23:00:00', freq='1h') # converted to local time
times_subset = np.concatenate((times_subset1), axis=None)

ds = handle_chi_file(ds,times_subset,hour_offset=6)

#%% 
elev_PYR = 5050 #m
elev_PYR_mod = ds_mask[idx_PYR_lat,idx_PYR_lon]

#%% create model and obs series
OBS_tem2 = pd.Series(df_metcom['temp.C'],index=df_metcom.index)
MOD_tem2 = extract_var(ds,'tem2',PYR_loc)

OBS_ws10 = pd.Series(df_metcom['ws.m.s'],index=df_metcom.index)
MOD_ws10 = extract_var(ds,'ws10m',PYR_loc)

OBS_wd = pd.Series(df_metcom['wd.degree'],index=df_metcom.index)
MOD_u10 = units.Quantity(extract_var(ds,'u10m',PYR_loc).values,'m/s')
MOD_v10 = units.Quantity(extract_var(ds,'v10m',PYR_loc).values,'m/s')
MOD_wd10 = pd.Series( wind_direction(MOD_u10, MOD_v10) , index=ds.local_times)

#%% tem2 
loc = mdates.DayLocator(interval=1)
fmt = mdates.DateFormatter('%d')

#fig,(ax1) = plt.subplots(1,1, figsize=(22,4))

fig = plt.figure(figsize=(22,4))
ax1 = plt.subplot()

tshift = 0.0 * 9.8
p_mod = ax1.plot(MOD_tem2.index,MOD_tem2-273.15-tshift,'b',label='WRF-CHIMERE')
# ax2 = ax1.twinx()
p_obs = ax1.plot(OBS_tem2.index,OBS_tem2,'r',label='Observations')

ax1.legend(fontsize=16)
ax1.set_ylabel('Temperature (°C)',color='k',fontsize=21)
ax1.tick_params(axis='y', colors='k')
ax1.set(ylim=(-30,23))
# ax2.set(ylim=(4-tshift,25-tshift))
ax1.set(xlim=(times_subset[0],times_subset[-1]))
ax1.xaxis.set_major_locator(loc)
ax1.xaxis.set_major_formatter(fmt)
ax1.grid(b = True, which = 'major', axis='x', color = '#666666', linestyle = '-', alpha = 0.2)
ax1.tick_params(axis='both', which='major', labelsize=18)
plt.title("NCO-P - 2 meters temperature", fontsize=25)
for i in range(len(MOD_tem2.index[18::24])-1):
    # print(i)
    ax1.fill_between([MOD_tem2.index[18::24][i],MOD_tem2.index[30::24][i]],-200,1000,color='grey',alpha=0.15)
ax1.fill_between([MOD_tem2.index[0],MOD_tem2.index[6]],-200,1000,color='grey',alpha=0.15)
ax1.fill_between([MOD_tem2.index[-6],MOD_tem2.index[-1]],-200,1000,color='grey',alpha=0.15)
plt.savefig('output_figures/tem2_timeseries_PYR.png',dpi=500)
# plt.show()


# diurnal

OBS_t2m_dc = OBS_tem2.groupby(OBS_tem2.index.hour).mean()
OBS_t2m_dc_st = OBS_tem2.groupby(OBS_tem2.index.hour).std()

MOD_t2m_dc = MOD_tem2.groupby(MOD_tem2.index.hour).mean()
MOD_t2m_dc_st = MOD_tem2.groupby(MOD_tem2.index.hour).std()


#fig,(ax1) = plt.subplots(1,1, figsize=(10,6))
fig = plt.figure(figsize=(12,9))
ax1 = plt.subplot()
p_mod = ax1.plot(MOD_t2m_dc.index,MOD_t2m_dc-273.15-tshift,'b', linewidth=3,label='WRF-CHIMERE')
p_obs = ax1.plot(OBS_t2m_dc.index,OBS_t2m_dc,'ro', markersize=8,label='Observations')
ax1.legend(fontsize=16)#,ncol=2, bbox_to_anchor=(0.88, 1.2))
ax1.set(ylim=(-15,10))
#ax1.set(xlim=(0,23))
ax1.set_ylabel('Temperature (°C)',color='k',fontsize=21)
ax1.set_xlabel('Datetime [Local time]',color='k',fontsize=21)
ax1.tick_params(axis='both', which='major', labelsize=18)
plt.title("NCO-P - 2 meters temperature", fontsize=25)
ax1.fill_between(MOD_t2m_dc.index,
                  MOD_t2m_dc-273.15-tshift-MOD_t2m_dc_st,
                  MOD_t2m_dc-273.15-tshift+MOD_t2m_dc_st,
                  color='b',alpha=0.1,label='std MOD')
ax1.fill_between(OBS_t2m_dc.index,
                  OBS_t2m_dc-OBS_t2m_dc_st,
                  OBS_t2m_dc+OBS_t2m_dc_st,
                  color='r',alpha=0.1,label='std OBS')
# ax1.legend(loc='upper left',fontsize=16,ncol=2)
plt.savefig('output_figures/tem2_diurnal_PYR.png',dpi=500)
# plt.show()

#%% Wind speed 

fig,(ax1) = plt.subplots(1,1, figsize=(22,4))

for i in range(len(MOD_ws10.index[18::24])-1):
    # print(i)
    ax1.fill_between([MOD_ws10.index[18::24][i],MOD_ws10.index[30::24][i]],0,1000000,color='grey',alpha=0.15)
ax1.fill_between([MOD_ws10.index[0],MOD_ws10.index[6]],0,1000000,color='grey',alpha=0.15)
ax1.fill_between([MOD_ws10.index[-6],MOD_ws10.index[-1]],0,1000000,color='grey',alpha=0.15)


p_mod = ax1.plot(MOD_ws10.index,MOD_ws10,'r',label='WS MOD')
p_obs = ax1.plot(OBS_ws10.index,OBS_ws10,'k',label='WS OBS (CNR)')
# p_obs = ax1.plot(OBS_ws_A.index,OBS_ws_A,'k--',label='WS OBS (AM)')


ax1.legend(loc='upper left',fontsize=16,ncol=3)
ax1.set_ylabel('Wind speed (m/s)',color='k',fontsize=18)
ax1.tick_params(axis='y', colors='k')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set(ylim=(0,10.5))
ax1.set(ylim=(0,20))
ax1.set(xlim=(times_subset[0],times_subset[-1]))
ax1.xaxis.set_major_locator(loc)
ax1.xaxis.set_major_formatter(fmt)
ax1.grid(b = True, which = 'major', axis='x', color = '#666666', linestyle = '-', alpha = 0.2)
#plt.savefig('output_figures/ws_timeseries_PYR.png',dpi=500)
# plt.show()

#%%
MOD_ws10_dc = MOD_ws10.groupby(MOD_ws10.index.hour).mean()
MOD_ws10_dc_st = MOD_ws10.groupby(MOD_ws10.index.hour).std()

OBS_ws_dc = OBS_ws10.groupby(OBS_ws10.index.hour).mean()
OBS_ws_dc_st = OBS_ws10.groupby(OBS_ws10.index.hour).std()

fig,(ax1) = plt.subplots(1,1, figsize=(10,6))
p_mod = ax1.plot(MOD_ws10_dc.index,MOD_ws10_dc,'r',label='WS MOD')
p_obs = ax1.plot(OBS_ws_dc.index,OBS_ws_dc,'k',label='WS OBS')
ax1.legend(loc="upper right",fontsize=16)#,ncol=2, bbox_to_anchor=(0.88, 1.2))
ax1.fill_between(MOD_ws10_dc.index,
                  MOD_ws10_dc-MOD_ws10_dc_st,
                  MOD_ws10_dc+MOD_ws10_dc_st,
                  color='red',alpha=0.1,label='std MOD')
ax1.fill_between(OBS_ws_dc.index,
                  OBS_ws_dc-OBS_ws_dc_st,
                  OBS_ws_dc+OBS_ws_dc_st,
                  color='grey',alpha=0.1,label='std OBS')

ax1.set(ylim=(0,10))
ax1.set(xlim=(0,23))
ax1.set_ylabel('WS (m/s)',color='k',fontsize=18)
ax1.set_xlabel('hour of the day',color='k',fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
# ax1.legend(loc='upper left',fontsize=16,ncol=2)
#plt.savefig('output_figures/ws_diurnal_PYR.png', dpi=500)
# plt.show()
#%% Wind direction 

fig,(ax1) = plt.subplots(1,1, figsize=(22,4))
for i in range(len(MOD_ws10.index[18::24])-1):
    # print(i)
    ax1.fill_between([MOD_ws10.index[18::24][i],MOD_ws10.index[30::24][i]],0,1000000,color='grey',alpha=0.15)
ax1.fill_between([MOD_ws10.index[0],MOD_ws10.index[6]],0,1000000,color='grey',alpha=0.25)
ax1.fill_between([MOD_ws10.index[-6],MOD_ws10.index[-1]],0,1000000,color='grey',alpha=0.15)


p_mod = ax1.scatter(MOD_wd10.index,MOD_wd10,color='r',label='WD MOD',s=12)
p_obs = ax1.scatter(OBS_wd.index,OBS_wd,color='k',label='WS OBS',s=12)

ax1.legend(loc='lower right',fontsize=16,ncol=2,bbox_to_anchor=(0.9,0.01))
ax1.set_ylabel('Wind dir (deg)',color='k',fontsize=18)
ax1.tick_params(axis='y', colors='k')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set(ylim=(0,360))
ax1.set(xlim=(times_subset[0],times_subset[-1]))
ax1.xaxis.set_major_locator(loc)
ax1.xaxis.set_major_formatter(fmt)
ax1.grid(b = True, which = 'major', axis='x', color = '#666666', linestyle = '-', alpha = 0.2)
#plt.savefig('output_figures/wd_timeseries_PYR.png', dpi=500)
# plt.show()

MOD_wd10_dc = MOD_wd10.groupby(MOD_wd10.index.hour).mean()
MOD_wd10_dc_st = MOD_wd10.groupby(MOD_wd10.index.hour).std()

OBS_wd_dc = OBS_wd.groupby(OBS_wd.index.hour).mean()
OBS_wd_dc_st = OBS_wd.groupby(OBS_wd.index.hour).std()

fig = plt.figure(figsize=(12,9))
ax1 = plt.subplot()
ax1.plot(MOD_wd10_dc.index,MOD_wd10_dc,color='b', linewidth=3,label='WRF-CHIMERE')
ax1.plot(OBS_wd_dc.index,OBS_wd_dc,'ro', markersize=8,label='Observations')
ax1.fill_between(OBS_wd_dc.index, OBS_wd_dc-OBS_wd_dc_st, OBS_wd_dc +OBS_wd_dc_st, color='r',alpha=0.2)
ax1.legend(loc='upper right',fontsize=16)
ax1.set_ylim([0,360])
ax1.set_ylabel('Wind direction (deg)',color='k',fontsize=21)
ax1.set_xlabel('Datetime [Lacal Time]',color='k',fontsize=21)
ax1.tick_params(axis='y', colors='k')
ax1.tick_params(axis='both', which='major', labelsize=18)
#ax1.set(ylim=(0,360))
plt.title("Diurnal wind direction - NCO-P", fontsize=25)
plt.savefig('output_figures/wd_diurnal__PYR.png')



#%% windroses

# # select day/night
OBS_ws_day = OBS_ws10.copy()
OBS_ws_day[(OBS_ws10.index.hour >= 16) | (OBS_ws10.index.hour < 15)] = np.nan
OBS_ws_night = OBS_ws10.copy()
OBS_ws_night[(OBS_ws10.index.hour >= 6) & (OBS_ws10.index.hour < 18)] = np.nan

OBS_wd_day = OBS_wd.copy()
OBS_wd_day[(OBS_wd.index.hour >= 16) | (OBS_wd.index.hour < 15)] = np.nan
OBS_wd_night = OBS_wd.copy()
OBS_wd_night[(OBS_wd.index.hour >= 6) & (OBS_wd.index.hour < 18)] = np.nan

MOD_ws_day = MOD_ws10.copy()
MOD_ws_day[(MOD_ws10.index.hour >= 16) | (MOD_ws10.index.hour < 15)] = np.nan
MOD_ws_night = MOD_ws10.copy()
MOD_ws_night[(MOD_ws10.index.hour >= 6) & (MOD_ws10.index.hour < 18)] = np.nan

MOD_wd_day = MOD_wd10.copy()
MOD_wd_day[(MOD_wd10.index.hour >= 16) | (MOD_wd10.index.hour < 15)] = np.nan
MOD_wd_night = MOD_wd10.copy()
MOD_wd_night[(MOD_wd10.index.hour >= 6) & (MOD_wd10.index.hour < 18)] = np.nan

# # model
import windrose
from windrose import WindroseAxes

fig = plt.figure(figsize=(12,6))
# # fig.suptitle('All Period', fontsize = 24)
ax1 = fig.add_subplot(projection='windrose')
ax1.bar(MOD_wd_day, MOD_ws_day, normed=True, bins = np.array([0,1,2,3,5,8,12,16]), opening=0.9, edgecolor='k')
ax1.set_title('NCO-P - WRF-CHIMERE [15 Local Time]', fontsize=24)
ax1.set_yticks(np.arange(10, 40, step=10))
ax1.set_yticklabels(np.arange(10, 40, step=10))
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.set_legend()
ax1.set_ylim(0,40)
plt.savefig('output_figures/wr_model_day_PYR.png', dpi=500)
# # plt.show()

fig = plt.figure(figsize=(12,6))
# # fig.suptitle('All Period', fontsize = 24)
ax1 = fig.add_subplot(projection='windrose')
ax1.bar(MOD_wd_night, MOD_ws_night, normed=True, bins = np.array([0,1,2,3,5,8,12,16]), opening=0.9, edgecolor='k')
ax1.set_title('Model NIGHT', fontsize=24)
ax1.set_yticks(np.arange(10, 40, step=10))
ax1.set_yticklabels(np.arange(10, 40, step=10))
ax1.tick_params(axis='both', which='major', labelsize=18)
#plt.savefig('output_figures/wr_model_night_PYR.png', dpi=500)
# # plt.show()
# #%% windrose obs

fig = plt.figure(figsize=(12,6))
# # fig.suptitle('All Period', fontsize = 24)
ax1 = fig.add_subplot(projection='windrose')
ax1.bar(OBS_wd_day, OBS_ws_day, normed=True, bins = np.array([0,1,2,3,5,8,12,16]), opening=0.9, edgecolor='k')
# # ax1.set_legend()
ax1.set_title('NCO-P - Observations [15 Local Time]', fontsize=24)
ax1.set_yticks(np.arange(10, 40, step=10))
ax1.set_yticklabels(np.arange(10, 40, step=10))
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.set_legend()
ax1.set_ylim(0,40)
plt.savefig('output_figures/wr_obs_day_PYR.png', dpi=500)
# # plt.show()

fig = plt.figure(figsize=(12,6))
# # fig.suptitle('All Period', fontsize = 24)
ax1 = fig.add_subplot(projection='windrose')
ax1.bar(OBS_wd_night, OBS_ws_night, normed=True, bins = np.array([0,1,2,3,5,8,12,16]), opening=0.9, edgecolor='k')
# # ax1.set_legend()
ax1.set_title('Observations NIGHT', fontsize=24)
ax1.set_yticks(np.arange(10, 40, step=10))
ax1.set_yticklabels(np.arange(10, 40, step=10))
ax1.tick_params(axis='both', which='major', labelsize=18)
#plt.savefig('output_figures/wr_obs_night_PYR.png', dpi=500)
# # plt.show()
