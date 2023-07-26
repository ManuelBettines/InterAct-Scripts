#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 08:55:42 2023

@author: bettines
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


#%% extract model
ds_name = 'heatflux.OUT.NEPALR4.nc'
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
MOD_SW = extract_var(ds,'swrd',PYR_loc)
MOD_LH = extract_var(ds,'slhf',PYR_loc) + extract_var(ds,'sshf',PYR_loc)
#MOD_SH = extract_var(ds,'sshf',PYR_loc)

MOD_SW_D = MOD_SW.groupby(MOD_SW.index.hour).mean()
MOD_LH_D = MOD_LH.groupby(MOD_LH.index.hour).mean()

fig = plt.figure(figsize=(12,9))
ax1 = plt.subplot()
lns2 = ax1.plot(MOD_SW_D.index,MOD_SW_D,'b', linewidth=3,label="WRF-CHIMERE (SW radiation)")
ax2 = ax1.twinx()
lns3 = ax2.plot(MOD_LH_D.index,MOD_LH_D,'k', linewidth=3,label="WRF-CHIMERE (Heat flux)")
lns =  lns2 +lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs,loc=0,prop={'size': 13})
ax1.set_xlabel('Datetime [Local Time]', fontsize = 21)
ax1.set_ylabel('Short wave incoming radiation (W m$^{-2}$)', fontsize = 21)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax2.set_ylabel('Heat flux (W m$^{-2}$)', fontsize = 21)
ax2.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('output_figures/radiation_flux_diurnal_mean.png',dpi=500)

