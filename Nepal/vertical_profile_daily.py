#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 1 14:17:13 2023

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


ds_name = 'bnum-NEPALR3.nc'
ds = xr.open_dataset('/scratch/project_2006279/GC/CHIMERE/chimere_out_online_nepal_vbs_homs/nest-NEPALR3/{}'.format(ds_name))
ds2 = xr.open_dataset('/scratch/project_2006279/GC/CHIMERE/chimere_out_online_nepal_vbs_homs/nest-NEPALR3/hlayer.nc')

geog = xr.open_dataset('geog_NEPALR3.nc')
ds_mask = geog.HGT_M[0,:,:]

times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times)
local_times = utc_times + DateOffset(hours=6)
ds['local_times'] = pd.DatetimeIndex(local_times)
ds['bnum_tot'] = ds.bnum_tot.swap_dims({'Time':'local_times'})

dailyCycle_chem = ds.bnum_tot.groupby(ds.local_times.dt.hour).mean()
local_hours = dailyCycle_chem.hour.values

chem_N = dailyCycle_chem

bnum = chem_N.where(ds.south_north > 100-33/140*ds.west_east)

hlay = ds2.hlay.mean('Time')
thlay = ds2.thlay.mean('Time')
elev = ds_mask
zlay = hlay - 0.5*thlay + elev

zlay = zlay.where(ds2.south_north > 100-33/140*ds2.west_east)

sub =  bnum.mean('south_north').mean('west_east')
h = zlay.mean('south_north').mean('west_east')

H = []
for i in range(len(h)):
    H.append(int(h.values[i]))

time = np.arange(0,24,1)
sub = sub.T


fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot()
a = ax.imshow(sub,interpolation='spline36' , cmap='magma_r', aspect='auto')
ax.invert_yaxis()
ax.xaxis.set_ticks(np.arange(0, 24), time)
ax.yaxis.set_ticks(np.arange(0, 15), H)
ax.set_ylabel('Height above mean sea level (m)', fontsize=18)
ax.set_xlabel('Datetime [Local Time]', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
cbar = fig.colorbar(a, extend='both')
cbar.set_label('Biogenic particles (particles cm$^{-3}$)',size=18)
cbar.ax.tick_params(labelsize=15)
#plt.show()
fig.savefig('output_figures/vertical_profile.png', dpi=500)
