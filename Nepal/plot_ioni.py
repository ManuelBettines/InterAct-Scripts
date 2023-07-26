#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 08:47:48 2023

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
ds_name = 'IONI-NEPALR4-ONL-VAL.nc'
ds = xr.open_dataset(ds_name)
ds_name = 'bnum.OUT.NEPALR4-ONL-VAL.nc'
ds_homs = xr.open_dataset(ds_name)


# get GEOG file
geog = xr.open_dataset('geog_NEPALR4.nc')
# elevation
ds_mask = geog.HGT_M[0,:,:]

##########

PYR_loc = np.array([86.69, 27.6]) # Bottom-valley long-section

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def extract_chi_loc(ds,location):
    idx_lon = find_nearest(ds.lon[10,:], location[0])
    idx_lat = find_nearest(ds.lat[:,10], location[1])
    chi_lon = ds.lon[10,idx_lon]
    chi_lat = ds.lat[idx_lat,10]
    return(chi_lat,chi_lon,idx_lat,idx_lon)

chi_lat_PYR,chi_lon_PYR, idx_PYR_lat,idx_PYR_lon = extract_chi_loc(ds,PYR_loc)


##########

hlay = ds.hlay.median('Time')
thlay = ds.thlay.median('Time')
elev = ds_mask
zlay = hlay - 0.5*thlay + elev
h = zlay.sel(south_north=idx_PYR_lat).sel(west_east=idx_PYR_lon)#median('south_north').median('west_east')

sub_ion = ds.ion_conc.median("Time").sel(south_north=idx_PYR_lat).sel(west_east=idx_PYR_lon)#.median("south_north").median("west_east") 
sub_j = ds.nucl_homs.mean("Time").sel(south_north=idx_PYR_lat).sel(west_east=idx_PYR_lon)#.mean("south_north").mean("west_east")
sub_homs = ds_homs.HOMSgas.median("Time").sel(south_north=idx_PYR_lat).sel(west_east=idx_PYR_lon)*2.46e10

fig = plt.figure(figsize=(11,13))
ax1 = plt.subplot()
ax2 = ax1.twiny()
#lns3 = ax1.plot(sub_crays, h, 'k', linewidth=3, label="Cosmic rays")
#lns2 = ax1.plot(sub_j,h,'ko',label="Nucleation rate")
c = ax1.scatter(sub_j,h,c=sub_homs, s=150,label="Nucleation rate", cmap="coolwarm")
lns1 = ax2.plot(sub_ion, h, 'r', linewidth=3,label="Ion concentration")
#lns = lns1 + lns2  #+ lns3
#labs = [l.get_label() for l in lns]
cbar = plt.colorbar(c, fraction = 0.042, pad = 0.10,extend='both')
cbar.set_label('HOMs concentration (molecules cm$^{-3}$)', fontsize=21, y=0.5)
cbar.ax.tick_params(labelsize=18)
#ax1.legend(lns,labs,loc='upper left',prop={'size': 15})
ax2.set_xlabel('Ion concentration (cm$^{-3}$)', fontsize = 21)
ax1.set_ylabel('Height above median sea level (m)', fontsize = 21)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.set_xlabel('Nucleation rate J (cm$^{-3}$ s$^{-1}$)', fontsize = 21)
ax2.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('output_figures/ioni_height_fondovalle.png',dpi=500)
plt.show()
