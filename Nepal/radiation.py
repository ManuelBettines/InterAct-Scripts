#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 8 15:57:01 2023

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

tmp = xr.open_dataset("cloud-NEPALR3.nc")
ds_name = 'radiation-NEPALR3.nc'
ds = xr.open_dataset(ds_name)
ds_name = 'radiation-BIOx10.nc'
ds1 = xr.open_dataset(ds_name)
# get GEOG file
geog = xr.open_dataset('geog_NEPALR3.nc')
# ds_mask = geog.LANDUSEF[:,20,:,:]
# sea
ds_sea = geog.LANDUSEF[:,16,:,:]
# elevation
ds_mask = geog.HGT_M[0,:,:]

#subset 
subset = ds.SWDNT.mean("Time") + ds.LWDNT.mean("Time") - ds.SWUPT.mean("Time") - ds.LWUPT.mean("Time")
subset_biox10 = ds1.SWDNT.mean("Time") + ds1.LWDNT.mean("Time") - ds1.SWUPT.mean("Time") - ds1.LWUPT.mean("Time")
    
#plotting
fig = plt.figure(figsize=(12,9))
ax0 = plt.subplot(projection=ccrs.PlateCarree())
levels = np.arange(-5, 5, 0.1)
chem_plot = ax0.contourf(tmp.lon,tmp.lat,subset_biox10-subset, transform=ccrs.PlateCarree(),levels=levels, cmap='bwr', extend='both');
cbar = plt.colorbar(chem_plot, fraction = 0.035, pad = 0.12)
cbar.set_label('Net radiative balance absolute change (W m$^{-2}$)', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)
levels = np.arange(10., 9000., 500.)
geogr = ax0.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.2)
ax0.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
ax0.coastlines(color='k', linewidth = 1);
ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
ax0.add_feature(cartopy.feature.STATES, linewidth = 0.6, alpha = 0.3)
ax0.add_feature(cartopy.feature.RIVERS, linewidth = 0.6,color='blue', alpha = 0.6)
gl = ax0.gridlines(draw_labels=True,alpha=0.5);
gl.xlabel_style = {'size': 18, 'color': 'k'}
gl.ylabel_style = {'size': 18, 'color': 'k'}
plt.title('Radiative balance', fontsize='31')
plt.show()
fig.savefig('output_figures/Radiation_balance.png',dpi=500)
