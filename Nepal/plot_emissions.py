#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 18:01:22 2022

@author: bvitali
"""

#%%
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature

#%%
# Settings:
emission_ncfile_name = 'EMISSIONI.APINEN-ONLYVALLEY.nc'
# Choose one: C5H8, APINEN, TERPEN...
spec='APINEN' # isoprene 
# set different colormap maximum for the 4 domains 
levels = np.arange(0, 1e-8, 1e-9) 

geog = xr.open_dataset('geog_NEPALR4.nc')
ds_mask = geog.HGT_M[0,:,:]

# get valleys location
df = pd.read_csv('khumbu_latlon_extended.txt',sep=',',header=None)
lats = np.flip(np.array(df[0]))
lons = np.flip(np.array(df[1]))
# ridges
df_e = pd.read_csv('khumbu_east_ridge_LATLON.txt',sep=',',header=None)
lats_e = np.flip(np.array(df_e[0]))
lons_e = np.flip(np.array(df_e[1]))
df_w = pd.read_csv('khumbu_west_ridge_LATLON.txt',sep=',',header=None)
lats_w = np.flip(np.array(df_w[0]))
lons_w = np.flip(np.array(df_w[1]))

#%% 
ds = xr.open_dataset(emission_ncfile_name)
ds_avg = ds.mean(dim='Time')
PYR_loc = np.array([27.95903, 86.81322]) # Pyramid
#%%
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
levels = np.arange(0, 1e-8, 1e-12)
plot = plt.contourf(ds_avg.lon, ds_avg.lat, ds_avg[spec], levels=levels, transform=ccrs.PlateCarree(), cmap='magma_r', extend='max')#,vmin=260,vmax=330); 
cbar = plt.colorbar(plot, fraction = 0.027, pad=0.11, extend='max')
cbar.ax.yaxis.offsetText.set(size=20)
cbar.ax.tick_params(labelsize=24)
cbar.ax.set_title('α-pinene\n (g m$^{-2}$ s$^{-1}$)\n',fontsize=24, y = 1.09)
cbar.ax.tick_params(labelsize=16)
ax.tick_params(axis='both', which='major', labelsize=24)

levels = np.arange(10., 9000., 500.)
geogr = ax.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.2)
ax.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
ax.coastlines(color='k', linewidth = 1);
ax.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
ax.add_feature(cartopy.feature.STATES, linewidth = 0.6, alpha = 0.3)
ax.add_feature(cartopy.feature.RIVERS, linewidth = 0.6,color='blue', alpha = 0.6)
gl = ax.gridlines(draw_labels=True,alpha=0.5);
gl.xlabel_style = {'size': 10, 'color': 'k'}
gl.ylabel_style = {'size': 10, 'color': 'k'}

#ax.coastlines('50m', linewidth=0.8)
#ax.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 1) ;
#gl = ax.gridlines(color="black", linestyle="dotted")
#gl.xlabel_style = {'size': 14, 'color': 'k'}
#gl.ylabel_style = {'size': 14, 'color': 'k'}
#for lat,lon in zip(lats,lons):
plt.plot(lons, lats, linewidth=2, color='w')
#for lat,lon in zip(lats_e,lons_e):
plt.plot(lons_e, lats_e, linewidth=2, color='b')
#for lat,lon in zip(lats_w,lons_w):
plt.plot(lons_w, lats_w, linewidth=2, color='r')
plt.title("Average α-pinene emissions",fontsize=32, y=1.03)
plt.plot(PYR_loc[1], PYR_loc[0], markersize=10, marker='x',color='k')
plt.annotate('NCO-P',(PYR_loc[1]+2*0.01, PYR_loc[0]+1*0.01),fontsize=24,color='k') # 3
title = 'output_figures/' + emission_ncfile_name + '.png'
plt.savefig(title)
