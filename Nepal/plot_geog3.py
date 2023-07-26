#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import metpy
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import metpy.calc
from metpy.plots import add_metpy_logo, add_timestamp
from shapely.geometry.polygon import LinearRing

# GEOG file
geog = xr.open_dataset('geog_NEPALR3.nc')


# geog = xr.open_dataset('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/downscaling/geog_POVALLEY4.nc')
# ds_mask = geog.LANDUSEF[:,20,:,:]
# sea 
ds_sea = geog.LANDUSEF[:,16,:,:]
# elevation
ds_mask = geog.HGT_M[0,:,:]
# max elevation
# idxMAX_lat,idxMAX_lon = np.where(ds_mask == ds_mask.max())

# get profile
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# idx_lon_CIM = find_nearest(geog.XLONG_M[0,0,:], 10.699915396767054)
# lon_CIM = geog.XLONG_M[0,0,idx_lon_CIM]
# max elevation (CIMONE?)
# idxMAX_lat,idxMAX_lon = np.where(ds_mask == ds_mask.max())

# geog_profile = ds_mask[:,idx_lon_CIM]
# lat_profile = geog.XLAT_M[0,:,idx_lon_CIM]
# np.save('profiles/prof_geog',geog_profile)
# np.save('profiles/lat_geog',lat_profile)


cproj = cartopy.crs.PlateCarree()    
# Plot - replacing .imshow with .pcolormesh is more accurate but slowier
plt.figure(figsize=(22, 16))
ax = plt.axes(projection=cproj)
# ax.set_extent([10, 11, 44.7, 43.9])
# ax.set_extent([10.2, 11.2, 44.5, 43.9])
# ax.set_extent([9, 13, 46.15, 43])

# ax.set_extent([9.5, 13.5, 41, 45.5])
# ax.set_extent([3, 19, 51, 36]) # italy
# emis = plt.pcolormesh(dms_file.longitude+1,dms_file.latitude+1, dms_file.DMS[6,:,:], transform=ccrs.PlateCarree(), cmap='magma_r'); 

# emis = plt.pcolormesh(ds.lon,ds.lat, chem, transform=ccrs.PlateCarree(), cmap='magma_r',vmin=0,vmax=1e11); 
# emis = plt.pcolormesh(ds.lon-1.0*dx,ds.lat-1.0*dy, chem, transform=ccrs.PlateCarree(), cmap='magma_r',vmin=0,vmax=1e11); # plot 700hPa geopotential height at 13:00 UTC
# levels = np.arange(10., 4000., 100.)
# geogr = plt.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.3, linewidths = 1.6)
# geogrf = plt.contourf(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, cmap='Greys', alpha=0.9, linewidths = 1.6)
geogr = plt.pcolormesh(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(), cmap='terrain',vmin=-2300,vmax=7500)
sea = plt.pcolormesh(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_sea[0,:,:],transform=ccrs.PlateCarree(), cmap = 'viridis_r', alpha=0.45)
# cams = plt.pcolor(LON_cams, LAT_cams, cams, transform=ccrs.PlateCarree(), cmap = 'magma_r',vmin=0,vmax=6e-10)
# cbar = plt.colorbar(emis, fraction = 0.028)#, cax=pos_cax)

# emis = plt.pcolormesh(ds.lon-2*dx,ds.lat-2*dy, chem, transform=ccrs.PlateCarree(), cmap='magma_r',vmin=0,vmax=1e11); # plot 700hPa geopotential height at 13:00 UTC
# ax.tick_params(axis='both', which='major', labelsize=48)
# plt.clabel(GEO, GEO.levels, inline=True, fontsize=10)
# Wind = ax.quiver(u_10.longitude, u_10.latitude, u_10.values, v_10.values, scale=150, transform=ccrs.PlateCarree()); # plot 700 hPa wind at 13:00 UTC
ax.coastlines(color='k', linewidth = 1.5); ax.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 2) ;ax.add_feature(cartopy.feature.RIVERS);#ax.add_feature(cartopy.feature.STATES, alpha = 0.3)
gl = ax.gridlines(draw_labels=True,alpha=0.5);
gl.xlabel_style = {'size': 24, 'color': 'k'}
gl.ylabel_style = {'size': 24, 'color': 'k'}
# offset = 0.010
# MTC_loc = np.array([44.1931632649319, 10.701766513804616]) # CMN 

# plt.plot(MTC_loc[1], MTC_loc[0], markersize=15, marker='o',color='k')
# plt.annotate('CMN',(MTC_loc[1]-10*offset, MTC_loc[0]+10*offset),fontsize=36,color='k')
# # plt.plot([lon_CIM, lon_CIM], [lat_profile[0], lat_profile[-1]],
# #          color='red', linewidth=2, marker='o', markersize=3,
# #          transform=ccrs.PlateCarree
# #cimone 44.19335658052407, 10.699915396767054
# # plt.plot(11.6213, 44.6557, markersize=11, marker='o', color='b')
# # SPC 44.65578505554352, 11.62135742257148
# lons = [10.02,11.4,11.4,10.02 ]
# lats = [43.72,43.72,44.68,44.68]
# # ring = LinearRing(list(zip(lons, lats)))
# # ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='k',linewidth=2.5,linestyle='--')

# # lons = [3.09777832, 19.46635628, 19.46635628, 3.09777832]
# # lats = [36.07337952,36.07337952,50.9799118,50.9799118]
# # ring = LinearRing(list(zip(lons, lats)))
# # ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='k',linewidth=2.5,linestyle='--')

from shapely.geometry.polygon import LinearRing
lons = [86.12929, 87.43071, 87.43071, 86.12929]
lats = [27.294151,27.294151,28.283554,28.283554]
ring = LinearRing(list(zip(lons, lats)))
ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='k',linewidth=2.5,linestyle='-')


plt.annotate('d3',(88.6,29.2),fontsize=42,color='k')
# plt.annotate('d4',(11.7,44.68),fontsize=42,color='k')

# scale_bar(ax, 100,location=(0.7,0.06))

# plt.colorbar()
# plt.title(f'emissions (molecules/cm2)', pad = 10, fontsize = 24)
plt.savefig('geog_nepal3.png')

plt.show()


#%% plot S-N profile:
    
# fig, ax1 = plt.subplots(1,1, figsize=(18,4))
# ax1.plot(lat_profile,geog_profile)
# # ax1.set_xlabel('Date', fontsize=18)
# ax1.set_ylabel('Elevation (m)', fontsize=18)
# # legendarray = ('Chl','NH4','NO3','SO4','Org')
# # ax1.set_title('Mt. Cimone', fontsize = 18)
# # ax1.legend(legendarray, loc='upper right', fontsize = 14)
# ax1.set(xlim=(44.0, 44.43))
# ax1.set(ylim=(0, 2500))
# # ax1.grid(b = True, which = 'major', axis='x', color = '#666666', linestyle = '-', alpha = 0.2)
# # ax1.xaxis.set_major_locator(loc)
# # ax1.xaxis.set_major_formatter(fmt)
# # plt.show()
