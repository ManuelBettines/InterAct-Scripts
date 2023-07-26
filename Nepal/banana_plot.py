#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 7 07:57:33 2023

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


ds_name = 'bnum-NEPALR4-NEW.nc'
ds = xr.open_dataset(ds_name)

minmax_diameters = ds.cut_off_diameters
center_diameters = ds.mmd

# get GEOG file
geog = xr.open_dataset('geog_NEPALR4.nc')
# elevation
ds_mask = geog.HGT_M[0,:,:]

###############

# get valleys location
df = pd.read_csv('VALLEY_INFLUENCE_AREA_2.txt',sep=',',header=None)
lats = np.flip(np.array(df[0]))
lons = np.flip(np.array(df[1]))
df = pd.read_csv('VALLEY_INFLUENCE_AREA_1.txt',sep=',',header=None)
lats1 = np.flip(np.array(df[0]))
lons1 = np.flip(np.array(df[1]))
df = pd.read_csv('VALLEY_INFLUENCE_AREA_2.txt',sep=',',header=None)
lats2 = np.flip(np.array(df[0]))
lons2 = np.flip(np.array(df[1]))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# get chimere lat-lon indexes
chi_lats = np.zeros((len(lats)))
for i in np.arange(len(lats)):
    chi_lats[i] = find_nearest(ds.lat[:,0],lats[i])
chi_lons = np.zeros((len(lons)))
for i in np.arange(len(lons)):
    chi_lons[i] = find_nearest(ds.lon[0,:],lons[i])

###############

PYR_loc = np.array([86.81322, 27.95903]) # Pyramid ori
# PYR_loc = np.array([86.76, 27.85]) # Pyramid valley P2
# PYR_loc = np.array([86.731, 27.88]) # 3rd point (adjacent valley)
#PYR_loc = np.array([86.71, 27.88]) # 3rd point (adjacent valley) P1
#PYR_loc = np.array([86.86,71456, 27.95903]) # 4rd point (adjacent valley) P3

idx_lon_PYR = find_nearest(ds.lon[10,:], PYR_loc[0])
idx_lat_PYR = find_nearest(ds.lat[:,10], PYR_loc[1])

ds = ds.sel(bottom_top=0).sel(Time=slice(0,312))

#ds = ds.sel(bottom_top=0).sel(south_north=idx_lat_PYR).sel(west_east=idx_lon_PYR).sel(Time=slice(0,312))
#ds = ds.sel(bottom_top=0).sel(Time=slice(0,312))
    
#ds = ds_new#.sel(bottom_top=0).sel(Time=slice(0,312))
times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times)
local_times = utc_times + DateOffset(hours=6)
ds['local_times'] = pd.DatetimeIndex(local_times)
ds['bnum'] = ds.bnum.swap_dims({'Time':'local_times'})

sub = ds.bnum.sel(south_north=int(chi_lats[0])).sel(west_east=int(chi_lons[0]))
for i in range(len(chi_lats)-1):
    sub = sub + ds.bnum.sel(south_north=int(chi_lats[i+1])).sel(west_east=int(chi_lons[i+1]))

sub = sub/len(chi_lats)

lat_max = 27.95903
lat_min = 27.80000
lon_max = 86.71956
lon_min = 86.70956

#ds_new = ds.where((ds.lat<lat_max) & (ds.lat>lat_min))
#ds_new = ds_new.where((ds_new.lon<lon_max) & (ds_new.lon>lon_min))
#ds_new = ds_new.where((ds_mask<=5000) & (ds_mask>4500))#ds_mask.sel(south_north=idx_lat_PYR).sel(west_east=idx_lon_PYR))

bnum_d = sub.groupby(sub.local_times.dt.hour).median()
#bnum_d = bnum_d.mean('south_north').mean('west_east')
#local_hours = bnum_d.hour.values

MOD_bnum = bnum_d.where((center_diameters > 1e-9) & (center_diameters < 50e-9))#.sel(south_north=idx_lat_PYR).sel(west_east=idx_lon_PYR)
#idx_2nm_minmax = np.where((center_diameters > 1.1e-9) & (center_diameters < 4.88e-7))
#minmax_diameters_2nm = minmax_diameters[idx_2nm_minmax]

# compute dlogDp
#dlogDp = np.zeros((1,30))
#for i in range(29):
#        dlogDp[0,i] = np.log10(minmax_diameters[i+1]) - np.log10(minmax_diameters[i])
        #print(dlogDp[0,i])
# compute dN/dlog(Dp)
PYR_dNdlogDp = np.array(MOD_bnum)/0.11683277902223121 #/ np.array(dlogDp))

sub = PYR_dNdlogDp.T
time = np.arange(0,24,1)

mmd = []
for i in range(len(center_diameters)):
    mmd.append(int(center_diameters.values[i]))


fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot()
levels = np.logspace(np.log10(100), np.log10(20000), num=500)
a = ax.contourf(time,ds.nbins,sub, levels=levels, locator=ticker.LogLocator(),cmap='YlGnBu', extend='both')
ax.yaxis.set_ticks(np.arange(0,30), mmd)
ax.set_ylabel('Diameter (m)', fontsize=18)
ax.set_xlabel('Datetime [Local Time]', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.set_ylim([0,9])
cbar = fig.colorbar(a, extend='both')
cbar.set_label('dN/d(logD$_p$) (cm$^-3$)',size=18)
cbar.ax.tick_params(labelsize=15)
tick_locations = [100, 1000, 10000]
tick_labels = ['$10^{{{}}}$'.format(int(np.log10(t))) for t in tick_locations]
cbar.ax.set_yticks(tick_locations)
cbar.ax.set_yticklabels(tick_labels)
#fig.autofmt_xdate(rotation=45)
ax.set_title('NCO-P Valley - WRF-CHIMERE - 1-13 December', fontsize=21)
plt.show()
fig.savefig('output_figures/Banana_NCO-Pvalley.png', dpi=500)


######################################################
#ds_avg = ds_new.mean(dim='local_times')
PYR_loc = np.array([27.95903, 86.81322]) # Pyramid
#subset = ds_avg.bnum.sum('nbins')
#%%
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
#plot = plt.contourf(ds_avg.lon, ds_avg.lat, ds_avg.bnum.sum('nbins'), levels=levels, transform=ccrs.PlateCarree(), cmap='magma_r')#,vmin=260,vmax=330);
#plot = plt.pcolormesh(ds.lon,ds.lat, subset, transform=ccrs.PlateCarree(), cmap='magma_r', shading='gouraud',vmin=0,vmax=1);
#cbar = plt.colorbar(plot, fraction = 0.027, pad=0.11, extend='max')
levels = np.arange(10., 9000., 250.)
geogr = ax.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.2)
plt.plot(lons1,lats1,linewidth=3 ,color='k')
plt.plot(lons2,lats2,linewidth=3 ,color='k')
ax.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
ax.coastlines(color='k', linewidth = 1);
ax.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
ax.add_feature(cartopy.feature.STATES, linewidth = 0.6, alpha = 0.3)
ax.add_feature(cartopy.feature.RIVERS, linewidth = 0.6,color='blue', alpha = 0.6)
gl = ax.gridlines(draw_labels=True,alpha=0.5);
gl.xlabel_style = {'size': 16, 'color': 'k'}
gl.ylabel_style = {'size': 16, 'color': 'k'}
plt.plot(PYR_loc[1], PYR_loc[0], markersize=10, marker='x',color='k')
plt.annotate('NCO-P',(PYR_loc[1]+2*0.01, PYR_loc[0]+1*0.01),fontsize=24,color='k') # 3
plt.plot(86.715, 27.802, markersize=10, marker='x', color='k') # Namche
plt.annotate('Namche',(86.715+2*0.01, 27.802-2*0.01),fontsize=24,color='k')

plt.savefig('output_figures/Location_area_two_valleys.png',dpi=500)
                                                                   
