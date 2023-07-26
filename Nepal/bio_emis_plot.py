#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 11:01:22 2023

@author: bettines
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
from pandas.tseries.offsets import DateOffset


#%%
# Settings:
#emission_ncfile_name = 'EMISSIONI.APINEN-ONLYVALLEY.nc'
emission_ncfile_name = 'BIO-EMIS-NEPALR4-NEW.nc'
# Choose one: C5H8, APINEN, TERPEN...
spec='APINEN' # isoprene 
# set different colormap maximum for the 4 domains 
#levels = np.arange(0, 1e-7, 1e-8) 

#geog = xr.open_dataset('geog_NEPALR4.nc')
#ds_mask = geog.HGT_M[0,:,:]

# get valleys location
#df = pd.read_csv('khumbu_latlon_extended.txt',sep=',',header=None)
#lats = np.flip(np.array(df[0]))
#lons = np.flip(np.array(df[1]))
# ridges
#df_e = pd.read_csv('khumbu_east_ridge_LATLON.txt',sep=',',header=None)
#lats_e = np.flip(np.array(df_e[0]))
#lons_e = np.flip(np.array(df_e[1]))
#df_w = pd.read_csv('khumbu_west_ridge_LATLON.txt',sep=',',header=None)
#lats_w = np.flip(np.array(df_w[0]))
#lons_w = np.flip(np.array(df_w[1]))

#%% 
ds = xr.open_dataset(emission_ncfile_name)
#ds_avg = ds.mean(dim='Time')
#PYR_loc = np.array([27.95903, 86.81322]) # Pyramid
#%%
#plt.figure(figsize=(15, 10))
#ax = plt.axes(projection=cartopy.crs.PlateCarree())
#levels = np.arange(0, 5*1e-8, 1e-9)
#plot = plt.contourf(ds_avg.lon, ds_avg.lat, ds_avg[spec], levels=levels, transform=ccrs.PlateCarree(), cmap='magma_r', extend='max')#,vmin=260,vmax=330); 
#cbar = plt.colorbar(plot, fraction = 0.027, pad=0.11, extend='max')
#cbar.ax.yaxis.offsetText.set(size=20)
#cbar.ax.tick_params(labelsize=24)
#cbar.ax.set_ylabel('α-pinene (g m$^{-2}$ s$^{-1}$)',fontsize=21)
#cbar.ax.tick_params(labelsize=16)
#ax.tick_params(axis='both', which='major', labelsize=24)

#levels = np.arange(10., 9000., 500.)
#geogr = ax.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.2)
#ax.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
#ax.coastlines(color='k', linewidth = 1);
#ax.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
#ax.add_feature(cartopy.feature.STATES, linewidth = 0.6, alpha = 0.3)
#ax.add_feature(cartopy.feature.RIVERS, linewidth = 0.6,color='blue', alpha = 0.6)
#gl = ax.gridlines(draw_labels=True,alpha=0.5);
#gl.xlabel_style = {'size': 16, 'color': 'k'}
#gl.ylabel_style = {'size': 16, 'color': 'k'}

###ax.coastlines('50m', linewidth=0.8)
###ax.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 1) ;
###gl = ax.gridlines(color="black", linestyle="dotted")
###gl.xlabel_style = {'size': 14, 'color': 'k'}
###gl.ylabel_style = {'size': 14, 'color': 'k'}
###for lat,lon in zip(lats,lons):
#plt.plot(lons, lats, linewidth=2, color='w')
###for lat,lon in zip(lats_e,lons_e):
#plt.plot(lons_e, lats_e, linewidth=2, color='b')
###for lat,lon in zip(lats_w,lons_w):
#plt.plot(lons_w, lats_w, linewidth=2, color='r')
#plt.title("Average α-pinene emissions - NEPALR4",fontsize=31, y=1.05)
#plt.plot(PYR_loc[1], PYR_loc[0], markersize=10, marker='x',color='k')
#plt.annotate('NCO-P',(PYR_loc[1]+2*0.01, PYR_loc[0]+1*0.01),fontsize=24,color='k') # 3
#plt.plot(86.715, 27.802, markersize=10, marker='x', color='k') # Namche
#plt.annotate('Namche',(86.715+2*0.01, 27.802+1*0.01),fontsize=24,color='k')

#plt.savefig('output_figures/APINEN_NEPALR4.png',dpi=500)

################################
#%% Diurnal

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

#times_subset1 = pd.date_range(start='2014-12-02 00:00:00', end='2014-12-25 23:00:00', freq='1h') # converted to local time
#times_subset = np.concatenate((times_subset1), axis=None)

#ds = handle_chi_file(ds,times_subset,hour_offset=6)

#ts = pd.Series(ds['APINEN'].mean('south_north').mean('west_east'),index=ds.local_times)

#MOD_APINEN = ts.groupby(ts.index.hour).mean()
#MOD_STD = ts.groupby(ts.index.hour).std()

#fig = plt.figure(figsize=(12,9))
#ax1 = plt.subplot()
#ax1.plot(MOD_APINEN.index,MOD_APINEN,'b', linewidth=3)
#ax1.fill_between(MOD_APINEN.index, MOD_APINEN-MOD_STD, MOD_APINEN+MOD_STD, color='b', alpha=0.2)
#ax1.set_xlabel('Datetime [Local Time]', fontsize = 21)
#ax1.set_ylabel('α-pinene emission (μg m$^{-2}$ s$^{-1}$)', fontsize = 21)
#ax1.tick_params(axis='both', which='major', labelsize=18)
#plt.title('Diurnal α-pinene emission', fontsize=25)
#plt.savefig('output_figures/APINEN_diurnal_mean.png',dpi=500)


################################
#%% relative contribution

apinen = ds['APINEN'].mean('south_north').mean('west_east').mean('Time')
bpinen = ds['BPINEN'].mean('south_north').mean('west_east').mean('Time')
limone = ds['LIMONE'].mean('south_north').mean('west_east').mean('Time')
humule = ds['HUMULE'].mean('south_north').mean('west_east').mean('Time')
ocimen = ds['OCIMEN'].mean('south_north').mean('west_east').mean('Time')
isoprene = ds['C5H8'].mean('south_north').mean('west_east').mean('Time')

labels = 'α-pinene', 'β-pinene', 'Limonene', 'Ocimenes', 'Humulene', 'Isoprene' 
sizes = [apinen,bpinen,limone,humule,ocimen,isoprene]
colors = ['dodgerblue', 'purple', 'brown', 'orange', 'gold', 'green'] 

fig = plt.figure(figsize=(12,9))
ax = plt.subplot()
title = plt.title('BVOC emissions - NEPALR4', fontsize=19)
title.set_ha("left")
pie = ax.pie(sizes, startangle=0, colors=colors,autopct='%1.1f%%')
#p, tx, autotexts = plt.pie(sizes, labels=labels, colors=colors, startangle=0,autopct="", shadow=True)
#for i, a in enumerate(autotexts):
#    a.set_text("{} ng m$^{}$ s$^{}$)".format(sizes[i], -2, -1))
plt.legend(pie[0],labels, bbox_to_anchor=(1,0.5), loc="center right", fontsize=12,bbox_transform=plt.gcf().transFigure)
#plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)

plt.savefig('output_figures/BVOC_mean.png',dpi=500)



