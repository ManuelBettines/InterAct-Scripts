#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 02:22:29 2023

@author: bettines
"""

import pandas as pd
from pandas.tseries.offsets import DateOffset
import metpy
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import metpy.calc
from metpy.plots import add_metpy_logo, add_timestamp
from metpy.calc import wind_direction, potential_temperature
from metpy.units import units
import os

#%% plot together
from pylab import *
from numpy import *

# single variable file:
ds_name = 'VALLEY-WIND-NEPALR4.nc'
ds = xr.open_dataset(ds_name)

# get GEOG file
geog = xr.open_dataset('geog_NEPALR4.nc')
# ds_mask = geog.LANDUSEF[:,20,:,:]
# sea 
ds_sea = geog.LANDUSEF[:,16,:,:]
# elevation
ds_mask = geog.HGT_M[0,:,:]

# get profiles coordinates
crss2_coords = [27.6 , 27.6 , 86.49, 86.88]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

crss2_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],crss2_coords[0])
crss2_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],crss2_coords[1])
crss2_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],crss2_coords[2])
crss2_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],crss2_coords[3])

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

# get chimere lat-lon indexes
chi_lats = np.zeros((len(lats)))
for i in np.arange(len(lats)):
    chi_lats[i] = find_nearest(ds.lat[:,0],lats[i])
chi_lons = np.zeros((len(lons)))
for i in np.arange(len(lons)):
    chi_lons[i] = find_nearest(ds.lon[0,:],lons[i])

chi_lats_e = np.zeros((len(lats_e)))
for i in np.arange(len(lats_e)):
    chi_lats_e[i] = find_nearest(ds.lat[:,0],lats_e[i])
chi_lons_e = np.zeros((len(lons_e)))
for i in np.arange(len(lons_e)):
    chi_lons_e[i] = find_nearest(ds.lon[0,:],lons_e[i])
    
chi_lats_w = np.zeros((len(lats_w)))
for i in np.arange(len(lats_w)):
    chi_lats_w[i] = find_nearest(ds.lat[:,0],lats_w[i])
chi_lons_w = np.zeros((len(lons_w)))
for i in np.arange(len(lons_w)):
    chi_lons_w[i] = find_nearest(ds.lon[0,:],lons_w[i])

# find along-valley intersection with cross profile
chi_lat_cross2 = find_nearest(lats,crss2_coords[0])

#%%
############
times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times)
local_times = utc_times + DateOffset(hours=6)

# create new auxiliary dimension
ds['local_times'] = pd.DatetimeIndex(local_times)
# swap it with the old time dimension of the main variable
ds['winm'] = ds.winm.swap_dims({'Time':'local_times'})
ds['winz'] = ds.winz.swap_dims({'Time':'local_times'})
#ds['winw'] = ds.winw.swap_dims({'Time':'local_times'})
ds['pres'] = ds.pres.swap_dims({'Time':'local_times'})
ds['temp'] = ds.temp.swap_dims({'Time':'local_times'})
ds['hlay'] = ds.hlay.swap_dims({'Time':'local_times'})
ds['thlay'] = ds.thlay.swap_dims({'Time':'local_times'})


ds = ds.drop_dims('Time')
# select period of interest
# ds = ds.sel(local_times=(slice('2017-07-01 00:00:00', '2017-07-09 23:00:00')))
times_subset1 = pd.date_range(start='2014-12-02 00:00:00', end='2014-12-25 23:00:00', freq='1h') # converted to local time
times_subset = np.concatenate((times_subset1), axis=None)
ds = ds.sel(local_times=times_subset)

# compute daily cycle
dailyCycle_u = ds.winz.groupby(ds.local_times.dt.hour).mean()
dailyCycle_v = ds.winm.groupby(ds.local_times.dt.hour).mean()
dailyCycle_pres = ds.pres.groupby(ds.local_times.dt.hour).mean()
dailyCycle_temp = ds.temp.groupby(ds.local_times.dt.hour).mean()
dailyCycle_hlay = ds.hlay.groupby(ds.local_times.dt.hour).mean()
dailyCycle_thlay = ds.thlay.groupby(ds.local_times.dt.hour).mean()
local_hours = dailyCycle_u.hour.values

u_D = dailyCycle_u.isel(hour=[9,10,11,12,13,14,15]).mean(dim='hour')
v_D = dailyCycle_v.isel(hour=[9,10,11,12,13,14,15]).mean(dim='hour')
pres_D = dailyCycle_pres.isel(hour=[9,10,11,12,13,14,15]).mean(dim='hour')
temp_D = dailyCycle_temp.isel(hour=[9,10,11,12,13,14,15]).mean(dim='hour')
#hlay_D = dailyCycle_hlay.isel(hour=[15]).mean(dim='hour')
#thlay_D = dailyCycle_thlay.isel(hour=[15]).mean(dim='hour')

#############
chem_DAY = u_D#ds.winw.mean('Time')
p0 = pres_D.sel(bottom_top=0)#ds.pres.sel(bottom_top=0).mean('Time')
pot = temp_D*((p0/pres_D)**0.286)

for daynight in ['DAILY AVERAGE']:#['5 LC','15 LC','DAILY AVERAGE']:
    print(daynight)
    if daynight == 'DAILY AVERAGE':
        chem = chem_DAY
        
    #%% extract 2d vertical profiles
    #### along valley profile num concentration
         
    profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    potential_t = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        profile[:,i] = chem[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
        potential_t[:,i] = pot[:,int(chi_lats[i]),int(chi_lons[i])].squeeze() 
        
    # extract topography along valley
    elev = np.zeros((len(lats)))
    for i in np.arange(len(lats)):
        elev[i] = ds_mask[int(chi_lats[i]),int(chi_lons[i])]
    # extract topography on ridges
    elev_e = np.zeros((len(lats_e)))
    for i in np.arange(len(lats_e)):
        elev_e[i] = ds_mask[int(chi_lats_e[i]),int(chi_lons_e[i])]
    elev_w = np.zeros((len(lats_w)))
    for i in np.arange(len(lats_w)):
        elev_w[i] = ds_mask[int(chi_lats_w[i]),int(chi_lons_w[i])]
    
    # extract layer heigth along valley
    hlay = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    thlay = np.zeros((len(ds.thlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        hlay[:,i] = ds.hlay[i,:,int(chi_lats[i]),int(chi_lons[i])]
        thlay[:,i] = ds.thlay[i,:,int(chi_lats[i]),int(chi_lons[i])]
        
    zlay = hlay - 0.5 * thlay + elev
    #################################################

    u_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        u_profile[:,i] = u_D[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()

    v_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        v_profile[:,i] = v_D[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()

    step_y = np.zeros((len(lats)))
    step_x = np.zeros((len(lons)))
    alpha = np.zeros((len(lons)))
    alpha[0:2] = np.nan
    alpha[-2:] = np.nan

    for i in np.arange(2,len(lats)-2):
        step_y[i] = lats[i+2] - lats[i-2]
        step_x[i] = lons[i+2] - lons[i-2]
        alpha[i] = np.arctan2(step_y[i], step_x[i]) # degrees

    horizontal_wind = (u_profile * np.cos(alpha)) + (v_profile * np.sin(alpha))

    # thermodynamics
    t_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        t_profile[:,i] = temp_D[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()

    p_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        p_profile[:,i] = pres_D[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()

    # compute potential temperature profile
    theta_profile = potential_temperature(p_profile * units.Pa, t_profile * units.kelvin)


    
    #### cross valley profiles 
    
    #2
    
    crss_profile_2 = np.zeros((len(ds.hlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    t_cross = np.zeros((len(ds.hlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    p_cross = np.zeros((len(ds.hlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
        # print(i)
    crss_profile_2[:,:] = chem[:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)].squeeze()
    t_cross[:,:] = temp_D[:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)].squeeze()
    p_cross[:,:] = pres_D[:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)].squeeze()

    potential_temp_cross = potential_temperature(p_cross * units.Pa, t_cross * units.kelvin)
    # extract topography cross valley
    crss_elev_2 = np.zeros((crss2_lon_idx_2-crss2_lon_idx_1))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    crss_elev_2[:] = ds_mask[int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)]
    
    # extract layer heigth cross valley
    crss_hlay_2 = np.zeros((len(ds.hlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    crss_thlay_2 = np.zeros((len(ds.thlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    crss_hlay_2[:,:] = ds.hlay[i,:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)]
    crss_thlay_2[:,:] = ds.thlay[i,:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)]
        
    crss_zlay_2 = crss_hlay_2 - 0.5 * crss_thlay_2 + crss_elev_2
    
    #%%
    
    #fig = plt.figure(figsize=(80,80))
    
    #%%
    # along-valley profile conc
    #ax1 = plt.subplot2grid((100, 100), (3, 14), colspan=12, rowspan=6)
    fig = plt.figure(figsize=(13,7))
    ax1 = fig.add_subplot()
    x = np.arange(1,len(lats)+1)
    x = np.tile(x,(15,1))
    levels = np.arange(-5,5,0.1)
    
    profile_plot = ax1.contourf(x,zlay,horizontal_wind,cmap='coolwarm',levels=levels, extend='both')
    cbar = plt.colorbar(profile_plot, fraction = 0.042, pad = 0.10)
    cbar.set_label('Wind speed (m s$^{-1}$)', fontsize=27, y=0.5)
    cbar.ax.tick_params(labelsize=18)
    level1 = np.arange(240., 350., 4.)
    contours = ax1.contour(x, zlay, theta_profile,levels=level1, colors="black", linewidths=0.5)
    ax1.clabel(contours, inline=1, fontsize=8, fmt="%i")
    ax1.set(xlim=(1,len(lats)))
    ax1.set(ylim=(0,10000))
    ax1.set_ylabel("Height above mean sea level (m)", fontsize=27)
    ax1.tick_params(labelsize=18)
    ax2=ax1.twinx()
    ax2.plot(x[0],elev,'k')
    ax2.fill_between(x[0], elev, color= 'none', hatch="\\\\\\\\",edgecolor="black")
    ax2.plot(x[0],elev_e,'b--')
    ax2.plot(x[0],elev_w,'r--')
    ax2.set(ylim=(0,10000))
    ax2.set_yticks([])
    
    ax1.set_title('Along-valley winds - Latitudinal section B-B', fontsize = 31)    
    fig.savefig('output_figures/Along-valley-wind.png',dpi=500)

    y = np.tile(geog.XLONG_M[0,0,crss2_lon_idx_1:crss2_lon_idx_2],(15,1))
    #%%
    # cross-section 2 conc 
    #ax5 = plt.subplot2grid((100, 100), (1, 1), colspan=10, rowspan=8)
    fig = plt.figure(figsize=(12,9))
    ax5 = fig.add_subplot()
    crss_profile_plot = ax5.contourf(y,crss_zlay_2,crss_profile_2,levels,cmap='coolwarm',extend='both')
    cbar = plt.colorbar(crss_profile_plot, fraction = 0.042, pad = 0.10)
    cbar.set_label('Wind speed (m s$^{-1}$)', fontsize=27, y=0.5)
    cbar.ax.tick_params(labelsize=18)
    level1 = np.arange(240., 350., 4.)
    contours = ax5.contour(y, crss_zlay_2, potential_temp_cross,levels=level1, colors="black", linewidths=0.5)
    ax5.clabel(contours, inline=1, fontsize=8, fmt="%i")
    ax5.set(xlim=(y[0,0],y[0,-1]))
    ax5.set(ylim=(0,7000))
    ax5.set_ylabel('Height above mean sea level (m)', fontsize=27)
    ax5.tick_params(labelsize=18)
    ax6=ax5.twinx()
    ax6.plot(y[0],crss_elev_2,'k')
    ax6.fill_between(y[0], crss_elev_2, color= 'none', hatch="\\\\\\\\",edgecolor="black")
    ax6.set(ylim=(0,7000))
    ax5.set_xticks([])
    ax6.set_yticks([])
    ax6.set_title('Along-ridge winds - Longitudinal section A-A', fontsize = 31)
    
    #title = 'output_figures/valley_wind' + daynight + '.png'
    plt.savefig('output_figures/Along-ridge-wind.png',dpi=500)
    #plt.show()

