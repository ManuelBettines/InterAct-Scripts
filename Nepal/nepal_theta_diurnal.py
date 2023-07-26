#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:36:29 2022

@author: bvitali
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
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import metpy.calc
from metpy.plots import add_metpy_logo, add_timestamp
from metpy.calc import wind_direction, potential_temperature
from metpy.units import units
import os

# plot together
        
# utility for nonlinear colorscale
        
from pylab import *
from numpy import *
        
class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""
        
    name = 'nlcmap'
        
    def __init__(self, cmap, levels):
                self.cmap = cmap
                self.monochrome = self.cmap.monochrome
                self.levels = asarray(levels, dtype='float64')
                self._x = self.levels/ self.levels.max()
                self.levmax = self.levels.max()
                self.levmin = self.levels.min()
                self._y = linspace(self.levmin, self.levmax, len(self.levels))
        
    def __call__(self, xi, alpha=1.0, **kw):
        yi = interp(xi, self._x, self._y)
        return self.cmap(yi/self.levmax, alpha)

#%%

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
# crss1_coords = [27.40 , 27.40 , 86.6 , 86.8]
# crss2_coords = [27.60 , 27.60 , 86.6, 86.8]
# crss3_coords = [43.60 , 44.60 , 10.30 , 10.30]
# crss4_coords = [28.00 , 28.00 , 86.6 , 86.8]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# crss1_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],crss1_coords[0])
# crss1_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],crss1_coords[1])
# crss1_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],crss1_coords[2])
# crss1_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],crss1_coords[3])   

# crss2_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],crss2_coords[0])
# crss2_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],crss2_coords[1])
# crss2_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],crss2_coords[2])
# crss2_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],crss2_coords[3])

# crss3_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],crss3_coords[0])
# crss3_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],crss3_coords[1])
# crss3_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],crss3_coords[2])
# crss3_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],crss3_coords[3])

# crss4_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],crss4_coords[0])
# crss4_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],crss4_coords[1])
# crss4_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],crss4_coords[2])
# crss4_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],crss4_coords[3])

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
# chi_lat_cross1 = find_nearest(lats,crss1_coords[0])
# chi_lat_cross2 = find_nearest(lats,crss2_coords[0])
# chi_lat_cross3 = find_nearest(lats,crss3_coords[0])
# chi_lat_cross4 = find_nearest(lats,crss4_coords[0])

#%%

times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times)
local_times = utc_times + DateOffset(hours=6)

# create new auxiliary dimension
ds['local_times'] = pd.DatetimeIndex(local_times)
# swap it with the old time dimension of the main variable  
ds['winm'] = ds.winm.swap_dims({'Time':'local_times'})
ds['winz'] = ds.winz.swap_dims({'Time':'local_times'})
# ds['winw'] = ds.winw.swap_dims({'Time':'local_times'})
ds['pres'] = ds.pres.swap_dims({'Time':'local_times'})
ds['temp'] = ds.temp.swap_dims({'Time':'local_times'})
ds['hlay'] = ds.hlay.swap_dims({'Time':'local_times'})
ds['thlay'] = ds.thlay.swap_dims({'Time':'local_times'})
# ds['pblh'] = ds.pblh.swap_dims({'Time':'local_times'})

ds = ds.drop_dims('Time')
# select period of interest
# ds = ds.sel(local_times=(slice('2017-07-01 00:00:00', '2017-07-09 23:00:00')))
times_subset1 = pd.date_range(start='2014-12-03 00:00:00', end='2014-12-10 23:00:00', freq='1h') # converted to local time
# times_subset2 = pd.date_range(start='2017-07-15 00:00:00', end='2017-07-20 23:00:00', freq='1h') # converted to local time
# times_subset3 = pd.date_range(start='2017-07-26 00:00:00', end='2017-07-28 23:00:00', freq='1h') # converted to local time

# times_subset1 = pd.date_range(start='2017-07-10 00:00:00', end='2017-07-14 23:00:00', freq='1h') # converted to local time
# times_subset2 = pd.date_range(start='2017-07-21 00:00:00', end='2017-07-25 23:00:00', freq='1h') # converted to local time

# times_subset = np.concatenate((times_subset1, times_subset2, times_subset3), axis=None)
# times_subset = np.concatenate((times_subset1, times_subset2), axis=None)
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

# # chem_N = dailyCycle_chem.isel(hour=[0,1,2,3,4,5,6,7,8,21,22,23]).mean(dim='hour')
# # chem_N = dailyCycle_chem.isel(hour=[3,4,5,6,7,8]).mean(dim='hour')
# u_N = dailyCycle_u.isel(hour=[5]).mean(dim='hour')
# # chem_D = dailyCycle_chem.isel(hour=[9,10,11,12,13,14,15,16,17,18,19,20]).mean(dim='hour')
# # chem_D = dailyCycle_chem.isel(hour=[12,13,14,15,16,17]).mean(dim='hour')
# u_D = dailyCycle_u.isel(hour=[14]).mean(dim='hour')

# v_N = dailyCycle_v.isel(hour=[5]).mean(dim='hour')
# v_D = dailyCycle_v.isel(hour=[14]).mean(dim='hour')

# pres_N = dailyCycle_pres.isel(hour=[5]).mean(dim='hour')
# pres_D = dailyCycle_pres.isel(hour=[14]).mean(dim='hour')

# temp_N = dailyCycle_temp.isel(hour=[5]).mean(dim='hour')
# temp_D = dailyCycle_temp.isel(hour=[14]).mean(dim='hour')

# hlay_N = dailyCycle_hlay.isel(hour=[5]).mean(dim='hour')
# hlay_D = dailyCycle_hlay.isel(hour=[14]).mean(dim='hour')

# thlay_N = dailyCycle_thlay.isel(hour=[5]).mean(dim='hour')
# thlay_D = dailyCycle_thlay.isel(hour=[14]).mean(dim='hour')

#%%
count = 1
for lh in (local_hours):
    # local_time = local_times[i] 
    print(lh)
    # concentration and vertical integral
    u =  dailyCycle_u[lh,...]
    v =  dailyCycle_v[lh,...]
    pres =  dailyCycle_pres[lh,...]
    temp =  dailyCycle_temp[lh,...]
    hlay =  dailyCycle_hlay[lh,...]
    thlay =  dailyCycle_thlay[lh,...]
    
    #% extract 2d vertical profiles
    
    #### along valley profile num concentration
         
    # profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    # for i in np.arange(len(lats)):
    #     profile[:,i] = chem_now[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
        
    u_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        u_profile[:,i] = u[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
    
    v_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        v_profile[:,i] = v[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()

    # alpha = np.arctan(0.46) # degrees
    # horizontal_wind = (u_profile * np.cos(alpha)) + (v_profile * np.sin(alpha))
    
    step_y = np.zeros((len(lats)))
    step_x = np.zeros((len(lons)))
    alpha = np.zeros((len(lons)))
    alpha[0:2] = np.nan
    alpha[-2:] = np.nan

    for i in np.arange(2,len(lats)-2):
        step_y[i] = lats[i+2] - lats[i-2]
        step_x[i] = lons[i+2] - lons[i-2]
        # tangent = step_y[i] / step_x[i]
        alpha[i] = np.arctan2(step_y[i], step_x[i]) # degrees
        
    # alpha = np.arctan(0.46) # degrees
    horizontal_wind = (u_profile * np.cos(alpha)) + (v_profile * np.sin(alpha))
    
    
    # thermodynamics
    t_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        t_profile[:,i] = temp[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
      
    p_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        p_profile[:,i] = pres[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
        
    # compute potential temperature profile
    theta_profile = potential_temperature(p_profile * units.Pa, t_profile * units.kelvin)
    
    
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
    hlay_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    thlay_profile = np.zeros((len(ds.thlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        hlay_profile[:,i] = hlay[:,int(chi_lats[i]),int(chi_lons[i])]
        thlay_profile[:,i] = thlay[:,int(chi_lats[i]),int(chi_lons[i])]
        
    zlay = hlay_profile - 0.5 * thlay_profile + elev
    
    
    #%
    
    fig = plt.figure(figsize=(80,80))
    
    ax0 = plt.subplot2grid((100, 100), (1, 1), colspan=13, rowspan=10, projection=ccrs.PlateCarree())
    # ax0.set_extent([9.7, 11.4, 43.6, 44.65])
    ax0.set_extent([86.14, 87.43, 27.295, 28.28])
    
    # free-profiles
    
    for lat,lon in zip(lats,lons):
        plt.plot(lon, lat, markersize=1.2, marker='o', color='k')
    for lat,lon in zip(lats_e,lons_e):
        plt.plot(lon, lat, markersize=1.2, marker='o', color='b')
    for lat,lon in zip(lats_w,lons_w):
        plt.plot(lon, lat, markersize=1.2, marker='o', color='r')    
    
    levels = np.arange(10., 9000., 250.)
    # levels2 = np.arange(10., 3000., 500.)
    geogr = ax0.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.3, linewidths = 1.6)
    # geogr2 = ax0.contourf(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, cmap='Greys', alpha=0.7)
    ax0.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
    ax0.coastlines(color='k', linewidth = 1); 
    # ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
    # ax0.add_feature(cartopy.feature.STATES, linewidth = 0.6, alpha = 0.3)
    # ax0.add_feature(cartopy.feature.RIVERS, linewidth = 0.6,color='blue', alpha = 0.6)
    gl = ax0.gridlines(draw_labels=True,alpha=0.5);
    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    # levels = np.arange(0,3.0,0.1)
    # plot = ax0.contourf(ds.lon,ds.lat, np.sqrt(np.square(u[0,:,:]) + np.square(v[0,:,:])), transform=ccrs.PlateCarree(), levels = levels, cmap='magma',alpha=0.5) 
    # cbar = plt.colorbar(plot, fraction = 0.028, location='right', pad = 0.10)
    # cbar.set_label('zonal wind speed', fontsize=12, rotation=270, labelpad=30, y=0.5) 
    levels = np.arange(0,15.5,0.5)
    plot = ax0.contourf(ds.lon,ds.lat, u[0,:,:], transform=ccrs.PlateCarree(),levels=levels, cmap='Reds',alpha=1) 
    cbar = plt.colorbar(plot, fraction = 0.028, location='right', pad = 0.10)
    cbar.set_label('wind speed', fontsize=12, rotation=270, labelpad=30, y=0.5) 
    # add wind 10 m 
    # gap = 6
    # plt.barbs((ds.lon[::gap,::gap]), (ds.lat[::gap,::gap]),
    #           (u10m_now[::gap, ::gap]*1.94), (v10m_now[::gap, ::gap]*1.94),
    #           transform=ccrs.PlateCarree(), length=5, alpha=0.5)
    # higher levels
    # gap = 2
    # plt.barbs((ds.lon[::gap,::gap]), (ds.lat[::gap,::gap]),
    #           (u_now[1,::gap, ::gap]*1.94), (v_now[1,::gap, ::gap]*1.94),
    #           transform=ccrs.PlateCarree(), length=5, alpha=0.5)
    plt.streamplot((ds.lon[1,:]), (ds.lat[:,1]),
              (np.array(u[0,:,:])), (np.array(v[0,:,:])),
              transform=ccrs.PlateCarree(),density=2,color='k',linewidth=0.8)

    offset = 1
    plt.plot(86.81322,  27.95903, markersize=5, marker='o', color='k') # Pyramid
    plt.annotate('NCO-P',(86.81322+0.01,  27.95903),fontsize=18,color='k')
    plt.plot(86.76, 27.85, markersize=5, marker='o', color='k') # Pyramid
    plt.annotate('P2',(86.76+0.02, 27.85),fontsize=18,color='k')
    plt.plot(86.71456, 27.80239, markersize=5, marker='o', color='k') # Namche
    plt.annotate('Namche',(86.71456-0.22, 27.80239-0.01),fontsize=18,color='k')
    plt.plot(86.72306, 27.69556, markersize=5, marker='o', color='k') 
    plt.annotate('Lukla',(86.72306-0.14, 27.69556-0.01),fontsize=18,color='k')
    plt.plot(86.71, 27.88, markersize=5, marker='o', color='k')  # original lon 86.731
    plt.annotate('P1',(86.731, 27.88),fontsize=18,color='k')
    
    title = f'wind; local time - ' + str(lh) 
    ax0.set_title(title, fontsize = 14)
    
    
    
    #%
    # along-valley profile 
    ax1 = plt.subplot2grid((100, 100), (11, 1), colspan=13, rowspan=4)
    x = np.arange(1,len(lats)+1)
    x = np.tile(x,(15,1))
    
    # horizontal speed
    
    levels = np.arange(-10.0,10.5,0.5)
    w_cont = ax1.contourf(x,zlay,horizontal_wind,levels=levels, cmap='RdBu_r', extend='both')
    # w_cont.cmap.set_under(w_cont.cmap.get_under())
    cbar2 = plt.colorbar(w_cont, fraction = 0.028, location='right', pad = 0.025)
    cbar2.set_label('Along-profile wind speed (m/s)', fontsize=12, rotation=270, labelpad=30, y=0.5)
    # theta
    levels = np.arange(280,330,2)
    theta = ax1.contour(x,zlay,theta_profile, colors='k',alpha=0.4,levels=levels)
    ax1.clabel(theta, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
    # cbar2 = plt.colorbar(w_cont, fraction = 0.028, location='right', pad = 0.10)
    # cbar2.set_label('Along-profile wind speed (m/s)', fontsize=12, rotation=270, labelpad=30, y=0.5)
    
    # gap = 5
    # plt.barbs(x[2:10,::gap], zlay[2:10,::gap],
    #           (horizontal_wind[2:,::gap]*1.94), (w_profile[2:,::gap]*1.94),
    #           length=5, alpha=0.5)
    ax1.set(xlim=(0,x[0,-1]))
    ax1.set(ylim=(0,9000))
    
    
    ax2=ax1.twinx()
    ax2.plot(x[0,:],elev,'k')
    ax2.plot(x[0],elev_e,'b--')
    ax2.plot(x[0],elev_w,'r--')
    # ax2.plot(x[0,:],PBLH+elev,'k--', linewidth=0.5)
    ax2.set(ylim=(0,9000))
    ax1.set(xlim=(1,len(lats)-2))
    ax2.set_yticks([])
    
    # ax1.set_title('Free-profile', fontsize = 12)
    title = f'Wind; local time - ' + str(lh) 
    ax1.set_title(title, fontsize = 14)
   
    
    
    
    # title = 'images/theta_wind_daynight/' + str(count) + '.png'
    #count += 1
    plt.savefig('output_figures/Valley_wind.png')
    #plt.show()
    
 


