#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:00:32 2022

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
import os

#%% plot together
        
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
ds_name = '/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/WRF-CHIMERE_OUTPUT/H2SO4/pH2SO4_PV4.nc'
ds = xr.open_dataset(ds_name)

# get GEOG file
geog = xr.open_dataset('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/downscaling/geog_POVALLEY4.nc')
# ds_mask = geog.LANDUSEF[:,20,:,:]
# sea 
ds_sea = geog.LANDUSEF[:,16,:,:]
# elevation
ds_mask = geog.HGT_M[0,:,:]

# get profiles coordinates
# crss1_coords = [27.40 , 27.40 , 86.6 , 86.8]
# crss2_coords = [27.60 , 27.60 , 86.6, 86.8]
# crss3_coords = [27.80 , 27.80 , 86.55 , 86.85]
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
# df = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/WRF-CHIMERE_OUTPUT/H2SO4/old/free_profiles/cimone_prof4_new7.txt',sep=',',header=None)
df = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/WRF-CHIMERE_OUTPUT/H2SO4/old/free_profiles/cimone_prof4_newlong.txt',sep=',',header=None)

lats = np.flip(np.array(df[0]))
lons = np.flip(np.array(df[1]))
# ridges
# df_e = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/NEPAL/NEW DOMAIN/khumbu_east_ridge_LATLON.txt',sep=',',header=None)
# lats_e = np.flip(np.array(df_e[0]))
# lons_e = np.flip(np.array(df_e[1]))
# df_w = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/NEPAL/NEW DOMAIN/khumbu_west_ridge_LATLON.txt',sep=',',header=None)
# lats_w = np.flip(np.array(df_w[0]))
# lons_w = np.flip(np.array(df_w[1]))

# get chimere lat-lon indexes
chi_lats = np.zeros((len(lats)))
for i in np.arange(len(lats)):
    chi_lats[i] = find_nearest(ds.lat[:,0],lats[i])
chi_lons = np.zeros((len(lons)))
for i in np.arange(len(lons)):
    chi_lons[i] = find_nearest(ds.lon[0,:],lons[i])


#%%


times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times)
local_times = utc_times + DateOffset(hours=2)

# create new auxiliary dimension
ds['local_times'] = pd.DatetimeIndex(local_times)
# swap it with the old time dimension of the main variable  
ds['pH2SO4'] = ds.pH2SO4.swap_dims({'Time':'local_times'})
ds['hlay'] = ds.hlay.swap_dims({'Time':'local_times'})
ds['thlay'] = ds.thlay.swap_dims({'Time':'local_times'})

ds = ds.drop_dims('Time')
# select period of interest
# select period of interest
# ds = ds.sel(local_times=(slice('2017-07-01 00:00:00', '2017-07-09 23:00:00')))
times_subset1 = pd.date_range(start='2017-07-04 00:00:00', end='2017-07-28 23:00:00', freq='1h') # converted to local time
# times_subset2 = pd.date_range(start='2017-07-13 00:00:00', end='2017-07-14 23:00:00', freq='1h') # converted to local time
# times_subset3 = pd.date_range(start='2017-07-18 00:00:00', end='2017-07-20 23:00:00', freq='1h') # converted to local time

# times_subset1 = pd.date_range(start='2017-07-10 00:00:00', end='2017-07-11 23:00:00', freq='1h') # converted to local time
# times_subset2 = pd.date_range(start='2017-07-21 00:00:00', end='2017-07-23 23:00:00', freq='1h') # converted to local time

# times_subset = np.concatenate((times_subset1, times_subset2, times_subset3), axis=None)
# times_subset = np.concatenate((times_subset1, times_subset2), axis=None)
times_subset = np.concatenate((times_subset1), axis=None)

ds = ds.sel(local_times=times_subset)

# compute daily cycle
dailyCycle_chem = ds.pH2SO4.groupby(ds.local_times.dt.hour).mean()
dailyCycle_hlay = ds.hlay.groupby(ds.local_times.dt.hour).mean()
dailyCycle_thlay = ds.thlay.groupby(ds.local_times.dt.hour).mean()
local_hours = dailyCycle_chem.hour.values

# chem_N = dailyCycle_chem.isel(hour=[0,1,2,3,4,5,6,7,8,21,22,23]).mean(dim='hour')
# chem_N = dailyCycle_chem.isel(hour=[3,4,5,6,7,8]).mean(dim='hour')
chem_N = dailyCycle_chem.isel(hour=[5]).mean(dim='hour')
# chem_D = dailyCycle_chem.isel(hour=[9,10,11,12,13,14,15,16,17,18,19,20]).mean(dim='hour')
# chem_D = dailyCycle_chem.isel(hour=[12,13,14,15,16,17]).mean(dim='hour')
chem_D = dailyCycle_chem.isel(hour=[14]).mean(dim='hour')

hlay_N = dailyCycle_hlay.isel(hour=[5]).mean(dim='hour')
hlay_D = dailyCycle_hlay.isel(hour=[14]).mean(dim='hour')

thlay_N = dailyCycle_thlay.isel(hour=[5]).mean(dim='hour')
thlay_D = dailyCycle_thlay.isel(hour=[14]).mean(dim='hour')

count = 1
for daynight in ['NIGHT [5 LT]','DAY [14 LT]']:
    # local_time = local_times[i] 
    print(daynight)
    # concentration and vertical integral
    if daynight == 'DAY [14 LT]':
        chem = chem_D 
        chem_vert_sum_now = chem_D[0,:,:]#.sum(dim='bottom_top').squeeze()
        hlay = hlay_D
        thlay = thlay_D
    elif daynight == 'NIGHT [5 LT]':
        chem = chem_N 
        chem_vert_sum_now = chem_N[0,:,:]#.sum(dim='bottom_top').squeeze()    
        hlay = hlay_N
        thlay = thlay_N
    #%% extract 2d vertical profiles
    
    #### along valley profile num concentration
         
    profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        profile[:,i] = chem[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
    
    # extract topography along valley
    elev = np.zeros((len(lats)))
    for i in np.arange(len(lats)):
        elev[i] = ds_mask[int(chi_lats[i]),int(chi_lons[i])]
    # extract topography on ridges
    # elev_e = np.zeros((len(lats_e)))
    # for i in np.arange(len(lats_e)):
    #     elev_e[i] = ds_mask[int(chi_lats_e[i]),int(chi_lons_e[i])]
    # elev_w = np.zeros((len(lats_w)))
    # for i in np.arange(len(lats_w)):
    #     elev_w[i] = ds_mask[int(chi_lats_w[i]),int(chi_lons_w[i])]
        
    # extract layer heigth along valley
    hlay_now = np.zeros((len(hlay.bottom_top), len(lats)))
    thlay_now = np.zeros((len(thlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        hlay_now[:,i] = hlay[:,int(chi_lats[i]),int(chi_lons[i])]
        thlay_now[:,i] = thlay[:,int(chi_lats[i]),int(chi_lons[i])]
        
    zlay = hlay_now - 0.5 * thlay_now + elev
    
    # along valley profile emissions

    # # extract emissions along valley
    # emis_profile = np.zeros((len(lats)))
    # for i in np.arange(len(lats)):
    #     emis_profile[i] = emis[int(chi_lats[i]),int(chi_lons[i])]
    
    fig = plt.figure(figsize=(80,80))
    
    #%%
    # along-valley profile conc
    ax1 = plt.subplot2grid((100, 100), (12, 1), colspan=13, rowspan=4)
    x = np.arange(1,len(lats)+1)
    x = np.tile(x,(15,1))
    # levels = np.logspace(-0.5,0.75)
    levels = np.linspace(0,2.5,26)
    # cmap_lin = cm.gnuplot2
    # cmap_nonlin = nlcmap(cmap_lin, levels)
    profile_plot = ax1.contourf(x,zlay,profile,levels=levels, cmap = 'YlOrRd',norm=colors.SymLogNorm(linthresh=0.6, linscale=0.2,
                                                    vmin=0, vmax=2.5),extend='max') 
    # profile_plot = ax1.contourf(x,zlay,profile,levels,cmap=cmap_nonlin)
    # cbar = plt.colorbar(profile_plot, fraction = 0.028, location='right', pad = 0.025)
    # cbar.set_label('(μg/m3)', fontsize=12, rotation=270, labelpad=30, y=0.5)
    cbar = plt.colorbar(profile_plot, fraction = 0.028, location='right', pad = 0.06)
    cbar.ax.tick_params(labelsize=12)
    # cbar.set_label('(μg/m3)', fontsize=12, rotation=270, labelpad=30, y=0.5) 
    cbar.ax.set_title('SO$_{4}^{2-}$ \n(μg/m$^{3}$)',fontsize=16, y = 1.1)
    ax1.set(xlim=(1,len(lats)))
    # ax1.set(ylim=(0,3000))
    ax1.set(ylim=(0,4000))
    # ax1.set(ylim=(0,4000))
    
    plt.annotate('CMN',(101.2,1110),fontsize=12,color='k')
    ax1.set_ylabel('m a.s.l.', fontsize=16)

    
    # ax1.plot([chi_lat_cross1, chi_lat_cross1], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    # ax1.plot([chi_lat_cross2, chi_lat_cross2], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    # ax1.plot([chi_lat_cross3, chi_lat_cross3], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    # ax1.plot([chi_lat_cross4, chi_lat_cross4], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    
    ax2=ax1.twinx()
    ax2.plot(x[0],elev,'k')
    # ax2.plot(x[0],elev_e,'b--')
    # ax2.plot(x[0],elev_w,'r--')

    # ax2.set(ylim=(0,3000))
    ax2.set(ylim=(0,4000))
    ax2.set_yticks([])
    
    # ax1.set_title('along-valley profile', fontsize = 12)
    title = daynight
    ax1.set_title(title, fontsize = 14)

    
    
    
    title = 'images/H2SO4_daynight/' + str(count) + '.png'
    count += 1
    # plt.savefig(title)
    plt.show()
    
 


