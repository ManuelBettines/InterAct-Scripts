#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 18:23:11 2022

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


# for i in out.*
# do 
# echo "Extracting variables for file:" ${i}
# ncks -v lat,lon,Times,hlay,thlay,pH2SO4 ${i} ${i}_sub
# done 

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
df = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/WRF-CHIMERE_OUTPUT/H2SO4/old/free_profiles/cimone_prof4_newlong.txt',sep=',',header=None)

# df = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/WRF-CHIMERE_OUTPUT/H2SO4/old/free_profiles/cimone_prof4_new_ori.txt',sep=',',header=None)
# df = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/WRF-CHIMERE_OUTPUT/H2SO4/old/free_profiles/cimone_prof4_new7.txt',sep=',',header=None)
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

# chi_lats_e = np.zeros((len(lats_e)))
# for i in np.arange(len(lats_e)):
#     chi_lats_e[i] = find_nearest(ds.lat[:,0],lats_e[i])
# chi_lons_e = np.zeros((len(lons_e)))
# for i in np.arange(len(lons_e)):
#     chi_lons_e[i] = find_nearest(ds.lon[0,:],lons_e[i])
    
# chi_lats_w = np.zeros((len(lats_w)))
# for i in np.arange(len(lats_w)):
#     chi_lats_w[i] = find_nearest(ds.lat[:,0],lats_w[i])
# chi_lons_w = np.zeros((len(lons_w)))
# for i in np.arange(len(lons_w)):
#     chi_lons_w[i] = find_nearest(ds.lon[0,:],lons_w[i])


# find along-valley intersection with cross profile
# chi_lat_cross1 = find_nearest(lats,crss1_coords[0])
# chi_lat_cross2 = find_nearest(lats,crss2_coords[0])
# chi_lat_cross3 = find_nearest(lats,crss3_coords[0])
# chi_lat_cross4 = find_nearest(lats,crss4_coords[0])

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
# ds = ds.sel(local_times=slice('2017-07-07 09:00:00', '2017-07-07 20:00:00'))

times_subset1 = pd.date_range(start='2017-07-06 06:00:00', end='2017-07-06 16:00:00', freq='1h') # converted to local time
# times_subset2 = pd.date_range(start='2017-07-18 00:00:00', end='2017-07-18 23:00:00', freq='1h') # converted to local time
# times_subset3 = pd.date_range(start='2017-07-26 00:00:00', end='2017-07-28 23:00:00', freq='1h') # converted to local time

# times_subset1 = pd.date_range(start='2017-07-09 00:00:00', end='2017-07-14 23:00:00', freq='1h') # converted to local time
# times_subset1 = pd.date_range(start='2017-07-24 04:00:00', end='2017-07-24 05:00:00', freq='1h') # converted to local time


# times_subset = np.concatenate((times_subset1, times_subset2, times_subset3), axis=None)
times_subset = np.concatenate((times_subset1), axis=None)
# times_subset = np.concatenate((times_subset1), axis=None)
ds = ds.sel(local_times=times_subset)


# idxTimei = list(local_times.astype(str)).index('2017-07-07 00:00:00') 
# idxTimef = list(local_times.astype(str)).index('2017-07-08 00:00:00') 


count = 1
for local_time in ds.local_times :
    
    # local_time = local_times[i] 
    print(local_time)
    # concentration 
    chem_now = ds.pH2SO4.sel(local_times=local_time)
    hlay = ds.hlay.sel(local_times=local_time)
    thlay = ds.thlay.sel(local_times=local_time)
    # chem = ds.pH2SO4[i,:,:,:] 
    # vertical integral or lowermost model layer
    chem_vert_sum_now = ds.pH2SO4.sel(local_times=local_time)[0,:,:]
    # chem_vert_sum_now = ds.pH2SO4[i,0,:,:]#.sum(dim='bottom_top').squeeze()
    

    #%% extract 2d vertical profiles
    
    #### along valley profile num concentration
         
    profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        profile[:,i] = chem_now[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
    
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
    hlay_now = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    thlay_now = np.zeros((len(ds.thlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        hlay_now[:,i] = hlay[:,int(chi_lats[i]),int(chi_lons[i])]
        thlay_now[:,i] = thlay[:,int(chi_lats[i]),int(chi_lons[i])]
        
    zlay = hlay_now - 0.5 * thlay_now + elev
    
    # along valley profile emissions

    # # extract emissions along valley
    # emis_profile = np.zeros((len(lats)))
    # for i in np.arange(len(lats)):
    #     emis_profile[i] = emis[int(chi_lats[i]),int(chi_lons[i])]
    
    
    #### cross valley profiles 
    
    #1
 
    # crss_profile_1 = np.zeros((len(ds.hlay.bottom_top), (crss1_lon_idx_2-crss1_lon_idx_1)))
    # crss_profile_1[:,:] = chem[:,int(crss1_lat_idx_1),int(crss1_lon_idx_1):int(crss1_lon_idx_2)].squeeze()
    
    # # extract topography cross valley
    # crss_elev_1 = np.zeros((crss1_lon_idx_2-crss1_lon_idx_1))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    # crss_elev_1[:] = ds_mask[int(crss1_lat_idx_1),int(crss1_lon_idx_1):int(crss1_lon_idx_2)]
    
    # # extract layer heigth cross valley
    # crss_hlay_1 = np.zeros((len(ds.hlay.bottom_top), (crss1_lon_idx_2-crss1_lon_idx_1)))
    # crss_thlay_1 = np.zeros((len(ds.thlay.bottom_top), (crss1_lon_idx_2-crss1_lon_idx_1)))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    # crss_hlay_1[:,:] = ds.hlay[i,:,int(crss1_lat_idx_1),int(crss1_lon_idx_1):int(crss1_lon_idx_2)]
    # crss_thlay_1[:,:] = ds.thlay[i,:,int(crss1_lat_idx_1),int(crss1_lon_idx_1):int(crss1_lon_idx_2)]
        
    # crss_zlay_1 = crss_hlay_1 - 0.5 * crss_thlay_1 + crss_elev_1
    
    # extract emissions cross valley
    # crss_emis_1 = np.zeros((crss1_lon_idx_2-crss1_lon_idx_1))
    # crss_emis_1[:] = emis[int(crss1_lat_idx_1),int(crss1_lon_idx_1):int(crss1_lon_idx_2)]
    
    #2
    
    # crss_profile_2 = np.zeros((len(ds.hlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    #     # print(i)
    # crss_profile_2[:,:] = chem[:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)].squeeze()
    
    # # extract topography cross valley
    # crss_elev_2 = np.zeros((crss2_lon_idx_2-crss2_lon_idx_1))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    # crss_elev_2[:] = ds_mask[int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)]
    
    # # extract layer heigth cross valley
    # crss_hlay_2 = np.zeros((len(ds.hlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    # crss_thlay_2 = np.zeros((len(ds.thlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    # crss_hlay_2[:,:] = ds.hlay[i,:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)]
    # crss_thlay_2[:,:] = ds.thlay[i,:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)]
        
    # crss_zlay_2 = crss_hlay_2 - 0.5 * crss_thlay_2 + crss_elev_2
    
    # extract emissions cross valley
    # crss_emis_2 = np.zeros((crss2_lon_idx_2-crss2_lon_idx_1))
    # crss_emis_2[:] = emis[int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)]
    
    #3

    # crss_profile_3 = np.zeros((len(ds.hlay.bottom_top), (crss3_lat_idx_2-crss3_lat_idx_1)))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    #     # print(i)
    # crss_profile_3[:,:] = chem[:,int(crss3_lon_idx_1),int(crss3_lat_idx_1):int(crss3_lat_idx_2)].squeeze()
    
    # # extract topography cross valley
    # crss_elev_3 = np.zeros((crss3_lat_idx_2-crss3_lat_idx_1))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    # crss_elev_3[:] = ds_mask[int(crss3_lon_idx_1),int(crss3_lat_idx_1):int(crss3_lat_idx_2)]
    
    # # extract layer heigth cross valley
    # crss_hlay_3 = np.zeros((len(ds.hlay.bottom_top), (crss3_lat_idx_2-crss3_lat_idx_1)))
    # crss_thlay_3 = np.zeros((len(ds.thlay.bottom_top), (crss3_lat_idx_2-crss3_lat_idx_1)))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    # crss_hlay_3[:,:] = ds.hlay[i,:,int(crss3_lon_idx_1),int(crss3_lat_idx_1):int(crss3_lat_idx_2)]
    # crss_thlay_3[:,:] = ds.thlay[i,:,int(crss3_lon_idx_1),int(crss3_lat_idx_1):int(crss3_lat_idx_2)]
        
    # crss_zlay_3 = crss_hlay_3 - 0.5 * crss_thlay_3 + crss_elev_3
    
    # extract emissions cross valley
    # crss_emis_3 = np.zeros((crss3_lon_idx_2-crss3_lon_idx_1))
    # crss_emis_3[:] = emis[int(crss3_lat_idx_1),int(crss3_lon_idx_1):int(crss3_lon_idx_2)]
    
    #4
    
    # crss_profile_4 = np.zeros((len(ds.hlay.bottom_top), (crss4_lon_idx_2-crss4_lon_idx_1)))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    #     # print(i)
    # crss_profile_4[:,:] = chem[:,int(crss4_lat_idx_1),int(crss4_lon_idx_1):int(crss4_lon_idx_2)].squeeze()
    
    # # extract topography cross valley
    # crss_elev_4 = np.zeros((crss4_lon_idx_2-crss4_lon_idx_1))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    # crss_elev_4[:] = ds_mask[int(crss4_lat_idx_1),int(crss4_lon_idx_1):int(crss4_lon_idx_2)]
    
    # # extract layer heigth cross valley
    # crss_hlay_4 = np.zeros((len(ds.hlay.bottom_top), (crss4_lon_idx_2-crss4_lon_idx_1)))
    # crss_thlay_4 = np.zeros((len(ds.thlay.bottom_top), (crss4_lon_idx_2-crss4_lon_idx_1)))
    # # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    # crss_hlay_4[:,:] = ds.hlay[i,:,int(crss4_lat_idx_1),int(crss4_lon_idx_1):int(crss4_lon_idx_2)]
    # crss_thlay_4[:,:] = ds.thlay[i,:,int(crss4_lat_idx_1),int(crss4_lon_idx_1):int(crss4_lon_idx_2)]
        
    # crss_zlay_4 = crss_hlay_4 - 0.5 * crss_thlay_4 + crss_elev_4
    
    # extract emissions cross valley
    # crss_emis_4 = np.zeros((crss4_lon_idx_2-crss4_lon_idx_1))
    # crss_emis_4[:] = emis[int(crss4_lat_idx_1),int(crss4_lon_idx_1):int(crss4_lon_idx_2)]
    
    #%%
    
    fig = plt.figure(figsize=(80,80))
    
    #%%
    ax0 = plt.subplot2grid((100, 100), (1, 1), colspan=10, rowspan=10, projection=ccrs.PlateCarree())
    # ax0.set_extent([9.9, 10.8, 43.7, 44.3])
    # ax0.set_extent([9.9, 11, 43.6, 44.45])
    # levels = np.array([0,0.1,0.2,0.3,0.4,0.5,0.8,1,1.2,1.4,1.6,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,12])#,14,16,18,20,24,28,32,36,40,44,48,52])
    # levels = np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10])
    levels = np.linspace(0,6,21)
    # cmap_lin = cm.gnuplot2
    # cmap_nonlin = nlcmap(cmap_lin, levels)
    chem_plot = ax0.contourf(ds.lon,ds.lat, chem_vert_sum_now[:,:], transform=ccrs.PlateCarree(), levels=levels, cmap = 'YlOrRd',norm=colors.SymLogNorm(linthresh=0.6, linscale=0.2,
                                                    vmin=0, vmax=4),extend='max',) 
    cbar = plt.colorbar(chem_plot, fraction = 0.028, location='right', pad = 0.12)
    cbar.set_label('vertical column (μg/m2)', fontsize=12, rotation=270, labelpad=30, y=0.5) 
    
    for lat,lon in zip(lats,lons):
        plt.plot(lon, lat, markersize=1.2, marker='o', color='k')
        
    # for lat,lon in zip(lats_e,lons_e):
    #     plt.plot(lon, lat, markersize=1.2, marker='o', color='b')
    # for lat,lon in zip(lats_w,lons_w):
    #     plt.plot(lon, lat, markersize=1.2, marker='o', color='r')
        
    # ax0.plot([crss1_coords[2], crss1_coords[3]], [crss1_coords[0], crss1_coords[1]],
    #           color='w', linewidth=1, marker='o', markersize=3,
    #           transform=ccrs.PlateCarree())
    # ax0.plot([crss2_coords[2], crss2_coords[3]], [crss2_coords[0], crss2_coords[1]],
    #           color='w', linewidth=1, marker='o', markersize=3,
    #           transform=ccrs.PlateCarree())
    # ax0.plot([crss3_coords[2], crss3_coords[3]], [crss3_coords[0], crss3_coords[1]],
    #           color='k', linewidth=1, marker='o', markersize=3,
    #           transform=ccrs.PlateCarree())
    # ax0.plot([crss4_coords[2], crss4_coords[3]], [crss4_coords[0], crss4_coords[1]],
    #           color='w', linewidth=1, marker='o', markersize=3,
    #           transform=ccrs.PlateCarree())
    # plt.plot(86.81322, 27.95903, markersize=10, marker='x', color='w') # Pyramid
    # plt.plot(86.71456, 27.80239, markersize=10, marker='x', color='w') # Namche
    # plt.plot(86.72306, 27.69556, markersize=10, marker='x', color='w') # Lukla
    
    levels = np.arange(10., 4000., 100.)
    geogr = ax0.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.2,extend='both')
    ax0.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
    ax0.coastlines(color='k', linewidth = 1); 
    ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
    ax0.add_feature(cartopy.feature.STATES, linewidth = 0.6, alpha = 0.3)
    ax0.add_feature(cartopy.feature.RIVERS, linewidth = 0.6,color='blue', alpha = 0.6)
    gl = ax0.gridlines(draw_labels=True,alpha=0.5);
    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}
    
    title = 'H2SO4; local time - ' + str(pd.to_datetime(local_time.values)) 
    ax0.set_title(title, fontsize = 14)
    
    #%%
    # ax9 = plt.subplot2grid((100, 100), (1, 1), colspan=10, rowspan=10, projection=ccrs.PlateCarree())
    # ax9.set_extent([85.6, 88.55, 26.4, 29.2])
    # levels = np.array([0,1e-10,2e-10,5e-10, 1e-9,2e-9,3e-9,5e-9,7e-9, 1e-8,2e-8,3e-8,5e-8,7e-8, 1e-7,2e-7,5e-7,1e-6])
    # # levels = np.arange(0,1e-7,2e-9)
    # cmap_lin = cm.YlGn
    # cmap_nonlin = nlcmap(cmap_lin, levels)
    # chem_plot = ax9.contourf(ds.lon,ds.lat, emis[:,:], transform=ccrs.PlateCarree(),levels=levels, cmap=cmap_nonlin); 
    # cbar = plt.colorbar(chem_plot, fraction = 0.028, location='right', pad = 0.10)
    # cbar.set_label('alpha-pinene emissions (g/m2/s)', fontsize=12, rotation=270, labelpad=30, y=0.5) 
    
    # for lat,lon in zip(lats,lons):
    #     plt.plot(lon, lat, markersize=1.2, marker='o', color='k')
        
    # ax9.plot([crss1_coords[2], crss1_coords[3]], [crss1_coords[0], crss1_coords[1]],
    #           color='k', linewidth=1, marker='o', markersize=3,
    #           transform=ccrs.PlateCarree())
    # ax9.plot([crss2_coords[2], crss2_coords[3]], [crss2_coords[0], crss2_coords[1]],
    #           color='k', linewidth=1, marker='o', markersize=3,
    #           transform=ccrs.PlateCarree())
    # ax9.plot([crss3_coords[2], crss3_coords[3]], [crss3_coords[0], crss3_coords[1]],
    #           color='k', linewidth=1, marker='o', markersize=3,
    #           transform=ccrs.PlateCarree())
        
    # levels = np.arange(10., 9000., 500.)
    # geogr = ax9.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.2)
    # ax9.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
    # ax9.coastlines(color='k', linewidth = 1); 
    # ax9.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
    # ax9.add_feature(cartopy.feature.STATES, linewidth = 0.6, alpha = 0.3)
    # ax9.add_feature(cartopy.feature.RIVERS, linewidth = 0.6,color='blue', alpha = 0.6)
    # gl = ax9.gridlines(draw_labels=True,alpha=0.5);
    # gl.xlabel_style = {'size': 10, 'color': 'k'}
    # gl.ylabel_style = {'size': 10, 'color': 'k'}
    
    # title = str(local_time[0].year) +'_'+ str(local_time[0].month) +'_'+ str(local_time[0].day) +'_'+ f"{local_time.hour[0]:02d}" +'00'
    # ax9.set_title(title, fontsize = 12)
    
    
    #%%
    fig = plt.figure(figsize=(80,80))

    # along-valley profile conc
    ax1 = plt.subplot2grid((100, 100), (1, 1), colspan=13, rowspan=4)
    x = np.arange(1,len(lats)+1)
    x = np.tile(x,(15,1))
    # levels = np.array([0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20])
    # levels = np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
    # levels = np.logspace(-0.5,0.75)
    levels = np.linspace(0,4,26)
    # cmap_lin = cm.gnuplot2
    # cmap_nonlin = nlcmap(cmap_lin, levels)
    
    profile_plot = ax1.contourf(x,zlay,profile,levels=levels, cmap = 'YlOrRd',norm=colors.SymLogNorm(linthresh=0.6, linscale=0.2,
                                                    vmin=0, vmax=3.5),extend='max') 
    cbar = plt.colorbar(profile_plot, fraction = 0.028, location='right', pad = 0.06)
    cbar.ax.tick_params(labelsize=12)
    # cbar.set_label('(μg/m3)', fontsize=12, rotation=270, labelpad=30, y=0.5) 
    cbar.ax.set_title('(μg/m$^{3}$)',fontsize=16, y = 1.1)
    # cbar.set_label('(μg/m$^{3}$)', fontsize=12, rotation=270, labelpad=30, y=0.5)
    ax1.set(xlim=(1,len(lats)))
    ax1.set(ylim=(0,4000))
    plt.xticks(ticks=[0,25,50,75,100,125,150,175])
    
    # ax1.plot([chi_lat_cross1, chi_lat_cross1], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    # ax1.plot([chi_lat_cross2, chi_lat_cross2], [0, 10000],
    # #           color='k', linewidth=1, marker='o', markersize=3)
    # ax1.plot([chi_lat_cross3, chi_lat_cross3], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    # ax1.plot([chi_lat_cross4, chi_lat_cross4], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    ax1.set_ylabel('m a.s.l.', fontsize=16)
    ax2=ax1.twinx()
    ax2.plot(x[0],elev,'k')
    # ax2.plot(x[0],elev_e,'b--')
    # ax2.plot(x[0],elev_w,'r--')
    # ax2.set_ylabel('alpha-pinene emissions (g/m2/s)', fontsize=12)
    ax2.set(ylim=(0,4000), xlim=(1,len(lats)))
    ax2.set_yticks([])
    plt.annotate('CMN',(101.2,1110),fontsize=12,color='k')
    # ax1.set_title('along-valley profile', fontsize = 12)
    # title = 'H2SO4; local time - ' + str(pd.to_datetime(local_time.values) + DateOffset(hours=0)) 
    title = 'SO$_{4}^{2-}$ - ' + str(pd.to_datetime(local_time.values))
    ax1.set_title(title, fontsize = 14)
    ax1.set_title(title, fontsize = 18)
    #%%
    # # along-valley profile emis
    # ax10 = plt.subplot2grid((100, 100), (11, 1), colspan=9, rowspan=2)
    
    # ax10.plot(elev,'k')
    # # ax11.set_ylabel('RH(%)', color="b", fontsize=18)
    # ax10.set(ylim=(0,10000))
    # # ax11.set_yticks([])
    
    
    # ax11=ax10.twinx()
    # x = np.arange(1,len(lats)+1)
    # x = np.tile(x,(15,1))
    # levels = np.array([0,1e-10,2e-10,5e-10, 1e-9,2e-9,3e-9,5e-9,7e-9, 1e-8,2e-8,3e-8,5e-8,7e-8, 1e-7,2e-7,5e-7,1e-6])
    # cmap_lin = cm.YlGn
    # cmap_nonlin = nlcmap(cmap_lin, levels)
    # ratio_emis_elev = 2e-7/10000
    # profile_plot = ax11.bar(x[0],emis_profile,bottom=elev*ratio_emis_elev,color='green',width=1.5)
    # # cbar = plt.colorbar(profile_plot, fraction = 0.028, location='right', pad = 0.10)
    # # cbar.set_label('num concentration (#/cm3)', fontsize=12, rotation=270, labelpad=30, y=0.5)
    # ax11.set(xlim=(0,len(lats)+1))
    # ax11.set(ylim=(0,2e-7))
    
    # ax11.plot([chi_lat_cross1, chi_lat_cross1], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    # ax11.plot([chi_lat_cross2, chi_lat_cross2], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    # ax11.plot([chi_lat_cross3, chi_lat_cross3], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    
    # ax11.set_ylabel('a-pinene emissions (g/m2/s)', fontsize=12, rotation=270,labelpad=30, y=0.5)

    # ax10.set_title('along-valley profile', fontsize = 12)
    
    
    #%%
    # # cross-section 1 conc
    # ax3 = plt.subplot2grid((100, 100), (11, 14), colspan=9, rowspan=2)
    
    # x = np.tile(geog.XLONG_M[0,0,crss3_lon_idx_1:crss3_lon_idx_2],(15,1))
    
    # # levels = np.array([0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,5,7.5,10,15,20])
    # cmap_lin = cm.magma_r
    # cmap_nonlin = nlcmap(cmap_lin, levels)
    
    # crss_profile_plot = ax3.contourf(x,crss_zlay_1,crss_profile_1,cmap=cmap_nonlin,levels=levels)
    # # cbar = plt.colorbar(crss_profile_plot, fraction = 0.028, location='right', pad = 0.10)
    # ax3.set(xlim=(x[0,0],x[0,-1]))
    # ax3.set(ylim=(0,10000))
    # ax4=ax3.twinx()
    # ax4.plot(x[0],crss_elev_1,'k')
    # ax4.set(ylim=(0,10000))
    # ax4.set_yticks([])
    # # ax3.set_title('section 1 ', fontsize = 12)
    
    #%%
    # # cross-section 1 emis
    # ax12 = plt.subplot2grid((100, 100), (20, 1), colspan=9, rowspan=2)
    
    
    # ax12.plot(x[0],crss_elev_1,'k')
    # ax12.set(ylim=(0,10000))
    # # ax12.set_yticks([])
    # ax12.set(xlim=(x[0,0],x[0,-1]))
    # ax12.set_title('section 1 ', fontsize = 12)
    
    # ax13=ax12.twinx()
    # x = np.tile(geog.XLONG_M[0,0,crss1_lon_idx_1:crss1_lon_idx_2],(15,1))
    # crss_profile_plot = ax13.bar(x[0], crss_emis_1, bottom=crss_elev_1*ratio_emis_elev,color='green', width=0.008)
    # # # cbar = plt.colorbar(crss_profile_plot, fraction = 0.028, location='right', pad = 0.10)
    # ax13.set_yticks([])

    # ax13.set(ylim=(0,2e-7))
    
    
    #%%
    # # cross-section 2 conc 
    # ax5 = plt.subplot2grid((100, 100), (8, 14), colspan=9, rowspan=2)
    
    # crss_profile_plot = ax5.contourf(x,crss_zlay_2,crss_profile_2,cmap=cmap_nonlin,levels=levels)
    # # cbar = plt.colorbar(crss_profile_plot, fraction = 0.028, location='right', pad = 0.10)
    # ax5.set(xlim=(x[0,0],x[0,-1]))
    # ax5.set(ylim=(0,10000))
    # ax6=ax5.twinx()
    # ax6.plot(x[0],crss_elev_2,'k')
    # ax6.set(ylim=(0,10000))
    # ax5.set_xticks([])
    # ax6.set_yticks([])
    # # ax6.set_title('section 2 ', fontsize = 12)
    
    #%%
    # # # cross-section 2 emis
    # # ax14 = plt.subplot2grid((100, 100), (17, 1), colspan=9, rowspan=2)
    
    
    # # ax14.plot(x[0],crss_elev_2,'k')
    # # ax14.set(ylim=(0,10000))
    # # ax14.set(xlim=(x[0,0],x[0,-1]))
    # # ax14.set_title('section 2 ', fontsize = 12)
    # # ax14.set_yticks([])

    # # ax15=ax14.twinx()
    # # x = np.tile(geog.XLONG_M[0,0,crss2_lon_idx_1:crss2_lon_idx_2],(15,1))
    # # crss_profile_plot = ax15.bar(x[0], crss_emis_2, bottom=crss_elev_2*ratio_emis_elev,color='green', width=0.008)
    # # # # cbar = plt.colorbar(crss_profile_plot, fraction = 0.028, location='right', pad = 0.10)
    # # ax15.set_yticks([])
    # # ax15.set(ylim=(0,2e-7))
    
    #%%
    # cross-section 3 conc
    # ax7 = plt.subplot2grid((100, 100), (3, 14), colspan=7, rowspan=5)
        
    # x = np.tile(geog.XLAT_M[0,crss3_lat_idx_1:crss3_lat_idx_2,0],(15,1))
    
    # levels = np.array([0,0.1,0.2,0.3,0.4,0.5,0.8,1,1.2,1.4,1.6,2,2.5,3,3.5,4,4.5,5,6,7,8])
    # cmap_lin = cm.gnuplot2
    # cmap_nonlin = nlcmap(cmap_lin, levels)
    
    # crss_profile_plot = ax7.contourf(x,crss_zlay_3,crss_profile_3,cmap=cmap_nonlin,levels=levels)
    # # cbar = plt.colorbar(crss_profile_plot, fraction = 0.028, location='right', pad = 0.10)
    # ax7.set(xlim=(x[0,0],x[0,-1]))
    # ax7.set(ylim=(0,3000))
    # ax8=ax7.twinx()
    # ax8.plot(x[0],crss_elev_3,'k')
    # ax8.set(ylim=(0,3000))
    # ax7.set_xticks([])
    # ax8.set_yticks([])
    # ax8.set_title('section 3 ', fontsize = 12)
    
    # %%
    # # # cross-section 3 emis
    # # ax16 = plt.subplot2grid((100, 100), (14, 1), colspan=9, rowspan=2)
    
    
    # # ax16.plot(x[0],crss_elev_3,'k')
    # # ax16.set(ylim=(0,10000))
    # # ax16.set(xlim=(x[0,0],x[0,-1]))
    # # ax16.set_title('section 3 ', fontsize = 12)
    # # ax16.set_yticks([])

    # # ax17=ax16.twinx()
    # # x = np.tile(geog.XLONG_M[0,0,crss3_lon_idx_1:crss3_lon_idx_2],(15,1))
    # # crss_profile_plot = ax17.bar(x[0], crss_emis_3, bottom=crss_elev_3*ratio_emis_elev,color='green', width=0.008)
    # # # # cbar = plt.colorbar(crss_profile_plot, fraction = 0.028, location='right', pad = 0.10)
    # # # ax17.set_yticks([])
    # # ax17.set(ylim=(0,2e-7))  
    
    #%%
    # # cross-section 4 conc 
    # ax5 = plt.subplot2grid((100, 100), (2, 14), colspan=9, rowspan=2)
           
    # crss_profile_plot = ax5.contourf(x,crss_zlay_4,crss_profile_4,cmap=cmap_nonlin,levels=levels)
    # # cbar = plt.colorbar(crss_profile_plot, fraction = 0.028, location='right', pad = 0.10)
    # ax5.set(xlim=(x[0,0],x[0,-1]))
    # ax5.set(ylim=(0,10000))
    # ax5.ylabel='try'
    # ax6=ax5.twinx()
    # ax6.plot(x[0],crss_elev_4,'k')
    # ax6.set(ylim=(0,10000))
    # ax5.set_xticks([])
    # ax6.set_yticks([])
    # # ax6.set_title('section 4 ', fontsize = 12)
    
    
    
    title = 'images/H2SO4/' + str(count) + '.png'
    count += 1
    # plt.savefig(title)
    # plt.show()
    
 


