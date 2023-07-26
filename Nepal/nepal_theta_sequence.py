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
ds_name = '/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/NEPAL/NEW DOMAIN/WRF-CHIMERE_output/BASE_CASE/out_total.NEPAL.BASE_CASE_WIND.nc'
ds = xr.open_dataset(ds_name)

# get GEOG file
geog = xr.open_dataset('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/NEPAL/NEW DOMAIN/geog_NEPALR4.nc')
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
df = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/NEPAL/NEW DOMAIN/khumbu_latlon_new.txt',sep=',',header=None)
lats = np.flip(np.array(df[0]))
lons = np.flip(np.array(df[1]))
# ridges
df_e = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/NEPAL/NEW DOMAIN/khumbu_east_ridge_LATLON.txt',sep=',',header=None)
lats_e = np.flip(np.array(df_e[0]))
lons_e = np.flip(np.array(df_e[1]))
df_w = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/NEPAL/NEW DOMAIN/khumbu_west_ridge_LATLON.txt',sep=',',header=None)
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

# times_subset1 = pd.date_range(start='2017-07-04 00:00:00', end='2017-07-28 23:00:00', freq='1h') # converted to local time
# times_subset2 = pd.date_range(start='2017-07-18 00:00:00', end='2017-07-18 23:00:00', freq='1h') # converted to local time
# times_subset3 = pd.date_range(start='2017-07-26 00:00:00', end='2017-07-28 23:00:00', freq='1h') # converted to local time

times_subset1 = pd.date_range(start='2014-12-03 00:00:00', end='2014-12-10 23:00:00', freq='1h') # converted to local time
# times_subset2 = pd.date_range(start='2017-07-21 00:00:00', end='2017-07-25 23:00:00', freq='1h') # converted to local time


# times_subset = np.concatenate((times_subset1, times_subset2, times_subset3), axis=None)
# times_subset = np.concatenate((times_subset1, times_subset2), axis=None)
times_subset = np.concatenate((times_subset1), axis=None)
ds = ds.sel(local_times=times_subset)


count = 1
for local_time in ds.local_times :
    
    # local_time = local_times[i] 
    print(local_time)
    # concentration 
    v_now = ds.winm.sel(local_times=local_time)
    u_now = ds.winz.sel(local_times=local_time)
    # winw_now = ds.winw.sel(local_times=local_time)
    pres_now = ds.pres.sel(local_times=local_time)
    temp_now = ds.temp.sel(local_times=local_time)
    # pblh_now = ds.pblh.sel(local_times=local_time)
    
    hlay_now = ds.hlay.sel(local_times=local_time)
    thlay_now = ds.thlay.sel(local_times=local_time)
    # chem = ds.pH2SO4[i,:,:,:] 
    # vertical integral or lowermost model layer
    # chem_vert_sum_now = ds.pH2SO4.sel(local_times=local_time)[0,:,:]
    # chem_vert_sum_now = ds.pH2SO4[i,0,:,:]#.sum(dim='bottom_top').squeeze()
    

    #%% extract 2d vertical profiles
    
    #### along valley profile num concentration
         
    # profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    # for i in np.arange(len(lats)):
    #     profile[:,i] = chem_now[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
        
    u_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        u_profile[:,i] = u_now[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
    
    v_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        v_profile[:,i] = v_now[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()

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
        t_profile[:,i] = temp_now[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
      
    p_profile = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        p_profile[:,i] = pres_now[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
        
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
        hlay_profile[:,i] = hlay_now[:,int(chi_lats[i]),int(chi_lons[i])]
        thlay_profile[:,i] = thlay_now[:,int(chi_lats[i]),int(chi_lons[i])]
        
    zlay = hlay_profile - 0.5 * thlay_profile + elev
    
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
    
    ax0 = plt.subplot2grid((100, 100), (1, 1), colspan=13, rowspan=10, projection=ccrs.PlateCarree())
    ax0.set_extent([86.14, 87.43, 27.295, 28.28])
    # ax0.set_extent([9.9, 11.4, 43.6, 44.6])
    
    # free-profiles
    
    for lat,lon in zip(lats,lons):
        plt.plot(lon, lat, markersize=1.2, marker='o', color='k')
    for lat,lon in zip(lats_e,lons_e):
        plt.plot(lon, lat, markersize=1.2, marker='o', color='b')
    for lat,lon in zip(lats_w,lons_w):
        plt.plot(lon, lat, markersize=1.2, marker='o', color='r')
    
    levels = np.arange(10., 9000., 250.)
    geogr = plt.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.3)
    ax0.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
    # filled = plt.contourf(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, cmap='binary', alpha=0.7)
    ax0.coastlines(color='k', linewidth = 1); 
    # ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
    # ax0.add_feature(cartopy.feature.STATES, linewidth = 0.6, alpha = 0.3)
    # ax0.add_feature(cartopy.feature.RIVERS, linewidth = 0.6,color='blue', alpha = 0.6)
    gl = ax0.gridlines(draw_labels=True,alpha=0.5);
    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    

    levels = np.arange(0,15.5,0.5)
    # cmap_lin = cm.gnuplot2
    # cmap_nonlin = nlcmap(cmap_lin, levels)    
    plot = ax0.contourf(ds.lon,ds.lat, np.sqrt(np.square(u_now[0,:,:])+np.square(v_now[0,:,:])), transform=ccrs.PlateCarree(), levels = levels, cmap='Reds') 
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
              (np.array(u_now[0,:,:])), (np.array(v_now[0,:,:])),
              transform=ccrs.PlateCarree(),density=2,color='k',linewidth=0.8)
    
    
    title = 'zonal wind; local time - ' + str(pd.to_datetime(local_time.values)) 
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
    # along-valley profile 
    ax1 = plt.subplot2grid((100, 100), (12, 1), colspan=13, rowspan=4)
    x = np.arange(1,len(lats)+1)
    x = np.tile(x,(15,1))
    
    # horizontal speed
    
    levels = np.arange(-10.5,10.5,0.5)
    w_cont = ax1.contourf(x,zlay,horizontal_wind,levels=levels, cmap='RdBu_r')
    cbar2 = plt.colorbar(w_cont, fraction = 0.028, location='right', pad = 0.025)
    cbar2.set_label('Along-profile wind speed (m/s)', fontsize=12, rotation=270, labelpad=30, y=0.5)
    # theta
    levels = np.arange(280,350,2)
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
    ax2.set(xlim=(1,x[0,-1]))
    ax2.set_yticks([])
    
    # ax1.set_title('Free-profile', fontsize = 12)
    title = 'zonal wind; local time - ' + str(pd.to_datetime(local_time.values)) 
    ax1.set_title(title, fontsize = 14)
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
    
    
    
    title = 'images/theta_wind/' + str(count) + '.png'
    count += 1
    # plt.savefig(title)
    plt.show()
    
 


