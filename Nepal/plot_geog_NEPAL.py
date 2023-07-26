#!/usr/bin/env python
# coding: utf-8

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

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby+1500, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom', fontsize=30)

#%%

# get valleys location
# df1 = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/NEPAL/valley_profiles/gauri_latlon.txt',sep=' ',header=None)
# lats1 = df1[0]
# lons1 = df1[1]

df2 = pd.read_csv('khumbu_latlon_extended.txt',sep=',',header=None)
lats2 = df2[0]
lons2 = df2[1]
# ridges
df2_e = pd.read_csv('khumbu_east_ridge_LATLON.txt',sep=',',header=None)
lats2_e = df2_e[0]
lons2_e = df2_e[1]
df2_w = pd.read_csv('khumbu_west_ridge_LATLON.txt',sep=',',header=None)
lats2_w = df2_w[0]
lons2_w = df2_w[1]

# df3 = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/NEPAL/valley_profiles/makalu_latlon.txt',sep=' ',header=None)
# lats3 = df3[0]
# lons3 = df3[1]
# df4 = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/NEPAL/valley_profiles/kancha_latlon.txt',sep=' ',header=None)
# lats4 = df4[0]
# lons4 = df4[1]

# get GEOG file
geog = xr.open_dataset('geog_NEPALR4.nc')
# ds_mask = geog.LANDUSEF[:,20,:,:]
# sea 
ds_sea = geog.LANDUSEF[:,16,:,:]
# elevation
ds_mask = geog.HGT_M[0,:,:]

PYR_loc = np.array([86.81322, 27.95903]) # Pyramid
# PYR_loc = np.array([86.76, 27.85]) # Pyramid valley
# PYR_loc = np.array([86.71456, 27.80239]) # Namche
# PYR_loc = np.array([86.72306, 27.69556]) # Lukla
# PYR_loc = np.array([86.715, 27.702]) # Valley

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon_PYR = find_nearest(geog.XLONG_M[0,10,:], PYR_loc[0]) 
idx_lat_PYR = find_nearest(geog.XLAT_M[0,:,10], PYR_loc[1]) 
height_pyramid_model = geog.HGT_M[0,idx_lat_PYR,idx_lon_PYR]
print('elev model   =',height_pyramid_model.values,'\nelev station = 5035')

chi_lon_PYR = geog.XLONG_M[0,10,idx_lon_PYR]
chi_lat_PYR = geog.XLAT_M[0,idx_lat_PYR,10]

# # cross section 1 
crss_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],27.643)
crss_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],27.643)
crss_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],87.305)
crss_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],87.42)

#%% triangular area
tri_idx = []
# first vertex
# vertex = [27.80239,86.71456]
vertex = [27.85,86.705]

idx_lat_V = find_nearest(geog.XLAT_M[0,:,10], vertex[0]) 
idx_lon_V = find_nearest(geog.XLONG_M[0,10,:], vertex[1]) 
tri_idx.append([idx_lat_V,idx_lon_V])

# start the funnel 
idx_lon_V_left = idx_lon_V
idx_lon_V_right = idx_lon_V

x = 1
for i in range(0,7): # latitudinal extent 
    #print(i)
    idx_lat_V = idx_lat_V + 1
    if x % 3 != 0:
         idx_lon_V_left = idx_lon_V_left - 1
         idx_lon_V_right = idx_lon_V_right + 1
        
    for j in np.arange(idx_lon_V_left,idx_lon_V_right+1,1):
        # print(j)
        new_couple = [idx_lat_V,j]
        # print(new_couple)
        tri_idx.append(new_couple)
    x += 1
    
# Test the extent of the triangular area by setting the elevation to 0 
# comment the following 5 lines if no need to visualize
for i in range(len(geog.XLAT_M[0,:])):
    for j in range(len(geog.XLONG_M[0,:])):
        # print(i,j)
        if [i,j] in tri_idx:
            ds_mask[i,j] = 0

# save area
np.save('tri_idx.npy', np.array(tri_idx))
#%%

cproj = cartopy.crs.PlateCarree()    
# Plot - replacing .imshow with .pcolormesh is more accurate but slowier
plt.figure(figsize=(22, 16))
ax = plt.axes(projection=cproj)
ax.set_extent([86.14, 87.43, 27.295, 28.28])

levels = np.arange(10., 9000., 250.)
geogr = plt.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.3)
plt.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
filled = plt.contourf(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, cmap='binary', alpha=0.7)

ax.tick_params(axis='both', which='major', labelsize=24)
ax.coastlines(color='k', linewidth = 1.5); ax.add_feature(cartopy.feature.RIVERS);#ax.add_feature(cartopy.feature.STATES, alpha = 0.3)
gl = ax.gridlines(draw_labels=True,alpha=0.5);
gl.xlabel_style = {'size': 24, 'color': 'k'}
gl.ylabel_style = {'size': 24, 'color': 'k'}
# for lat,lon in zip(lats1,lons1):
    # plt.plot(lon, lat, markersize=1.2, marker='o', color='r')
for lat,lon in zip(lats2,lons2):
    plt.plot(lon, lat, markersize=3, marker='o', color='k')
for lat,lon in zip(lats2_e,lons2_e):
    plt.plot(lon, lat, markersize=2.2, marker='o', color='k')   
for lat,lon in zip(lats2_w,lons2_w):
    plt.plot(lon, lat, markersize=2.2, marker='o', color='k')  
    
    
# for lat,lon in zip(lats3,lons3):
    # plt.plot(lon, lat, markersize=1.2, marker='o', color='b')
# for lat,lon in zip(lats4,lons4):
    # plt.plot(lon, lat, markersize=1.2, marker='o', color='k')

fsize = 32
offset = 1.5
plt.plot(chi_lon_PYR, chi_lat_PYR, markersize=8, marker='o', color='k') # Pyramid
plt.annotate('PYR',(chi_lon_PYR-0.07, chi_lat_PYR),fontsize=fsize,color='k')
plt.plot(86.76, 27.85, markersize=8, marker='o', color='k') # Pyramid
plt.annotate('P2',(86.76+0.02, 27.85),fontsize=fsize,color='k')
plt.plot(86.71456, 27.80239, markersize=8, marker='o', color='k') # Namche
plt.annotate('Namche',(86.71456-0.2, 27.80239-0.01),fontsize=fsize,color='k')
plt.plot(86.72306, 27.69556, markersize=8, marker='o', color='k') 
plt.annotate('Lukla',(86.72306-0.14, 27.69556-0.01),fontsize=fsize,color='k')
plt.plot(86.71, 27.88, markersize=8, marker='o', color='k')  # original lon 86.731
plt.annotate('P1',(86.731, 27.88),fontsize=fsize,color='k')
plt.annotate('Khumbu Valley',(86.72, 27.5),fontsize=fsize,color='k')
plt.plot(86.9248, 27.98737, markersize=8, marker='o', color='k') # Everest
plt.annotate('Mt. Everest',(86.94, 27.98737),fontsize=fsize,color='k')

scale_bar(ax, 30,location=(0.75,0.05))


plt.plot(86.9248, 27.98737, markersize=10, marker='s', color='k') # Everest



# 27.956906562754874, 86.81374754760827
# plt.plot([86.6, 86.8], [27.4, 27.4],
#          color='k', linewidth=1, marker='o', markersize=3,
#          transform=ccrs.PlateCarree())

# plt.title(title,fontsize=18)
plt.savefig('topo_nepal4.png')
# plt.show()

