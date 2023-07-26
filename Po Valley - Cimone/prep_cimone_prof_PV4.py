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


#%%

df1 = pd.read_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/WRF-CHIMERE_OUTPUT/H2SO4/old/free_profiles/cimone_prof4.txt',sep=',',header=None)

for i in np.arange(25):
    new_lat = df1.loc[0,0] + 0.0089035 * 0.25
    new_lon = df1.loc[0,1] + 0.00918579 
    df1.loc[-1] = [new_lat, new_lon]  # adding a row
    df1.index = df1.index + 1  # shifting index
    df1.sort_index(inplace=True) 
    
for i in np.arange(20):
    new_lat = df1.loc[0,0] + 0.0089035 * 0.4
    new_lon = df1.loc[0,1] + 0.00918579 
    df1.loc[-1] = [new_lat, new_lon]  # adding a row
    df1.index = df1.index + 1  # shifting index
    df1.sort_index(inplace=True) 

for i in np.arange(10):
    new_lat = df1.loc[0,0] + 0.0089035 * 0.5
    new_lon = df1.loc[0,1] + 0.00918579 
    df1.loc[-1] = [new_lat, new_lon]  # adding a row
    df1.index = df1.index + 1  # shifting index
    df1.sort_index(inplace=True) 
    
for i in np.arange(10):
    new_lat = df1.loc[0,0] + 0.0089035 * 0.65
    new_lon = df1.loc[0,1] + 0.00918579 
    df1.loc[-1] = [new_lat, new_lon]  # adding a row
    df1.index = df1.index + 1  # shifting index
    df1.sort_index(inplace=True) 
    
    
# for i in np.arange(6):
#     new_lat = df1.loc[0,0] + 0.0089035 * 0.22
#     new_lon = df1.loc[0,1] + 0.00918579
#     df1.loc[-1] = [new_lat, new_lon]  # adding a row
#     df1.index = df1.index + 1  # shifting index
#     df1.sort_index(inplace=True)     
    

# for i in np.arange(50):
#     new_lat = df1.loc[0,0] + 0.0089035 * 0.5
#     new_lon = df1.loc[0,1] + 0.00918579
#     df1.loc[-1] = [new_lat, new_lon]  # adding a row
#     df1.index = df1.index + 1  # shifting index
#     df1.sort_index(inplace=True)     

# for i in np.arange(4):
#     new_lat = df1.loc[0,0] + 0.0089035 * 0.2
#     new_lon = df1.loc[0,1] + 0.00918579
#     df1.loc[-1] = [new_lat, new_lon]  # adding a row
#     df1.index = df1.index + 1  # shifting index
#     df1.sort_index(inplace=True)    
    
for i in np.arange(110):
    new_lat = df1.loc[0,0] + 0.0089035 * 0.885
    new_lon = df1.loc[0,1] + 0.00918579 * 0.7
    df1.loc[-1] = [new_lat, new_lon]  # adding a row
    df1.index = df1.index + 1  # shifting index
    df1.sort_index(inplace=True)   
    
# # for i in np.arange(5):                                        # ??
# #     new_lat = df1.loc[0,0] + 0.0089035 * 1
# #     new_lon = df1.loc[0,1] + 0.00918579 * -0.5
# #     df1.loc[-1] = [new_lat, new_lon]  # adding a row
# #     df1.index = df1.index + 1  # shifting index
# #     df1.sort_index(inplace=True)   
    
# for i in np.arange(8):                                          
#     new_lat = df1.loc[0,0] + 0.0089035 * 0.9
#     new_lon = df1.loc[0,1] + 0.00918579 * 1
#     df1.loc[-1] = [new_lat, new_lon]  # adding a row
#     df1.index = df1.index + 1  # shifting index
#     df1.sort_index(inplace=True)   

# # for i in np.arange(30):                                      # direct II               
# #     new_lat = df1.loc[0,0] + 0.0089035 * 1
# #     new_lon = df1.loc[0,1] + 0.00918579 * 0.75
# #     df1.loc[-1] = [new_lat, new_lon]  # adding a row
# #     df1.index = df1.index + 1  # shifting index
# #     df1.sort_index(inplace=True)   

# for i in np.arange(6):
#     new_lat = df1.loc[0,0] + 0.0089035 * 0.2
#     new_lon = df1.loc[0,1] + 0.00918579 * 1
#     df1.loc[-1] = [new_lat, new_lon]  # adding a row
#     df1.index = df1.index + 1  # shifting index
#     df1.sort_index(inplace=True)   

# for i in np.arange(5):
#     new_lat = df1.loc[0,0] + 0.0089035 * 0.7
#     new_lon = df1.loc[0,1] + 0.00918579 * 1
#     df1.loc[-1] = [new_lat, new_lon]  # adding a row
#     df1.index = df1.index + 1  # shifting index
#     df1.sort_index(inplace=True)   
    
# for i in np.arange(5):                                         # direct
#     new_lat = df1.loc[0,0] + 0.0089035 * 1
#     new_lon = df1.loc[0,1] + 0.00918579 * 0.45
#     df1.loc[-1] = [new_lat, new_lon]  # adding a row
#     df1.index = df1.index + 1  # shifting index
#     df1.sort_index(inplace=True)   
    
# for i in np.arange(25):                                        ## direct
#     new_lat = df1.loc[0,0] + 0.0089035 * 1
#     new_lon = df1.loc[0,1] + 0.00918579 * 0.05
#     df1.loc[-1] = [new_lat, new_lon]  # adding a row
#     df1.index = df1.index + 1  # shifting index
#     df1.sort_index(inplace=True)   
    
    
# # for i in np.arange(2):
# #     new_lat = df1.loc[0,0] + 0.0089035 * 0
# #     new_lon = df1.loc[0,1] + 0.00918579 * 1
# #     df1.loc[-1] = [new_lat, new_lon]  # adding a row
# #     df1.index = df1.index + 1  # shifting index
# #     df1.sort_index(inplace=True)   

# # for i in np.arange(6):
# #     new_lat = df1.loc[0,0] + 0.0089035 * -0.2
# #     new_lon = df1.loc[0,1] + 0.00918579 * 1
# #     df1.loc[-1] = [new_lat, new_lon]  # adding a row
# #     df1.index = df1.index + 1  # shifting index
# #     df1.sort_index(inplace=True)   

# # for i in np.arange(1):
# #     new_lat = df1.loc[0,0] + 0.0089035 * 0.2
# #     new_lon = df1.loc[0,1] + 0.00918579 * 1
# #     df1.loc[-1] = [new_lat, new_lon]  # adding a row
# #     df1.index = df1.index + 1  # shifting index
# #     df1.sort_index(inplace=True)   

# # for i in np.arange(2):
# #     new_lat = df1.loc[0,0] + 0.0089035 * 1
# #     new_lon = df1.loc[0,1] + 0.00918579 * 0.8
# #     df1.loc[-1] = [new_lat, new_lon]  # adding a row
# #     df1.index = df1.index + 1  # shifting index
# #     df1.sort_index(inplace=True)   
    
# # for i in np.arange(2):
# #     new_lat = df1.loc[0,0] + 0.0089035 * 1
# #     new_lon = df1.loc[0,1] + 0.00918579 * 0
# #     df1.loc[-1] = [new_lat, new_lon]  # adding a row
# #     df1.index = df1.index + 1  # shifting index
# #     df1.sort_index(inplace=True)   
    
# # for i in np.arange(8):
# #     new_lat = df1.loc[0,0] + 0.0089035 * 1
# #     new_lon = df1.loc[0,1] + 0.00918579 * -0.7
# #     df1.loc[-1] = [new_lat, new_lon]  # adding a row
# #     df1.index = df1.index + 1  # shifting index
# #     df1.sort_index(inplace=True)   
    
# # for i in np.arange(20):
# #     new_lat = df1.loc[0,0] + 0.0089035 * 1
# #     new_lon = df1.loc[0,1] + 0.00918579 * -0.4
# #     df1.loc[-1] = [new_lat, new_lon]  # adding a row
# #     df1.index = df1.index + 1  # shifting index
# #     df1.sort_index(inplace=True)   
        

lats1 = df1[0]
lons1 = df1[1]
df1.to_csv('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/WRF-CHIMERE_OUTPUT/H2SO4/old/free_profiles/cimone_prof4_newlong.txt',header=False,index=False)

SPC_loc = np.array([44.65578505554352, 11.62135742257148])
MTC_loc = np.array([44.1931632649319, 10.701766513804616])
MOB_loc = np.array([43.97286160339424, 10.18486379591059]) # 1 Massa 
MOB_loc2 = np.array([44.04, 10.33]) # 2 Apuan Alps
MOB_loc3 = np.array([44.0064308015, 10.25743189795525]) # 3 intermediate

geog = xr.open_dataset('/home/bvitali/Desktop/Università/AAA_Tesi/Materiali/downscaling/geog_POVALLEY4.nc')
ds_mask = geog.HGT_M[0,:,:]

#%%

# Get the cartopy mapping object
cproj = cartopy.crs.PlateCarree()  

# Figure 
cproj = ccrs.PlateCarree()    
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=cproj)
# ax.set_extent([9.8, 11, 43.75, 44.4])
# ax.set_extent([9.8, 10.8, 43.65, 44.25])


ax.tick_params(axis='both', which='major', labelsize=24)
ax.coastlines(color='blue', linewidth = 1.5, alpha=0.5);
ax.add_feature(cartopy.feature.BORDERS,color='blue', linewidth = 2, alpha = 0.5);
# ax.add_feature(cartopy.feature.RIVERS);
ax.add_feature(cartopy.feature.STATES, alpha = 0.5)
gl = ax.gridlines(draw_labels=True,alpha=0.5);
gl.xlabel_style = {'size': 14, 'color': 'k'}
gl.ylabel_style = {'size': 14, 'color': 'k'}
plt.plot(MTC_loc[1], MTC_loc[0], markersize=10, marker='o', color='r')
plt.plot(MOB_loc[1], MOB_loc[0], markersize=10, marker='o', color='b')
plt.plot(MOB_loc2[1], MOB_loc2[0], markersize=10, marker='o', color='g')
plt.plot(MOB_loc3[1], MOB_loc3[0], markersize=10, marker='o', color='orange')

levels = np.arange(10., 3000., 50.)
geogr = ax.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.2)
ax.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')

for lat,lon in zip(lats1,lons1):
    # print(lat,lon)
    plt.plot(lon, lat, markersize=2, marker='o', color='r')


