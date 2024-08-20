import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature

# Load geog file
geog = xr.open_dataset('/projappl/project_2008324/MB/CHIMERE/chimere_v2023r1/domains/FIN18/geog_FIN18.nc')
geog_nested = xr.open_dataset('/projappl/project_2008324/MB/CHIMERE/chimere_v2023r1/domains/FIN6/geog_FIN6.nc')

# Sea location
ds_sea = geog.LANDUSEF[:,16,:,:]
# Elevation
ds_mask = geog.HGT_M[0,:,:]

# Sea location
ds_sea_nested = geog_nested.LANDUSEF[:,16,:,:]
# Elevation
ds_mask_nested = geog_nested.HGT_M[0,:,:]

# Plot nested domain
fig, axes = plt.subplots(1, 2, figsize=(18, 11), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=24.2896, central_latitude=61.8417)})
ax, ax1 = axes

# Plot domain
#cproj = cartopy.crs.LambertConformal(central_longitude=-146.521, central_latitude=61.282)
#fig = plt.figure(figsize=(9,11))
#ax = plt.subplot(projection=cproj)
geogr = ax.pcolormesh(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(), cmap='terrain',vmin=-400,vmax=2000)
sea = ax.pcolormesh(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_sea[0,:,:],transform=ccrs.PlateCarree(), cmap = 'viridis_r', alpha=0.4)
ax.coastlines(color='k', linewidth = 1.5); 
ax.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 1);
ax.add_feature(cartopy.feature.RIVERS);
ax.add_feature(cartopy.feature.LAKES);
ax.set_title('18km x 18km  grid', fontsize=18, fontweight='bold')
gl = ax.gridlines(draw_labels=True, alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlocator = plt.FixedLocator(np.linspace(0, 40, 5))
gl.ylabels_left = True
gl.ylabels_right = False
gl.xlabels_top = False
gl.xlabels_bottom = True
gl.xlabel_style = {'size': 14, 'color': 'k'}
gl.ylabel_style = {'size': 14, 'color': 'k'}
ax.plot(24.2896, 61.8417, transform=ccrs.PlateCarree(),markersize=10, marker='x',color='k')

# Extract nested domain boundaries (all edge points)
nested_lons = geog_nested.XLONG_M[0, :, :].values
nested_lats = geog_nested.XLAT_M[0, :, :].values

# Get the boundary points
left_edge = list(zip(nested_lons[:, 0], nested_lats[:, 0]))
right_edge = list(zip(nested_lons[:, -1], nested_lats[:, -1]))
bottom_edge = list(zip(nested_lons[0, :], nested_lats[0, :]))
top_edge = list(zip(nested_lons[-1, :], nested_lats[-1, :]))

# Combine edges into a single boundary
boundary_points = left_edge + top_edge + right_edge[::-1] + bottom_edge[::-1]

# Separate into lons and lats
boundary_lons, boundary_lats = zip(*boundary_points)

# Plot the nested domain boundary
ax.plot(boundary_lons, boundary_lats, transform=ccrs.PlateCarree(), color='k', linewidth=2, linestyle='--')

# Plot nested domain 
geogr_nested = ax1.pcolormesh(geog_nested.XLONG_M[0,:,:], geog_nested.XLAT_M[0,:,:], ds_mask_nested[:,:], transform=ccrs.PlateCarree(), cmap='terrain', vmin=-400, vmax=2000)
sea_nested = ax1.pcolormesh(geog_nested.XLONG_M[0,:,:], geog_nested.XLAT_M[0,:,:], ds_sea_nested[0,:,:], transform=ccrs.PlateCarree(), cmap='viridis_r', alpha=0.4)
ax1.coastlines(color='k', linewidth=1.5)
ax1.add_feature(cartopy.feature.BORDERS, color='k', linewidth=2)
ax1.add_feature(cartopy.feature.RIVERS)
ax1.add_feature(cartopy.feature.LAKES)
ax1.set_title('6km x 6 km grid', fontsize=18, fontweight='bold')
gl1 = ax1.gridlines(draw_labels=True, alpha=0.3, dms=False, x_inline=False, y_inline=False)
gl1.xlocator = plt.FixedLocator(np.linspace(20, 30, 3))
gl1.ylabels_left = False
gl1.ylabels_right = True
gl1.xlabels_top = False
gl1.xlabels_bottom = True
gl1.xlabel_style = {'rotation': 0}
gl1.ylabel_style = {'rotation': 90}
gl1.xlabel_style = {'size': 12, 'color': 'k'}
gl1.ylabel_style = {'size': 12, 'color': 'k'}
ax1.plot(24.2896, 61.8417, transform=ccrs.PlateCarree(),markersize=15, marker='x',color='k')

ax1.annotate('Hyytiälä',(24.2896+2*0.01, 61.8417+10*0.01), transform=ccrs.PlateCarree(),fontsize=15,color='k')

plt.savefig('../figures/geog_finland.png', dpi=500)
