from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.cm import get_cmap
import matplotlib.patheffects as pe
import cartopy
import cartopy.crs as crs
import glob
from wrf import (getvar, ALL_TIMES, interplevel, to_np, latlon_coords, get_cartopy,
                 cartopy_xlim, cartopy_ylim)

# Open the NetCDF file
filename = 'NEPALR4-WIND-NEW.nc'
ncfile = Dataset("/scratch/project_2006279/GC/CHIMERE/chimere_out_online_nepal_vbs_homs/nest-NEPALR4/{}".format(filename))

# get GEOG file
geog = xr.open_dataset('geog_NEPALR4.nc')

# get valleys location
df = pd.read_csv('khumbu_latlon_extended.txt',sep=',',header=None)
lats_b = np.flip(np.array(df[0]))
lons_b = np.flip(np.array(df[1]))
# ridges
df_e = pd.read_csv('khumbu_east_ridge_LATLON.txt',sep=',',header=None)
lats_e = np.flip(np.array(df_e[0]))
lons_e = np.flip(np.array(df_e[1]))
df_w = pd.read_csv('khumbu_west_ridge_LATLON.txt',sep=',',header=None)
lats_w = np.flip(np.array(df_w[0]))
lons_w = np.flip(np.array(df_w[1]))

crss2_coords = [27.6 , 27.6 , 86.49, 86.88]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

crss2_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],crss2_coords[0])
crss2_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],crss2_coords[1])
crss2_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],crss2_coords[2])
crss2_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],crss2_coords[3])

# Extract the pressure, geopotential height, and wind variables
p = getvar(ncfile, "pressure")#, method="join")
z = getvar(ncfile, "z")#, method="join")
ua = getvar(ncfile, "ua")#, method="join")
va = getvar(ncfile, "va")#, method="join")
wspd = getvar(ncfile, "wspd_wdir")#, method="join")[0,:]

# Interpolate geopotential height, u, and v winds to 400 hPa
ht_500 = interplevel(z, p, 400)
u_500 = interplevel(ua, p, 400)
v_500 = interplevel(va, p, 400)
wspd_500 = interplevel(wspd, p, 400)[0,:]

# Get the lat/lon coordinates
lats, lons = latlon_coords(ht_500)

# Get the map projection information
cart_proj = get_cartopy(ht_500)

# Create the figure
fig = plt.figure(figsize=(12,9))
ax = plt.axes(projection=cart_proj)
ax.tick_params(axis='both', which='major', labelsize=27)
ax.coastlines(color='k', linewidth = 1);
gl = ax.gridlines(draw_labels=True,linewidth=0.5, color='gray', alpha=0.05, linestyle='--');
gl.xlabel_style = {'size': 15, 'color': 'k'}
gl.ylabel_style = {'size': 15, 'color': 'k'}
ax.plot([crss2_coords[2], crss2_coords[3]], [crss2_coords[0], crss2_coords[1]],color='w', linewidth=3, marker='o', markersize=3,transform=crs.PlateCarree())

# Plot valley ridges and bottom
plt.plot(lons_b, lats_b, linewidth=3, color='w', transform=crs.PlateCarree())
plt.plot(lons_e, lats_e, linewidth=3, color='b', transform=crs.PlateCarree())
plt.plot(lons_w, lats_w, linewidth=3, color='r', transform=crs.PlateCarree())


# Add the 400 hPa geopotential height contours
levels = np.arange(7000., 8000., 15.)
contours = plt.contour(to_np(lons), to_np(lats), to_np(ht_500),
                       levels=levels, colors="black",
                       transform=crs.PlateCarree())
plt.clabel(contours, inline=1, fontsize=10, fmt="%i")


# Add the wind speed contours
levels = np.arange(20,40,1) #[0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30,32.5,35,37.5,40,42.5,45,47.5,50]
wspd_contours = plt.contourf(to_np(lons), to_np(lats), to_np(wspd_500),
                             levels=levels,
                             cmap=get_cmap("viridis"),
                             transform=crs.PlateCarree(),
                             extend="both")
cbar = plt.colorbar(wspd_contours, fraction = 0.042, pad = 0.15)
cbar.ax.set_ylabel('Wind speed (m s$^{-1}$)',fontsize=27,y=0.5)
cbar.ax.tick_params(labelsize=18)
plt.quiver(to_np(lons[::6,::6]), to_np(lats[::6,::6]),to_np(u_500[::6, ::6]), to_np(v_500[::6, ::6]),transform=crs.PlateCarree(), scale = 1000, color="white")

# Plot information
ax.plot(86.813747, 27.956906, markersize=16, marker='x', color='w',path_effects=[pe.withStroke(linewidth=2, foreground="k")], transform=crs.PlateCarree()) # Pyramid
ax.plot(86.715, 27.802, markersize=16, marker='x', color='w',path_effects=[pe.withStroke(linewidth=2, foreground="k")], transform=crs.PlateCarree()) # Namche
ax.text(86.823747, 27.956906, s="NCO-P", fontsize=21,color='w',path_effects=[pe.withStroke(linewidth=2, foreground="k")], transform=crs.PlateCarree())
ax.text(86.725, 27.802, s="Namche", fontsize=21, color='w', path_effects=[pe.withStroke(linewidth=2, foreground="k")],transform=crs.PlateCarree())

# Set the map bounds
ax.set_xlim(cartopy_xlim(ht_500))
ax.set_ylim(cartopy_ylim(ht_500))

ax.gridlines()

plt.title("400 hPa Geopotential Height (m)", fontsize="31")

plt.show()
fig.savefig('output_figures/NEPALR3_geopotential.png', dpi=500)

