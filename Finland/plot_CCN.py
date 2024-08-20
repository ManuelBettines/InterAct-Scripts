import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime
import pandas as pd

base = xr.open_dataset("../data/ALASKA6-OA.nc")
ccn_base = xr.open_dataset("../data/ALASKA6-CCN.nc")
#ccn_cc = xr.open_dataset("../data/FINLAND6-CC-CCN.nc")

sub1 = ccn_base.CCN6.sel(bottom_top=slice(0,20)).sel(Time=slice(0,1000)).sum('bottom_top').mean('Time')
sub2 = ccn_base.CCN6.sel(bottom_top=slice(0,20)).sel(Time=slice(1001,1728)).sum('bottom_top').mean('Time')
#sub3 = ccn_base.CCN1.sel(bottom_top=slice(0,20)).sel(Time=slice(2001,3000)).sum('bottom_top').mean('Time')
#sub4 = ccn_base.CCN1.sel(bottom_top=slice(0,20)).sel(Time=slice(3001,4000)).sum('bottom_top').mean('Time')
#sub5 = ccn_base.CCN1.sel(bottom_top=slice(0,20)).sel(Time=slice(4001,5000)).sum('bottom_top').mean('Time')
#sub6 = ccn_base.CCN1.sel(bottom_top=slice(0,20)).sel(Time=slice(5001,6000)).sum('bottom_top').mean('Time')

sub_base = (sub1+sub2)/2#+sub3+sub4+sub5+sub6)/6

#sub1 = ccn_cc.CCN1.sel(bottom_top=slice(0,20)).sel(Time=slice(0,1000)).sum('bottom_top').mean('Time')
#sub2 = ccn_cc.CCN1.sel(bottom_top=slice(0,20)).sel(Time=slice(1001,2000)).sum('bottom_top').mean('Time')
#sub3 = ccn_cc.CCN1.sel(bottom_top=slice(0,20)).sel(Time=slice(2001,3000)).sum('bottom_top').mean('Time')
#sub4 = ccn_cc.CCN1.sel(bottom_top=slice(0,20)).sel(Time=slice(3001,4000)).sum('bottom_top').mean('Time')
#sub5 = ccn_cc.CCN1.sel(bottom_top=slice(0,20)).sel(Time=slice(4001,5000)).sum('bottom_top').mean('Time')
#sub6 = ccn_cc.CCN1.sel(bottom_top=slice(0,20)).sel(Time=slice(5001,6000)).sum('bottom_top').mean('Time')

#sub_cc = (sub1+sub2+sub3+sub4+sub5+sub6)/6

#sub = ((sub_cc - sub_base) / sub_base) * 100

# Plot
#cproj = cartopy.crs.LambertConformal(central_longitude=24.3, central_latitude=61.8)
cproj = cartopy.crs.LambertConformal(central_longitude=-146.521, central_latitude=61.282)
fig = plt.figure(figsize=(9,11))
ax0 = plt.subplot(projection=cproj)
c = plt.pcolormesh(base.nav_lon,base.nav_lat, sub_base, cmap='turbo', transform=ccrs.PlateCarree(),shading='gouraud', vmin=0, vmax=80000)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.12,  extend="both")
cbar.set_label(label='CCN concentration (# cm$^{-3}$)', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)
ax0.coastlines(color='k', linewidth = 1);
ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.7, alpha = 0.5);
ax0.set_title('Cloud condensation nuclei at S=1%', fontweight="bold", fontsize=25)
gl = ax0.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlabel_style = {'rotation': 0};
plt.savefig('../figures/ccn_base_alaska.png', dpi=400, bbox_inches='tight')
