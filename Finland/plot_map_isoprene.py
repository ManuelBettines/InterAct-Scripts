import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature

ds_base = xr.open_dataset('../data/FINLAND6_BVOC.nc', chunks={"time_counter": 200})
#ds = xr.open_dataset('../data/FINLAND6-UPDATED-noCC_BVOC.nc', chunks={"time_counter": 200})
thlay = xr.open_dataset('../data/FIN6_thaly.nc', chunks={"time_counter": 200})

isoprene = ds_base.APINEN.mean('time_counter') + ds_base.BPINEN.mean('time_counter') + ds_base.LIMONE.mean('time_counter') + ds_base.OCIMEN.mean('time_counter')
#data = ds.C5H8.mean('time_counter') 
thlay_cm = thlay['thlay'].mean('time_counter')*100
pressure = thlay['pres'].mean('time_counter')
temperature = thlay['temp'].mean('time_counter')

isoprene = isoprene * 2.46e10 * (pressure/101325) * (298/temperature)
#data = data * 2.46e10 * (pressure/101325) * (298/temperature)

sub = (isoprene * thlay_cm).sum(dim="bottom_top")
#sub_2 = (data * thlay_cm).sum(dim="bottom_top")
#sub = 100*(sub_2-sub_1)/sub_1

# Plot nested domain
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=24.2896, central_latitude=61.8417)})
c = ax.pcolormesh(ds_base.nav_lon, ds_base.nav_lat, sub,transform=ccrs.PlateCarree(),cmap='turbo')
ax.coastlines(color='k', linewidth = 1); 
ax.add_feature(cartopy.feature.BORDERS,color='k', alpha=0.5, linewidth = 0.8);
cbar = plt.colorbar(c, fraction = 0.042, pad = 0.21,  extend="both")
cbar.set_label(label='Monoterpenes column [molecules cm$^{-2}$]', fontsize=10, y=0.5)
cbar.ax.tick_params(labelsize=12)
gl = ax.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlabel_style = {'rotation': 0}
plt.show()
fig.savefig('../figures/monoterpenes_map_baseline.png', dpi=350, bbox_inches='tight')
