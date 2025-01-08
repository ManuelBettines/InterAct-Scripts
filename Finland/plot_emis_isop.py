import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature

ds_base = xr.open_dataset('../data/FINLAND6_emis_iso.nc')
cams = xr.open_dataset('/scratch/project_2008324/MB/CHIMERE/BIGFILES2023/CAMS_BIO/CAMS-GLOB-BIO_Glb_0.25x0.25_bio_isoprene_v3.1_monthly.nc')
#sub = ds_base.C5H8_b.mean('time_counter')

#for i in range(len(ds_base.nav_lat)-2):
#    for j in range(118):
#        sub[i+1,j+1] = sub[i+1,j+1]
#        sub[0,j] = sub[1,j]
#        sub[i,0] = sub[i,1]
#        sub[i, 119] = sub[i,118]
#        sub[188,j] = sub[187,j]

date = ['2017-06-01T00:00:00.000000000',  '2017-07-01T00:00:00.000000000',  '2017-08-01T00:00:00.000000000',  '2018-06-01T00:00:00.000000000',  '2018-07-01T00:00:00.000000000',  '2018-08-01T00:00:00.000000000',  '2019-06-01T00:00:00.000000000',  '2019-07-01T00:00:00.000000000',  '2019-08-01T00:00:00.000000000']

sub = cams.emiss_bio.sel(time=date, method='nearest').mean('time')*1e3

# Plot nested domain
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=24.2896, central_latitude=61.8417)})
#fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': ccrs.PlateCarree()})
#c = ax.pcolormesh(ds_base.nav_lon, ds_base.nav_lat, sub,transform=ccrs.PlateCarree(), cmap='turbo', shading='gouraud', vmin=0, vmax=1.5e-8)
c = ax.pcolormesh(cams.lon, cams.lat, sub,transform=ccrs.PlateCarree(), cmap='turbo', vmin=0, vmax=1.5e-8)
ax.coastlines(color='k', linewidth = 1);
ax.set_extent([18.9, 31.8, 59.4, 69.7], crs=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.BORDERS,color='k', alpha=0.5, linewidth = 0.8);
cbar = plt.colorbar(c, fraction = 0.042, pad = 0.21,  extend="both")
cbar.set_label(label='Isoprene emissions [g m$^{-2}$ s$^{-1}$]', fontsize=10, y=0.5)
cbar.ax.tick_params(labelsize=12)
gl = ax.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlabel_style = {'rotation': 0}
plt.show()
fig.savefig('../figures/isoprene_emis_cams.png', dpi=350, bbox_inches='tight')
