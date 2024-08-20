import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime
import pandas as pd
from matplotlib.colors import LogNorm
from datetime import timedelta
import matplotlib.ticker as ticker
import dask

base = xr.open_dataset('/projappl/project_2005956/CHIMERE/chimere_v2020r3_modified/domains/POVALLEY1/geog_POVALLEY1.nc')

ss = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_ssoff/nest-POVALLEY1/out_total.PV1.nossalt.nc')
ship = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_shipoff/nest-POVALLEY1/out_total.PV1.noships.nc')
indus = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_indusoff/nest-POVALLEY1/out_total.PV1.noindus.nc')
#dms = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_dmsoff/nest-POVALLEY1/out_total.PV1.nodms.nc')
bound = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_bdroff/nest-POVALLEY1/out_total.PV1.nobdr.nc')
base1 = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley/nest-POVALLEY1/out_total.PV1.nc').sel(Time=slice(7*24, 840))


BASE = base1.H2SO4.sel(bottom_top=0).mean('Time') #/(base1.pH2SO4.sel(bottom_top=0).mean('Time')*0.25445)
BDR = BASE - bound.H2SO4.sel(bottom_top=0).mean('Time')
#DMS = BASE - dms.H2SO4.sel(bottom_top=0).mean('Time')
IND = BASE - indus.H2SO4.sel(bottom_top=0).mean('Time')
SHIP = BASE - ship.H2SO4.sel(bottom_top=0).mean('Time')
SS = BASE - ss.H2SO4.sel(bottom_top=0).mean('Time')

# Plot
cproj = cartopy.crs.Mercator()#central_longitude=10.7, central_latitude=44.2)
fig = plt.figure(figsize=(11,11))
ax0 = plt.subplot2grid((22, 22), (0, 0), rowspan=9, colspan=9, projection=cproj)
#ax0 = plt.subplot(projection=cproj)
c = plt.pcolormesh(base.XLONG_M[0,:,:], base.XLAT_M[0,:,:], BASE, cmap='turbo', transform=ccrs.PlateCarree(),shading='gouraud', vmin=0, vmax=0.005)
ax0.coastlines(color='k', linewidth = 1);
ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
ax0.set_title('Industries', fontweight="bold", fontsize=23, y=1.05)
gl = ax0.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlabel_style = {'rotation': 0};

ax1 = plt.subplot2grid((22, 22), (0, 11), rowspan=9, colspan=9, projection=cproj)
c = plt.pcolormesh(base.XLONG_M[0,:,:], base.XLAT_M[0,:,:], SHIP, cmap='turbo', transform=ccrs.PlateCarree(),shading='gouraud', vmin=0,vmax=0.005)
ax1.coastlines(color='k', linewidth = 1);
ax1.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
ax1.set_title('Ships', fontweight="bold", fontsize=23, y=1.05)
gl = ax1.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlabel_style = {'rotation': 0};

ax2 = plt.subplot2grid((22, 22), (11, 0), rowspan=9, colspan=9, projection=cproj)
c = plt.pcolormesh(base.XLONG_M[0,:,:], base.XLAT_M[0,:,:], SS, cmap='turbo', transform=ccrs.PlateCarree(),shading='gouraud', vmin=0,vmax=0.005)
ax2.coastlines(color='k', linewidth = 1);
ax2.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
ax2.set_title('Sea Salt', fontweight="bold", fontsize=23, y=1.05)
gl = ax2.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlabel_style = {'rotation': 0};

ax3 = plt.subplot2grid((22, 22), (11, 11), rowspan=9, colspan=9, projection=cproj)
c = plt.pcolormesh(base.XLONG_M[0,:,:], base.XLAT_M[0,:,:], BDR, cmap='turbo', transform=ccrs.PlateCarree(),shading='gouraud', vmin=0,vmax=0.005)
ax3.coastlines(color='k', linewidth = 1);
ax3.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
ax3.set_title('DMS', fontweight="bold", fontsize=23, y=1.05)
gl = ax3.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlabel_style = {'rotation': 0};

cbar = fig.colorbar(c, ax=[ax0, ax1, ax2, ax3], orientation='horizontal', fraction=0.05, pad=0.04, extend="both")
#cbar = fig.colorbar(c, ax=ax0, orientation='horizontal', fraction=0.05, pad=0.04, extend="both")
cbar.set_label(label='Contribution to H$_2$SO$_4$ (ppbv)', fontsize=16)
cbar.ax.tick_params(labelsize=18)

plt.show()




