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

#base = xr.open_dataset('/projappl/project_2008324/MB/CHIMERE/chimere_v2023r1/domains/FIN18/geog_FIN18.nc')
#ds = xr.open_dataset('/scratch/project_2008324/MB/FLEXPART/output/test/flxout_d01_20190815_221500.nc', chunks={'Time': 10}).isel(releases=slice(-120, None))

#ds.to_netcdf('../data/flexpart_test.nc')

#ds = xr.open_dataset('../data/flexpart_test.nc', chunks={'Time': 10})
#ds = ds.isel(releases=slice(None, None, -1))
#ds.to_netcdf('../data/flexpart_test_final.nc')

#ds = xr.open_dataset('../data/flexpart_test_final.nc', chunks={'Time': 10})
#ds = ds.isel(Time=slice(None, None, -1))
#ds.to_netcdf('../data/flexpart_test_finale.nc')

#ds1 = xr.open_dataset('../data/flxout_d01_20190825_221500.nc', chunks={'Time': 10})
#ds2 = xr.open_dataset('../data/flxout_d01_20190820_221500.nc', chunks={'Time': 10})

#sub = ds.CONC.sel(ageclass=0).sel(releases=slice(2014, 2134)).sum('releases').sum('bottom_top').mean('Time').compute()
#sub1 = ds1.CONC.sel(ageclass=0).sel(releases=slice(1893, 2013)).sum('releases').sum('bottom_top').mean('Time').compute()
#sub2 = ds2.CONC.sel(ageclass=0).sel(releases=slice(1773, 1892)).sum('releases').sum('bottom_top').mean('Time').compute()

#sub_f = (sub+sub1+sub2)/3

#small_positive_value = 1e-1
#sub_safe = np.where(sub <= 0, small_positive_value, sub_f) 

# Plot
#cproj = cartopy.crs.LambertConformal(central_longitude=24.3, central_latitude=61.8)
#fig = plt.figure(figsize=(9,11))
#ax0 = plt.subplot(projection=cproj)
#norm = LogNorm(vmin=1e-1, vmax=400)
#c = plt.pcolormesh(base.XLONG_M[0,:-1,:-1], base.XLAT_M[0,:-1,:-1], sub_safe, cmap='turbo', transform=ccrs.PlateCarree(),shading='gouraud',norm=norm)
#cbar = plt.colorbar(c, fraction = 0.040, pad = 0.12,  extend="both")
#cbar.set_label(label='SRR (s)', fontsize=18, y=0.5)
#cbar.ax.tick_params(labelsize=15)
#ax0.coastlines(color='k', linewidth = 1);
#ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
#ax0.set_title('Back trajectories', fontweight="bold", fontsize=25)
#gl = ax0.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
#gl.xlabel_style = {'rotation': 0};
#plt.savefig('../figures/mappa_backtraj.png', dpi=400)
#plt.show()

# AME calculation
ds_emis = xr.open_dataset('../data/FINLAND18.nc')
ds_back = xr.open_dataset('../data/flexpart_test_finale.nc', chunks={'Time': 20})

# Define the start and end dates
start_date = datetime(2019, 8, 11, 0, 0, 0)
time = [start_date + timedelta(hours=i) for i in range(120)]

def calculate_AME_single(traj, start_date, ds_back, ds_emis):
    AME = 0
    # Adjusting for different dimension names
    ds_back_corrected = ds_back.rename({'west_east': 'x', 'south_north': 'y'})
    sub = ds_back_corrected.CONC.sel(ageclass=0).sel(releases=traj).sum('bottom_top')
    emis = ds_emis.APINEN*1e6

    # Loop over all time steps for the specific trajectory
    for TIME in range(traj + 1):
        date = start_date + timedelta(hours=TIME)
        
        # Ignoring the last value of x and y in ds_emis
        tmp0 = sub.sel(Time=TIME)
        tmp1 = emis.isel(x=slice(-1), y=slice(-1)).sel(time_counter=date)
 
        # Calculate the product and sum over x and y dimensions
        tmp = tmp0 * tmp1
        somma = tmp.sum(['x', 'y'])

        # Accumulate the sum into AME
        AME += somma
    
    # Compute the final result
    AME = AME.compute()

    return AME.values

AME = [calculate_AME_single(i, start_date, ds_back, ds_emis) for i in range(120)]

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot()
ax.plot(time,AME, linewidth=3)
ax.set_ylabel("AME to APINEN", fontsize=18)
fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.grid()
fig.savefig("../figures/timeseries_AME_test.png", dpi=500)
plt.show()














