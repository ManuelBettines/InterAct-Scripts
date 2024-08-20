import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime
import matplotlib.ticker as ticker
import pandas as pd
from datetime import timedelta
from math import isnan
from pandas.tseries.offsets import DateOffset


# Load simulations output
base = xr.open_dataset("../data/FINLAND6-OA-CC.nc")
emis = xr.open_dataset("../data/FINLAND6-UPDATED-EMIS.nc")
ozono = xr.open_dataset("../data/FINLAND6-OZONE.nc")

# Account for time shift
times = base.Times.astype(str)
times_e = emis.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
times_e = np.core.defchararray.replace(times_e,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
utc_times_e = pd.to_datetime(times_e, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=2)
local_times_e = utc_times_e + DateOffset(hours=2)
base['local_times'] = pd.DatetimeIndex(local_times)
ozono['local_times'] = pd.DatetimeIndex(local_times)
emis['local_times'] = pd.DatetimeIndex(local_times_e)
base['OA'] = base.OA.swap_dims({'time_counter':'local_times'})
emis['MONOT'] = emis.MONOT.swap_dims({'time_counter':'local_times'})
emis['C5H8'] = emis.C5H8.swap_dims({'time_counter':'local_times'})
ozono['O3'] = ozono.O3.swap_dims({'time_counter':'local_times'})

# Define the start and end dates
start_date = datetime(2019, 7, 2, 0, 0, 0)
end_date = datetime(2019, 7, 30, 23, 0, 0)

time = [start_date + timedelta(hours=x) for x in range(744)]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

sub_base = base.OA.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))*1.7
sub_emis = emis.C5H8.sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date)) + emis.MONOT.sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))
sub_ozono = ozono.O3.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot()
c = ax.scatter(sub_emis, sub_base, s=50, c=sub_ozono, cmap='viridis', vmin=15, vmax=50)
ax.set_ylabel("OA (µg m$^{-3}$)", fontsize=18)
#ax.set_xlabel("C$_{5}$H$_{8}$ emissions (μg m$^{-2}$ s$^{-1}$)", fontsize=18)
ax.set_xlabel("Isoprene/Monoterpenes emissions ratio", fontsize=18)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.12,  extend="both")
cbar.set_label(label='Ozone (ppbv)', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Organic Aerosol vs BVOCs emissions',fontsize=21)
ax.grid()
ax.set_ylim([0,12])
fig.savefig("../figures/oa_vs_sum_emission.png", dpi=500)
