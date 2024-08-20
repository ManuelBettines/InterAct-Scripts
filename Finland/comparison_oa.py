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

# Concetrazioni O3
#ozone = pd.read_csv("../data/OZONO/O3_hyy.csv", na_values=["-"])
#conc = ozone.Ozone/1.96

oa = pd.read_csv("../data/SMEAR/HydeOA.txt", sep="\t",na_values=["NaN"])
oa['TimelistLT_com'] = pd.to_datetime(oa['TimelistLT_com'])
oa = oa[(oa['TimelistLT_com'] >= '2019-06-01') & (oa['TimelistLT_com'] < '2019-09-01')]
conc = oa["OA_com"].values
time_mis = oa['TimelistLT_com']

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-BASE-OA.nc")
final = xr.open_dataset("../data/FINLAND6-CC-OA.nc")
#megan = xr.open_dataset("../data/FINLAND6-MEGANv3.nc")
#update = xr.open_dataset("../data/FINLAND6-UPDATED-OA.nc")

# Account for time shift
times = base.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=2)
base['local_times'] = pd.DatetimeIndex(local_times)
base['OA'] = base.OA.swap_dims({'time_counter':'local_times'})
final['local_times'] = pd.DatetimeIndex(local_times)
final['OA'] = final.OA.swap_dims({'time_counter':'local_times'})

# Define the start and end dates
start_date = datetime(2019, 6, 1, 0, 0, 0)
end_date = datetime(2019, 8, 31, 23, 0, 0)

time = [start_date + timedelta(hours=x) for x in range(2184)]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def mean_bias_error(actual, predicted):
    # Filter out pairs where 'actual' is NaN
    filtered_data = [(a, p) for a, p in zip(actual, predicted) if not isnan(a)]

    # Separate the filtered 'actual' and 'predicted' values
    actual_filtered, predicted_filtered = zip(*filtered_data)

    # Calculate the error using the filtered data
    error = sum(p - a for p, a in zip(predicted_filtered, actual_filtered)) / len(actual_filtered)
    return error

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

print(idx_lon)
print(idx_lat)

sub_base = base.OA.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))*1.7
sub_final = final.OA.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))*1.7

mbe_base = mean_bias_error(conc,sub_base)
mbe_final = mean_bias_error(conc,sub_final)

fig = plt.figure(figsize=(30,6))
ax = fig.add_subplot()
#ax.plot(time,sub_base, linewidth=5, label="WRF-CHIMERE")
#ax.plot(time,sub_megan, linewidth=3, label="WRF-CHIMERE (MEGANv3.2)")
ax.plot(time,sub_base, linewidth=5, label="WRF-CHIMERE")
ax.plot(time,sub_final, linewidth=5, label="WRF-CHIMERE (Updated)")
ax.plot(time_mis, conc, 'ko', markersize=5, label="Observations")
ax.legend()
ax.text(0.25, 0.9, f'MBE = {mbe_base:.2f} µg m-3', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.text(0.25, 0.85, f'MBE (Updated) = {mbe_final:.2f} µg m-3', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.set_ylabel("OA (µg m$^{-3}$)", fontsize=18)
fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Organic aerosol',fontsize=21)
ax.grid()
ax.set_ylim([0,13])
fig.savefig("../figures/timeseries_OA_hyytiälä.png", dpi=500)
