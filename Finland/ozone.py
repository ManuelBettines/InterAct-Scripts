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

# Concetrazioni O3
#ozone = pd.read_csv("../data/OZONO/O3_hyy.csv", na_values=["-"])
#conc = ozone.Ozone/1.96

ozone = pd.read_csv("../data/OZONO/smeardata_20240430.txt", sep="\t",na_values=["NaN"])
conc = ozone["HYY_META.O342"].values

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-OZONE.nc")
#megan = xr.open_dataset("../data/FINLAND6-MEGANv3.nc")
#update = xr.open_dataset("../data/FINLAND6-UPDATED.nc")

# Define the start and end dates
start_date = datetime(2019, 7, 1, 0, 0, 0)
end_date = datetime(2019, 7, 31, 23, 0, 0)

time = [start_date + timedelta(hours=x) for x in range(744)]

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

sub_base = base.O3.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))
#sub_megan = megan.MONOT.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))
#sub_upd_2 = update.MONOT.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))

mbe = mean_bias_error(conc,sub_base)

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot()
ax.plot(time,sub_base, linewidth=5, label="WRF-CHIMERE")
#ax.plot(time,sub_megan, linewidth=3, label="WRF-CHIMERE (MEGANv3.2)")
#ax.plot(time,sub_upd_2, linewidth=3, label="WRF-CHIMERE (MEGANv3.2 updated)")
ax.plot(time, conc, 'ko', markersize=5, label="Observations")
ax.legend()
ax.text(0.05, 0.1, f'MBE = {mbe:.2f} ppbv', transform=ax.transAxes, fontsize=18, verticalalignment='top')
ax.set_ylabel("Ozone (ppbv)", fontsize=18)
fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Ozone',fontsize=21)
ax.grid()
ax.set_ylim([0,55])
fig.savefig("../figures/timeseries_ozone.png", dpi=500)
