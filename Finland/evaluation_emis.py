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
from pandas.tseries.offsets import DateOffset
from math import isnan

# Emissioni VOC
VOC_em = pd.read_csv("../data/VOC/VOC_fluxes/VOC_profileflux_all_data_2010_2023.txt", na_values=["NaN"])

for col in VOC_em.columns:
    VOC_em[col] = VOC_em[col].astype(float)

VOC_em = VOC_em[VOC_em.Year.isin([2019])]
VOC_em = VOC_em[VOC_em.Month.isin([6,7,8])]
#VOC_em = VOC_em[VOC_em.Day.isin([19,20,21,22,23,24,25,26,27])]

daily_mis = VOC_em.Isoprene.values
datetime_data = {
    'Year': VOC_em['Year'],
    'Month': VOC_em['Month'],
    'Day': VOC_em['Day'],
    'Hour': VOC_em['Hour']
}

# Create datetime vector
time2 = pd.to_datetime(datetime_data)

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-BVOC-BASE.nc")
update = xr.open_dataset("../data/FINLAND6-BVOC-CC.nc")

# Account for time shift
times = update.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=2)
base['local_times'] = pd.DatetimeIndex(local_times)
update['local_times'] = pd.DatetimeIndex(local_times)
base['C5H8_b'] = base.C5H8_b.swap_dims({'time_counter':'local_times'})
update['C5H8_b'] = update.C5H8_b.swap_dims({'time_counter':'local_times'})

# Define the start and end dates
start_date = datetime(2019, 6, 1, 0, 0, 0)
end_date = datetime(2019, 8, 30, 23, 0, 0)

time = [start_date + timedelta(hours=x) for x in range(2181)]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def mean_bias_error(actual, predicted, actual_times, predicted_times):
    # Create a dictionary to map actual times to actual values
    actual_dict = {t: a for t, a in zip(actual_times, actual)}

    # Filter out predicted values that have corresponding actual values
    filtered_predicted = [predicted[i] for i, t in enumerate(predicted_times) if t in actual_dict]

    # Filter out actual values that have corresponding predicted values
    filtered_actual = [actual_dict[t] for t in predicted_times if t in actual_dict]

    # Filter out pairs where 'actual' is NaN
    filtered_data = [(a, p) for a, p in zip(filtered_actual, filtered_predicted) if not isnan(a)]

    # Separate the filtered 'actual' and 'predicted' values
    actual_filtered, predicted_filtered = zip(*filtered_data)

    # Calculate the error using the filtered data
    error = sum(p - a for p, a in zip(predicted_filtered, actual_filtered)) / len(actual_filtered)
    return error

idx_lon = find_nearest(update.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(update.nav_lat[:,52], 61.8417)

# Select subset
sub_base = base.C5H8_b.sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))*1e6
sub_upd_2 = update.C5H8_b.sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))*1e6

mbe = mean_bias_error(daily_mis,sub_base,time2,time)
mbe_cc = mean_bias_error(daily_mis,sub_upd_2,time2,time)


# Plot results
fig = plt.figure(figsize=(30,6))
ax = fig.add_subplot()
ax.plot(time,sub_base, linewidth=5, label="WRF-CHIMERE")
ax.plot(time,sub_upd_2, linewidth=5, label="WRF-CHIMERE (Updated)")
ax.plot(time2, daily_mis, "ko", markersize=5, label="Observations")
ax.legend()
ax.set_ylabel("Isoprene emissions (μg m$^{-2}$ s$^{-1}$)", fontsize=18)
ax.text(0.25, 0.85, f'MBE = {mbe:.2f} ppbv', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.text(0.25, 0.8, f'MBE (updated) = {mbe_cc:.2f} ppbv', transform=ax.transAxes, fontsize=12, verticalalignment='top')
fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Isoprene emissions',fontsize=21)
ax.grid()
ax.set_ylim([0,0.5])
fig.savefig("../figures/Isoprene_emissions_ts.png", dpi=500)
