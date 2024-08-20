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

# Concetrazioni VOC
VOC = pd.read_csv("../data/VOC/Mastdata_VOC_2019.txt", na_values=["NaN"])

for col in VOC.columns:
    VOC[col] = VOC[col].astype(float)

VOC = VOC[VOC.Height.isin([42])]
VOC = VOC[VOC.Month.isin([6,7,8])]
#VOC = VOC[VOC.Day.isin([19,20,21,22,23,24,25,26,27])]
#VOC = VOC.groupby([VOC.Day, VOC.Hour]).mean()

datetime_data = {
    'Year': VOC['Year'],
    'Month': VOC['Month'],
    'Day': VOC['Day'],
    'Hour': VOC['Hour']
}

# Create datetime vector
time2 = pd.to_datetime(datetime_data)

conc = VOC.Monoterpenes.values
#daily_mis = VOC.groupby([VOC.Hour]).mean().Monoterpenes.values
#ore_mis = [2,5,8,11,14,17,20,23]

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-BVOC-BASE.nc")
update = xr.open_dataset("../data/FINLAND6-BVOC-CC.nc")

# Account for time shift
times = update.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=2)
update['local_times'] = pd.DatetimeIndex(local_times)
base['local_times'] = pd.DatetimeIndex(local_times)
update['MONOT'] = update.MONOT.swap_dims({'time_counter':'local_times'})
base['MONOT'] = base.MONOT.swap_dims({'time_counter':'local_times'})

# Define the start and end dates
start_date = datetime(2019, 6, 1, 0, 0, 0)
end_date = datetime(2019, 8, 30, 23, 0, 0)

start_1 = datetime(2019, 6, 1, 2, 0, 0)
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

print(idx_lon)
print(idx_lat)

sub_base = base.MONOT.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))
sub_upd_2 = update.MONOT.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))

mbe_base = mean_bias_error(conc,sub_base,time2,time)
mbe_cc = mean_bias_error(conc,sub_upd_2,time2,time)

fig = plt.figure(figsize=(30,6))
ax = fig.add_subplot()
ax.plot(time,sub_base, linewidth=5, label="WRF-CHIMERE")
ax.plot(time,sub_upd_2, linewidth=5, label="WRF-CHIMERE (Updated)")
ax.plot(time2, conc, "ko", markersize=3, label="Observations")
ax.text(0.25, 0.85, f'MBE = {mbe_base:.2f} ppbv', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.text(0.25, 0.8, f'MBE (Updated) = {mbe_cc:.2f} ppbv', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.legend()
ax.set_ylabel("Monoterpenes (ppbv)", fontsize=18)
fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.grid()
ax.set_ylim([0,10])
fig.savefig("../figures/timeseries_Monoterpenes_cc.png", dpi=500)

#modello = sub_upd_2.values[1::3]

#fig = plt.figure(figsize=(9,9))
#ax1 = fig.add_subplot()
#ax1.scatter(conc,modello,s=25)
#t = np.linspace(0,100)
#ax1.plot(t,t, color='black')
#r0 = np.linspace(0,100)
#y0 = 2*r0
#y1 = 0.5*r0
#ax1.plot(r0,y0,'k--')
#ax1.plot(r0,y1,'k--')
#ax1.set_ylabel("Model Monoterpenes (ppbv)", fontsize=18)
#ax1.set_xlabel("Measured Monoterpenes (ppbv)", fontsize=18)
#plt.xlim(0, 6.2)
#plt.ylim(0, 6.2)
#ax1.grid()
#ax1.tick_params(axis='both', which='major', labelsize=15)
#ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
#fig.savefig("../figures/scatter_monotepenes_megan_updated_no_age.png")

