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
import pyaerocom as pya
import glob

# Concetrazioni VOC
files = ["PALLAS1", "PALLAS2"]
all_data = []
for file in files:
    VOC = pya.io.EbasNasaAmesFile(file=f"../data/VOC/{file}.nas",
                                  only_head=False, 
                                  replace_invalid_nan=True, 
                                  convert_timestamps=True, 
                                  decode_flags=True)

    ds = VOC.data[:, 3] * 3.589254e-4
    time2 = pd.to_datetime(VOC.time_stamps)

    # Combine into a DataFrame
    df = pd.DataFrame({"time": time2, "data": ds})
    all_data.append(df)

# Concatenate all data
combined_data = pd.concat(all_data, ignore_index=True)

ds = combined_data["data"]
time2 = combined_data["time"]
# Load simulations output
base = xr.open_dataset("../data/FINLAND6_BVOC.nc")
update = xr.open_dataset("../data/FINLAND6-CC_BVOC.nc")
#megan = xr.open_dataset("../data/FINLAND6-MEGAN.nc")
megan = xr.open_dataset("../data/FINLAND6-UPDATED-noCC_BVOC.nc")


# Account for time shift
times = update.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=2)
update['local_times'] = pd.DatetimeIndex(local_times)
base['local_times'] = pd.DatetimeIndex(local_times)
megan['local_times'] = pd.DatetimeIndex(local_times)
update['C5H8'] = update.C5H8.swap_dims({'time_counter':'local_times'})
base['C5H8'] = base.C5H8.swap_dims({'time_counter':'local_times'})
megan['C5H8'] = megan.C5H8.swap_dims({'time_counter':'local_times'})

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

idx_lon = find_nearest(update.nav_lon[156,:], 24.11504)
idx_lat = find_nearest(update.nav_lat[:,50], 67.97310)

print(idx_lon)
print(idx_lat)

sub_base = base.C5H8.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat)
sub_megan = megan.C5H8.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat)
sub_upd_2 = update.C5H8.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat)

from collections import defaultdict

def scatter_filtering(actual, predicted, actual_times, predicted_times):
    # Group actual values by hour
    actual_by_hour = defaultdict(list)
    for a, t in zip(actual, actual_times):
        # Normalize time to hour resolution by removing minutes and seconds
        hour_time = t.replace(minute=0, second=0, microsecond=0)
        actual_by_hour[hour_time].append(a)
    
    # Calculate the average of actual values for each hour
    averaged_actual = {hour: np.mean(values) for hour, values in actual_by_hour.items()}
    
    # Filter out predicted values that have corresponding hourly averaged actual values
    filtered_predicted = [predicted[i] for i, t in enumerate(predicted_times) if t in averaged_actual]
    
    # Filter out averaged actual values that have corresponding predicted values
    filtered_actual = [averaged_actual[t] for t in predicted_times if t in averaged_actual]
    
    # Filter out pairs where 'actual' is NaN
    filtered_data = [(a, p) for a, p in zip(filtered_actual, filtered_predicted) if not isnan(a)]
    
    # Separate the filtered 'actual' and 'predicted' values
    if filtered_data:
        actual_filtered, predicted_filtered = zip(*filtered_data)
    else:
        actual_filtered, predicted_filtered = [], []

    return actual_filtered, predicted_filtered

conc, modello = scatter_filtering(ds,sub_upd_2,time2,local_times)
modello = np.array(modello)
conc = np.array(conc)
mbe = np.mean(modello - conc)

fig = plt.figure(figsize=(3,3))
ax1 = fig.add_subplot()
#c = ax1.scatter(conc,modello, s=10, color='green', edgecolor='black', linewidth=0.2)
c = ax1.scatter(conc,modello, s=10, color='blue', edgecolor='black', linewidth=0.2)
#c = ax1.scatter(conc,modello, s=10, color='orangered', edgecolor='black', linewidth=0.2)
#c = ax1.scatter(conc,modello, s=10, color='purple', edgecolor='black', linewidth=0.2)
t = np.linspace(0,20)
r0 = np.linspace(0,20)
y0 = 2*r0
y1 = 0.5*r0
#plt.text(0.05, 0.95, f"MBE: {mbe:.2f} ppbv", ha='left', va='top',
#         transform=plt.gca().transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.text(0.49, 0.11, f"MBE: {mbe:.2f} ppbv", ha='left', va='top',
         transform=plt.gca().transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
#cbar = plt.colorbar(c, fraction = 0.040, pad = 0.07,  extend="both")
#cbar.set_label(label='Data points density', y=0.5)
ax1.plot(r0,y0,'k--', alpha=0.8, linewidth=0.4)
ax1.plot(r0,y1,'k--', alpha=0.8, linewidth=0.4)
ax1.plot(r0,r0,'k--', alpha=0.8, linewidth=0.4)
ax1.grid(alpha=0.3, linewidth=0.4, zorder=-10)
ax1.set_ylabel("Model Isoprene [ppbv]",fontsize=12)
ax1.set_xlabel("Measured Isoprene [ppbv]",fontsize=12)
plt.xlim(0, 1)
plt.ylim(0, 1)
#ax1.set_yscale('log')
plt.yticks([0.0,0.3,0.6,0.9], [0.0,0.3,0.6,0.9], fontsize=12)
plt.xticks([0.0,0.3,0.6,0.9], [0.0,0.3,0.6,0.9], fontsize=12)
#ax1.set_xscale('log')
ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
fig.savefig("../figures/scatter_isoprene_pallas_UPD-cc.png", dpi=350, bbox_inches='tight')
plt.show()



