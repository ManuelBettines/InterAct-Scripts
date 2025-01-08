import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
import glob

# Load model output
ds = xr.open_dataset("../data/TMP_10m.nc")

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(ds.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(ds.nav_lat[:,52], 61.8417)

times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=2)
ds['local_times'] = pd.DatetimeIndex(local_times)
ds['u10m'] = ds.u10m.swap_dims({'time_counter':'local_times'})
ds['v10m'] = ds.v10m.swap_dims({'time_counter':'local_times'})
ds = ds.assign_coords(hour=('local_times', local_times.hour))

#tem2_data = (ds['winz'].sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat)**2 + ds['winm'].sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat)**2)**(1/2)
tem2_data = (ds['u10m'].sel(x=idx_lon).sel(y=idx_lat)**2 + ds['v10m'].sel(x=idx_lon).sel(y=idx_lat)**2)**(1/2)
local_time = local_times

# Load observation
file_path_pattern = "../data/Meteo/METEO*" 

file_list = glob.glob(file_path_pattern)

all_data = []
for file in file_list:
    df = pd.read_csv(file, sep="\t", na_values=["NaN"])
    all_data.append(df)

observational_data_t = pd.concat(all_data, axis=0, ignore_index=True)

datetime_data = {
    'Year': observational_data_t['Year'],
    'Month': observational_data_t['Month'],
    'Day': observational_data_t['Day'],
    'Hour': observational_data_t['Hour'],
    'Minute': observational_data_t['Minute'],
    'Second': observational_data_t['Second']
}

time = pd.to_datetime(datetime_data)

observational_data_t['datetime'] = time
observational_data_t['date'] = observational_data_t['datetime'].dt.date
temp = observational_data_t['HYY_META.WSU84']

# Calculate diurnal
local_time = pd.to_datetime(local_time)
diurnal_cycle_model = tem2_data.groupby('hour').mean()
std_model = tem2_data.groupby('hour').std()

observational_data_t['hour'] = observational_data_t['datetime'].dt.hour 
diurnal_cycle_observation = observational_data_t.groupby('hour')['HYY_META.WSU84'].mean()
std_observation = observational_data_t.groupby('hour')['HYY_META.WSU84'].std()

time = np.linspace(0,23,24)

# Plot
fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot()
ax.plot(time, diurnal_cycle_observation, linewidth=2, color='#4169E1', label="Observations")
ax.plot(time, diurnal_cycle_model, linewidth=2, color='orangered', label="WRF-CHIMERE")
ax.fill_between(time, diurnal_cycle_observation - std_observation, diurnal_cycle_observation + std_observation, color='#4169E1', alpha=0.2)
ax.fill_between(time, diurnal_cycle_model - std_model, diurnal_cycle_model + std_model, color='orangered', alpha=0.2)
ax.legend(frameon=True, edgecolor="black", framealpha=0.8, facecolor="white", fontsize=9)
ax.set_ylabel("Wind speed [m s$^{-1}$]")
ax.set_xlabel("Datetime [Local Time]")
#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#ax.tick_params(axis='both', which='major', labelsize=15)
#ax.set_title('4 Meter Temperature', fontsize=21, fontweight='bold')
ax.grid(alpha=0.4, linewidth=0.5, zorder=-5)
ax.set_ylim([0,4.5])
ax.set_xlim([0,23])
fig.savefig("../figures/wind_cc10m_15.png", dpi=350, bbox_inches='tight')
plt.show()

