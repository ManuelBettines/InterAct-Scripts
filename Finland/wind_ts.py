import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
import glob
from scipy.stats import gaussian_kde


# Load model output
ds = xr.open_dataset("../data/tmp_win.nc")
ds1 = xr.open_dataset("../data/tmp_usta.nc")
ds2 = xr.open_dataset("../data/TMP_10m.nc")


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(ds.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(ds.nav_lat[:,52], 61.8417)

times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
#local_times = utc_times + DateOffset(hours=2)
#ds['local_times'] = pd.DatetimeIndex(local_times)
#ds['winz'] = ds.winz.swap_dims({'time_counter':'local_times'})
#ds['winm'] = ds.winm.swap_dims({'time_counter':'local_times'})
#ds = ds.assign_coords(hour=('local_times', local_times.hour))
#ds2['local_times'] = pd.DatetimeIndex(local_times)
#ds2['u10m'] = ds2.u10m.swap_dims({'time_counter':'local_times'})
#ds2['v10m'] = ds2.v10m.swap_dims({'time_counter':'local_times'})
#ds2 = ds2.assign_coords(hour=('local_times', local_times.hour))
#ds1['local_times'] = pd.DatetimeIndex(local_times)
#ds1['usta'] = ds1.usta.swap_dims({'time_counter':'local_times'})
#ds1 = ds1.assign_coords(hour=('local_times', local_times.hour))


model = (ds['winz'].sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat)**2 + ds['winm'].sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat)**2)**(1/2)
model_times = ds['Times'].values  
model_times = pd.to_datetime(model_times)

model_10 = (ds2['u10m'].sel(x=idx_lon).sel(y=idx_lat)**2 + ds2['v10m'].sel(x=idx_lon).sel(y=idx_lat)**2)**(1/2)
model_usta = ds1['usta'].sel(x=idx_lon).sel(y=idx_lat)


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

# Remove NaN data and align model and observation
#observational_data_t = observational_data_t.sort_values(by='datetime')

#observational_data_t['datetime'] = pd.to_datetime(observational_data_t['datetime'], errors='coerce')
#model_times = pd.to_datetime(ds['Times'].values, errors='coerce')

#model_times_trimmed = model_times[model_times <= observational_data_t['datetime'].max()]
#model_trimmed = model.sel(local_times=model_times_trimmed)
#model_interpolated = pd.Series(model_trimmed.values, index=model_times_trimmed).reindex(observational_data_t['datetime']).interpolate('time')

#observational_data_t.set_index('datetime', inplace=True)
#observational_temps = observational_data_t['HYY_META.T42'].resample('D').mean()
#observational_temps = observational_data_t['HYY_META.T42']

#model_df = pd.DataFrame({
#    'model': model_interpolated
#}, index=observational_data_t['datetime'])

#model_daily_avg = model_df.resample('H').mean()

#observational_data_t.set_index('datetime', inplace=True)
#observational_temps = observational_data_t['HYY_META.WSU84'].resample('H').mean()

#observational_temps_array = np.asarray(observational_temps).flatten()
#model_daily_avg_array = np.asarray(model_daily_avg).flatten()

#valid_indices = ~np.isnan(observational_temps_array) & ~np.isinf(observational_temps_array) & \
#                ~np.isnan(model_daily_avg_array) & ~np.isinf(model_daily_avg_array)

#observational_temps_clean = observational_temps_array[valid_indices]
#model_daily_avg_clean = model_daily_avg_array[valid_indices]

#time_index = observational_temps.index[valid_indices]


# Plot
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot()
ax.plot(utc_times,model, color='green', label='Wind first layer = 8m')
ax1 = ax.twinx()
ax1.plot(utc_times,model_usta[:-1],color='blue', label='ustar')
ax.plot(utc_times,model_10[:-1], color='red',label='Wind 10 m')
#ax.plot(time_index, observational_temps_clean, 'ko', markersize=2)
ax.set_xlabel("Date")
ax.set_ylabel("Wind speed [m s$^{-1}$]")
ax1.set_ylabel("u* [m s$^{-1}$]")
# Highlight night hours with fill_between
night_start = 6  # Start of night (e.g., 6 PM)
night_end = 18     # End of night (e.g., 6 AM)

#or date in .normalize().unique():  # Normalize to get unique dates
#    night_start_time = date + pd.Timedelta(hours=night_start)
#    night_end_time = date + pd.Timedelta(hours=24) if night_end < night_start else date + pd.Timedelta(hours=night_end)
#    ax.fill_betweenx(model.min(),model.max(),night_start_time, night_end_time,color='gray', alpha=0.1)
ax.legend(loc=0)
ax1.legend(loc='upper left')
#ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
fig.savefig("../figures/tmp_all.png", dpi=350, bbox_inches='tight')
plt.show()

