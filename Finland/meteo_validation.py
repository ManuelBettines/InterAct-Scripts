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


# Meteo Hyy
#meteo = pd.read_csv("../data/Meteo/smeardata_20240609.txt", sep="\t" ,na_values=["NaN"])
#meteo = meteo[meteo.Year.isin([2019])]
#meteo = meteo[meteo.Month.isin([6,7,8])]
#misure = meteo["HYY_META.WSU84"].values

#meteo['datetime'] = pd.to_datetime(meteo[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
#meteo.set_index('datetime', inplace=True)
#hourly_data = meteo.resample('H').mean()
#misure = hourly_data["HYY_META.WDU336"].values
#misure = hourly_data.reset_index()

#datetime_data = {
#    'Year': hourly_data['Year'],
#    'Month': hourly_data['Month'],
#    'Day': hourly_data['Day'],
#    'Hour': hourly_data['Hour']
#}

# Create datetime vector
#time2 = pd.to_datetime(datetime_data)

#misure = meteo.groupby([meteo.Hour]).mean()["HYY_META.T42"].values
#std = meteo.groupby([meteo.Hour]).std()["HYY_META.WSU84"].values
#lower_bound = misure - std
#upper_bound = misure + std

meteo = pd.read_csv("../data/Meteo/hyy_misure_fmi.csv", na_values=["-"])
#misure = meteo['Precipitation [mm]']
#misure = meteo['Average temperature [°C]']
misure = meteo['Average relative humidity [%]']

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-meteo.nc")
upd = xr.open_dataset("../data/FINLAND6-CC-meteo.nc")

# Account for time shift
times = base.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=3)
base['local_times'] = pd.DatetimeIndex(local_times)
upd['local_times'] = pd.DatetimeIndex(local_times)
base['relh'] = base.relh.swap_dims({'time_counter':'local_times'})
#base['winm'] = base.winm.swap_dims({'time_counter':'local_times'})
#base['topc'] = base.topc.swap_dims({'time_counter':'local_times'})
#upd['winz'] = upd.winz.swap_dims({'time_counter':'local_times'})
#upd['winm'] = upd.winm.swap_dims({'time_counter':'local_times'})

# Define the start and end dates
start_date = datetime(2019, 6, 1, 0, 0, 0)
end_date = datetime(2019, 8, 30, 23, 0, 0)

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

def calculate_wind_direction(us, vs):
    # Calculate angles using arctan2, which is vectorized
    angles_radians = np.arctan2(vs, us)
    # Convert radians to degrees and adjust to meteorological convention
    wind_directions = 270 - np.degrees(angles_radians)
    # Ensure wind directions are within the range [0, 360)
    wind_directions = np.mod(wind_directions, 360)
    return wind_directions

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

# Select subset
#winz = base.winz.sel(bottom_top=2).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))
#winm = base.winm.sel(bottom_top=2).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))
#sub = (winm**2+winz**2)**(1/2)
#winz = upd.winz.sel(bottom_top=2).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))
#winm = upd.winm.sel(bottom_top=2).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))
#sub_cc = (winm**2+winz**2)**(1/2)
#sub = calculate_wind_direction(winz,winm)
sub = base.relh.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))*100
#sub_base = np.array(sub.values).reshape(-1,24)
#model = np.mean(sub_base, axis=0)

time = [start_date + timedelta(hours=x) for x in range(2180)]
time2 = [start_date + timedelta(hours=x) for x in range(2184)]

mbe = mean_bias_error(misure,sub, time, time2)


# Plot results
fig = plt.figure(figsize=(30,6))
ax = fig.add_subplot()
ax.plot(time, sub, linewidth=5, label="WRF-CHIMERE")
#ax.plot(time, sub_cc, linewidth=5, label="WRF-CHIMERE with canopty correction")
ax.plot(time2, misure, "ko", markersize=3, label="Observations")
ax.text(0.25, 0.1, f'MBE = {mbe:.2f} %', transform=ax.transAxes, fontsize=18, verticalalignment='top')
#ax.text(0.55, 0.85, f'MBE (Updated) = {mbe_cc:.2f} m s-1', transform=ax.transAxes, fontsize=18, verticalalignment='top')
#ax.fill_between(time,lower_bound, upper_bound, color='gray', alpha=0.3)
ax.legend(loc="upper right")
ax.set_ylabel("Realtive Humidity (%)", fontsize=18)
#ax.set_ylabel("Precipitation (mm h$^{-1}$)", fontsize=18)
fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Relative Humidity',fontsize=21)
ax.grid()
ax.set_ylim([0,104])
fig.savefig("../figures/relh_hyy_ts.png", dpi=500)
