import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
import glob
from scipy.stats import gaussian_kde


# Load model output
ds = xr.open_dataset("../data/meteo_FINLAND6.nc")

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
ds['winz'] = ds.winz.swap_dims({'time_counter':'local_times'})
ds['winm'] = ds.winm.swap_dims({'time_counter':'local_times'})
ds = ds.assign_coords(hour=('local_times', local_times.hour))

winm = ds.winm.sel(bottom_top=2).sel(x=idx_lon).sel(y=idx_lat)
winz = ds.winz.sel(bottom_top=2).sel(x=idx_lon).sel(y=idx_lat)

model = (np.arctan2(-winz, -winm) * 180 / np.pi) % 360
model_times = ds['local_times'].values  
model_times = pd.to_datetime(model_times)

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
observational_data_t = observational_data_t.sort_values(by='datetime')

observational_data_t['datetime'] = pd.to_datetime(observational_data_t['datetime'], errors='coerce')
model_times = pd.to_datetime(ds['local_times'].values, errors='coerce')

model_times_trimmed = model_times[model_times <= observational_data_t['datetime'].max()]
model_trimmed = model.sel(local_times=model_times_trimmed)
model_interpolated = pd.Series(model_trimmed.values, index=model_times_trimmed).reindex(observational_data_t['datetime']).interpolate('time')

#observational_data_t.set_index('datetime', inplace=True)
#observational_temps = observational_data_t['HYY_META.T42'].resample('D').mean()
#observational_temps = observational_data_t['HYY_META.T42']

model_df = pd.DataFrame({
    'model': model_interpolated
}, index=observational_data_t['datetime'])

def circular_mean(degrees):
    radians = np.deg2rad(degrees)
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))
    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.rad2deg(mean_angle)
    return (mean_angle_deg + 360) % 360

model_daily_avg = model_df.resample('H').apply(circular_mean)

observational_data_t.set_index('datetime', inplace=True)
observational_temps = observational_data_t['HYY_META.WDU336'].resample('H').apply(circular_mean)

def compute_density(x, y):
    positions = np.vstack([x, y])
    kde = gaussian_kde(positions)
    densities = kde(positions)
    return densities

observational_temps_array = np.asarray(observational_temps).flatten()
model_daily_avg_array = np.asarray(model_daily_avg).flatten()

valid_indices = ~np.isnan(observational_temps_array) & ~np.isinf(observational_temps_array) & \
                ~np.isnan(model_daily_avg_array) & ~np.isinf(model_daily_avg_array)

observational_temps_clean = observational_temps_array[valid_indices]
model_daily_avg_clean = model_daily_avg_array[valid_indices]

densities = compute_density(observational_temps_clean, model_daily_avg_clean)

#mbe = np.mean(model_daily_avg_clean - observational_temps_clean)

def circular_mbe(predicted, observed):
    # Convert to radians
    predicted_radians = np.deg2rad(predicted)
    observed_radians = np.deg2rad(observed)
    
    # Calculate the difference (Model - Observation)
    bias_radians = predicted_radians - observed_radians
    
    # Normalize the differences to the range [-pi, pi] (i.e., [-180°, 180°])
    bias_radians = np.arctan2(np.sin(bias_radians), np.cos(bias_radians))
    
    # Convert back to degrees
    bias_degrees = np.rad2deg(bias_radians)
    
    # Calculate the Mean Bias Error (MBE) as the mean of these differences
    mbe = np.mean(bias_degrees)
    
    return mbe

mbe = circular_mbe(model_daily_avg_clean, observational_temps_clean)

# Plot
fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot()
c = ax.scatter(observational_temps_clean, model_daily_avg_clean, c=100*densities, cmap='viridis', s=1)
ax.set_xlabel("Measured wind direction [°]")
ax.set_ylabel("Model wind direction [°]")
r0 = np.linspace(0,360)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.07,  extend="both")
cbar.set_label(label='Data points density', y=0.5)
#plt.text(0.05, 0.95, f"MBE: {mbe:.2f} m s$^{-1}$", ha='left', va='top', transform=plt.gca().transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.text(0.49, 0.11, r"MBE: {:.2f} °".format(mbe), ha='left', va='top', transform=plt.gca().transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
y0 = 2*r0
y1 = 0.5*r0
#ax.plot(r0,y0,'k--', alpha=0.8, linewidth=0.4)
#ax.plot(r0,y1,'k--', alpha=0.8, linewidth=0.4)
ax.plot(r0,r0,'k--', alpha=0.8, linewidth=0.4)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#plt.xticks([0,1.5,3,4.5,6], [0.0,1.5,3.0,4.5,6.0])  # Custom x-ticks
plt.yticks([0,100,200,300], [0,100,200,300])
ax.grid(alpha=0.3, linewidth=0.4, zorder=-10)
ax.set_ylim([0,360])
ax.set_xlim([0,360])
plt.gca().set_aspect('equal', adjustable='box')
fig.savefig("../figures/wind_dir_scatter.png", dpi=350, bbox_inches='tight')
plt.show()

