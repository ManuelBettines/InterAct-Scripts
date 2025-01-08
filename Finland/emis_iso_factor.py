import numpy as np
from scipy.stats import gaussian_kde
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
from matplotlib.ticker import LogLocator, FuncFormatter
import glob

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
}

time = pd.to_datetime(datetime_data)

observational_data_t['datetime'] = time
observational_data_t['date'] = observational_data_t['datetime'].dt.date


# Load observation
file_path_pattern = "../data/Meteo/OZO*"

file_list = glob.glob(file_path_pattern)

all_data = []
for file in file_list:
    df = pd.read_csv(file, sep="\t", na_values=["NaN"])
    all_data.append(df)

observational_data_co2 = pd.concat(all_data, axis=0, ignore_index=True)

datetime_data = {
    'Year': observational_data_co2['Year'],
    'Month': observational_data_co2['Month'],
    'Day': observational_data_co2['Day'],
    'Hour': observational_data_co2['Hour'],
}

time = pd.to_datetime(datetime_data)

observational_data_co2['datetime'] = time
observational_data_co2['date'] = observational_data_co2['datetime'].dt.date

# Load observation
file_path_pattern = "../data/Meteo/RAD*"

file_list = glob.glob(file_path_pattern)

all_data = []
for file in file_list:
    df = pd.read_csv(file, sep="\t", na_values=["NaN"])
    all_data.append(df)

observational_data_rad = pd.concat(all_data, axis=0, ignore_index=True)

datetime_data = {
    'Year': observational_data_rad['Year'],
    'Month': observational_data_rad['Month'],
    'Day': observational_data_rad['Day'],
    'Hour': observational_data_rad['Hour'],
}

time = pd.to_datetime(datetime_data)

observational_data_rad['datetime'] = time
observational_data_rad['date'] = observational_data_rad['datetime'].dt.date


# Emissioni VOC
VOC_em = pd.read_csv("../data/VOC/VOC_fluxes/VOC_profileflux_all_data_2010_2023.txt", na_values=["NaN"])

for col in VOC_em.columns:
    VOC_em[col] = VOC_em[col].astype(float)

VOC_em = VOC_em[VOC_em.Year.isin([2017,2018,2019])]
VOC_em = VOC_em[VOC_em.Month.isin([6,7,8])]

datetime_data = {
    'Year': VOC_em['Year'],
    'Month': VOC_em['Month'],
    'Day': VOC_em['Day'],
    'Hour': VOC_em['Hour'],
}

time = pd.to_datetime(datetime_data)

VOC_em['datetime'] = time
VOC_em['date'] = VOC_em['datetime'].dt.date

# Load simulations output
update = xr.open_dataset("../data/FINLAND6_emis_iso.nc")
temp = xr.open_dataset("../data/FIN6-temp.nc")

# Account for time shift
times = update.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=3)
update['local_times'] = pd.DatetimeIndex(local_times)
update['C5H8_b'] = update.C5H8_b.swap_dims({'time_counter':'local_times'})
temp['local_times'] = pd.DatetimeIndex(local_times)
temp['tem2'] = temp.tem2.swap_dims({'time_counter':'local_times'})

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(update.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(update.nav_lat[:,52], 61.8417)

# Select subset
model = update.C5H8_b.sel(x=idx_lon).sel(y=idx_lat)*1e6
temp_model = temp.tem2.sel(x=idx_lon).sel(y=idx_lat) - 273.15

# Remove NaN data and align model and observation
VOC_em = VOC_em.sort_values(by='datetime')
observational_data_t = observational_data_t.sort_values(by='datetime')
observational_data_rad = observational_data_rad.sort_values(by='datetime')
observational_data_co2 = observational_data_co2.sort_values(by='datetime')

observational_data_t['datetime'] = pd.to_datetime(observational_data_t['datetime'], errors='coerce')
observational_data_rad['datetime'] = pd.to_datetime(observational_data_rad['datetime'], errors='coerce')
observational_data_co2['datetime'] = pd.to_datetime(observational_data_co2['datetime'], errors='coerce')

VOC_em = VOC_em[['datetime', 'Isoprene']]
observational_data_t = observational_data_t[['datetime', 'HYY_META.T42']]
observational_data_rad = observational_data_rad[['datetime', 'HYY_META.Glob']]
observational_data_co2 = observational_data_co2[['datetime', 'HYY_META.CO2icos168']]

merged_data = observational_data_co2.merge(observational_data_rad, on='datetime', how='outer')
tmp = merged_data[merged_data['HYY_META.Glob']>0]

tmp.set_index('datetime', inplace=True)
tmp = tmp.resample('h').first().reset_index()
#tmp.drop_duplicates(subset='datetime', inplace=True)
observational_data_t.drop_duplicates(subset='datetime', inplace=True)

tmp = tmp.merge(observational_data_t, on='datetime', how='outer')

merged_data = VOC_em.merge(tmp, on='datetime', how='outer')

VOC_em = merged_data

VOC_em['datetime'] = pd.to_datetime(VOC_em['datetime'], errors='coerce')
model_times = pd.to_datetime(update['local_times'].values, errors='coerce')

model_times_trimmed = model_times[model_times <= VOC_em['datetime'].max()]
model_trimmed = model.sel(local_times=model_times_trimmed)
model_interpolated = pd.Series(model_trimmed.values, index=model_times_trimmed).reindex(VOC_em['datetime']).interpolate('time')

temp_trimmed = temp_model.sel(local_times=model_times_trimmed)
temp_interpolated = pd.Series(temp_trimmed.values, index=model_times_trimmed).reindex(VOC_em['datetime']).interpolate('time')


model_df = pd.DataFrame({
    'model': model_interpolated
}, index=VOC_em['datetime'])

model_temp_df = pd.DataFrame({
    'model_temp': temp_interpolated
}, index=VOC_em['datetime'])


model_daily_avg = model_df.resample('h').mean()
temp_daily_avg = model_temp_df.resample('h').mean()

VOC_em.set_index('datetime', inplace=True)

observational_iso = VOC_em['Isoprene'].resample('h').mean()
observational_rad = VOC_em['HYY_META.Glob'].resample('h').mean()
observational_temp = VOC_em['HYY_META.T42'].resample('h').mean()
observational_co2 = VOC_em['HYY_META.CO2icos168'].resample('h').mean()

print(observational_co2)

observational_co2_array = np.asarray(observational_co2).flatten()
observational_rad_array = np.asarray(observational_rad).flatten()
observational_temp_array = np.asarray(observational_temp).flatten()
observational_iso_array = np.asarray(observational_iso).flatten()
model_daily_avg_array = np.asarray(model_daily_avg).flatten()
temp_daily_avg_array = np.asarray(temp_daily_avg).flatten()

valid_indices = ~np.isnan(observational_iso_array) & ~np.isinf(observational_iso_array) & \
                ~np.isnan(model_daily_avg_array) & ~np.isinf(model_daily_avg_array)

temp_daily_avg_clean = temp_daily_avg_array[valid_indices]
observational_iso_clean = observational_iso_array[valid_indices]
model_daily_avg_clean = model_daily_avg_array[valid_indices]
observational_temp_clean = observational_temp_array[valid_indices]
observational_rad_clean = observational_rad_array[valid_indices]
observational_co2_clean = observational_co2_array[valid_indices]

gamma = 1.344 - ((1.344*(0.7*observational_co2_clean)**1.4614)/(585**1.4614+(0.7*observational_co2_clean)**1.4614))
model_corrected = model_daily_avg_clean/gamma
print(gamma)


# Plot
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot()
c = ax.scatter(observational_iso_clean, model_daily_avg_clean, s=10, edgecolor='black', linewidth=0.2, c=observational_co2_clean, cmap='viridis', vmin=390, vmax=420)
ax.set_xlabel("Measured emissions [μg m$^{-2}$ s$^{-1}$]")
ax.set_ylabel("Model emissions [μg m$^{-2}$ s$^{-1}$]")
r0 = np.linspace(0,2)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.07,  extend="both")
#cbar.set_label(label='Temperature [°C]', y=0.5)
cbar.set_label(label='CO$_2$ [ppmv]', y=0.5)
y0 = 10*r0
y1 = (1/10)*r0
ax.plot(r0,y0,'k--', alpha=0.8, linewidth=0.4)
ax.plot(r0,y1,'k--', alpha=0.8, linewidth=0.4)
ax.plot(r0,r0,'k--', alpha=0.8, linewidth=0.4)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.grid(alpha=0.3, linewidth=0.4, zorder=-10)
#ax.set_ylim(top=0.15)
#ax.set_xlim(right=0.15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([1e-7,2])
ax.set_xlim([1e-7,2])
x_major = LogLocator(base=10.0, subs=(1.0,), numticks=10)
x_minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
plt.gca().xaxis.set_major_locator(x_major)
plt.gca().xaxis.set_minor_locator(x_minor)
y_major = LogLocator(base=10.0, subs=(1.0,), numticks=10)
y_minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
plt.gca().yaxis.set_major_locator(y_major)
plt.gca().yaxis.set_minor_locator(y_minor)
tick_locations = [1e-7, 1e-5, 1e-3, 1e-1]
plt.xticks(tick_locations)
plt.yticks(tick_locations)
formatter = FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x != 0 else '0')
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().set_aspect('equal', adjustable='box')
#fig.savefig("../figures/isoprene_emissions_scatter_with_co2_corrected.png", dpi=350, bbox_inches='tight')
plt.show()

bias = (model_daily_avg_clean - observational_iso_clean)

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot()
c = ax.scatter(observational_rad_clean, observational_iso_clean, c=observational_co2_clean, cmap='viridis', s=10, edgecolor='black', linewidth=0.2, vmin=385,vmax=420)
ax.set_xlabel("Surface radiation [W m$^{-2}$]")
ax.grid(alpha=0.3, linewidth=0.4, zorder=-10)
#ax.set_xlabel("Model temperature [°C]")
ax.set_ylabel("Isoprene emissions [μg m$^{-2}$ s$^{-1}$]")
ax.set_yscale('log')
#ax.set_xscale('log')
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.07,  extend="both")
#cbar.set_label(label='Observed temperature [°C]', y=0.5)
cbar.set_label(label='CO$_2$ [ppmv]', y=0.5)
#cbar.set_label(label='Surface radiation [W m$^{-2}$]', y=0.5)
ax.set_ylim([1e-7,2])
tick_locations = [1e-7, 1e-5, 1e-3, 1e-1]
plt.yticks(tick_locations)
#ax.set_xlim([1e-7,])
#fig.savefig("../figures/isoprene_emissions_model_radiations_log.png", dpi=350, bbox_inches='tight')
plt.show()

#bias = (model_daily_avg_clean - observational_iso_clean)
bias = np.where(observational_iso_clean > 0, np.abs(np.log(model_daily_avg_clean) - np.log(observational_iso_clean)), np.nan)
observational_rad_clean = np.where(observational_iso_clean > 0, observational_co2_clean, np.nan)

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot()
c = ax.scatter(observational_rad_clean, bias, color='deepskyblue', s=10, edgecolor='black', linewidth=0.2)
ax.set_xlabel("CO$_2$ [ppmv]")
#ax.set_xlabel("O$_3$ [ppbv]")
#ax.set_xlabel("Model temperature [°C]")
#ax.set_xlabel("Surface radiation [W m$^{-2}$]")
ax.grid(alpha=0.3, linewidth=0.4, zorder=-10)
ax.set_ylabel("Isoprene emissions LAD")
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_ylim(bottom=0)
#ax.set_xlim(left=0)
#tick_locations = [1e-7, 1e-5, 1e-3, 1e-1]
#plt.yticks(tick_locations)
#ax.set_xlim([1e-7,])
fig.savefig("../figures/isoprene_emissions_model_bias_co2.png", dpi=350, bbox_inches='tight')
plt.show()
