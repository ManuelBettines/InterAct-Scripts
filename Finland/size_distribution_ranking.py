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
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter


# Size distribution
observational_data = pd.read_csv("../data/SMEAR/smeardata_20240618.txt",sep="\t" ,na_values=["NaN"])

datetime_data = {
    'Year': observational_data['Year'],
    'Month': observational_data['Month'],
    'Day': observational_data['Day'],
    'Hour': observational_data['Hour'],
    'Minute': observational_data['Minute'],
    'Second': observational_data['Second']
}

# Create datetime vector
time = pd.to_datetime(datetime_data)

# Adding time to the dataframe
observational_data['datetime'] = time

particle_sizes = ['HYY_DMPS.d100e1', 'HYY_DMPS.d112e1', 'HYY_DMPS.d126e1', 'HYY_DMPS.d141e1', 'HYY_DMPS.d158e1', 'HYY_DMPS.d178e1', 'HYY_DMPS.d200e1', 'HYY_DMPS.d224e1', 'HYY_DMPS.d251e1', 'HYY_DMPS.d282e1', 'HYY_DMPS.d316e1', 'HYY_DMPS.d355e1', 'HYY_DMPS.d398e1', 'HYY_DMPS.d447e1', 'HYY_DMPS.d562e1', 'HYY_DMPS.d631e1', 'HYY_DMPS.d708e1', 'HYY_DMPS.d794e1', 'HYY_DMPS.d891e1','HYY_DMPS.d100e2', 'HYY_DMPS.d112e2', 'HYY_DMPS.d126e2', 'HYY_DMPS.d141e2', 'HYY_DMPS.d158e2', 'HYY_DMPS.d178e2', 'HYY_DMPS.d200e2', 'HYY_DMPS.d224e2', 'HYY_DMPS.d251e2', 'HYY_DMPS.d282e2', 'HYY_DMPS.d316e2', 'HYY_DMPS.d355e2', 'HYY_DMPS.d398e2', 'HYY_DMPS.d447e2', 'HYY_DMPS.d562e2', 'HYY_DMPS.d631e2', 'HYY_DMPS.d708e2', 'HYY_DMPS.d794e2', 'HYY_DMPS.d891e2','HYY_DMPS.d100e3', 'HYY_DMPS.d112e3', 'HYY_DMPS.d126e3', 'HYY_DMPS.d141e3', 'HYY_DMPS.d158e3', 'HYY_DMPS.d178e3', 'HYY_DMPS.d200e3', 'HYY_DMPS.d224e3', 'HYY_DMPS.d251e3', 'HYY_DMPS.d282e3', 'HYY_DMPS.d316e3', 'HYY_DMPS.d355e3', 'HYY_DMPS.d398e3', 'HYY_DMPS.d447e3', 'HYY_DMPS.d562e3', 'HYY_DMPS.d631e3', 'HYY_DMPS.d708e3', 'HYY_DMPS.d794e3', 'HYY_DMPS.d891e3','HYY_DMPS.d100e4']

# Ranking
sizes = ['HYY_DMPS.d251e1', 'HYY_DMPS.d282e1', 'HYY_DMPS.d316e1', 'HYY_DMPS.d355e1', 'HYY_DMPS.d398e1', 'HYY_DMPS.d447e1']
size_values = [2.24e-9, 2.51e-9, 2.82e-9, 3.16e-9, 3.55e-9, 3.98e-9, 4.47e-9]

# 2. Calculate bin widths (logarithmic)
size_values_log = np.log10(size_values)
bin_widths = np.diff(size_values_log)
modified_data = observational_data.copy()

# 3. Convert dn/dlog(dp) to number concentration
for i, size in enumerate(sizes):
    modified_data[size] *= bin_widths[i]

# 4. Sum the concentrations to get the total concentration
observational_data['summed_concentration'] = modified_data[sizes].sum(axis=1)
observational_data['summed_concentration_smoothed'] = observational_data['summed_concentration'].rolling(12).median()

# Calculate daily min and max differences
observational_data.index = pd.to_datetime(time)
daily_diff = observational_data['summed_concentration_smoothed'].resample('D').agg(lambda x: x.max() - x.min())
observational_data['daily_diff'] = observational_data.index.normalize().map(daily_diff)

# Calculate percentiles based on daily differences
percentiles = daily_diff.rank(pct=True) * 100

# Assign ranking values based on percentiles
observational_data['ranking'] = observational_data.index.normalize().map(percentiles)
#observational_data = observational_data[observational_data['ranking'] > 90]

# Diurnal banana plot
# Extract size bins and convert to arrays
obs_particle_sizes = np.array([float(col.split('.')[1][1:]) for col in particle_sizes]) * 1e-3
obs_concentrations = observational_data[particle_sizes].mean().values  # Take mean across time or adjust as needed

obs_bin_widths = np.diff(np.log(obs_particle_sizes))
obs_concentrations = obs_concentrations
obs_concentrations_dNdlog = obs_concentrations

# Create a column for hour and minute
observational_data['time_of_day'] = observational_data['datetime'].dt.hour + observational_data['datetime'].dt.minute / 60.0
diurnal_data = observational_data.groupby('time_of_day')[particle_sizes].mean()

# Plot Diurnal
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot()
norm = LogNorm(vmin=1, vmax=1e4)
c = plt.pcolormesh(diurnal_data.index, obs_particle_sizes, diurnal_data.T, cmap='RdYlBu_r',shading='gouraud', norm=norm)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.01,  extend="both")
cbar.set_label(label='dN/d(logD$_p$) (cm$^{-3}$)', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)
ax.set_xlabel('Datetime', fontsize=18)
ax.set_ylabel('Particles size (nm)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.yscale('log')
plt.ylim([2.82,1e3])
#fig.savefig('../figures/Banana_hyy_diurnal_rank95.png', dpi=500, bbox_inches='tight') 

# Load meteorological data
meteo = pd.read_csv("../data/Meteo/smeardata_20240609.txt", sep="\t", na_values=["NaN"])
meteo['datetime'] = pd.to_datetime(meteo[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']], errors='coerce')
meteo.set_index('datetime', inplace=True)
meteo = meteo[(meteo.index >= '2019-06-01') & (meteo.index < '2019-09-01')]

# Resample both datasets to hourly and interpolate missing values
#meteo_hourly = meteo.resample('H').mean()
meteo_sel = meteo[(meteo.index.hour >= 0) | (meteo.index.hour <= 24)]
#meteo_sel = meteo[(meteo.index.hour >= 0) & (meteo.index.hour <= 7)]

# Aggregate particle size data by wind speed
observational_data['wind_dir'] = meteo_sel["HYY_META.WDU336"]

# Define wind dairection bins (e.g., every 45 degrees)
direction_bins = np.arange(-22.5, 360, 45)
direction_centers = np.arange(0, 360, 45)
theta = np.radians(direction_centers)

bins = np.arange(0, 360, 10)
# Function to assign wind direction to bins
def wind_direction_bins(direction):
    return bins[np.digitize(direction, bins) - 1]

# Assign wind direction bins
observational_data['wind_dir_bin'] = observational_data['wind_dir'].apply(wind_direction_bins)

# Group by wind direaction bins and compute mean particle sizes
diurnal_data = observational_data.groupby('wind_dir_bin')[particle_sizes].mean()
total_particle_sum = diurnal_data.sum(axis=1)
total_particle_sum = total_particle_sum.append(pd.Series(total_particle_sum.iloc[0]), ignore_index=True)

# Plotting
fig, ax =  plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})

norm = LogNorm(vmin=100, vmax=7*1e3)
c = ax.pcolormesh(np.radians(np.linspace(0, 360, len(diurnal_data))), obs_particle_sizes, diurnal_data.values.T, cmap='RdYlBu_r', norm=norm)
cbar = plt.colorbar(c, fraction=0.040, pad=0.08, extend="both")
cbar.set_label(label='dN/d(logD$_p$) (cm$^{-3}$)', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)

ax2 = fig.add_axes(ax.get_position(), projection='polar', frame_on=False)
ax2.plot(np.radians(np.linspace(0, 360, len(total_particle_sum))), total_particle_sum, color='black', marker='o', linewidth=2, label='Average total particle concentration (x 10$^4$ cm$^{-3}$)')

ax.set_title('Particle Size Distribution by Wind Direction', fontsize=18, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yscale('log')
ax.set_ylim([2.82, 2.4*1e2])
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
ax.set_xticks(theta)
ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
#ax.set_yticks([])
ax2.set_ylim([0,60000])
#ax2.set_yscale('log')
#ax2.set_yticks([])
ax2.grid(True, which='both', color='black', linestyle='--', linewidth=0.5)
ax2.patch.set_alpha(0)
ax2.set_theta_direction(-1)
ax2.set_theta_zero_location('N')
ax2.set_xticks([])
# Adjust tick labels manually
yticks = [10000, 30000, 50000]#, 70000, 90000, 110000] 
ax2.set_yticks(yticks)

def scientific_format(x, pos):
    if x >= 10000:
        return f'{int(x/10000)}'
    else:
        return f'{int(x)}'

formatter = FuncFormatter(scientific_format)
ax2.yaxis.set_major_formatter(formatter)
ax2.tick_params(axis='y', labelsize=15)

# Positioning the labels
for label, tick in zip(ax2.get_yticklabels(), yticks):
    label.set_verticalalignment('bottom')
    label.set_position((550, tick)) 
ax2.tick_params(axis='y', labelsize=15) 
lines, labels = ax2.get_legend_handles_labels()
ax2.legend(lines, labels, loc='lower left', bbox_to_anchor=(-0.1, -0.1))
fig.savefig('../figures/windrose_sizedis_par.png', dpi=500, bbox_inches='tight')
plt.show()

