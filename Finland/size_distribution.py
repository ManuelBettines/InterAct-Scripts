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

# Extract particle sizes from the column names
#particle_sizes = ['HYY_DMPS.d100e2', 'HYY_DMPS.d112e2', 'HYY_DMPS.d126e2', 'HYY_DMPS.d141e2', 'HYY_DMPS.d158e2', 'HYY_DMPS.d178e2', 'HYY_DMPS.d200e2', 'HYY_DMPS.d224e2', 'HYY_DMPS.d251e2', 'HYY_DMPS.d282e2', 'HYY_DMPS.d316e2', 'HYY_DMPS.d355e2', 'HYY_DMPS.d398e2', 'HYY_DMPS.d447e2', 'HYY_DMPS.d501e2', 'HYY_DMPS.d562e2', 'HYY_DMPS.d631e2', 'HYY_DMPS.d708e2', 'HYY_DMPS.d794e2', 'HYY_DMPS.d891e2', 'HYY_DMPS.d100e3', 'HYY_DMPS.d112e3', 'HYY_DMPS.d126e3', 'HYY_DMPS.d141e3', 'HYY_DMPS.d158e3', 'HYY_DMPS.d178e3', 'HYY_DMPS.d200e3', 'HYY_DMPS.d224e3', 'HYY_DMPS.d251e3', 'HYY_DMPS.d282e3', 'HYY_DMPS.d316e3', 'HYY_DMPS.d355e3', 'HYY_DMPS.d398e3', 'HYY_DMPS.d447e3', 'HYY_DMPS.d501e3', 'HYY_DMPS.d562e3', 'HYY_DMPS.d631e3', 'HYY_DMPS.d708e3', 'HYY_DMPS.d794e3', 'HYY_DMPS.d891e3', 'HYY_DMPS.d100e4']

particle_sizes = ['HYY_DMPS.d100e1', 'HYY_DMPS.d112e1', 'HYY_DMPS.d126e1', 'HYY_DMPS.d141e1', 'HYY_DMPS.d158e1', 'HYY_DMPS.d178e1', 'HYY_DMPS.d200e1', 'HYY_DMPS.d224e1', 'HYY_DMPS.d251e1', 'HYY_DMPS.d282e1', 'HYY_DMPS.d316e1', 'HYY_DMPS.d355e1', 'HYY_DMPS.d398e1', 'HYY_DMPS.d447e1', 'HYY_DMPS.d562e1', 'HYY_DMPS.d631e1', 'HYY_DMPS.d708e1', 'HYY_DMPS.d794e1', 'HYY_DMPS.d891e1','HYY_DMPS.d100e2', 'HYY_DMPS.d112e2', 'HYY_DMPS.d126e2', 'HYY_DMPS.d141e2', 'HYY_DMPS.d158e2', 'HYY_DMPS.d178e2', 'HYY_DMPS.d200e2', 'HYY_DMPS.d224e2', 'HYY_DMPS.d251e2', 'HYY_DMPS.d282e2', 'HYY_DMPS.d316e2', 'HYY_DMPS.d355e2', 'HYY_DMPS.d398e2', 'HYY_DMPS.d447e2', 'HYY_DMPS.d562e2', 'HYY_DMPS.d631e2', 'HYY_DMPS.d708e2', 'HYY_DMPS.d794e2', 'HYY_DMPS.d891e2','HYY_DMPS.d100e3', 'HYY_DMPS.d112e3', 'HYY_DMPS.d126e3', 'HYY_DMPS.d141e3', 'HYY_DMPS.d158e3', 'HYY_DMPS.d178e3', 'HYY_DMPS.d200e3', 'HYY_DMPS.d224e3', 'HYY_DMPS.d251e3', 'HYY_DMPS.d282e3', 'HYY_DMPS.d316e3', 'HYY_DMPS.d355e3', 'HYY_DMPS.d398e3', 'HYY_DMPS.d447e3', 'HYY_DMPS.d562e3', 'HYY_DMPS.d631e3', 'HYY_DMPS.d708e3', 'HYY_DMPS.d794e3', 'HYY_DMPS.d891e3','HYY_DMPS.d100e4']

# Extract size bins and convert to arrays
obs_particle_sizes = np.array([float(col.split('.')[1][1:]) for col in particle_sizes]) * 1e-3
obs_concentrations = observational_data[particle_sizes].mean().values  # Take mean across time or adjust as needed

obs_bin_widths = np.diff(np.log(obs_particle_sizes))
obs_concentrations = obs_concentrations
obs_concentrations_dNdlog = obs_concentrations

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot()
ax.plot(obs_particle_sizes, obs_concentrations_dNdlog, "ko", markersize=5, label="Observations")
ax.legend()
ax.set_ylabel("dN/dlogDp (# cm$^{-3}$)", fontsize=18)
ax.set_xlabel("Particles size (nm)", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.grid()
ax.set_xscale("log")
ax.set_ylim([0,3000])
#fig.savefig("../figures/size_distribution.png", dpi=500)

# Banana plot
fig = plt.figure(figsize=(50,10))
ax = fig.add_subplot()
norm = LogNorm(vmin=1, vmax=0.3e5)
c = plt.pcolormesh(time,obs_particle_sizes,observational_data[particle_sizes].T, cmap='RdYlBu_r',shading='gouraud', norm=norm)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.01,  extend="both")
cbar.set_label(label='dN/d(logD$_p$) (cm$^{-3}$)', fontsize=25, y=0.5)
cbar.ax.tick_params(labelsize=15)
plt.yscale("log")
ax.set_ylabel('Diameter (nm)', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=25)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
fig.autofmt_xdate(rotation=45)
ax.set_ylim([2.82,1e3])
ax.set_title('Hyytiälä size distribution', fontsize=31, fontweight='bold')
#plt.show()
fig.savefig('../figures/Banana_hyy_timeserie.png', dpi=500, bbox_inches='tight')


# Diurnal banana plot
# Create a column for hour and minute
observational_data['time_of_day'] = observational_data['datetime'].dt.hour + observational_data['datetime'].dt.minute / 60.0

# Group by time of day and average
diurnal_data = observational_data.groupby('time_of_day')[particle_sizes].mean()

# Plotting
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
fig.savefig('../figures/Banana_hyy_diurnal.png', dpi=500, bbox_inches='tight') 

# Ranking
sizes = ['HYY_DMPS.d251e1', 'HYY_DMPS.d282e1', 'HYY_DMPS.d316e1', 'HYY_DMPS.d355e1', 'HYY_DMPS.d398e1', 'HYY_DMPS.d447e1']
size_values = [2.24e-9, 2.51e-9, 2.82e-9, 3.16e-9, 3.55e-9, 3.98e-9, 4.47e-9]

# 2. Calculate bin widths (logarithmic)
size_values_log = np.log10(size_values)
bin_widths = np.diff(size_values_log)  

# 3. Convert dn/dlog(dp) to number concentration
for i, size in enumerate(sizes):
    observational_data[size] *= bin_widths[i]

# 4. Sum the concentrations to get the total concentration
observational_data['summed_concentration'] = observational_data[sizes].sum(axis=1)
observational_data['summed_concentration_smoothed'] = observational_data['summed_concentration'].rolling(12).median()

# Calculate daily min and max differences
observational_data.index = pd.to_datetime(time)
daily_diff = observational_data['summed_concentration_smoothed'].resample('D').agg(lambda x: x.max() - x.min())
observational_data['daily_diff'] = observational_data.index.normalize().map(daily_diff)

# Calculate percentiles based on daily differences
percentiles = daily_diff.rank(pct=True) * 100

# Assign ranking values based on percentiles
observational_data['ranking'] = observational_data.index.normalize().map(percentiles)

fig = plt.figure(figsize=(50,10))
ax = fig.add_subplot()
ax.plot(time,observational_data['summed_concentration_smoothed'], "ko", markersize=5, label="ΔN$_{max-min}$")
ax1 = ax.twinx()
ax1.plot(time,observational_data['ranking'], markersize=5, label="Ranking")
ax.legend()
ax.set_ylabel("Number Concentration (2.5 to 5 nm)", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.grid()
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
fig.savefig('../figures/numb_conc25_5.png', dpi=500, bbox_inches='tight')
#plt.show()

# Load meteorological data
meteo = pd.read_csv("../data/Meteo/smeardata_20240609.txt", sep="\t", na_values=["NaN"])
meteo['datetime'] = pd.to_datetime(meteo[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']], errors='coerce')
meteo.set_index('datetime', inplace=True)
meteo = meteo[(meteo.index >= '2019-06-01') & (meteo.index < '2019-09-01')]

# Resample both datasets to hourly and interpolate missing values
meteo_hourly = meteo.resample('H').mean()
misure = meteo_hourly["HYY_META.WDU336"].values
rank = observational_data.resample('H').mean()

combined_data = pd.concat([rank['ranking'], meteo_hourly["HYY_META.WDU336"]], axis=1).dropna()
combined_data = combined_data[(combined_data.index.hour >= 6) & (combined_data.index.hour <= 9)]

misure = combined_data["HYY_META.WDU336"].values
rank = combined_data["ranking"].values

# Define wind direction bins (e.g., every 45 degrees)
direction_bins = np.arange(-15, 360, 30)
direction_centers = np.arange(0, 360, 30)

# Bin wind directions and calculate average OA concentrations and counts
avg_ranking_by_direction = []
count_by_direction = []
for i in range(len(direction_bins) - 1):
    lower = direction_bins[i]
    upper = direction_bins[i + 1]
    mask = ((misure >= lower) & (misure < upper)) | ((lower < 0) & (misure >= 360 + lower))
    avg_r = np.mean(rank[mask])
    #if len(rank[mask]) > 0:
    #    avg_r = np.max(rank[mask])
    #else:
    #    avg_r = np.nan
    count = np.sum(mask)
    avg_ranking_by_direction.append(avg_r)
    count_by_direction.append(count)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

# Convert degrees to radians for polar plot
theta = np.radians(direction_centers)

# Normalize concentrations for colormap (adjust vmin and vmax as needed)
norm = plt.Normalize(vmin=40, vmax=70)
bars = ax.bar(theta, count_by_direction, width=np.radians(30), color=plt.cm.viridis(norm(avg_ranking_by_direction)), edgecolor='black', linewidth=0.8)

# Add a colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, orientation='vertical', pad=0.1, extend='both')
cbar.set_label('Average Ranking Percentile', fontsize=15)

# Add labels and title
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
ax.set_xticks(theta)
ax.set_xticklabels(['N', '30°', '60°', 'E', '120°', '150°', 'S', '210°', '240°', 'W', '300°', '330°'])
ax.set_yticks([])
#ax.set_ylim([0,350])
ax.set_title('Average Ranking value by Wind Direction', pad = 20, fontweight='bold', fontsize=18)
#fig.savefig('../figures/DeltaN_windrose.png', dpi=500, bbox_inches='tight')
#plt.show()

#####
# Aggregate particle size data by wind speed
observational_data['wind_dir'] = meteo["HYY_META.WDU336"]

# Define wind direction bins (e.g., every 45 degrees)
direction_bins = np.arange(-22.5, 360, 45)
direction_centers = np.arange(0, 360, 45)
theta = np.radians(direction_centers)

bins = np.arange(0, 360, 10)
# Function to assign wind direction to bins
def wind_direction_bins(direction):
    return bins[np.digitize(direction, bins) - 1]

# Assign wind direction bins
observational_data['wind_dir_bin'] = observational_data['wind_dir'].apply(wind_direction_bins)

# Group by wind direction bins and compute mean particle sizes
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

#ax2 = fig.add_axes(ax.get_position(), projection='polar', frame_on=False)
#ax2.plot(np.radians(np.linspace(0, 360, len(total_particle_sum))), total_particle_sum, color='black', marker='o', linewidth=2, label='Average particle concentration')

ax.set_title('Particle Size Distribution by Wind Direction', fontsize=18, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yscale('log')
ax.set_ylim([2.82, 2.4*1e2])
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
ax.set_xticks(theta)
ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
#ax.set_yticks([])
#ax2.set_ylim([0,60000])
#ax2.set_yscale('log')
#ax2.set_yticks([])
#ax2.grid(False)  
#ax2.patch.set_alpha(0)
#ax2.set_theta_direction(-1)
#ax2.set_theta_zero_location('N')
#ax2.set_xticks([])
#ax2.tick_params(axis='y', labelsize=15, direction='in', pad=-150) 
#lines, labels = ax2.get_legend_handles_labels()
#ax2.legend(lines, labels, loc='upper left')
fig.savefig('../figures/windrose_sizedis_rank90.png', dpi=500, bbox_inches='tight')
plt.show()

