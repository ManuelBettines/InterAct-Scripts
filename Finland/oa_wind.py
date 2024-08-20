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

# Load misure OA
oa = pd.read_csv("../data/SMEAR/HydeOA.txt", sep="\t", na_values=["NaN"])
oa['TimelistLT_com'] = pd.to_datetime(oa['TimelistLT_com'], errors='coerce')  # Coerce errors if any invalid datetime strings
oa.set_index('TimelistLT_com', inplace=True)
oa = oa[(oa.index >= '2019-06-01') & (oa.index < '2019-09-01')]

# Load meteorological data
meteo = pd.read_csv("../data/Meteo/smeardata_20240609.txt", sep="\t", na_values=["NaN"])
meteo['datetime'] = pd.to_datetime(meteo[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']], errors='coerce')  # Coerce errors if any invalid datetime strings
meteo.set_index('datetime', inplace=True)
meteo = meteo[(meteo.index >= '2019-06-01') & (meteo.index < '2019-09-01')]

# Resample both datasets to hourly and interpolate missing values
oa_hourly = oa.resample('H').mean()
meteo_hourly = meteo.resample('H').mean()

# Ensure both datasets have the same time index and remove any rows with NaNs
combined_data = pd.concat([oa["OA_com"],meteo_hourly["HYY_META.WDU84"], meteo_hourly["HYY_META.WSU84"]], axis=1).dropna()

# Extract aligned data
conc = combined_data["OA_com"].values
misure = combined_data["HYY_META.WDU84"].values
speed = combined_data["HYY_META.WSU84"].values

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot()
scatter = ax.scatter(misure,conc, c=speed,s=50,cmap='viridis', vmin=0, vmax=2)
cbar = fig.colorbar(scatter, ax=ax, extend='both')
cbar.set_label('Wind Speed (m s$^{-1}$)', fontsize=18)
cbar.ax.tick_params(labelsize=15)
ax.set_ylabel('OA (µg m$^{-3}$)', fontsize=18)
ax.set_xlabel('Wind Direction (°)', fontsize=18)
ax.grid(True)
ax.tick_params(labelsize=15)
ax.set_ylim([0,17])
ax.set_xlim([0,360])
fig.savefig('../figures/scatter_oa_wind_inside.png', dpi=400, bbox_inches='tight')

# Load simulations output
oa = xr.open_dataset("../data/FINLAND6-CC-OA.nc")
base = xr.open_dataset("../data/FINLAND6-CC-meteo.nc")

def calculate_wind_direction(us, vs):
    # Calculate angles using arctan2, which is vectorized
    angles_radians = np.arctan2(vs, us)
    # Convert radians to degrees and adjust to meteorological convention
    wind_directions = 270 - np.degrees(angles_radians)
    # Ensure wind directions are within the range [0, 360)
    wind_directions = np.mod(wind_directions, 360)
    return wind_directions

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

# Select subset
winz = base.winz.sel(bottom_top=2).sel(x=idx_lon).sel(y=idx_lat)
winm = base.winm.sel(bottom_top=2).sel(x=idx_lon).sel(y=idx_lat)
sub = calculate_wind_direction(winz,winm)
speed = (winz**2+winm**2)**(1/2)
oa_sub = oa.OA.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat)*1.7

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot()
scatter = ax.scatter(sub,oa_sub, c=speed,s=50,cmap='viridis', vmin=0, vmax=6)
cbar = fig.colorbar(scatter, ax=ax, extend='both')
cbar.set_label('Wind Speed (m s$^{-1}$)', fontsize=18)
cbar.ax.tick_params(labelsize=15)
ax.set_ylabel('OA (µg m$^{-3}$)', fontsize=18)
ax.set_xlabel('Wind Direction (°)', fontsize=18)
ax.grid(True)
ax.tick_params(labelsize=15)
ax.set_ylim([0,17])
ax.set_xlim([0,360])
fig.savefig('../figures/scatter_oa_wind_inside_outside.png', dpi=400, bbox_inches='tight')


import matplotlib.colors as mcolors
import matplotlib.cm as cmx

# Define bins for wind speed (you can adjust the bins as needed)
speed_bins = [0,0.25,0.5,1,1.5,2]

# Define bins for wind direction (in degrees)
direction_bins = np.arange(-15, 360, 30)
direction_bins_wrapped = np.mod(direction_bins, 360)
# Adjust bins to be in [0, 360) range
direction_bins_ = np.arange(-22.5, 360, 45)
direction_bins_wrapped_c = np.mod(direction_bins_, 360)
direction_centers = (direction_bins_wrapped_c[:-1] + direction_bins_wrapped_c[1:]) / 2


# Convert centers to radians for plotting
theta = np.radians(direction_centers)

def plot_wind_rose(direction, speed, direction_bins, speed_bins):
    # Normalize direction to be within 0-360 degrees
    direction = np.mod(direction, 360)
    
    # Create the wind rose figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Compute histogram
    hist, _, _ = np.histogram2d(direction, speed, bins=[direction_bins, speed_bins])

    # Create a colormap
    cmap = plt.get_cmap('viridis')
    c_norm = mcolors.Normalize(vmin=min(speed_bins), vmax=max(speed_bins))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)

    # Define the width of each bar in radians
    width = np.deg2rad(np.diff(np.mod(direction_bins, 360)))

    # Adjust direction bins to match histogram shape
    direction_bins_wrapped_ = np.deg2rad(direction_bins_wrapped[:-1])

    # Plot each speed bin as a bar
    x = [4,3,2,1,0]
    for i in x:
        bars = ax.bar(
            direction_bins_wrapped_, hist[:, i], 
            width=width, color= scalar_map.to_rgba(speed_bins[i]),
            edgecolor='k', align='edge'
        )


    # Add color bar
    cbar = fig.colorbar(scalar_map, ax=ax, orientation='vertical', pad=0.1, extend='both')
    cbar.set_label('Wind Speed (m s$^{-1}$)', fontsize=18)
    
    # Add labels and title
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Windrose Inside Canopy', fontweight='bold', fontsize=21)
    #ax.set_xticks(np.radians(np.mod(direction_centers, 360)))
    #ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    
     # Fix ticks to ensure 'N' is displayed
    tick_locs = np.linspace(0, 2*np.pi, num=8, endpoint=False)
    tick_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=12)

    #ax.set_xticklabels([f'{int(np.degrees(angle))}°' for angle in np.radians(np.mod(direction_centers, 360))])
    ax.set_yticks([])
    #ax.set_title('Wind Rose', va='bottom')
    fig.savefig('../figures/windrose_windspeed_in.png', dpi=500)
    plt.show()

#plot_wind_rose(misure, speed, direction_bins, speed_bins)

# Define wind direction bins (e.g., every 45 degrees)
direction_bins = np.arange(-15, 360, 30)
direction_centers = np.arange(0, 360, 30)

# Bin wind directions and calculate average OA concentrations and counts
avg_oa_by_direction = []
count_by_direction = []
for i in range(len(direction_bins) - 1):
    lower = direction_bins[i]
    upper = direction_bins[i + 1]
    mask = ((misure >= lower) & (misure < upper)) | ((lower < 0) & (misure >= 360 + lower))
    avg_oa = np.mean(conc[mask])
    count = np.sum(mask)
    avg_oa_by_direction.append(avg_oa)
    count_by_direction.append(count)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

# Convert degrees to radians for polar plot
theta = np.radians(direction_centers)

# Normalize concentrations for colormap (adjust vmin and vmax as needed)
norm = plt.Normalize(vmin=min(avg_oa_by_direction), vmax=max(avg_oa_by_direction))

bars = ax.bar(theta, count_by_direction, width=np.radians(30), color=plt.cm.viridis(norm(avg_oa_by_direction)), edgecolor='black', linewidth=0.8)

# Add a colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, orientation='vertical', pad=0.1, extend='both')
cbar.set_label('NO$_3$ Concentration (µg m$^{-3}$)', fontsize=15)

# Add labels and title
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
ax.set_xticks(theta)
#ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
ax.set_xticklabels(['N', '30°', '60°', 'E', '120°', '150°', 'S', '210°', '240°', 'W', '300°', '330°'])
ax.set_yticks([])
ax.set_ylim([0,350])
ax.set_title('Average NO$_3$ Concentration by Wind Direction', pad = 20, fontweight='bold', fontsize=18)
#fig.savefig('../figures/NO3_windrose.png', dpi=500, bbox_inches='tight')
#plt.show()


#plt.show()
