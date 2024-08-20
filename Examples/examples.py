import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime
from pandas.tseries.offsets import DateOffset

# Extract lat/lon CHIMERE index (for Mercatore/LatLon projections) 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(ds.lon[0,:], INSERT_LON)
idx_lat = find_nearest(ds.lat[:,0], INSERT_LAT)

# Load CHIMERE output
ds = xr.open_dataset("path_to_output/file_name.nc")

# Subset CHIMERE output
sub_PM25_ts = ds.PM25.sel(bottom_top=0).sel(south_north=idx_lat).sel(west_east=idx_lon) # Ground level PM2.5 timeseries at specific lat lon
sub_PM25_map = ds.PM25.sel(bottom_top=0).mean('Time') # Average over Time for each grid cell

# Convert Output from UTC to Local Time (if needed)
times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f') # This format works woth the new version of CHIMERE, old version might be different
local_times = utc_times + DateOffset(hours=3) # Insert difference between UTC and Local Time
ds['local_times'] = pd.DatetimeIndex(local_times)
ds['PM25'] = ds.PM25.swap_dims({'Time':'local_times'})

# Time series plot
fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot()
ax.plot(time, sub, linewidth=5, label="WRF-CHIMERE")
ax.plot(time_obs, obs, "ko", markersize=7, label="Observations")
ax.legend()
ax.set_ylabel("PM$_{2.5} (µg m$^{-3}$)$", fontsize=18)
fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.grid()
ax.set_ylim([0,6.2])
fig.savefig("path_output__dir/name_plot.png", dpi=500)

# Scatter plot
fig = plt.figure(figsize=(9,9))
ax1 = fig.add_subplot()
ax1.scatter(obs , sub, s=25) # obs and sub need to have same lenght and ordered so that the time for each element is the same
t = np.linspace(0,100)
ax1.plot(t,t, color='black')
r0 = np.linspace(0,100)
y0 = 2*r0
y1 = 0.5*r0
ax1.plot(r0,y0,'k--')
ax1.plot(r0,y1,'k--')
ax1.set_ylabel("Model PM$_{2.5} (µg m$^{-3}$)$", fontsize=18)
ax1.set_xlabel("Measured PM$_{2.5} (µg m$^{-3}$)$", fontsize=18)
plt.xlim(0, 6.2)
plt.ylim(0, 6.2)
ax1.grid()
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
fig.savefig("path_output__dir/name_plot.png", dpi=500)

# Boxplots
data_top_plot = [list1_A, list2_B]
data_top_plot_category2 = [list3_A, list4_B]
fig, ax = plt.subplots(figsize=(9, 9))
boxprops_red = dict(color='red', linewidth=2)
boxprops_purple = dict(color='purple', linewidth=2)
whiskerprops = dict(color='black', linewidth=1)
capprops = dict(color='black', linewidth=1)
medianprops = dict(color='black', linewidth=2)
meanprops = dict(markerfacecolor='green', markeredgecolor='black', markersize=10)
bp1 = ax.boxplot(data_to_plot[0], positions=[1.2], widths=0.35, showmeans=True,
                 boxprops=boxprops_purple, whiskerprops=whiskerprops, capprops=capprops,
                 medianprops=medianprops, meanprops=meanprops, showfliers=False)
bp2 = ax.boxplot(data_to_plot[1], positions=[1.8], widths=0.35, showmeans=True,
                 boxprops=boxprops_red, whiskerprops=whiskerprops, capprops=capprops,
                 medianprops=medianprops, meanprops=meanprops, showfliers=False)

bp5 = ax.boxplot(data_to_plot_category2[0], positions=[3], widths=0.35, showmeans=True,
                 boxprops=boxprops_purple, whiskerprops=whiskerprops, capprops=capprops,
                 medianprops=medianprops, meanprops=meanprops, showfliers=False)
bp6 = ax.boxplot(data_to_plot_category2[1], positions=[3.6], widths=0.35, showmeans=True,
                 boxprops=boxprops_red, whiskerprops=whiskerprops, capprops=capprops,
                 medianprops=medianprops, meanprops=meanprops, showfliers=False)
# Customize the x-ticks
ax.set_xticks([1.5, 3.3])
ax.set_xticklabels(['Category 1', 'Category2'], fontsize=18)
# Grid, labels, and title
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_ylabel('Y label Name', fontsize=18)
# Create a unified legend
purple_patch = plt.Line2D([0], [0], color='purple', lw=4, label='Object A')
red_patch = plt.Line2D([0], [0], color='red', lw=4, label='Object B')
ax.legend(loc='upper center', handles=[purple_patch, red_patch], fontsize=15)
# Set y-axis limits if needed
ax.set_ylim(bottom=0)
plt.show()

# Map plot
def create_plot(sub, title="", label="", cmap='bwr', savefig=None, vmin, vmax):

    cproj = cartopy.crs.LambertConformal(central_longitude=24.3, central_latitude=61.8)
    fig = plt.figure(figsize=(9,11))
    ax0 = plt.subplot(projection=cproj)
    c = plt.pcolormesh(ds.lon,ds.lat, sub, cmap=cmap, transform=ccrs.PlateCarree(),shading='gouraud', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(c, fraction = 0.040, pad = 0.12,  extend="both")
    cbar.set_label(label=label, fontsize=18, y=0.5)
    cbar.ax.tick_params(labelsize=15)
    ax0.coastlines(color='k', linewidth = 1);
    ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
    ax0.set_title(title, fontweight="bold", fontsize=25)
    gl = ax0.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
    gl.xlabel_style = {'rotation': 0};

    if savefig:
        plt.savefig(savefig, dpi=400)
    else:
        plt.show()
