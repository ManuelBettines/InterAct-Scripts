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
from pandas.tseries.offsets import DateOffset

# Load simulations output
update = xr.open_dataset("../data/FINLAND6-BNUM-CC.nc")

# Account for time shift
times = update.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=2)
update['local_times'] = pd.DatetimeIndex(local_times)
update['bnum'] = update.bnum.swap_dims({'time_counter':'local_times'})

# Define the start and end dates
start_date = datetime(2019, 6, 1, 0, 0, 0)
end_date = datetime(2019, 8, 31, 2, 0, 0)

time = [start_date + timedelta(hours=x) for x in range(2184)]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(update.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(update.nav_lat[:,52], 61.8417)

print(idx_lon)
print(idx_lat)

sub_upd = update.bnum.sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))

mmd_data = xr.open_dataset("../../nest-FINLAND6/chim_nest-FINLAND6_2019071400_24_reduit.nc")
mmd = mmd_data.mmd.values
cut_off_diameters = mmd_data.cut_off_diameters.values

# Compute log space bin widths
log_bin_edges = np.log(cut_off_diameters)
log_bin_widths = np.diff(log_bin_edges)
log_bin_widths = xr.DataArray(log_bin_widths, dims=('nbins',))

# Transform number concentrations to dN/dlog(D)
sub_upd_t_dNdlog = (sub_upd / log_bin_widths)


mmd = cut_off_diameters[:-1]
# Banana plot
fig = plt.figure(figsize=(50,10))
ax = fig.add_subplot()
norm = LogNorm(vmin=1, vmax=0.3e5)
c = plt.pcolormesh(time,mmd*1e9,sub_upd_t_dNdlog[:,:,0].T, cmap='RdYlBu_r',shading='gouraud', norm=norm)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.01,  extend="both")
cbar.set_label(label='dN/d(logD$_p$) (cm$^{-3}$)', fontsize=25, y=0.5)
cbar.ax.tick_params(labelsize=15)
plt.yscale("log")
ax.set_ylabel('Diameter (nm)', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=25)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
fig.autofmt_xdate(rotation=45)
ax.set_ylim([10,1e3])
ax.set_title('Hyytiälä size distribution', fontsize=31, fontweight='bold')
#plt.show()
#fig.savefig('../figures/Banana_hyy_timeserie_model.png', dpi=500, bbox_inches='tight')


# Diurnal banana plot
print(len(update.local_times))
print(len(sub_upd_t_dNdlog))
diurnal_data = sub_upd_t_dNdlog.groupby(update.local_times.dt.hour).mean()
ore = np.linspace(0,24,24)

# Plotting
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot()
norm = LogNorm(vmin=1, vmax=1e4)
c = plt.pcolormesh(ore, mmd*1e9, diurnal_data[:,:,0].T, cmap='RdYlBu_r',shading='gouraud', norm=norm)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.01,  extend="both")
cbar.set_label(label='dN/d(logD$_p$) (cm$^{-3}$)', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)
ax.set_xlabel('Datetime', fontsize=18)
ax.set_ylabel('Particles size (nm)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.yscale('log')
plt.ylim([10,1e3])
#fig.savefig('../figures/Banana_hyy_diurnal_model.png', dpi=500, bbox_inches='tight') 
#plt.show()

# Windrose
meteo = xr.open_dataset("../data/FINLAND6-CC-meteo.nc")
meteo['local_times'] = pd.DatetimeIndex(local_times)
meteo['winz'] = meteo.winz.swap_dims({'time_counter':'local_times'})
meteo['winm'] = meteo.winm.swap_dims({'time_counter':'local_times'})

def calculate_wind_direction(us, vs):
    # Calculate angles using arctan2, which is vectorized
    angles_radians = np.arctan2(vs, us)
    # Convert radians to degrees and adjust to meteorological convention
    wind_directions = 270 - np.degrees(angles_radians)
    # Ensure wind directions are within the range [0, 360)
    wind_directions = np.mod(wind_directions, 360)
    return wind_directions

winz = meteo.winz.sel(bottom_top=2).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))
winm = meteo.winm.sel(bottom_top=2).sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))
meteo['wind_dir'] = calculate_wind_direction(winz,winm)

bins = np.arange(0, 360, 10)
# Function to assign wind direction to bins
def wind_direction_bins(direction):
    return bins[np.digitize(direction, bins) - 1]

# Assign wind direction bins
wind_dir_bin = xr.apply_ufunc(wind_direction_bins, meteo['wind_dir'])

# Group by wind direction bins and compute mean particle sizes
diurnal_data = sub_upd_t_dNdlog.groupby(wind_dir_bin).mean()

direction_centers = np.arange(0, 360, 45)
theta = np.radians(direction_centers)

# Plotting
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, polar=True)
norm = LogNorm(vmin=100, vmax=7*1e3)
rad = np.radians(np.linspace(0, 360, len(diurnal_data)))
c = ax.pcolormesh(rad, mmd, diurnal_data.values[:,:,0].T, cmap='RdYlBu_r', norm=norm)
cbar = plt.colorbar(c, fraction=0.040, pad=0.08, extend="both")
cbar.set_label(label='dN/d(logD$_p$) (cm$^{-3}$)', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)
ax.set_title('Particle Size Distribution by Wind Direction (WRF-CHIMERE)', fontsize=18, fontweight='bold')#
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
ax.set_xticks(theta)
ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
plt.yscale('log')
plt.ylim([0, 235])
fig.savefig('../figures/windrose_sizedis_model.png', dpi=500, bbox_inches='tight')
plt.show()

