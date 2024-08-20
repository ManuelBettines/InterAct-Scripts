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

# Concetrazioni VOC
ozone = pd.read_csv("../data/OZONO/ozono_vertical.txt", sep="\t",na_values=["NaN"])
height = [4.2,8.4,16.8,33.6,50.4,67.2,101,125]

ozone = ozone[ozone.Year.isin([2019])]
ozone = ozone[ozone.Month.isin([7])]
#ozone = ozone[ozone.Hour.isin([6,7,8,9,10,11,12,13,14,15,16,17])] #Day
ozone = ozone[ozone.Hour.isin([18,19,20,21,22,23,0,1,2,3,4,5])]  #Night

mis_4 = np.nanmean(ozone["HYY_META.O342"].values)
mis_8 = np.nanmean(ozone["HYY_META.O384"].values)
mis_16 = np.nanmean(ozone["HYY_META.O3168"].values)
mis_33 = np.nanmean(ozone["HYY_META.O3336"].values)
mis_50 = np.nanmean(ozone["HYY_META.O3504"].values)
mis_67 = np.nanmean(ozone["HYY_META.O3672"].values)
mis_101 = np.nanmean(ozone["HYY_META.O31010"].values)
mis_125 = np.nanmean(ozone["HYY_META.O31250"].values)
print(mis_4)

mis = np.array([mis_4,mis_8,mis_16,mis_33,mis_50,mis_67,mis_101,mis_125])

nanstd_4 = np.nanstd(ozone["HYY_META.O342"].values)
nanstd_8 = np.nanstd(ozone["HYY_META.O384"].values)
nanstd_16 = np.nanstd(ozone["HYY_META.O3168"].values)
nanstd_33 = np.nanstd(ozone["HYY_META.O3336"].values)
nanstd_50 = np.nanstd(ozone["HYY_META.O3504"].values)
nanstd_67 = np.nanstd(ozone["HYY_META.O3672"].values)
nanstd_101 = np.nanstd(ozone["HYY_META.O31010"].values)
nanstd_125 = np.nanstd(ozone["HYY_META.O31250"].values)

nanstd = np.array([nanstd_4,nanstd_8,nanstd_16,nanstd_33,nanstd_50,nanstd_67,nanstd_101,nanstd_125])

lower = mis - nanstd
upper = mis + nanstd

# Load simulations output
update_cc = xr.open_dataset("../data/FINLAND6-CC-O3.nc")

# Account for time shift
times = update_cc.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=2)
update_cc['local_times'] = pd.DatetimeIndex(local_times)
update_cc['O3'] = update_cc.O3.swap_dims({'time_counter':'local_times'})

# Define the start and end dates
start_date = datetime(2019, 7, 1, 0, 0, 0)
end_date = datetime(2019, 7, 30, 23, 0, 0)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(update_cc.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(update_cc.nav_lat[:,52], 61.8417)

# Calculate altitude layer
hlay = update_cc.hlay.sel(x=idx_lon).sel(y=idx_lat)
thlay = update_cc.thlay.sel(x=idx_lon).sel(y=idx_lat)
zlay = hlay - 0.5 * thlay

# Define hours from 6 am to 6 pm
start_hour = 18
end_hour = 6

# Empty lists to store subsets for each day
sub_base = []
sub_base_list = []
zlay_list = []

# Loop through each day and select only the hours between 6 am to 6 pm
current_date = start_date
while current_date <= end_date:
    next_date = current_date + timedelta(days=1)
    hours = pd.date_range(start=current_date.replace(hour=start_hour, minute=0, second=0), 
                          end=next_date.replace(hour=end_hour, minute=0, second=0), freq='H')
    
    sub_base = update_cc.O3.sel(x=idx_lon).sel(y=idx_lat).sel(local_times=hours)
    zlay_sub = zlay.sel(time_counter=hours)
    
    sub_base_list.append(sub_base)
    zlay_list.append(zlay_sub)
    
    current_date += timedelta(days=1)

# Concatenate the lists to create xarray datasets
sub_base = xr.concat(sub_base_list, dim='local_times').mean(dim='local_times')
zlay = xr.concat(zlay_list, dim='time_counter').mean(dim='time_counter')

# Plot results
fig = plt.figure(figsize=(8,11))
ax = fig.add_subplot()
ax.plot(sub_base, zlay, marker='o',linewidth=5, label="WRF-CHIMERE (with canopy correction)")
ax.plot(mis, height, "ko", markersize=7, label="Observations")
ax.fill_betweenx(height, lower, upper,color='gray', alpha=0.3)
ax.legend()
ax.set_xlabel("Ozone (ppbv)", fontsize=18)
ax.set_ylabel("Height above ground level (m)", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Ozone vertical profile - Night [18-6 Local Time]',fontsize=21)
ax.grid()
ax.set_ylim([0,130])
ax.set_xlim([0,70])
fig.savefig("../figures/ozone_vertical_profile_night_cc.png", dpi=500)
