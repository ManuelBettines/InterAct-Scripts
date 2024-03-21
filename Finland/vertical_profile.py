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

# Concetrazioni VOC
VOC = pd.read_csv("../data/VOC/Mastdata_VOC_2019.txt", na_values=["NaN"])

for col in VOC.columns:
    VOC[col] = VOC[col].astype(float)

VOC = VOC[VOC.Month.isin([7])]
#VOC = VOC[VOC.Hour.isin([2,5,20,23])]
VOC = VOC[VOC.Hour.isin([8,11,14,17])]
VOC = VOC[VOC.Day.isin([19,20,21,22,23,24,25,26,27])]

val_mis = VOC.groupby([VOC.Height]).mean().Monoterpenes.values
std = VOC.groupby([VOC.Height]).std().Monoterpenes.values
height = [4.2,8.4,16.8,33.6,50.4,67.2,101,125]

upper = val_mis + std
lower = val_mis - std

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-VOC.nc")
megan = xr.open_dataset("../data/FINLAND6-MEGAN-VOC.nc")
update = xr.open_dataset("../data/FINLAND6-MEGAN-UPDATED-VOC.nc")

# Calculate altitude layer
hlay = base.hlay
thlay = base.thlay
elev = base.HGT_M
zlay = hlay -0.5*thlay

# Define the start and end dates
start_date = datetime(2019, 7, 19, 0, 0, 0)
end_date = datetime(2019, 7, 27, 23, 0, 0)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

# Select subset (All Day)
#sub_base = base.monoterpenes.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')
#sub_megan = megan.monoterpenes.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')
#sub_upd_2 = update.monoterpenes.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')
#zlay = zlay.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')

# Define hours from 6 am to 6 pm
start_hour = 6
end_hour = 18

# Empty lists to store subsets for each day
sub_base_list = []
sub_megan_list = []
sub_upd_2_list = []
zlay_list = []

# Loop through each day and select only the hours between 6 am to 6 pm
current_date = start_date
while current_date <= end_date:
    next_date = current_date + timedelta(days=1)
    hours = pd.date_range(start=current_date.replace(hour=start_hour, minute=0, second=0), 
                          end=current_date.replace(hour=end_hour, minute=0, second=0), freq='H')
    
    sub_base = base.monoterpenes.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=hours)
    sub_megan = megan.monoterpenes.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=hours)
    sub_upd_2 = update.monoterpenes.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=hours)
    zlay_sub = zlay.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=hours)
    
    sub_base_list.append(sub_base)
    sub_megan_list.append(sub_megan)
    sub_upd_2_list.append(sub_upd_2)
    zlay_list.append(zlay_sub)
    
    current_date += timedelta(days=1)

# Concatenate the lists to create xarray datasets
sub_base = xr.concat(sub_base_list, dim='time_counter').mean(dim='time_counter')
sub_megan = xr.concat(sub_megan_list, dim='time_counter').mean(dim='time_counter')
sub_upd_2 = xr.concat(sub_upd_2_list, dim='time_counter').mean(dim='time_counter')
zlay = xr.concat(zlay_list, dim='time_counter').mean(dim='time_counter')

# Plot results
fig = plt.figure(figsize=(8,11))
ax = fig.add_subplot()
ax.plot(sub_base, zlay, marker='o',linewidth=5, label="WRF-CHIMERE (MEGANv2.1)")
ax.plot(sub_megan, zlay, marker='o',linewidth=5, label="WRF-CHIMERE (MEGANv3.2)")
ax.plot(sub_upd_2, zlay, marker='o',linewidth=5, label="WRF-CHIMERE (MEGANv3.2 updated no age function)")
ax.plot(val_mis, height, "ko", markersize=7, label="Observations")
ax.fill_betweenx(height, lower, upper,color='gray', alpha=0.3)
ax.legend()
ax.set_xlabel("Monoterpenes (ppbv)", fontsize=18)
ax.set_ylabel("Height above ground level (m)", fontsize=18)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Monoterpenes vertical profile - Day [6-18 Local Time]',fontsize=21)
ax.grid()
ax.set_ylim([0,130])
ax.set_xlim([0,3])
fig.savefig("../figures/monoterpenes_vertical_profile_day.png", dpi=500)
