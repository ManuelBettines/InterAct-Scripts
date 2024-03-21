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

VOC = VOC[VOC.Height.isin([42])]
VOC = VOC[VOC.Month.isin([7])]
VOC = VOC[VOC.Day.isin([19,20,21,22,23,24,25,26,27])]

daily_mis = VOC.groupby([VOC.Hour]).mean().Isoprene.values
ore_mis = [2,5,8,11,14,17,20,23]

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
print(zlay[0,:,idx_lat,idx_lon])
idx_alt = find_nearest(zlay[0,:,idx_lat,idx_lon], 4.2)
print(idx_alt)

# Select subset
sub_base = base.C5H8.sel(bottom_top=idx_alt+1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))
sub_megan = megan.C5H8.sel(bottom_top=idx_alt+1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))
sub_upd_2 = update.C5H8.sel(bottom_top=idx_alt+1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))

sub_base = np.array(sub_base.values).reshape(-1,24)
sub_megan = np.array(sub_megan.values).reshape(-1,24)
sub_upd_2 = np.array(sub_upd_2.values).reshape(-1,24)

base_h = np.mean(sub_base, axis=0)
megan_h = np.mean(sub_megan, axis=0)
upd_h = np.mean(sub_upd_2, axis=0)

time = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

# Plot results
fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot()
ax.plot(time,base_h, linewidth=5, label="WRF-CHIMERE (MEGANv2.1)")
ax.plot(time,megan_h, linewidth=5, label="WRF-CHIMERE (MEGANv3.2)")
ax.plot(time,upd_h, linewidth=5, label="WRF-CHIMERE (MEGANv3.2 updated no age function)")
ax.plot(ore_mis, daily_mis, "ko", markersize=7, label="Observations")
ax.legend()
ax.set_ylabel("Isoprene (ppbv)", fontsize=18)
ax.set_xlabel("Datetime (Local Time)", fontsize=18)
#fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Isoprene 4.2 meters',fontsize=21)
ax.grid()
ax.set_ylim([0,1])
ax.set_xlim([0,24])
fig.savefig("../figures/daily_isoprene_4.2.png", dpi=500)
