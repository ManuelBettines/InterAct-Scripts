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


# Emissioni VOC
VOC_em = pd.read_csv("../data/VOC/VOC_fluxes/VOC_profileflux_all_data_2010_2023.txt", na_values=["NaN"])

for col in VOC_em.columns:
    VOC_em[col] = VOC_em[col].astype(float)

VOC_em = VOC_em[VOC_em.Year.isin([2017,2018,2019])]
VOC_em = VOC_em[VOC_em.Month.isin([6,7,8])]
#VOC_em = VOC_em[VOC_em.Day.isin([19,20,21,22,23,24,25,26,27])]

#daily_mis = VOC_em.groupby([VOC_em.Hour]).mean().Isoprene.values
#std =  VOC_em.groupby([VOC_em.Hour]).std().Isoprene.values

daily_mis = VOC_em.dropna(subset=["Isoprene"]).groupby([VOC_em.Hour]).mean().Isoprene.values
std = VOC_em.dropna(subset=["Isoprene"]).groupby([VOC_em.Hour]).std().Isoprene.values


ore_mis = [2,5,8,11,14,17,20,23]


upper = daily_mis + std
lower = daily_mis - std

# Load simulations output
base = xr.open_dataset("../data/FINLAND6_isoprene.nc")
megan = xr.open_dataset("../data/FINLAND6-MEGAN3_isoprene.nc")
update = xr.open_dataset("../data/FINLAND6-UPD_isoprene.nc")

# Account for time shift
times = megan.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=0)
base['local_times'] = pd.DatetimeIndex(local_times)
base['C5H8_b'] = base.C5H8_b.swap_dims({'time_counter':'local_times'})
megan['local_times'] = pd.DatetimeIndex(local_times)
megan['C5H8_b'] = megan.C5H8_b.swap_dims({'time_counter':'local_times'})
update['local_times'] = pd.DatetimeIndex(local_times)
update['C5H8_b'] = update.C5H8_b.swap_dims({'time_counter':'local_times'})

# Define the start and end dates
#start_date = datetime(2019, 6, 1, 0, 0, 0)
#end_date = datetime(2019, 8, 31, 23, 0, 0)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

# Select subset
sub_base = base.C5H8_b.sel(x=idx_lon).sel(y=idx_lat)*1e6
sub_base = np.array(sub_base.values).reshape(-1,24)

sub_megan = megan.C5H8_b.sel(x=idx_lon).sel(y=idx_lat)*1e6
sub_megan = np.array(sub_megan.values).reshape(-1,24)

sub_update = update.C5H8_b.sel(x=idx_lon).sel(y=idx_lat)*1e6
sub_update = np.array(sub_update.values).reshape(-1,24)

sub_base = np.roll(sub_base, shift=3, axis=1)
base_h = np.mean(sub_base, axis=0)
base_std = np.std(sub_base, axis=0)

sub_megan = np.roll(sub_megan, shift=3, axis=1)
megan_h = np.mean(sub_megan, axis=0)
megan_std = np.std(sub_megan, axis=0)

sub_update = np.roll(sub_update, shift=3, axis=1)
update_h = np.mean(sub_update, axis=0)
update_std = np.std(sub_update, axis=0)

time = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

# Plot results
fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot()
ax.plot(ore_mis, daily_mis, linewidth=2, color='#4169E1', label="Observations")
ax.plot(time, base_h, linewidth=2, color='orangered', label="Baseline")
ax.plot(time, megan_h, linewidth=2, color='purple', label="MEG3")
ax.plot(time, update_h, linewidth=2, color='green', label="MEG3-UPD")
ax.fill_between(ore_mis,lower, upper, color='#4169E1', alpha=0.15)
#ax.fill_between(time, base_h-base_std, base_h+base_std, color='orangered', alpha=0.15)
#ax.fill_between(time, megan_h-megan_std, megan_h+megan_std, color='purple', alpha=0.15)
#ax.fill_between(time, update_h-update_std, update_h+update_std, color='green', alpha=0.15)
ax.set_ylabel("Isoprene emissions [μg m$^{-2}$ s$^{-1}$]")
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
#ax.set_title('Isoprene emissions')
ax.legend(frameon=True,loc='upper left', edgecolor="black", framealpha=0.8, facecolor="white", fontsize=9)
ax.set_xlabel("Datetime [Local Time]")
ax.grid(alpha=0.4, linewidth=0.5, zorder=-5)
ax.set_ylim([0,0.17])
ax.set_xlim([0,24])
fig.savefig("../figures/daily_isoprene_emissions_hyy.png", dpi=350, bbox_inches='tight')
plt.show()
