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

VOC_em = VOC_em[VOC_em.Year.isin([2019])]
VOC_em = VOC_em[VOC_em.Month.isin([6,7,8])]
#VOC_em = VOC_em[VOC_em.Day.isin([19,20,21,22,23,24,25,26,27])]

daily_mis = VOC_em.groupby([VOC_em.Hour]).mean().Isoprene.values
std =  VOC_em.groupby([VOC_em.Hour]).std().Isoprene.values
ore_mis = [2,5,8,11,14,17,20,23]

upper = daily_mis + std
lower = daily_mis - std

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-BVOC-BASE.nc")
#megan = xr.open_dataset("../data/FINLAND6-BEMIS-MEGANv3.nc")
#update = xr.open_dataset("../data/FINLAND6-UPDATED-EMIS.nc")

# Account for time shift
times = base.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=0)
base['local_times'] = pd.DatetimeIndex(local_times)
base['C5H8_b'] = base.C5H8_b.swap_dims({'time_counter':'local_times'})

# Define the start and end dates
start_date = datetime(2019, 6, 1, 0, 0, 0)
end_date = datetime(2019, 8, 31, 23, 0, 0)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

# Select subset
sub_base = base.C5H8_b.sel(x=idx_lon).sel(y=idx_lat).sel(local_times=slice(start_date,end_date))*1e6
sub_base = np.array(sub_base.values).reshape(-1,24)

sub_base = np.roll(sub_base, shift=2, axis=1)
base_h = np.mean(sub_base, axis=0)

time = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

# Plot results
fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot()
ax.plot(time,base_h, linewidth=5, label="WRF-CHIMERE (MEGANv2.1)")
ax.plot(ore_mis, daily_mis, "ko", markersize=7, label="Observations Hyytiälä")
ax.fill_between(ore_mis,lower, upper, color='gray', alpha=0.3)
ax.legend()
ax.set_ylabel("Isoprene emissions (μg m$^{-2}$ s$^{-1}$)", fontsize=18)
ax.set_xlabel("Datetime (Local Time)", fontsize=18)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Isoprene emissions',fontsize=21)
ax.grid()
ax.set_ylim([0,0.175])
ax.set_xlim([0,24])
fig.savefig("../figures/daily_isoprene_emissions_hyy.png", dpi=500)
