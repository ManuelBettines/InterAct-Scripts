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


# Meteo Hyy
meteo = pd.read_csv("../data/Meteo/smeardata_20240321.txt", sep="\t" ,na_values=["NaN"])

meteo = meteo[meteo.Year.isin([2019])]
meteo = meteo[meteo.Month.isin([7])]
meteo = meteo[meteo.Day.isin([19,20,21,22,23,24,25,26,27])]
#meteo = meteo[meteo.Hour.isin([6,7,8,9,10,11,12,13,14,15,16,17])] #Day
meteo = meteo[meteo.Hour.isin([18,19,20,21,22,23,0,1,2,3,4,5])]  #Night

mis_4 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU42"].values)
mis_8 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU84"].values)
mis_16 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU168"].values)
mis_33 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU336"].values)
#mis_50 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU504"].values)
mis_67 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU672"].values)
#mis_101 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU1010"].values)
mis_125 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU1250"].values)

mis = np.array([mis_4,mis_8,mis_16,mis_33,mis_67,mis_125])
height = [4.2,8.4,16.8,33.6,67.2,125]

std_4 = np.std(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU42"].values)
std_8 = np.std(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU84"].values)
std_16 = np.std(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU168"].values)
std_33 = np.std(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU336"].values)
std_67 = np.std(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU672"].values)
std_125 = np.std(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU1250"].values)

std = np.array([std_4,std_8,std_16,std_33,std_67,std_125])

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-meteo.nc")
ds = xr.open_dataset("../data/FINLAND6-VOC.nc")

# Calculate altitude layer
hlay = ds.hlay
thlay = ds.thlay
elev = ds.HGT_M
zlay = hlay -0.5*thlay

# Define the start and end dates
start_date = datetime(2019, 7, 19, 0, 0, 0)
end_date = datetime(2019, 7, 26, 23, 0, 0)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

# Define hours from 6 am to 6 pm
start_hour = 18
end_hour = 6

# Empty lists to store subsets for each day
sub_base_list = []
zlay_list = []

# Loop through each day and select only the hours between 6 am to 6 pm
current_date = start_date
while current_date <= end_date:
    next_date = current_date + timedelta(days=1)
    hours = pd.date_range(start=current_date.replace(hour=start_hour, minute=0, second=0),
                          end=next_date.replace(hour=end_hour, minute=0, second=0), freq='H')

    winz = base.winz.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=hours)
    winm = base.winm.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=hours)
    sub_base = (winz**2 + winm**2)**(1/2)
    zlay_sub = zlay.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=hours)

    sub_base_list.append(sub_base)
    zlay_list.append(zlay_sub)

    current_date += timedelta(days=1)

# Concatenate the lists to create xarray datasets
sub_base = xr.concat(sub_base_list, dim='time_counter').mean(dim='time_counter')
zlay = xr.concat(zlay_list, dim='time_counter').mean(dim='time_counter')

lower_bound = mis - std
upper_bound = mis + std

# Plot results
fig = plt.figure(figsize=(7,11))
ax = fig.add_subplot()
ax.plot(sub_base, zlay, marker='o', markersize=9,linewidth=5, label="WRF-CHIMERE")
ax.plot(mis, height, "ko", markersize=9, label="Observations")
ax.fill_betweenx(height,lower_bound, upper_bound, color='gray', alpha=0.3)
ax.legend(fontsize=15)
ax.set_ylabel("Height above ground level (m)", fontsize=18)
ax.set_xlabel("Wind speed (m s$^{-1}$)", fontsize=18)
#fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Wind speed vertical profile - Night',fontsize=21)
ax.grid()
ax.set_ylim([0,130])
ax.set_xlim([0,5])
fig.savefig("../figures/wind_vertical_Night.png", dpi=500)
