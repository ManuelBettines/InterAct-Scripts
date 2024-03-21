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
meteo = meteo[meteo.Hour.isin([6,7,8,9,10,11,12,13,14,15,16,17])] #Day
#meteo = meteo[meteo.Hour.isin([18,19,20,21,22,23,0,1,2,3,4,5])]  #Night

mis_4 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU42"].values)
mis_8 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU84"].values)
mis_16 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU168"].values)
mis_33 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU336"].values)
mis_50 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU504"].values)
mis_67 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU672"].values)
mis_101 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU1010"].values)
mis_125 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.WSU1250"].values)

mis = np.array([mis_4,mis_8,mis_16,mis_33,mis_50,mis_67,mis_101,mis_125])

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-meteo.nc")

# Define the start and end dates
start_date = datetime(2019, 7, 19, 0, 0, 0)
end_date = datetime(2019, 7, 27, 23, 0, 0)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

# Select subset
winz = base.winz.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')
winm = base.winm.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')

sub_base = (winz**2 + winm**2)**(1/2)
sub_base = np.array(sub_base.values).reshape(-1,24)

base_h = np.mean(sub_base, axis=0)

# Plot results
fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot()
ax.plot(time,base_h, linewidth=5, label="WRF-CHIMERE")
ax.plot(time, daily_mis, "ko", markersize=7, label="Observations")
ax.legend()
ax.set_ylabel("Temperature (K)", fontsize=18)
ax.set_xlabel("Datetime (Local Time)", fontsize=18)
#fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Wind speed vertical profile',fontsize=21)
ax.grid()
ax.set_ylim([0,35])
ax.set_xlim([0,24])
fig.savefig("../figures/temperature_above.png", dpi=500)
