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

# Meteo Hyy
meteo = pd.read_csv("../data/Meteo/smeardata_20240321.txt", sep="\t" ,na_values=["NaN"])

meteo = meteo[meteo.Year.isin([2019])]
meteo = meteo[meteo.Month.isin([7])]
meteo = meteo[meteo.Day.isin([1,2,3,4,5,6,7])]

mis_4 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.T42"].values)
#mis_8 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.T84"].values)
mis_16 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.T168"].values)
mis_33 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.T336"].values)
mis_50 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.T504"].values)
mis_67 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.T672"].values)
#mis_101 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.T1010"].values)
mis_125 = np.mean(meteo.groupby([meteo.Hour]).mean()["HYY_META.T1250"].values)

mis = np.array([mis_4,mis_16,mis_33,mis_50,mis_67,mis_125])
height = [4.2,16.8,33.6,50,67.2,125]

std_4 = np.std(meteo.groupby([meteo.Hour]).std()["HYY_META.T42"].values)
#std_8 = np.std(meteo.groupby([meteo.Hour]).std()["HYY_META.T84"].values)
std_16 = np.std(meteo.groupby([meteo.Hour]).std()["HYY_META.T168"].values)
std_33 = np.std(meteo.groupby([meteo.Hour]).std()["HYY_META.T336"].values)
std_50 = np.std(meteo.groupby([meteo.Hour]).std()["HYY_META.T504"].values)
std_67 = np.std(meteo.groupby([meteo.Hour]).std()["HYY_META.T672"].values)
#std_101 = np.std(meteo.groupby([meteo.Hour]).mean()["HYY_META.T1010"].values)
std_125 = np.std(meteo.groupby([meteo.Hour]).std()["HYY_META.T1250"].values)

std = np.array([std_4,std_16,std_33,std_50,std_67,std_125])

# Load simulations output
ds = xr.open_dataset("../data/FINLAND6-FIXED.nc")
#ds_pres = xr.open_dataset("../data/FINLAND6-pres.nc")
wrf = xr.open_dataset("../data/FINLAND6-METEO-WRF.nc")
#wrf_w = xr.open_dataset("../data/FINLAND6-WIND.nc")
tms = xr.open_dataset("../data/FINLAND6-wrf-times.nc")

# Account for time shift
times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=0)
#ds_pres['local_times'] = pd.DatetimeIndex(local_times)
ds['local_times'] = pd.DatetimeIndex(local_times)
#ds['winz'] = ds.winz.swap_dims({'time_counter':'local_times'})
#ds['winm'] = ds.winm.swap_dims({'time_counter':'local_times'})
ds['temp'] = ds.temp.swap_dims({'time_counter':'local_times'})
#ds_pres['pres'] = ds_pres.pres.swap_dims({'time_counter':'local_times'})
# Wrf
times = tms.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y-%m-%d %H:%M:%S')
local_times = utc_times + DateOffset(hours=0)
wrf['local_times'] = pd.DatetimeIndex(local_times)
#wrf_w['local_times'] = pd.DatetimeIndex(local_times)
wrf['P'] = wrf.P.swap_dims({'Time':'local_times'})
wrf['PB'] = wrf.PB.swap_dims({'Time':'local_times'})
wrf['T'] = wrf.T.swap_dims({'Time':'local_times'})
wrf['PH'] = wrf.PH.swap_dims({'Time':'local_times'})
wrf['PHB'] = wrf.PHB.swap_dims({'Time':'local_times'})
#wrf_w['U'] = wrf_w.U.swap_dims({'Time':'local_times'})
#wrf_w['V'] = wrf_w.V.swap_dims({'Time':'local_times'})


# Define the start and end dates
start_date = datetime(2019, 7, 1, 0, 0, 0)
end_date = datetime(2019, 7, 7, 23, 0, 0)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(ds.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(ds.nav_lat[:,52], 61.8417)

# Calculate altitude layer
hlay = ds.hlay.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')
thlay = ds.thlay.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')
zlay = hlay - 0.5 * thlay

# Calculate layers altitude WRF
ph = wrf.PH.sel(south_north=idx_lat).sel(west_east=idx_lon).sel(local_times=slice(start_date,end_date)).mean('local_times')
phb = wrf.PHB.sel(south_north=idx_lat).sel(west_east=idx_lon).sel(local_times=slice(start_date,end_date)).mean('local_times')
h_wrf = (ph + phb) / 9.81 - 166.08

# Calculate levels pressure WRF
p = wrf.P.sel(local_times=slice(start_date,end_date)).sel(south_north=idx_lat).sel(west_east=idx_lon)
pb = wrf.PB.sel(local_times=slice(start_date,end_date)).sel(south_north=idx_lat).sel(west_east=idx_lon)
pres = (p + pb)*0.01

# Calculcate levels temperature WRF
theta = 300 + wrf.T.sel(local_times=slice(start_date,end_date)).sel(south_north=idx_lat).sel(west_east=idx_lon)
t_wrf = theta*((pres / 1000)**0.286) - 273.15
t_wrf = t_wrf.mean('local_times')

# Calculate Wind Speed WRF
#u =  wrf_w.U.sel(south_north=idx_lat).sel(west_east_stag=idx_lon).sel(local_times=slice(start_date,end_date))#.mean('local_times')
#v =  wrf_w.V.sel(south_north_stag=idx_lat).sel(west_east=idx_lon).sel(local_times=slice(start_date,end_date))#.mean('local_times')
#sub_base_wrf = (u**2+v**2)**(1/2)
#sub_base_wrf = sub_base_wrf.mean('local_times')

# Calculate Wind Speed CHIMERE
#winm =  ds.winm.sel(y=idx_lat).sel(x=idx_lon).sel(local_times=slice(start_date,end_date))#.mean('local_times')
#winz=  ds.winz.sel(y=idx_lat).sel(x=idx_lon).sel(local_times=slice(start_date,end_date))#.mean('local_times')
#sub_base = (winm**2+winz**2)**(1/2)

# Calculate Temperature CHIMERE
sub_base = ds.temp.sel(y=idx_lat).sel(x=idx_lon).sel(local_times=slice(start_date,end_date))
sub_base = sub_base.mean('local_times') - 273.15

def calculate_midpoints(arr):
    midpoints = []
    for i in range(len(arr) - 1):
        mid = (arr[i] + arr[i + 1]) / 2
        midpoints.append(mid)
    return midpoints

zlay_wrf = calculate_midpoints(h_wrf)

lower_bound = mis - std
upper_bound = mis + std

# Plot results
fig = plt.figure(figsize=(7,11))
ax = fig.add_subplot()
ax.plot(sub_base, zlay, marker='o', markersize=9,linewidth=5, label="CHIMERE")
ax.plot(t_wrf, zlay_wrf, marker='o', markersize=9,linewidth=5, label="WRF")
ax.plot(mis, height, "ko", markersize=9, label="Observations")
ax.fill_betweenx(height,lower_bound, upper_bound, color='gray', alpha=0.3)
ax.legend(fontsize=15)
ax.set_ylabel("Height above ground level (m)", fontsize=18)
#ax.set_xlabel("Wind speed (m s$^{-1}$)", fontsize=18)
ax.set_xlabel("Temperature (°C)", fontsize=18)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Temperature vertical profile',fontsize=21)
ax.grid()
ax.set_ylim([0,130])
ax.set_xlim([12,15])
fig.savefig("../figures/temperature_vertical.png", dpi=500)
