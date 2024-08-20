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
meteo = meteo[meteo.Hour.isin([6,7,8,9,10,11,12,13,14,15,16,17])] #Day
#meteo = meteo[meteo.Hour.isin([18,19,20,21,22,23,0,1,2,3,4,5])]  #Night

mis_4 = np.mean(meteo["HYY_META.WSU42"].values)
mis_8 = np.mean(meteo["HYY_META.WSU84"].values)
mis_16 = np.mean(meteo["HYY_META.WSU168"].values)
mis_33 = np.mean(meteo["HYY_META.WSU336"].values)
#mis_50 = np.mean(meteo["HYY_META.WSU504"].values)
mis_67 = np.mean(meteo["HYY_META.WSU672"].values)
#mis_101 = np.mean(meteo["HYY_META.WSU1010"].values)
mis_125 = np.mean(meteo["HYY_META.WSU1250"].values)

mis = np.array([mis_4,mis_8,mis_16,mis_33,mis_67,mis_125])
height = [4.2,8.6,16.8,33.6,67.2,125]

std_4 = np.std(meteo["HYY_META.WSU42"].values)
std_8 = np.std(meteo["HYY_META.WSU84"].values)
std_16 = np.std(meteo["HYY_META.WSU168"].values)
std_33 = np.std(meteo["HYY_META.WSU336"].values)
#std_50 = np.std(meteo["HYY_META.WSU504"].values)
std_67 = np.std(meteo["HYY_META.WSU672"].values)
#std_101 = np.std(meteo["HYY_META.WSU1010"].values)
std_125 = np.std(meteo["HYY_META.WSU1250"].values)

std = np.array([std_4,std_8,std_16,std_33,std_67,std_125])

# Load simulations output
ds = xr.open_dataset("../data/FINLAND6-CC.nc")
ds_wrf = xr.open_dataset("../data/FINLAND6-METEO-WRF.nc")

# Account for time shift
times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=3)
ds['local_times'] = pd.DatetimeIndex(local_times)
ds['winz'] = ds.winz.swap_dims({'time_counter':'local_times'})
ds['winm'] = ds.winm.swap_dims({'time_counter':'local_times'})
#ds['temp'] = ds.temp.swap_dims({'time_counter':'local_times'})
# Account for time shift in WRF
times = ds_wrf.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y-%m-%d %H:%M:%S')
local_times = utc_times + DateOffset(hours=3)
ds_wrf['local_times'] = pd.DatetimeIndex(local_times)
#ds_wrf['T'] = ds_wrf.T.swap_dims({'Time':'local_times'})
#ds_wrf['P'] = ds_wrf.P.swap_dims({'Time':'local_times'})
ds_wrf['PH'] = ds_wrf.PH.swap_dims({'Time':'local_times'})
#ds_wrf['PB'] = ds_wrf.PB.swap_dims({'Time':'local_times'})
ds_wrf['PHB'] = ds_wrf.PHB.swap_dims({'Time':'local_times'})
ds_wrf['U'] = ds_wrf.U.swap_dims({'Time':'local_times'})
ds_wrf['V'] = ds_wrf.V.swap_dims({'Time':'local_times'})

# Define the start and end dates
start_date = datetime(2019, 7, 1, 0, 0, 0)
end_date = datetime(2019, 7, 31, 23, 0, 0)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(ds.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(ds.nav_lat[:,52], 61.8417)

# Calculate altitude layer
hlay = ds.hlay.sel(x=idx_lon).sel(y=idx_lat)
thlay = ds.thlay.sel(x=idx_lon).sel(y=idx_lat)
zlay = hlay -0.5*thlay

# Calculate layers altitude WRF
ph = ds_wrf.PH.sel(south_north=idx_lat).sel(west_east=idx_lon)
phb = ds_wrf.PHB.sel(south_north=idx_lat).sel(west_east=idx_lon)
h_wrf = (ph+phb)/9.81

# Calculate levels pressure WRF
#p = ds_wrf.P.sel(south_north=idx_lat).sel(west_east=idx_lon)
#pb = ds_wrf.PB.sel(south_north=idx_lat).sel(west_east=idx_lon)
#pres_wrf = (p+pb)*0.01

# Calculcate levels temp WRF
#theta = 300 + ds_wrf.T.sel(south_north=idx_lat).sel(west_east=idx_lon)
#t_wrf = theta/((1000/pres_wrf)**(0.286))

# Calculate Wind Speed WRF
u =  ds_wrf.U.sel(south_north=idx_lat).sel(west_east_stag=idx_lon)
v =  ds_wrf.V.sel(south_north_stag=idx_lat).sel(west_east=idx_lon)

# Define hours from 6 am to 6 pm
start_hour = 6
end_hour = 18

# Empty lists to store subsets for each day
sub_base_list = []
zlay_list = []
sub_base_list_wrf = []
zlay_list_wrf = []

# Loop through each day and select only the hours between 6 am to 6 pm
current_date = start_date
while current_date <= end_date:
    next_date = current_date + timedelta(days=1)
    hours = pd.date_range(start=current_date.replace(hour=start_hour, minute=0, second=0),
                          end=current_date.replace(hour=end_hour, minute=0, second=0), freq='H')

    winz = ds.winz.sel(x=idx_lon).sel(y=idx_lat).sel(local_times=hours)
    winm = ds.winm.sel(x=idx_lon).sel(y=idx_lat).sel(local_times=hours)
    sub_base = (winz**2 + winm**2)**(1/2)
    #temp = ds.temp.sel(x=idx_lon).sel(y=idx_lat).sel(local_times=hours) - 273.15
    zlay_sub = zlay.sel(time_counter=hours)
    #temp_wrf = t_wrf.sel(local_times=hours) - 273.15
    U = u.sel(local_times=hours)
    V = v.sel(local_times=hours)
    wind_wrf = (U**2 + V**2)**(1/2)
    zlay_sub_wrf = h_wrf.sel(local_times=hours) - 166.08

    sub_base_list.append(sub_base)
    zlay_list.append(zlay_sub)
    sub_base_list_wrf.append(wind_wrf)
    zlay_list_wrf.append(zlay_sub_wrf)

    current_date += timedelta(days=1)

# Concatenate the lists to create xarray datasets
sub_base = xr.concat(sub_base_list, dim='local_times').mean(dim='local_times')
zlay = xr.concat(zlay_list, dim='time_counter').mean(dim='time_counter')

sub_base_wrf = xr.concat(sub_base_list_wrf, dim='local_times').mean(dim='local_times')
zlay_wrf = xr.concat(zlay_list_wrf, dim='local_times').mean(dim='local_times')

def calculate_midpoints(arr):
    midpoints = []
    for i in range(len(arr) - 1):
        mid = (arr[i] + arr[i + 1]) / 2
        midpoints.append(mid)
    return midpoints

zlay_wrf = calculate_midpoints(zlay_wrf)

lower_bound = mis - std
upper_bound = mis + std

# Plot results
fig = plt.figure(figsize=(7,11))
ax = fig.add_subplot()
ax.plot(sub_base, zlay, marker='o', markersize=12,linewidth=5, label="CHIMERE (with canopy correction)")
ax.plot(sub_base_wrf, zlay_wrf, marker='o', markersize=12,linewidth=5, label="WRF")
ax.plot(mis, height, "ko", markersize=9, label="Observations")
ax.fill_betweenx(height,lower_bound, upper_bound, color='gray', alpha=0.3)
ax.legend(fontsize=15)
ax.set_ylabel("Height above ground level (m)", fontsize=18)
ax.set_xlabel("Wind speed (m s$^{-1}$)", fontsize=18)
#ax.set_xlabel("Temperarture (°C)", fontsize=18)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('Wind speed vertical profile - Day',fontsize=21)
ax.grid()
ax.set_ylim([0,130])
ax.set_xlim([0,7])
fig.savefig("../figures/wind_vertical_day_cc.png", dpi=500)
