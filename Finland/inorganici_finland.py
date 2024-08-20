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
from math import isnan
from pandas.tseries.offsets import DateOffset

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-inorganici.nc")
base_oa = xr.open_dataset("../data/FINLAND6-CC-OA.nc")

# Define the start and end dates
start_date = datetime(2019, 6, 7, 0, 0, 0)
end_date = datetime(2019, 8, 30, 0, 0, 0)

time = [start_date + timedelta(hours=x) for x in range(576)]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

print(idx_lon)
print(idx_lat)

# pie
oa = pd.read_csv("../data/SMEAR/HydeOA.txt", sep="\t",na_values=["NaN"])
oa['TimelistLT_com'] = pd.to_datetime(oa['TimelistLT_com'])
oa = oa[(oa['TimelistLT_com'] >= '2019-06-01') & (oa['TimelistLT_com'] < '2019-09-01')]

sub1 = np.mean(oa['OA_com'])
sub2 = np.mean(oa['SO4_com'])
sub3 = np.mean(oa['NO3_com'])
sub4 = np.mean(oa['NH4_com'])

#sub1 = base_oa.OA.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean("time_counter")*1.7
#sub2 = base.pH2SO4.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean("time_counter")
#sub3 = base.pHNO3.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean("time_counter")
#sub4 = base.pNH3.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean("time_counter")

sizes = [sub1,sub2,sub3,sub4]
labels = ['OA', 'SO$_4$', 'NO$_3$', 'NH$_4$']
colors = ['limegreen', 'dodgerblue', 'mediumorchid', 'chocolate']

fig, ax = plt.subplots()
pie = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=0, colors=colors)

for i, p in enumerate(pie[0]):
    ang = (p.theta2 - p.theta1) / 2. + p.theta1
    y = 1.1 * np.sin(np.deg2rad(ang))
    x = 1.1 * np.cos(np.deg2rad(ang))

ax.axis('equal')
plt.title('Hyytiälä - Observations', fontsize=15, fontweight="bold")
plt.savefig('../figures/hyy_aerosol_composition_obs.png', dpi=500)
