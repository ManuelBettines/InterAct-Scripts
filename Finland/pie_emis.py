import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime

base = xr.open_dataset("../data/FINLAND6-BEMIS.nc")
megan = xr.open_dataset("../data/FINLAND6-MEGAN-BEMIS.nc")
mymegan = xr.open_dataset("../data/FINLAND6-myMEGAN-BEMIS.nc")

# Define the start and end dates
start_date = datetime(2021, 5, 27, 0, 0, 0)
end_date = datetime(2021, 6, 2, 23, 0, 0)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896) 
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417) 

print(idx_lon)
print(idx_lat)

sub1 = mymegan.C5H8.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')*1e9
sub2 = mymegan.APINEN.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')*1e9
sub3 = mymegan.BPINEN.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')*1e9
sub4 = mymegan.LIMONE.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')*1e9
sub5 = mymegan.OCIMEN.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')*1e9
sub6 = mymegan.HUMULE.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date)).mean('time_counter')*1e9

sizes = [sub1,sub2,sub3,sub4,sub5,sub6]
labels = ['Isoprene', 'Alpha-pinene', 'Beta-pinene', 'Limonene', 'Ocimene', 'Humulene']
colors = ['limegreen', 'dodgerblue', 'mediumorchid', 'chocolate', 'darkorange', 'gold']

fig, ax = plt.subplots()
pie = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=0, colors=colors)

for i, p in enumerate(pie[0]):
    ang = (p.theta2 - p.theta1) / 2. + p.theta1
    y = 1.1 * np.sin(np.deg2rad(ang))
    x = 1.1 * np.cos(np.deg2rad(ang))

ax.axis('equal')
plt.title('BVOC emissions - MEGANv3.2 updated', fontsize=18, fontweight="bold")
plt.savefig('../figures/BVOC_emis_mymegan.png', dpi=500)

