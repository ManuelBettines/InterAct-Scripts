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
VOC = VOC.groupby([VOC.Day, VOC.Hour]).mean()

conc = VOC.Isoprene.values
#daily_mis = VOC.groupby([VOC.Hour]).mean().Monoterpenes.values
#ore_mis = [2,5,8,11,14,17,20,23]

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-VOC.nc")
megan = xr.open_dataset("../data/FINLAND6-MEGAN-VOC.nc")
#mymeg = xr.open_dataset("../data/FINLAND6-CONC-megan.nc")
update = xr.open_dataset("../data/FINLAND6-MEGAN-UPDATED-VOC.nc")

# Define the start and end dates
start_date = datetime(2019, 7, 19, 0, 0, 0)
end_date = datetime(2019, 7, 27, 23, 0, 0)

start_1 = datetime(2019, 7, 19, 2, 0, 0)
time = [start_date + timedelta(hours=x) for x in range(216)]
time2 = [start_1 + timedelta(hours=3*x) for x in range(72)]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(base.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(base.nav_lat[:,52], 61.8417)

sub_base = base.C5H8.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))
sub_megan = megan.C5H8.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))
#sub_upd = mymeg.monotepenes.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))
sub_upd_2 = update.C5H8.sel(bottom_top=1).sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))

#time = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot()
ax.plot(time,sub_base, linewidth=5, label="WRF-CHIMERE (MEGANv2.1)")
ax.plot(time,sub_megan, linewidth=5, label="WRF-CHIMERE (MEGANv3.2)")
#ax.plot(time,sub_upd, linewidth=5, label="WRF-CHIMERE (MEGANv3.2 updated with age function)")
ax.plot(time,sub_upd_2, linewidth=5, label="WRF-CHIMERE (MEGANv3.2 updated no age function)")
ax.plot(time2, conc, "ko", markersize=7, label="Observations")
ax.legend()
ax.set_ylabel("Isoprene (ppbv)", fontsize=18)
fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.grid()
ax.set_ylim([0,6.2])
fig.savefig("../figures/timeseries_isoprene.png", dpi=500)

#modello = sub_upd_2.values[1::3]

#fig = plt.figure(figsize=(9,9))
#ax1 = fig.add_subplot()
#ax1.scatter(conc,modello,s=25)
#t = np.linspace(0,100)
#ax1.plot(t,t, color='black')
#r0 = np.linspace(0,100)
#y0 = 2*r0
#y1 = 0.5*r0
#ax1.plot(r0,y0,'k--')
#ax1.plot(r0,y1,'k--')
#ax1.set_ylabel("Model Isoprene (ppbv)", fontsize=18)
#ax1.set_xlabel("Measured Isoprene (ppbv)", fontsize=18)
#plt.xlim(0, 6.2)
#plt.ylim(0, 6.2)
#ax1.grid()
#ax1.tick_params(axis='both', which='major', labelsize=15)
#ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
#fig.savefig("../figures/scatter_isoprene_megan_updated_no_age.png")

