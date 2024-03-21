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

# Emissioni VOC
VOC_em = pd.read_csv("../data/VOC/VOC_fluxes/VOC_profileflux_all_data_2010_2023.txt", na_values=["NaN"])

for col in VOC_em.columns:
    VOC_em[col] = VOC_em[col].astype(float)

VOC_em = VOC_em[VOC_em.Year.isin([2019])]
VOC_em = VOC_em[VOC_em.Month.isin([7])]
VOC_em = VOC_em[VOC_em.Day.isin([19,20,21,22,23,24,25,26,27])]

emis = [-0.00080753,0.0015314,0.01183,-0.01546,0.011725,-0.010643,0.0042018,np.nan,0.00149,0.065938,0.044778,-0.0077749,0.0045107,0.0075016,np.nan,np.nan,np.nan,np.nan,0.033706,0.087706,np.nan,0.017916,0.016586,np.nan,np.nan,np.nan,0.04789,0.042484,0.046601,-0.0030606,np.nan,np.nan,np.nan,0.0092692,np.nan,np.nan,0.0016577,0.026242,0.0065192,np.nan,-0.00033931,0.0030741,0.049047,0.073944,0.020032,0.015864,np.nan,np.nan,0.002438,np.nan,-0.012415,0.012171,0.0076342,0.054967,0.0063015,np.nan,np.nan,0.0071584,0.034566,0.061187,0.077781,0.084155,np.nan,np.nan,np.nan,np.nan,0.04664,0.07043,0.013899,0.044225,np.nan,0.0064133]

# Load simulations output
base = xr.open_dataset("../data/FINLAND6-EMIS-base.nc")
megan = xr.open_dataset("../data/FINLAND6-EMIS-megan.nc")
mymeg = xr.open_dataset("../data/FINLAND6-EMIS-updated.nc")
upd = xr.open_dataset("../data/FINLAND6-EMIS-updated_2.nc")

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

sub_base = base.C5H8.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))*1e6
sub_megan = megan.C5H8.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))*1e6
sub_upd = mymeg.C5H8.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))*1e6
sub_upd_2 = upd.C5H8.sel(x=idx_lon).sel(y=idx_lat).sel(time_counter=slice(start_date,end_date))*1e6


fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot()
ax.plot(time,sub_base, linewidth=5, label="WRF-CHIMERE (MEGANv2.1)")
ax.plot(time,sub_megan, linewidth=5, label="WRF-CHIMERE (MEGANv3.2)")
ax.plot(time,sub_upd, linewidth=5, label="WRF-CHIMERE (MEGANv3.2 updated with age function)")
ax.plot(time,sub_upd_2, linewidth=5, label="WRF-CHIMERE (MEGANv3.2 updated no age function)")
ax.plot(time2, emis, "ko", markersize=8, label="Observations")
ax.legend()
ax.set_ylabel("Isoprene emission (μg m$^{-2}$ s$^{-1}$)", fontsize=18)
fig.autofmt_xdate(rotation=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.grid()
ax.set_ylim([0,0.4])
fig.savefig("../figures/timeseries_emis_isoprene.png", dpi=500)

#modello = sub_upd.values[1::3]

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
#fig.savefig("../figures/scatter_isoprene_megan_updated.png")

