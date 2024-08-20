import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import datetime
from datetime import timedelta

ss = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_ssoff/nest-POVALLEY4/out_total.PV4.nossalt.nc')
ship = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_shipoff/nest-POVALLEY4/out_total.PV4.noships.nc')
indus = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_indusoff/nest-POVALLEY4/out_total.PV4.noindus.nc')
dms = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_dmsoff/nest-POVALLEY4/out_total.PV4.nodms.nc')
bound = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_bdroff/nest-POVALLEY4/out_total.PV4.nobdr.nc')
base = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley/nest-POVALLEY4/out_total_SO4.PV4_new.nc').sel(Time=slice(7*24, 840))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Get chimere lat-lon indexes
idx_lats = find_nearest(ss.lat[:, 0], 44.1938)
idx_lons = find_nearest(ss.lon[0, :], 10.7015)

BASE = base.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)
BDR = BASE - bound.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)
DMS = BASE - dms.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)
IND = BASE - indus.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)
SHIP = BASE - ship.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)
SS = BASE - ss.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)

start = datetime.datetime(2019, 7, 1, 0, 0, 0)
time = [start + timedelta(hours=x) for x in range(672)]

plt.figure(figsize=(14, 7))

# Plotting the time series with fill between each dataset
#plt.fill_between(time, 0, IND, color='purple', label='Industries')
#plt.fill_between(time, IND, IND+BDR, color='blue', label='Boundary')
#plt.fill_between(time, IND+BDR, IND+BDR+SHIP, color='red', label='Ships')
plt.fill_between(time, 0, DMS, color='green', label='DMS')
#plt.fill_between(time, IND+BDR+SHIP+DMS, IND+BDR+SHIP+DMS+SS, color='gray', label='Sea Salt')
plt.plot(time, BASE, 'k--', label='Base SO$_4$')

# Adding labels and legend
plt.xlabel('Time', fontsize=18)
plt.ylabel('SO$_4$ (µg m$^{-3}$)', fontsize=18)
plt.title('Source apportionment SO$_4$', fontsize=21, fontweight='bold')
plt.legend(loc='upper right', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.ylim([0,2.5])

plt.show()
