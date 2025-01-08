import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
from pandas.tseries.offsets import DateOffset
from matplotlib.colors import LogNorm

ds = xr.open_dataset('../data/FINLAND6-UPDATED-noCC_BVOC.nc')
ds1 = xr.open_dataset('../data/FINLAND6-CC_BVOC.nc')
thlay = xr.open_dataset('../data/FIN6_thaly.nc')

times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=2)
#ds['local_times'] = pd.DatetimeIndex(local_times)
#ds['APINEN'] = ds.C5H8.swap_dims({'time_counter':'local_times'})
#ds['BPINEN'] = ds.BPINEN.swap_dims({'time_counter':'local_times'})
#ds['OCIMEN'] = ds.OCIMEN.swap_dims({'time_counter':'local_times'})
#ds['LIMONE'] = ds.LIMONE.swap_dims({'time_counter':'local_times'})

ds1['local_times'] = pd.DatetimeIndex(local_times)
ds1['APINEN'] = ds1.C5H8.swap_dims({'time_counter':'local_times'})
ds1['BPINEN'] = ds1.BPINEN.swap_dims({'time_counter':'local_times'})
ds1['OCIMEN'] = ds1.OCIMEN.swap_dims({'time_counter':'local_times'})
ds1['LIMONE'] = ds1.LIMONE.swap_dims({'time_counter':'local_times'})

#ds = ds.assign_coords(hour=('local_times', local_times.hour))
ds1 = ds1.assign_coords(hour=('local_times', local_times.hour))

thlay['local_times'] = pd.DatetimeIndex(local_times)
thlay['thlay'] = thlay.thlay.swap_dims({'time_counter':'local_times'})
#thlay['temp'] = thlay.temp.swap_dims({'time_counter':'local_times'})
thlay = thlay.assign_coords(hour=('local_times', local_times.hour))

#isoprene = ds.APINEN.sel(x=52).sel(y=43) + ds.LIMONE.sel(x=52).sel(y=43) + ds.BPINEN.sel(x=52).sel(y=43) + ds.OCIMEN.sel(x=52).sel(y=43)
cc = ds1.APINEN.sel(x=52).sel(y=43) + ds1.LIMONE.sel(x=52).sel(y=43) + ds1.BPINEN.sel(x=52).sel(y=43) + ds1.OCIMEN.sel(x=52).sel(y=43)
th = thlay.thlay.sel(x=52).sel(y=43)
hlay = th.cumsum(dim="bottom_top")
z = hlay - 1*th

z = z.sel(bottom_top=slice(1,6))
#isoprene = isoprene.sel(bottom_top=slice(1,6))
cc = cc.sel(bottom_top=slice(1,6))

z_diurnal = z.mean('local_times')
print(z_diurnal)
#iso_diurnal = isoprene.groupby('hour').mean()
iso_diurnal_cc = cc.groupby('hour').mean()
#temp_layer = thlay.temp.sel(bottom_top=1).sel(x=52).sel(y=43)
#temp_bins = np.arange(temp_layer.min(), temp_layer.max() + 2, 2)
#iso_grouped = isoprene.groupby(temp_bins).mean()

H, T = np.meshgrid(z_diurnal, iso_diurnal_cc['hour'])
#H, T = np.meshgrid(z_diurnal, temp_bins[:-1] + 1)


# Plot
fig, ax = plt.subplots(figsize=(4, 3))
norm = LogNorm(1e-1,1)
c = ax.pcolormesh(T, H, iso_diurnal_cc, cmap='jet', shading='gouraud', vmin=0, vmax=1)
ax.axhline(y=16, color='black', linestyle='--', linewidth=0.5)
cbar = plt.colorbar(c, fraction = 0.042, pad = 0.1,  extend="max")
cbar.set_label(label='Monoterpenes [ppbv]', y=0.5)
ax.set_xlabel('Datetime [Local Time]')
ax.set_ylabel('Height above ground level [m]')
ax.set_ylim([0,110])
#ax.set_yscale('log')
cbar.ax.tick_params(labelsize=12)
plt.show()
fig.savefig('../figures/monot_vertical_profile.png', dpi=350, bbox_inches='tight')
