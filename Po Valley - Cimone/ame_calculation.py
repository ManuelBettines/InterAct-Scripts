import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime
import pandas as pd
from matplotlib.colors import LogNorm
from datetime import timedelta
import matplotlib.ticker as ticker

# AME calculation
ds_emis_pv1 = xr.open_dataset('/projappl/project_2005956/CHIMERE/chimere_v2020r3_modified/domains/POVALLEY1/geog_POVALLEY1.nc')
#ds_emis_pv4 = xr.open_dataset('../../../nest-POVALLEY4/data_POVALLEY4_nest-POVALLEY4/emis_dms_pv4.nc')
#ds_back_pv1 = xr.open_dataset('../flxout_d01_20170728_220000.nc', chunks={'Time': 5}).sum('bottom_top')
ds_back_pv1 = xr.open_dataset('pv1_conc_reduit.nc')#.sel(bottom_top=slice(0,4))
#ds_back_pv4 = xr.open_dataset('../flxout_d02_20170728_220000.nc', chunks={'Time': 1}).sel(bottom_top=slice(0,4))
#ds_emis = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley/nest-POVALLEY1/out_total.PV1.nc', chunks={'Time': 320}).sum('bottom_top')

# Time dimension swap FLEXPART output
#times = ds_back_pv1.Times.astype(str)
#times = np.core.defchararray.replace(times,'_',' ')
#utc_times = pd.to_datetime(times, format='%Y%m%d %H%M%S')
#ds_back_pv1['local_times'] = pd.DatetimeIndex(utc_times)
#ds_back_pv4['local_times'] = pd.DatetimeIndex(utc_times)
#ds_back_pv1['CONC'] = ds_back_pv1.CONC.swap_dims({'Time':'local_times'})
#ds_back_pv4['CONC'] = ds_back_pv4.CONC.swap_dims({'Time':'local_times'})

# Time dimension swap CHIMERE output
#times = ds_emis.Times.astype(str)
#times = np.core.defchararray.replace(times,'_',' ')
#utc_times = pd.to_datetime(times, format='%Y-%m-%d %H:%M:%S')
#ds_emis['local_times'] = pd.DatetimeIndex(utc_times)
#ds_emis_pv4['local_times'] = pd.DatetimeIndex(utc_times)
#ds_emis['cliq'] = ds_emis.cliq.swap_dims({'Time':'local_times'})
#ds_emis_pv4['DMS'] = ds_emis_pv4.DMS.swap_dims({'Time':'local_times'})

#emis = ds_emis.cliq#.mean('Time')# + ds_emis.H2O2.mean('Time') + ds_emis.OH.mean('Time')
#emis = emis.mean('bottom_top')
emis = ds_emis_pv1.LANDMASK#USEF[:,16,:,:]#.SO2.sel(nsectors=7).mean('Time').mean('type_day').sum('nlevel_emep')
def calculate_AME_single(traj, ds_back, ds_emis):
    AME = 0
    start_date = datetime(2017, 7, 3, 0, 0, 0) + timedelta(hours=traj)
    
    back = ds_back.CONC.sel(releases=traj).sum('bottom_top')*600
    emis = ds_emis #ds_emis.SO2.sel(nsectors=2).mean('Time').mean('type_day').sum('nlevel_emep')
    
    layer_flex = [0, 3, 4, 4, 5, 5, 6, 7, 8, 9, 11]

    #for layer in range(len(layer_flex)):
    #    model = emis.sel(bottom_top=layer_flex[layer]).isel(west_east=slice(-1), south_north=slice(-1))
    #    flex = back.sel(bottom_top=layer)
    #    somma = model*flex
    #    somma = somma.sum(['west_east', 'south_north'])
    #    AME += somma

    # Loop over all time steps for the specific trajectory
    #for TIME in range(72):
    #    date = start_date - timedelta(hours=TIME)
#
 #       tmp0 = back.sel(local_times=date)
  #      tmp1 = emis.sel(local_times=date).isel(west_east=slice(-1), south_north=slice(-1))

        # Calculate the product and sum over x and y dimensions
   #     tmp = tmp0 * tmp1
    #    somma = tmp.sum(['west_east', 'south_north'])

        # Accumulate the sum into AME
 #       AME += somma
     
    tmp0 = back
    tmp1 = emis.isel(west_east=slice(-1), south_north=slice(-1))
    AME = tmp0*tmp1
    AME = AME.sum(['west_east', 'south_north'])

    # Compute the final result
    AME = AME.compute()
    AME = float(AME.values)
    print(f'{start_date}, {AME}')

    return start_date, AME


#start, AME = [calculate_AME_single(i, ds_back, ds_emis) for i in range(0,744)]
with open('ame_results_cimone_ToL.txt', 'w') as f:
    for i in range(48, 623):
        start_date, AME = calculate_AME_single(i, ds_back_pv1, emis)
        f.write(f'{start_date}, {AME}\n')

