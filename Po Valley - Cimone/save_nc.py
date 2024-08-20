import pandas as pd
import xarray as xr
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar

# Load datasets with Dask
ds_name = 'flxout_d01_20170728_220000.nc'
ds = xr.open_dataset(ds_name, chunks={'Time': 10, 'bottom_top': 1})

# Process dataset
ds = ds.squeeze('ageclass')
result = ds.CONC.mean('Time')

print('Start writing...')
# Enable a progress bar to monitor the writing process
with ProgressBar():
    result.to_netcdf('pv4_higher_res_conc_tmp.nc', format='NETCDF4')

print('File writing complete')

