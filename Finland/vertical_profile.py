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
import glob
from matplotlib.colors import LogNorm


# Load observation
file_path_pattern = "../data/VOC/Mastdata_VOC_201*.txt"

file_list = glob.glob(file_path_pattern)

all_data = []
for file in file_list:
    if '2017' in file:
        df = pd.read_csv(file, sep="\t", na_values=["NaN"])
    else:
        df = pd.read_csv(file, sep=",", na_values=["NaN"])

    all_data.append(df)

VOC = pd.concat(all_data, axis=0, ignore_index=True)

VOC = VOC[VOC.Month.isin([6,7,8])]

datetime_data = {
    'Year': VOC['Year'],
    'Month': VOC['Month'],
    'Day': VOC['Day'],
    'Hour': VOC['Hour']
}

val_mis = VOC.groupby([VOC.Height]).mean().Monoterpenes.values
height = [4.2,8.4,16.8,33.6,50.4,67.2,101,125]

VOC_grouped = VOC.groupby(['Height', 'Hour'])['Monoterpenes'].mean().reset_index()
iso_diurnal = VOC_grouped.pivot(index='Height', columns='Hour', values='Monoterpenes')

#h = [0,4.2,8.4,16.8,33.6,50.4,67.2,101]
h = [2.1,6.3,12.6,25.2,42,58.8,84.1,113]

# Plotting
fig, ax = plt.subplots(figsize=(4, 3))
norm = LogNorm(1e-1,1)
c = ax.pcolormesh(iso_diurnal.columns, h, iso_diurnal, cmap='jet', shading='gouraud', norm=norm)
ax.axhline(y=16, color='black', linestyle='--', linewidth=0.5)
cbar = plt.colorbar(c, fraction = 0.042, pad = 0.1,  extend="max")
cbar.set_label(label='Monoterpenes [ppbv]', y=0.5)
ax.set_xlabel('Datetime [Local time]')
ax.set_ylabel('Height above ground level [m]')
ax.set_ylim([0,110])
#ax.set_yscale('log')
cbar.ax.tick_params(labelsize=12)
plt.show()
fig.savefig('../figures/monot_vertical_profile_observation.png', dpi=350, bbox_inches='tight')
