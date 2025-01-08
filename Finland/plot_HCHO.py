import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import xarray as xr
import h5py
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Directory path
data_dir = "/scratch/project_2008324/MB/CHIMERE/chimere_output/analisi/data/HCHO_data/files/"

# Initialize lists to store data
hcho_values = []
longitudes = []
latitudes = []
AMF_ = []

# Loop through files and read data
for file in glob.glob(os.path.join(data_dir, "*.he5")):
    with h5py.File(file, 'r') as f:
        # Access datasets with corrected path
        hcho_data = f['HDFEOS']['GRIDS']['OMI Total Column Amount HCHO']['Data Fields']['ColumnAmountHCHO'][:]
        lon_data = f['HDFEOS']['GRIDS']['OMI Total Column Amount HCHO']['Data Fields']['Longitude'][:]
        lat_data = f['HDFEOS']['GRIDS']['OMI Total Column Amount HCHO']['Data Fields']['Latitude'][:]
        quality = f['HDFEOS']['GRIDS']['OMI Total Column Amount HCHO']['Data Fields']['MainDataQualityFlag'][:]
        cloud_cover = f['HDFEOS']['GRIDS']['OMI Total Column Amount HCHO']['Data Fields']['AMFCloudFraction'][:]
        solar_zenith = f['HDFEOS']['GRIDS']['OMI Total Column Amount HCHO']['Data Fields']['SolarZenithAngle'][:]
        AMF = f['HDFEOS']['GRIDS']['OMI Total Column Amount HCHO']['Data Fields']['AirMassFactor'][:]

        # Filter out bad data (assuming negative values are invalid)
        valid_mask = (quality == 0) & (cloud_cover < 40) & (solar_zenith < 70) & (hcho_data>0)
        hcho_values.append(hcho_data[valid_mask].flatten())
        longitudes.append(lon_data[valid_mask].flatten())
        latitudes.append(lat_data[valid_mask].flatten())
        AMF_.append(AMF[valid_mask].flatten())

# Convert lists to arrays
hcho_values = np.concatenate(hcho_values)
longitudes = np.concatenate(longitudes)
latitudes = np.concatenate(latitudes)

# Calculate the mean of HCHO
mean_hcho = np.mean(hcho_values)

# Create a grid for pcolormesh
lon_bins = np.linspace(longitudes.min(), longitudes.max(), 52)
lat_bins = np.linspace(latitudes.min(), latitudes.max(), 41)
HCHO_grid, lon_edges, lat_edges = np.histogram2d(longitudes, latitudes, bins=[lon_bins, lat_bins], weights=hcho_values)
counts, _, _ = np.histogram2d(longitudes, latitudes, bins=[lon_bins, lat_bins])
mean_hcho_grid = np.divide(HCHO_grid, counts, out=np.zeros_like(HCHO_grid), where=counts != 0)

# Plot
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=24.2896, central_latitude=61.8417)})
c = ax.pcolormesh(lon_edges[:-1], lat_edges[:-1], mean_hcho_grid.T, transform=ccrs.PlateCarree(), cmap='turbo', vmin=0, vmax=1e16)
#c = ax.pcolormesh(longitudes, latitudes, hcho_values, transform=ccrs.PlateCarree(), cmap='turbo', shading='gouraud', vmin=1e17, vmax=1e18)
ax.coastlines(color='k', linewidth=1)
ax.set_extent([18.9, 31.8, 59.4, 69.7], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, color='k', alpha=0.5, linewidth=0.8)

cbar = plt.colorbar(c, fraction=0.042, pad=0.21, extend="both")
cbar.set_label(label='HCHO column [molecules cm$^{-2}$]', fontsize=10, y=0.5)
cbar.ax.tick_params(labelsize=12)

gl = ax.gridlines(draw_labels=True, alpha=0.3, dms=False, x_inline=False, y_inline=False)
gl.xlabel_style = {'rotation': 0}
plt.show()
fig.savefig('../figures/hcho_column_map.png', dpi=350, bbox_inches='tight')

