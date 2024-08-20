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
import netCDF4 as nc


base = xr.open_dataset('/projappl/project_2005956/CHIMERE/chimere_v2020r3_modified/domains/POVALLEY1/geog_POVALLEY1.nc')

file_path = 'pv1_conc_reduit.nc'
ds = nc.Dataset(file_path, 'r')

print(ds)

# Check available variables
print(ds.variables.keys())

m = -119 / 185
c = 119  # since it passes through (0, 119)

# Extract data dimensions and variables
releases_dim = ds.dimensions['releases'].size
x_dim = ds.dimensions['west_east'].size
y_dim = ds.dimensions['south_north'].size

# Extract data variable (assuming it's named 'data')
data_var = ds.variables['CONC']
data_var = np.sum(data_var[:, :, :, :], axis=1)

# Function to check if a point is in the lower left part
def is_lower_left(x, y):
    return y < m * x + c

selection = []

#for release in range(releases_dim):
#    lower_left_count = 0
#    upper_right_count = 0

    # Extract the data for the current release
#    data = data_var[release, :, :]

#    for x in range(x_dim):
#        for y in range(y_dim):
#            if data[y, x] != 0: 
#                if is_lower_left(x, y):
#                    lower_left_count += data[y, x]
#                else:
#                    upper_right_count += data[y, x]

    # Check if the release is mainly in the lower left part
#    if lower_left_count > upper_right_count:
#        selection.append(release)

print(len(selection))
ds = xr.open_dataset('pv1_conc_reduit.nc')

selection = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 373, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622]

sub = ds.CONC.sel(releases=selection)
sub = sub.sum(dim='bottom_top').sum(dim='releases')
sub = sub.persist()  # Persist intermediate results in memory
sub_computed = sub.compute()  # Compute final result

small_positive_value = 1e-4
sub_safe = np.where(sub <= 0, small_positive_value, sub_computed) 

# Plot
cproj = cartopy.crs.Mercator()#central_longitude=10.7, central_latitude=44.2)
fig = plt.figure(figsize=(11,11))
ax0 = plt.subplot(projection=cproj)
norm = LogNorm(vmin=1e-1, vmax=1000)
c = plt.pcolormesh(base.XLONG_M[0,:-1,:-1], base.XLAT_M[0,:-1,:-1], sub_safe, cmap='turbo', transform=ccrs.PlateCarree(),shading='gouraud',norm=norm)#,vmin=0,vmax=0.75)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.12,  extend="both")
cbar.set_label(label='SRR (s)', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)
ax0.coastlines(color='k', linewidth = 1);
ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
ax0.set_title('SRR', fontweight="bold", fontsize=25)
gl = ax0.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlabel_style = {'rotation': 0};
plt.savefig('../figures/mappa_SRR_sea.png', dpi=400)
plt.show()
