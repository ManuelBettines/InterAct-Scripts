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
mymeg = xr.open_dataset("../data/FINLAND6-myMEGAN-BEMIS.nc")

def create_plot(sub, title="", label="", cmap='bwr', savefig=None):

    cproj = cartopy.crs.LambertConformal(central_longitude=24.3, central_latitude=61.8)
    fig = plt.figure(figsize=(9,11))
    ax0 = plt.subplot(projection=cproj)
    c = plt.pcolormesh(base.nav_lon,base.nav_lat, sub, cmap=cmap, transform=ccrs.PlateCarree(),shading='gouraud', vmin=0, vmax=30)
    cbar = plt.colorbar(c, fraction = 0.040, pad = 0.12,  extend="both")
    cbar.set_label(label=label, fontsize=18, y=0.5)
    cbar.ax.tick_params(labelsize=15)
    ax0.coastlines(color='k', linewidth = 1);
    ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
    ax0.set_title(title, fontweight="bold", fontsize=25)
    gl = ax0.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
    gl.xlabel_style = {'rotation': 0};

    if savefig:
        plt.savefig(savefig, dpi=400)
    else:
        plt.show()

#for i in range(187):
#    for j in range(118):
#        sub[i+1,j+1] = 100*(data2.values[i,j] - data1.values[i,j])/data1.values[i,j]
#        sub[0,j] = 0
#        sub[i,0] = 0
#        sub[i, 119] = 0
#        sub[188,j] = 0

# Define the start and end dates
start_date = datetime(2021, 5, 27, 0, 0, 0)
end_date = datetime(2021, 6, 2, 23, 0, 0)

sub1 = base.C5H8.sel(time_counter=slice(start_date,end_date)).mean('time_counter')*1e9
sub2 = mymeg.C5H8.sel(time_counter=slice(start_date,end_date)).mean('time_counter')*1e9

sub = (sub2-sub1)

#create_plot(sub,title="Base simulation", label="α-pinene concentration (ppbv)", cmap='magma_r', savefig='../figures/apinen_base.png')
create_plot(sub1,title="Base simulation", label="Isoprene emission (ng m$^{-2}$ s$^{-1}$)", cmap='turbo', savefig='../figures/iso_emis_base.png')
