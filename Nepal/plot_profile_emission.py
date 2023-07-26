import pandas as pd
from pandas.tseries.offsets import DateOffset
import metpy
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import numpy as np
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import metpy.calc
from metpy.plots import add_metpy_logo, add_timestamp
import os

#%% plot together

# utility for nonlinear colorscale

from pylab import *
from numpy import *

class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""

    name = 'nlcmap'

    def __init__(self, cmap, levels):
                self.cmap = cmap
                self.monochrome = self.cmap.monochrome
                self.levels = asarray(levels, dtype='float64')
                self._x = self.levels/ self.levels.max()
                self.levmax = self.levels.max()
                self.levmin = self.levels.min()
                self._y = linspace(self.levmin, self.levmax, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = interp(xi, self._x, self._y)
        return self.cmap(yi/self.levmax, alpha)

#%%

# single variable file:
#ds_name = 'BNUM_HOMS_APIN.out.201412-NEPALR4_tot.nc'
ds_name = 'BNUM.out.201412-NEPALR4-ONLYVALLEY.nc'
ds = xr.open_dataset(ds_name)

# get GEOG file
geog = xr.open_dataset('geog_NEPALR4.nc')
# ds_mask = geog.LANDUSEF[:,20,:,:]
# sea
ds_sea = geog.LANDUSEF[:,16,:,:]
# elevation
ds_mask = geog.HGT_M[0,:,:]

emission_ncfile_name = 'EMISSIONI.APINEN-ONLYVALLEY.nc'
# Choose one: C5H8, APINEN, TERPEN...
spec='APINEN' # isoprene
# set different colormap maximum for the 4 domains

#%%
ds1 = xr.open_dataset(emission_ncfile_name)
ds_avg = ds1.mean(dim='Time')


# get profiles coordinates
crss1_coords = [27.40 , 27.40 , 86.6 , 86.8]
crss2_coords = [27.6 , 27.6 , 86.49, 86.88]
crss3_coords = [27.80 , 27.80 , 86.55 , 86.85]
crss4_coords = [28.00 , 28.00 , 86.6 , 86.8]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

crss1_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],crss1_coords[0])
crss1_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],crss1_coords[1])
crss1_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],crss1_coords[2])
crss1_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],crss1_coords[3])

crss2_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],crss2_coords[0])
crss2_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],crss2_coords[1])
crss2_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],crss2_coords[2])
crss2_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],crss2_coords[3])

colori = ds_avg.sel(south_north=crss2_lat_idx_1).sel(west_east=slice(crss2_lon_idx_1,crss2_lon_idx_2))

crss3_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],crss3_coords[0])
crss3_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],crss3_coords[1])
crss3_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],crss3_coords[2])
crss3_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],crss3_coords[3])

crss4_lat_idx_1 = find_nearest(geog.XLAT_M[0,:,0],crss4_coords[0])
crss4_lat_idx_2 = find_nearest(geog.XLAT_M[0,:,0],crss4_coords[1])
crss4_lon_idx_1 = find_nearest(geog.XLONG_M[0,0,:],crss4_coords[2])
crss4_lon_idx_2 = find_nearest(geog.XLONG_M[0,0,:],crss4_coords[3])

# get valleys location
df = pd.read_csv('khumbu_latlon_extended.txt',sep=',',header=None)
lats = np.flip(np.array(df[0]))
lons = np.flip(np.array(df[1]))
# ridges
df_e = pd.read_csv('khumbu_east_ridge_LATLON.txt',sep=',',header=None)
lats_e = np.flip(np.array(df_e[0]))
lons_e = np.flip(np.array(df_e[1]))
df_w = pd.read_csv('khumbu_west_ridge_LATLON.txt',sep=',',header=None)
lats_w = np.flip(np.array(df_w[0]))
lons_w = np.flip(np.array(df_w[1]))

# get chimere lat-lon indexes
chi_lats = np.zeros((len(lats)))
for i in np.arange(len(lats)):
    chi_lats[i] = find_nearest(ds.lat[:,0],lats[i])
chi_lons = np.zeros((len(lons)))
for i in np.arange(len(lons)):
    chi_lons[i] = find_nearest(ds.lon[0,:],lons[i])

colori_lat = []
for i in range(len(chi_lats)):
    colori_lat.append(ds_avg.APINEN.sel(south_north=int(chi_lats[i])).sel(west_east=int(chi_lons[i])))

chi_lats_e = np.zeros((len(lats_e)))
for i in np.arange(len(lats_e)):
    chi_lats_e[i] = find_nearest(ds.lat[:,0],lats_e[i])
chi_lons_e = np.zeros((len(lons_e)))
for i in np.arange(len(lons_e)):
    chi_lons_e[i] = find_nearest(ds.lon[0,:],lons_e[i])

chi_lats_w = np.zeros((len(lats_w)))
for i in np.arange(len(lats_w)):
    chi_lats_w[i] = find_nearest(ds.lat[:,0],lats_w[i])
chi_lons_w = np.zeros((len(lons_w)))
for i in np.arange(len(lons_w)):
    chi_lons_w[i] = find_nearest(ds.lon[0,:],lons_w[i])

# find along-valley intersection with cross profile
#chi_lat_cross1 = find_nearest(lats,crss1_coords[0])
chi_lat_cross2 = find_nearest(lats,crss2_coords[0])
#chi_lat_cross3 = find_nearest(lats,crss3_coords[0])
#chi_lat_cross4 = find_nearest(lats,crss4_coords[0])

#%%

times = ds.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times)
local_times = utc_times + DateOffset(hours=6)


# create new auxiliary dimension
ds['local_times'] = pd.DatetimeIndex(local_times)
# swap it with the old time dimension of the main variable
ds['bnum_tot'] = ds.bnum_tot.swap_dims({'Time':'local_times'})

# compute daily cycle
dailyCycle_chem = ds.bnum_tot.groupby(ds.local_times.dt.hour).mean()
local_hours = dailyCycle_chem.hour.values

# chem_N = dailyCycle_chem.isel(hour=[0,1,2,3,4,5,6,7,8,21,22,23]).mean(dim='hour')
# chem_N = dailyCycle_chem.isel(hour=[3,4,5,6,7,8]).mean(dim='hour')
#chem_N = dailyCycle_chem.isel(hour=[5]).mean(dim='hour')

# chem_D = dailyCycle_chem.isel(hour=[9,10,11,12,13,14,15,16,17,18,19,20]).mean(dim='hour')
# chem_D = dailyCycle_chem.isel(hour=[12,13,14,15,16,17]).mean(dim='hour')
#chem_D = dailyCycle_chem.isel(hour=[15]).mean(dim='hour')

chem_DAY = dailyCycle_chem.mean(dim='hour')

for daynight in ['DAILY AVERAGE']:
    # local_time = local_times[i]
    print(daynight)
    if daynight == 'DAILY AVERAGE':
        chem = chem_DAY

    #%% extract 2d vertical profiles

    #### along valley profile num concentration

    #profile = np.zeros((1, len(lats)))
    #for i in np.arange(len(lats)):
    #    profile[:,i] = chem[:,int(chi_lats[i]),int(chi_lons[i])].squeeze()

    # extract topography along valley
    elev = np.zeros((len(lats)))
    for i in np.arange(len(lats)):
        elev[i] = ds_mask[int(chi_lats[i]),int(chi_lons[i])]
    # extract topography on ridges
    elev_e = np.zeros((len(lats_e)))
    for i in np.arange(len(lats_e)):
        elev_e[i] = ds_mask[int(chi_lats_e[i]),int(chi_lons_e[i])]
    elev_w = np.zeros((len(lats_w)))
    for i in np.arange(len(lats_w)):
        elev_w[i] = ds_mask[int(chi_lats_w[i]),int(chi_lons_w[i])]

    # extract layer heigth along valley
    hlay = np.zeros((len(ds.hlay.bottom_top), len(lats)))
    thlay = np.zeros((len(ds.thlay.bottom_top), len(lats)))
    for i in np.arange(len(lats)):
        hlay[:,i] = ds.hlay[i,:,int(chi_lats[i]),int(chi_lons[i])]
        thlay[:,i] = ds.thlay[i,:,int(chi_lats[i]),int(chi_lons[i])]

    zlay = hlay - 0.5 * thlay + elev


    #### cross valley profiles

    #1

    #crss_profile_1 = np.zeros((len(ds.hlay.bottom_top), (crss1_lon_idx_2-crss1_lon_idx_1)))
    #crss_profile_1[:,:] = chem[:,int(crss1_lat_idx_1),int(crss1_lon_idx_1):int(crss1_lon_idx_2)].squeeze()

    # extract topography cross valley
    #crss_elev_1 = np.zeros((crss1_lon_idx_2-crss1_lon_idx_1))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    #crss_elev_1[:] = ds_mask[int(crss1_lat_idx_1),int(crss1_lon_idx_1):int(crss1_lon_idx_2)]

    # extract layer heigth cross valley
    #crss_hlay_1 = np.zeros((len(ds.hlay.bottom_top), (crss1_lon_idx_2-crss1_lon_idx_1)))
    #crss_thlay_1 = np.zeros((len(ds.thlay.bottom_top), (crss1_lon_idx_2-crss1_lon_idx_1)))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    #crss_hlay_1[:,:] = ds.hlay[i,:,int(crss1_lat_idx_1),int(crss1_lon_idx_1):int(crss1_lon_idx_2)]
    #crss_thlay_1[:,:] = ds.thlay[i,:,int(crss1_lat_idx_1),int(crss1_lon_idx_1):int(crss1_lon_idx_2)]

    #crss_zlay_1 = crss_hlay_1 - 0.5 * crss_thlay_1 + crss_elev_1

    #2

    crss_profile_2 = np.zeros((len(ds.hlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
        # print(i)
    crss_profile_2[:,:] = chem[:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)].squeeze()

    # extract topography cross valley
    crss_elev_2 = np.zeros((crss2_lon_idx_2-crss2_lon_idx_1))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    crss_elev_2[:] = ds_mask[int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)]

    # extract layer heigth cross valley
    crss_hlay_2 = np.zeros((len(ds.hlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    crss_thlay_2 = np.zeros((len(ds.thlay.bottom_top), (crss2_lon_idx_2-crss2_lon_idx_1)))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    crss_hlay_2[:,:] = ds.hlay[i,:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)]
    crss_thlay_2[:,:] = ds.thlay[i,:,int(crss2_lat_idx_1),int(crss2_lon_idx_1):int(crss2_lon_idx_2)]

    crss_zlay_2 = crss_hlay_2 - 0.5 * crss_thlay_2 + crss_elev_2

    #3

    #crss_profile_3 = np.zeros((len(ds.hlay.bottom_top), (crss3_lon_idx_2-crss3_lon_idx_1)))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
        # print(i)
    #crss_profile_3[:,:] = chem[:,int(crss3_lat_idx_1),int(crss3_lon_idx_1):int(crss3_lon_idx_2)].squeeze()

    # extract topography cross valley
    #crss_elev_3 = np.zeros((crss3_lon_idx_2-crss3_lon_idx_1))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    #crss_elev_3[:] = ds_mask[int(crss3_lat_idx_1),int(crss3_lon_idx_1):int(crss3_lon_idx_2)]

    # extract layer heigth cross valley
    #crss_hlay_3 = np.zeros((len(ds.hlay.bottom_top), (crss3_lon_idx_2-crss3_lon_idx_1)))
    #crss_thlay_3 = np.zeros((len(ds.thlay.bottom_top), (crss3_lon_idx_2-crss3_lon_idx_1)))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    #crss_hlay_3[:,:] = ds.hlay[i,:,int(crss3_lat_idx_1),int(crss3_lon_idx_1):int(crss3_lon_idx_2)]
    #crss_thlay_3[:,:] = ds.thlay[i,:,int(crss3_lat_idx_1),int(crss3_lon_idx_1):int(crss3_lon_idx_2)]

    #crss_zlay_3 = crss_hlay_3 - 0.5 * crss_thlay_3 + crss_elev_3

    #4

    #crss_profile_4 = np.zeros((len(ds.hlay.bottom_top), (crss4_lon_idx_2-crss4_lon_idx_1)))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
        # print(i)
    #crss_profile_4[:,:] = chem[:,int(crss4_lat_idx_1),int(crss4_lon_idx_1):int(crss4_lon_idx_2)].squeeze()

    # extract topography cross valley
    #crss_elev_4 = np.zeros((crss4_lon_idx_2-crss4_lon_idx_1))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    #crss_elev_4[:] = ds_mask[int(crss4_lat_idx_1),int(crss4_lon_idx_1):int(crss4_lon_idx_2)]

    # extract layer heigth cross valley
    #crss_hlay_4 = np.zeros((len(ds.hlay.bottom_top), (crss4_lon_idx_2-crss4_lon_idx_1)))
    #crss_thlay_4 = np.zeros((len(ds.thlay.bottom_top), (crss4_lon_idx_2-crss4_lon_idx_1)))
    # for i in np.arange((crss_lon_idx_2-crss_lon_idx_1)):
    #crss_hlay_4[:,:] = ds.hlay[i,:,int(crss4_lat_idx_1),int(crss4_lon_idx_1):int(crss4_lon_idx_2)]
    #crss_thlay_4[:,:] = ds.thlay[i,:,int(crss4_lat_idx_1),int(crss4_lon_idx_1):int(crss4_lon_idx_2)]

    #crss_zlay_4 = crss_hlay_4 - 0.5 * crss_thlay_4 + crss_elev_4

    #%%

    fig = plt.figure(figsize=(80,80))

    #%%
    ax0 = plt.subplot2grid((100, 100), (1, 1), colspan=10, rowspan=10, projection=ccrs.PlateCarree())
    # ax0.set_extent([86.4, 87.2, 27.45, 28.0])
    #levels = np.array([0,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,2000000,5000000])
    #levels = np.array([0,0.0001*10e-4,0.0002*10e-4,0.0005*10e-4,0.001*10e-4,0.002*10e-4,0.005*10e-4,0.01*10e-4,0.02*10e-4,0.05*10e-4,0.1*10e-4,0.2*10e-4,0.5*10e-4,1*10e-4,2*10e-4 ,5*10e-4,10e-3, 2*10e-3, 5*10e-3])
    levels = np.arange(0, 1e-8, 1e-9)
    cmap_lin = cm.magma_r
    cmap_nonlin = nlcmap(cmap_lin, levels)
    chem_plot = ax0.contourf(ds_avg.lon,ds_avg.lat, ds_avg[spec], transform=ccrs.PlateCarree(),levels=levels, cmap='magma_r', extend='max');
    cbar = plt.colorbar(chem_plot, fraction = 0.028, location='right', pad = 0.12)
    cbar.set_label('α-pinene emission (g m$^{-2}$ s$^{-1}$)', fontsize=12, rotation=270, labelpad=30, y=0.5)
    cbar.formatter.set_powerlimits((0, 0))

    #for lat,lon in zip(lats,lons):
    plt.plot(lons, lats, linewidth=2, color='k')
    #for lat,lon in zip(lats_e,lons_e):
    plt.plot(lons_e, lats_e, linewidth=2, color='b')
    #for lat,lon in zip(lats_w,lons_w):
    plt.plot(lons_w, lats_w, linewidth=2, color='r')

    # ax0.plot([crss1_coords[2], crss1_coords[3]], [crss1_coords[0], crss1_coords[1]],
    #           color='w', linewidth=1, marker='o', markersize=3,
    #           transform=ccrs.PlateCarree())
    ax0.plot([crss2_coords[2], crss2_coords[3]], [crss2_coords[0], crss2_coords[1]],
               color='k', linewidth=2, marker='o', markersize=3,
               transform=ccrs.PlateCarree())
    #ax0.plot([crss3_coords[2], crss3_coords[3]], [crss3_coords[0], crss3_coords[1]],
    #          color='w', linewidth=1, marker='o', markersize=3,
    #          transform=ccrs.PlateCarree())
    # ax0.plot([crss4_coords[2], crss4_coords[3]], [crss4_coords[0], crss4_coords[1]],
    #           color='w', linewidth=1, marker='o', markersize=3,
    #           transform=ccrs.PlateCarree())
    plt.plot(86.813747, 27.956906, markersize=12, marker='x', color='k') # Pyramid
    plt.plot(86.715, 27.802, markersize=12, marker='x', color='k') # Namche
    # plt.plot(86.715, 27.702, markersize=10, marker='x', color='w') # valley
    #plt.text(86.85, 27.8,s='A' , fontsize=12, color='w')
    #plt.text(86.52, 27.8,s='A', fontsize=12, color='w')
    plt.text(86.70, 27.3,s='B', fontsize=12, color='k')
    plt.text(86.81, 28.23,s='B', fontsize=12, color='k')
    plt.text(86.45, 27.6,s='A', fontsize=12, color='k')
    plt.text(86.88, 27.6,s='A', fontsize=12, color='k')
    plt.text(86.813747, 27.976906, fontsize=12, s='NCO-P', color='k') # Pyramid
    plt.text(86.715, 27.822, fontsize=12, s='Namche', color='k') 

    levels = np.arange(10., 9000., 500.)
    geogr = ax0.contour(geog.XLONG_M[0,:,:], geog.XLAT_M[0,:,:], ds_mask[:,:],transform=ccrs.PlateCarree(),levels=levels, colors='k', alpha=0.2)
    ax0.clabel(geogr, fontsize=7, inline=1,fmt = '%1.0f',colors='k')
    ax0.coastlines(color='k', linewidth = 1);
    ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
    ax0.add_feature(cartopy.feature.STATES, linewidth = 0.6, alpha = 0.3)
    ax0.add_feature(cartopy.feature.RIVERS, linewidth = 0.6,color='blue', alpha = 0.6)
    gl = ax0.gridlines(draw_labels=True,alpha=0.5);
    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    title = f'Average α-pinene emission'
    ax0.set_title(title, fontsize = 12)

    #%%
    # along-valley profile conc
    ax1 = plt.subplot2grid((100, 100), (3, 14), colspan=12, rowspan=6)
    x = np.arange(1,len(lats)+1)
    x = np.tile(x,(15,1))
    #levels = np.array([0,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,50000])
    #levels = np.array([0,0.0001*10e-3, 0.0002*10e-3,0.0005*10e-3,0.001*10e-3,0.002*10e-3,0.005*10e-3,0.01*10e-3,0.02*10e-3,0.05*10e-3,0.1*10e-3,0.2*10e-3,0.5*10e-3,1*10e-3,2*10e-3,5*10e-3])
    
    levels = np.arange(0, 1e-8, 1e-9)
    cmap_lin = cm.magma_r
    cmap_nonlin = nlcmap(cmap_lin, levels)
    cc = ax1.scatter(x[0],zlay[0],c=colori_lat, s=100 , cmap='magma_r', vmin=0, vmax=1e-8)
    cbar =  plt.colorbar(cc, fraction = 0.028, location='right', pad = 0.10, extend='max')
    cbar.set_label('α-pinene emission (g m$^{-3}$ s$^{-1})$', fontsize=12, rotation=270, labelpad=30, y=0.5)
    cbar.formatter.set_powerlimits((0, 0))
    ax1.set(xlim=(1,len(lats)))
    ax1.set(ylim=(0,10000))
    ax1.set_ylabel("Altitude (m)", fontsize=12)
    #ax1.set_xlabel("Latitude", fontsize=12)

    # ax1.plot([chi_lat_cross1, chi_lat_cross1], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)
    ax1.plot([chi_lat_cross2, chi_lat_cross2], [0, 10000],
               color='k', linewidth=1, marker='o', markersize=3)
    #ax1.plot([chi_lat_cross3, chi_lat_cross3], [0, 10000],
    #          color='k', linewidth=1, marker='o', markersize=3)
    # ax1.plot([chi_lat_cross4, chi_lat_cross4], [0, 10000],
    #           color='k', linewidth=1, marker='o', markersize=3)

    ax2=ax1.twinx()
    ax2.plot(x[0],elev,'k')
    ax2.fill_between(x[0], elev, color= 'none', hatch="\\\\\\\\",edgecolor="black")
    ax2.plot(x[0],elev_e,'b--')
    ax2.plot(x[0],elev_w,'r--')
    #ax2.set_ylabel('alpha-pinene emissions (g/m2/s)', fontsize=12)
    ax2.set(ylim=(0,10000))
    ax2.set_yticks([])
    #ax2.set_xticks(ticks=x[0],labels=x[0])
    #ax2.xaxis.set_major_locator(ticker.MultipleLocator(8))
    #ax2.set_ylabel("Altitude (m)", fontsize=12)

    ax1.set_title('Latitudinal section B-B', fontsize = 12)

    #%%
    # # cross-section 1 conc
    # ax3 = plt.subplot2grid((100, 100), (11, 14), colspan=9, rowspan=2)

    #x = np.tile(geog.XLONG_M[0,0,crss3_lon_idx_1:crss3_lon_idx_2],(15,1))
    y = np.tile(geog.XLONG_M[0,0,crss2_lon_idx_1:crss2_lon_idx_2],(15,1))

    levels = np.array([0,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,50000])
    #levels = np.array([0,0.0001*10e-3, 0.0002*10e-3,0.0005*10e-3,0.001*10e-3,0.002*10e-3,0.005*10e-3,0.01*10e-3,0.02*10e-3,0.05*10e-3,0.1*10e-3,0.2*10e-3,0.5*10e-3,1*10e-3,2*10e-3,5*10e-3])
    cmap_lin = cm.magma_r
    cmap_nonlin = nlcmap(cmap_lin, levels)

    #%%
    # cross-section 2 conc
    ax5 = plt.subplot2grid((100, 100), (11, 1), colspan=10, rowspan=8)

    #crss_profile_plot = ax5.contourf(y,crss_zlay_2,crss_profile_2,cmap=cmap_nonlin,levels=levels,extend='max')
    #cbar = plt.colorbar(crss_profile_plot, fraction = 0.028, location='right', pad = 0.10)
    #cbar.set_label('particles cm$^{-3}$', fontsize=12, rotation=270, labelpad=30, y=0.5)
    #cbar.formatter.set_powerlimits((0, 0))
    #levels = np.arange(0, 1e-8, 1e-9)
    cc = ax5.scatter(y[0], crss_elev_2, s=100 , c=colori.APINEN, cmap='magma_r', vmin=0, vmax=1e-8)
    cbar =  plt.colorbar(cc, fraction = 0.028, location='right', pad = 0.10, extend='max')
    cbar.set_label('α-pinene emission (g m$^{-3}$ s$^{-1})$', fontsize=12, rotation=270, labelpad=30, y=0.5)
    cbar.formatter.set_powerlimits((0, 0))
    ax5.set(xlim=(y[0,0],y[0,-1]))
    ax5.set(ylim=(0,7000))
    ax5.set_ylabel('Altitude (m)', fontsize=12)
    ax6=ax5.twinx()
    ax6.plot(y[0],crss_elev_2,'k')
    ax6.fill_between(y[0], crss_elev_2, color= 'none', hatch="\\\\\\\\",edgecolor="black")
    ax6.set(ylim=(0,7000))
    ax5.set_xticks([])
    ax6.set_yticks([])
    ax6.set_title('Longitudinal section A-A', fontsize = 12)

    title = 'output_figures/Emission_profile__' + daynight + '.png'
    plt.savefig(title)
