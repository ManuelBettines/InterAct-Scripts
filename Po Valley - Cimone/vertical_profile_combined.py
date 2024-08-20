import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import os
import datetime
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature

class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""
    name = 'nlcmap'
    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels / self.levels.max()
        self.levmax = self.levels.max()
        self.levmin = self.levels.min()
        self._y = np.linspace(self.levmin, self.levmax, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self._y)
        return self.cmap(yi / self.levmax, alpha)


geog = xr.open_dataset('/projappl/project_2005956/CHIMERE/chimere_v2020r3_modified/domains/POVALLEY1/geog_POVALLEY1.nc')
geog_pv4 = xr.open_dataset('/projappl/project_2005956/CHIMERE/chimere_v2020r3_modified/domains/POVALLEY4/geog_POVALLEY4.nc')
ds_mask = geog.HGT_M[0,:,:]
ds_mask_pv4 = geog_pv4.HGT_M[0,:,:]

folder_path = '../'

df = []

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # Assicurati che il file sia di tipo .txt
        file_path = os.path.join(folder_path, filename)

        data_list = []

        with open(file_path, 'r') as file:
            for line in file:
                # Split della linea per spazi multipli
                values = line.split()

                # Estrai solo i primi cinque valori dalla lista values
                selected_values = values[:6]
                float_values = [float(val) for val in selected_values]

                # Uniscili in una stringa e aggiungile alla lista dei dati
                data_list.append(float_values)

        # Crea un DataFrame dai dati
        df1 = pd.DataFrame(data_list, columns=['Release', 'Time', 'Lon', 'Lat', 'Height', 'Ground'])

        df.append(df1)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Ora all_data_frames è una lista di DataFrame, uno per ciascun file nella cartella
day=22
ora=18
rel = (day-3)*24 + ora - 1
fed = (day-1)*24 + ora
lon_new = df[0].loc[df[0]['Release'] == rel, 'Lon'][:72]
lon_new = lon_new.reset_index(drop=True)
lat_new = df[0].loc[df[0]['Release'] == rel, 'Lat'][:72]
lat_new = lat_new.reset_index(drop=True)
height = df[0].loc[df[0]['Release'] == rel, 'Height'][:72]
height = height.reset_index(drop=True)
ground = df[0].loc[df[0]['Release'] == rel, 'Ground'][:72]
ground = ground.reset_index(drop=True)
time_ = df[0].loc[df[0]['Release'] == rel, 'Time'][:72]
time_ = time_.reset_index(drop=True)

def ALT(lon, lat):
    alt = []
    lon = lon[::-1].reset_index(drop=True)
    lat = lat[::-1].reset_index(drop=True)
    for i in range(len(lon)):
        idx_lon = find_nearest(geog.XLONG_M[0,0,:], lon[i])
        idx_lat = find_nearest(geog.XLAT_M[0,:,0], lat[i])
        alt.append(float(ds_mask.sel(south_north=idx_lat).sel(west_east=idx_lon).values))
    return alt

alt = ALT(lon_new, lat_new)

time = np.arange(-72,0,1)
time  = time[::-1]
alt = alt[::-1]

# Load datasets with Dask
ds_name = 'pv1_conc_reduit.nc'
ds = xr.open_dataset(ds_name, chunks={'Time': 10, 'bottom_top': 1})

ds_dms = xr.open_dataset('../../../nest-POVALLEY1/POVALLEY1_DMS.nc')
ds_so2 = xr.open_dataset('../../../nest-POVALLEY1/out_total.PV1.nc')
ds_pblh = xr.open_dataset('../../../nest-POVALLEY1/reduced_METEO.nc')

ds1 = xr.open_dataset('../../../nest-POVALLEY1/data_POVALLEY1_nest-POVALLEY1/emis_dms.nc')
ds2 = xr.open_dataset('../../../nest-POVALLEY4/data_POVALLEY4_nest-POVALLEY4/emis_dms_pv4.nc')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Get valley locations
df = pd.read_csv('coordi.txt', sep=',', header=None)
lats = lat_new[::-1]# np.flip(np.array(df[0]))
lons = lon_new[::-1]#np.flip(np.array(df[1]))

# Get chimere lat-lon indexes
chi_lats = np.array([find_nearest(ds1.lat[:, 0], lat) for lat in lats])
chi_lons = np.array([find_nearest(ds1.lon[0, :], lon) for lon in lons])

chi_lats_pv4 = np.array([find_nearest(ds2.lat[:, 0], lat) for lat in lats])
chi_lons_pv4 = np.array([find_nearest(ds2.lon[0, :], lon) for lon in lons])

print(chi_lats_pv4)
print(chi_lons_pv4)

import math

def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    
    return distance

def cumulative_distances(lat, lon):
    cumulative_dists = [0]
    total_distance = 0.0
    
    for i in range(1, len(lat)):
        lat1, lon1 = lat[i-1], lon[i-1]
        lat2, lon2 = lat[i], lon[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        total_distance += distance
        cumulative_dists.append(total_distance)
        
    return cumulative_dists


cumulative_dists = cumulative_distances(lat_new, lon_new)
cumulative_dists = cumulative_dists[::-1]
#print(cumulative_dists)

bt = xr.open_dataset('pv1_conc_reduit.nc').isel(releases=rel)
sub = ds_dms.sel(Time=slice(fed - 72, fed))
sub_so2 = ds_so2.sel(Time=slice(fed - 72, fed))
sub_ratio = sub_so2.SO2/(sub_so2.SO2 + sub_so2.pH2SO4*0.25445)

chem_now = bt.CONC#.sum('releases')
profile_SRR = np.zeros((len(bt.CONC.bottom_top), len(lats)))
profile = np.zeros((len(ds_dms.hlay.bottom_top), len(lats)))
profile_so2 = np.zeros((len(ds_dms.hlay.bottom_top), len(lats)))
profile_so4 = np.zeros((len(ds_dms.hlay.bottom_top), len(lats)))
profile_ratio = np.zeros((len(ds_dms.hlay.bottom_top), len(lats)))
hlay = np.zeros((len(ds_dms.hlay.bottom_top), len(lats)))
thlay = np.zeros((len(ds_dms.thlay.bottom_top), len(lats)))
for i in range(len(lats)):
    profile_SRR[:, i] = chem_now[:, int(chi_lats[i]), int(chi_lons[i])].squeeze()
    profile[:,i] = sub.DMS[i,:,int(chi_lats[i]),int(chi_lons[i])].squeeze()   
    profile_so2[:,i] = sub_so2.SO2[i,:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
    profile_so4[:,i] = sub_so2.pH2SO4[i,:,int(chi_lats[i]),int(chi_lons[i])].squeeze()
    profile_ratio[:,i] = sub_ratio[i,:,int(chi_lats[i]),int(chi_lons[i])].squeeze()

    hlay[:,i] = sub.hlay[i,:,int(chi_lats[i]),int(chi_lons[i])]
    thlay[:,i] = sub.thlay[i,:,int(chi_lats[i]),int(chi_lons[i])]

#zlay = hlay - 0.5 * thlay + elev
print('Profile calculated')

hlay_srr = [50, 100, 200, 350, 500, 750, 1000, 2000, 3000, 4500, 6000, 9000]
thlay_srr = [50, 50, 100, 150, 150, 250, 250, 1000, 1000, 1500, 1500, 3000]

elev = np.array([ds_mask[int(chi_lats[i]), int(chi_lons[i])] for i in range(len(lats))])
elev_pv4 = np.array([ds_mask_pv4[int(chi_lats_pv4[i]), int(chi_lons_pv4[i])] for i in range(len(lats))])
    
final_elev = []
for i in range(len(elev)):
    if chi_lats_pv4[i] > 1: 
        final_elev.append(elev_pv4[i])
    else:
        final_elev.append(elev[i])

zlay = hlay - 0.5 * thlay + final_elev
hlay_now = np.tile(hlay_srr, (len(lats), 1)).T
thlay_now = np.tile(thlay_srr, (len(lats), 1)).T
zlay_srr = hlay_now - 0.5 * thlay_now + final_elev

fig = plt.figure(figsize=(50, 50))
ax1 = plt.subplot2grid((40, 30), (13, 0), rowspan=12, colspan=14)
x = np.arange(1, len(lats) + 1)
x = np.tile(x, (15, 1))
cmap_lin = cm.turbo
norm = LogNorm(vmin=1e-6, vmax=2e-4)

profile_plot = plt.pcolormesh(x, zlay, profile_ratio, cmap='viridis',shading='gouraud',vmin=0.1, vmax=0.5)
cbar = plt.colorbar(profile_plot, fraction=0.028, location='right', pad=0.06, extend='both')
cbar.ax.tick_params(labelsize=15)
cbar.set_label(label='SO$_2$ / (SO$_2$ + SO$_4$)', fontsize=18, y=0.5)
    
ax1.set(xlim=(1, len(lats)), ylim=(0, 4000))
ax1.set_ylabel('Height above sea level (m)', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=15)
    
start = pd.to_datetime('20170722 180000', format='%Y%m%d %H%M%S', errors='ignore')
time = [start - datetime.timedelta(hours=x) for x in range(72)]
time_str = [t.strftime('%Y-%m-%d %H:%M:%S') for t in time]
time_str = time_str[::-1]

ax1.set_xticks(ticks=range(0, 72, 12), labels=time_str[0:72:12], rotation=45, ha="right")
    
ticks = np.round(cumulative_dists).astype(int)
ticks = ticks[::12]
ax3 = ax1.twiny()

zeros = [0 for _ in range(72)]
lin = np.arange(0, 72, 1)
ax3.plot(lin, zeros)

#ax3.set_xlabel('Distance from Cimone (km)', fontsize=18)
position = np.arange(0,72,12)
ax3.set_xticks(position)
ax3.set_xticklabels(ticks)
ax3.tick_params(axis='both', which='major', labelsize=15)


ax2 = ax1.twinx()
ax2.plot(x[0], final_elev, 'k')
ax2.fill_between(x[0], final_elev, color= 'none', hatch="\\\\\\\\",edgecolor="black")
ax2.set(ylim=(0, 4000), xlim=(1, len(lats)))
ax2.set_yticks([])
x1 = np.arange(1, len(lats) + 1)
x1 = x1[::-1]
final_elev = final_elev[::-1]
ax2.plot(x1, height-ground+final_elev,'k--', linewidth=3, label='Trajectory height')
ax2.legend(loc=0, fontsize=15)
#plt.title('Trajectory arrival time - 22 July 4:00 [Local Time]', fontweight='bold', fontsize=23)

# Part Srr
ax10 = plt.subplot2grid((40, 30), (0, 0), rowspan=12, colspan=14, sharex=ax1)
x2 = np.arange(1, len(lats) + 1)
x2 = np.tile(x2, (12, 1))
cmap_lin = cm.turbo
norm = LogNorm(vmin=1e-5, vmax=10)

profile_plot = plt.pcolormesh(x2, zlay_srr, profile_SRR, cmap='turbo',shading='gouraud',norm=norm)
cbar = plt.colorbar(profile_plot, fraction=0.028, location='right', pad=0.06, extend='both')
cbar.ax.tick_params(labelsize=15)
cbar.set_label(label='SRR (s)', fontsize=18, y=0.5)

ax10.set(xlim=(1, len(lats)), ylim=(0, 4000))
#ax10.set_ylabel('Height above sea level (m)', fontsize=18)
ax10.tick_params(axis='both', which='major', labelsize=15)

ax10.set_xticks(ticks=range(0, 72, 12), labels=time_str[0:72:12], rotation=45, ha="right")

ticks = np.round(cumulative_dists).astype(int)
ticks = ticks[::12]
ax30 = ax10.twiny()

zeros = [0 for _ in range(72)]
lin = np.arange(0, 72, 1)
ax30.plot(lin, zeros)

ax30.set_xlabel('Distance from Cimone (km)', fontsize=18)
position = np.arange(0,72,12)
ax30.set_xticks(position)
ax30.set_xticklabels(ticks)
ax30.tick_params(axis='both', which='major', labelsize=15)

ax20 = ax10.twinx()
fin = final_elev[::-1]
ax20.plot(x[0], fin, 'k')
ax20.fill_between(x[0], fin, color= 'none', hatch="\\\\\\\\",edgecolor="black")
ax20.set(ylim=(0, 4000), xlim=(1, len(lats)))
ax20.set_yticks([])
x3 = np.arange(1, len(lats) + 1)
x3 = x3[::-1]
ax20.plot(x3, height-ground+final_elev,'k--', linewidth=3, label='Trajectory height')
ax20.legend(loc=0, fontsize=15)
plt.title('                                                                     Trajectory arrival time - 22 July 18:00 [Local Time]', fontweight='bold', fontsize=23)


# Part 2
ax4 = plt.subplot2grid((40, 30), (26, 0), rowspan=12, colspan=14, sharex=ax1)
cmap_lin = cm.turbo
norm = LogNorm(vmin=1e-1, vmax=3)

profile_plot = plt.pcolormesh(x, zlay, profile_so4, cmap='viridis',shading='gouraud', vmin=0, vmax=3)
cbar = plt.colorbar(profile_plot, fraction=0.028, location='right', pad=0.06, extend='both')
cbar.ax.tick_params(labelsize=15)
cbar.set_label(label='SO$_4$ (µg m$^{-3}$)', fontsize=18, y=0.5)
    
ax4.set(xlim=(1, len(lats)), ylim=(0, 4000))
#ax4.set_ylabel('Height above sea level (m)', fontsize=18)
ax4.tick_params(axis='both', which='major', labelsize=15)
    
ax4.set_xticks(ticks=range(0, 72, 12), labels=time_str[0:72:12], rotation=45, ha="right")
    
ticks = np.round(cumulative_dists).astype(int)
ticks = ticks[::12]
ax5 = ax4.twiny()

zeros = [0 for _ in range(72)]
lin = np.arange(0, 72, 1)
ax5.plot(lin, zeros)

position = np.arange(0,72,12)
ax5.set_xticks(position)
ax5.set_xticklabels(ticks)
ax5.tick_params(axis='both', which='major', labelsize=15)

ax6 = ax4.twinx()
final_elev_ = final_elev[::-1]
ax6.plot(x[0], final_elev_, 'k')
ax6.fill_between(x[0], final_elev_, color= 'none', hatch="\\\\\\\\",edgecolor="black")
ax6.set(ylim=(0, 4000), xlim=(1, len(lats)))
ax6.set_yticks([])
ax6.plot(x1, height-ground+final_elev,'k--', linewidth=3, label='Trajectory height')
ax6.legend(loc=0, fontsize=15)
plt.setp(ax10.get_xticklabels(), visible=False)
plt.setp(ax5.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)

# Part 3
base = xr.open_dataset('/projappl/project_2005956/CHIMERE/chimere_v2020r3_modified/domains/POVALLEY1/geog_POVALLEY1.nc')
ds = xr.open_dataset('pv1_conc_reduit.nc').isel(releases=rel)

sub = ds.CONC#.sel(Time=slice(0,60))
sub = sub.sum(dim='bottom_top')#.sum(dim='releases')
sub = sub.persist()  # Persist intermediate results in memory
sub_computed = sub.compute()  # Compute final result

small_positive_value = 1e-4
sub_safe = np.where(sub <= 0, small_positive_value, sub_computed)

# Plot
cproj = cartopy.crs.Mercator()
ax0 = plt.subplot2grid((35, 30), (0, 16), rowspan=20, colspan=35, projection=cproj)
norm = LogNorm(vmin=1e-1, vmax=10)
c = plt.pcolormesh(base.XLONG_M[0,:-1,:-1], base.XLAT_M[0,:-1,:-1], sub_safe, cmap='turbo', transform=ccrs.PlateCarree(),shading='gouraud',norm=norm)#,vmin=0,vmax=0.75)
ax0.plot(lon_new,lat_new,'k--', transform=ccrs.PlateCarree(),linewidth=1.5)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.12,  extend="both")
cbar.set_label(label='SRR (s)', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)
ax0.coastlines(color='k', linewidth = 1);
ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
ax0.set_title('', fontweight="bold", fontsize=25)
gl = ax0.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlabel_style = {'rotation': 0};
#plt.savefig('../figures/mappa_traj_10_july_11.png', dpi=400)
#plt.show()

# Bar plot
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

BASE = base.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons).sel(Time=fed)
BDR = BASE - bound.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons).sel(Time=fed)
DMS = BASE - dms.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons).sel(Time=fed)
IND = BASE - indus.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons).sel(Time=fed)
SHIP = BASE - ship.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons).sel(Time=fed)
SS = BASE - ss.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons).sel(Time=fed)

labels = ['Industries', 'Boundary', 'Ships', 'DMS', 'Sea Salt']
means = [IND, BDR, SHIP, DMS, SS]

ax01 = plt.subplot2grid((35, 30), (23, 16), rowspan=10, colspan=15)
ax01.yaxis.grid(linestyle='--', linewidth=1, alpha=0.4)
ax01.bar(labels, means, color=['purple', 'blue', 'red', 'green', 'grey'])
ax01.set_ylabel('Contribution to SO$_4$ (µg m$^{-3}$)', fontsize=16)
ax01.tick_params(axis='both', which='major', labelsize=15)
ax01.yaxis.tick_right()
ax01.yaxis.set_label_position("right")
ax01.set_ylim([0,0.6])
#plt.savefig('../figures/mappa_traj_10_july_11.png', dpi=400)
plt.show()

