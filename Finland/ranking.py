import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime
import pandas as pd

base = xr.open_dataset("../../nest-FINLAND6-TEST/chim_nest-FINLAND6-TEST_2019060200_24_reduit.nc")

# Extract size distribution between 2 to 5 nm
sub_25 = base.bnum.sel(bottom_top=slice(1,1)).mean('bottom_top').sel(nbins=slice(2,5)).sum('nbins')

# Calcola la differenza tra il valore massimo e minimo per ciascun punto della griglia
max_values = np.max(sub_25, axis=0)
min_values = np.min(sub_25, axis=0)
diff_values = max_values - min_values

# Appiattisci l'array delle differenze per poter ordinare i valori
flatten_diff_values = diff_values.values.flatten()

# Calcola i percentili per ogni valore di diff_values
percentiles = pd.Series(flatten_diff_values).rank(pct=True).values * 100

# Reshape the percentile map back to the original grid shape
percentile_map_first = percentiles.reshape(diff_values.shape)

# Plot
cproj = cartopy.crs.LambertConformal(central_longitude=24.3, central_latitude=61.8)
fig = plt.figure(figsize=(9,11))
ax0 = plt.subplot(projection=cproj)
c = plt.pcolormesh(base.nav_lon,base.nav_lat, percentile_map_first, cmap='turbo', transform=ccrs.PlateCarree(),shading='gouraud', vmin=0, vmax=100)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.12,  extend="both")
cbar.set_label(label='Percentile (%)', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)
ax0.coastlines(color='k', linewidth = 1);
ax0.add_feature(cartopy.feature.BORDERS,color='k', linewidth = 0.6, alpha = 0.3);
ax0.set_title('Spatial ranking', fontweight="bold", fontsize=25)
gl = ax0.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
gl.xlabel_style = {'rotation': 0};
plt.savefig('../figures/ranking.png', dpi=400)

# Calcolo ranking vs landuse
var = base.APINEN_b + base.BPINEN_b + base.LIMONE_b + base.OCIMEN_b
var = var.mean("time_counter") * 1e6 
var_lin = var.values.flatten()

col = base.temp.sel(bottom_top=1).mean("time_counter")
color = col.values.flatten()

fig, ax = plt.subplots(figsize=(9, 9))
scatter = ax.scatter(percentiles, var_lin, marker='x', c=color, cmap="coolwarm")
cbar = fig.colorbar(scatter, ax=ax, extend='both')
cbar.set_label('Temperature (K)', fontsize=18)
cbar.ax.tick_params(labelsize=15)
ax.set_ylabel('Monoterpenes emissions (µg m$^{-2}$ s$^{-1}$)', fontsize=18)
ax.set_xlabel('Percentile', fontsize=18)
#ax.set_title('percentile', fontsize=21, fontweight='bold')
ax.grid(True)
#ax.legend(fontsize=18)
ax.tick_params(labelsize=15)
#ax.set_ylim([0,1.2])
ax.set_xlim([0,100])
ax.set_yscale('log')
#ax.set_aspect('box')
plt.tight_layout()
fig.savefig('../figures/scatter_ranking.png', dpi=400, bbox_inches='tight')


# Calcolo classi per ranking
# Carica i dati di temperatura
temp = base.pBSOA1.sel(bottom_top=1) + base.pBSOA2.sel(bottom_top=1) + base.pBSOA3.sel(bottom_top=1) + base.pBSOA4.sel(bottom_top=1) + base.pASOA1.sel(bottom_top=1) + base.pASOA2.sel(bottom_top=1) + base.pASOA3.sel(bottom_top=1) + base.pASOA4.sel(bottom_top=1) + base.pOPOA1.sel(bottom_top=1) + base.pOPOA2.sel(bottom_top=1) + base.pOPOA3.sel(bottom_top=1) + base.pOPOA4.sel(bottom_top=1) + base.pOPOA5.sel(bottom_top=1) + base.pOPOA6.sel(bottom_top=1) + base.pPOA1.sel(bottom_top=1) + base.pPOA2.sel(bottom_top=1) + base.pPOA3.sel(bottom_top=1) + base.pPOA4.sel(bottom_top=1) + base.pPOA5.sel(bottom_top=1) + base.pPOA6.sel(bottom_top=1)
temp = temp.mean('time_counter')

# Estrai i valori di temperatura come array
temp_values = temp.values.flatten() 

# Crea le classi di percentili
#bins = [0, 20, 40, 60, 80, 100]
bins = [0,10,20,30,40,50,60,70,80,90,100]
classes = np.digitize(percentile_map_first.flatten(), bins=bins, right=True)

# Inizializza un dizionario per contenere i valori di temperatura per ciascuna classe
temp_classes = {i: [] for i in range(1, 11)}

# Assegna i valori di temperatura alle rispettive classi
for i, class_label in enumerate(classes):
    if class_label in temp_classes:
        temp_classes[class_label].append(temp_values[i])

# Plot
fig, ax = plt.subplots(figsize=(15, 6))
boxprops = dict(color='red', linewidth=1.5)
ax.boxplot([temp_classes[i] for i in range(1, 11)], boxprops=boxprops, showfliers=False, showmeans=True)
#ax.set_xticklabels(['0-20', '20-40', '40-60', '60-80', '80-100'])
ax.set_xticklabels(['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim([0, 0.42])
ax.set_xlabel('Percentile Classes', fontsize=18)
ax.set_ylabel('OA (µg m$^{-3}$)', fontsize=18)
ax.set_title('OA for each ranking class', fontsize=21, fontweight='bold')
fig.savefig('../figures/ranking_vs_oa_10c.png', dpi=400)

# Calcolo ranking verticale
# Estrai la distribuzione di dimensione tra 2 a 5 nm su tutti i livelli
sub_25 = base.bnum.sel(bottom_top=slice(1, 30)).sel(nbins=slice(2, 5)).sum('nbins')

# Calcola la differenza tra il valore massimo e minimo per ciascun punto della griglia attraverso tutti i livelli
#max_values = sub_25.max(dim='bottom_top')
#min_values = sub_25.min(dim='bottom_top')
max_values = np.max(sub_25, axis=0)
min_values = np.min(sub_25, axis=0)
diff_values = max_values - min_values

# Appiattisci l'array delle differenze per calcolare i percentili
flatten_diff_values = diff_values.values.flatten()

# Calcola i percentili per ogni valore di diff_values
percentiles = pd.Series(flatten_diff_values).rank(pct=True).values * 100

# Reshape the percentile map back to the original grid shape
percentile_map = percentiles.reshape(diff_values.shape)

# Seleziona un punto specifico della griglia (ad esempio, lat=10, lon=10)
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

lon_index = find_nearest(base.nav_lon[43,:], 24.2888)
lat_index = find_nearest(base.nav_lat[:,52], 61.8439)

# Estrai i valori di sub_25 per il punto specifico su tutti i livelli
specific_point_values = percentile_map[:, lat_index, lon_index]
all_points = np.mean(percentile_map, axis=(1,2))

# Calcola altitudine layers
hlay = base.hlay.sel(x=lon_index).sel(y=lat_index).mean('time_counter')
thlay = base.thlay.sel(x=lon_index).sel(y=lat_index).mean('time_counter')
zlay = hlay - 0.5 * thlay

# Plot dei ranking per il punto specifico lungo la coordinata bottom_top
fig, ax = plt.subplots(figsize=(9, 10))
ax.plot(specific_point_values, zlay.values, marker='o', color='red', label="Hyytiälä")
ax.plot(all_points, zlay.values, marker='o', color='black', label="All domain")
ax.set_xlabel('Percentile', fontsize=18)
ax.set_ylabel('Height from ground level (m)', fontsize=18)
ax.set_title('Vertical ranking', fontsize=21, fontweight='bold')
ax.grid(True)
ax.legend(fontsize=18)
#ax.set_ylim([0,12000])
ax.tick_params(labelsize=15)
ax.set_yscale('log')
fig.savefig('../figures/ranking_vertical_profile.png', dpi=400)

# Vertical profile per ranking class
# Crea le classi di percentili solo per il primo livello
bins = [0, 20, 40, 60, 80, 100]
classes_first_level = np.digitize(percentile_map_first.flatten(), bins=bins, right=True)
# Reshape classes_first_level to the original grid shape
classes_first_level = classes_first_level.reshape(percentile_map_first.shape)

# Plot unico dei profili verticali per tutte le classi di ranking basate sul primo livello
fig, ax = plt.subplots(figsize=(9, 10))

for i in range(1, 6):
    # Trova gli indici dei punti della griglia appartenenti alla classe di ranking corrente
    indices = np.where(classes_first_level == i)
    
    # Prepara un array per memorizzare i profili verticali per questa classe
    class_profiles = []
    
    # Estrai i profili verticali per ciascun punto appartenente a questa classe
    for idx in zip(*indices):
        lat_idx, lon_idx = idx
        profile = percentile_map[:, lat_idx, lon_idx]
        class_profiles.append(profile)
    
    # Converte class_profiles in un array numpy e calcola il profilo medio
    class_profiles = np.array(class_profiles)
    mean_profile = np.mean(class_profiles, axis=0)
    
    # Plot del profilo verticale
    ax.plot(mean_profile, zlay.values, marker='o', label=f'Class {i}')

ax.set_xlabel('Percentile', fontsize=18)
ax.set_ylabel('Height from ground level (m)', fontsize=18)
ax.set_title('Vertical Ranking', fontsize=21, fontweight='bold')
ax.grid(True)
ax.legend(fontsize=15)
ax.tick_params(labelsize=15)
ax.set_xlim([0, 100])
ax.set_yscale('log')
fig.savefig('../figures/vertical_percentile_profiles_combined.png', dpi=400)
