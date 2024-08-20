import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import pandas as pd
import matplotlib.ticker as ticker
import os
import datetime


geog = xr.open_dataset('/projappl/project_2005956/CHIMERE/chimere_v2020r3_modified/domains/POVALLEY1/geog_POVALLEY1.nc')
ds_mask = geog.HGT_M[0,:,:]

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

# Ora all_data_frames è una lista di DataFrame, uno per ciascun file nella cartella
lon_new = df[0].loc[df[0]['Release'] == 170, 'Lon'][:72]
lon_new = lon_new.reset_index(drop=True)
lat_new = df[0].loc[df[0]['Release'] == 170, 'Lat'][:72]
lat_new = lat_new.reset_index(drop=True)
height = df[0].loc[df[0]['Release'] == 170, 'Height'][:72]
height = height.reset_index(drop=True)
ground = df[0].loc[df[0]['Release'] == 170, 'Ground'][:72]
ground = ground.reset_index(drop=True)
time_ = df[0].loc[df[0]['Release'] == 170, 'Time'][:72]
time_ = time_.reset_index(drop=True)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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
fig = plt.figure(figsize=(12,9))
ax1 = plt.subplot()
#
ax1.plot(time, height-ground+alt, linewidth=5)#, label="/07 12")
ax1.fill_between(time, alt, color= 'none', hatch="\\\\\\\\",edgecolor="black")
ax1.set_ylabel("Trajectory height above sea level (m)", fontsize=18)
ax1.set_xlabel("Hours prior to the release (h)", fontsize=18)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(8))
ax1.tick_params(axis='both', which='major', labelsize=18)
plt.show()
#fig.savefig('C:/Users/manue/Desktop/Vertical_test_bio_2.png', dpi=100)
