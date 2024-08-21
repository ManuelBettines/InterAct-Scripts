import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import pandas as pd
import matplotlib.ticker as ticker
import os
import datetime
import pyaerocom as pya

ame = pd.read_csv('ame_results_cimone_TOL.txt', sep=',')
ame_values = ame.ToL#*0.5159e-6
ame_time = ame.Time
ame_time = pd.to_datetime(ame_time)

df_list = []
df_list.append(pya.io.EbasNasaAmesFile(file='cimone_SMPS.lev2.nas',
        only_head=False,          #set True if you only want to import header
        replace_invalid_nan=True, #replace invalid values with NaNs
        convert_timestamps=True,  #compute datetime64 timestamps from numerical values
        decode_flags=True))

#time = df_list[0].time_stamps
selected_columns = df_list[0].data[:, 2::3]
diam = df_list[0].var_defs[2::3]
d = []
for i in range(71):
    d.append(diam[i].d)
    
def converti_in_float(stringa):
    valore_float = float(stringa.replace(" nm", ""))
    return valore_float

lista_d = [converti_in_float(stringa) for stringa in d]
log = np.log(lista_d) # list diameters
differenze = np.diff(log)

result_matrix = selected_columns.T
result_matrix = result_matrix[:-1, :]
result_matrix = result_matrix[:-1, :]

risultato = result_matrix / differenze[:, np.newaxis] #dN/d(logD$_p$) (cm$^{-3}$)

lista_d = lista_d[:-1]
print(lista_d)

time = df_list[0].time_stamps

X, Y = np.meshgrid(time, lista_d)

from datetime import datetime

fig = plt.figure(figsize=(40,8))
ax = fig.add_subplot()
ax1 = ax.twinx()
ax1.spines['right'].set_color('black')
ax1.spines['right'].set_linewidth(2)
#ax2 = ax.twinx()
#ax2.spines['right'].set_position(('outward', 120))
#ax2.spines['right'].set_color('green')
#ax2.spines['right'].set_linewidth(2)
#ax2.tick_params(axis='y', colors='green', which='both')
levels = np.logspace(np.log10(100), np.log10(500000), num=1000)
a = ax.contourf(X,Y,risultato,levels=levels,locator=ticker.LogLocator(),cmap='jet', extend='both')
#lns1 = ax1.plot(time, my_array_, color="black",label='AME$_{SO2}$', linewidth=5)
#lns2 = ax2.plot(time, my_array, color="green",label='AME$_{BIO}$', linewidth=5)
#lns = lns1 + lns2 
#labs = [l.get_label() for l in lns]
ax1.plot(ame_time, ame_values, color='black', label='AME to DMS', linewidth=3)
ax.set_yscale("log")
ax1.set_yscale("log")
#ax2.set_yscale("log")
ax.set_ylabel('Diameter (nm)', fontsize=19)
ax.set_xlabel('Date', fontsize=19)
ax1.set_ylabel('AME$_{DMS}$ (g S m$^{-2}$)', fontsize=19)
#ax2.set_ylabel('AME$_{BIO}$ (g m$^{-2}$)', color="green",fontsize=19)
ax.tick_params(axis='both', which='major', labelsize=23)
ax1.tick_params(axis='both', which='major', labelsize=23)
#ax2.tick_params(axis='both', which='major', labelsize=23)
cbar = fig.colorbar(a, location="top", pad=0.1, aspect=50)
cbar.set_label('dN/d(logD$_p$) (cm$^{-3}$)',size=25)
cbar.ax.tick_params(labelsize=15)
tick_locations = [100, 1000, 10000, 100000]
tick_labels = ['$10^{{{}}}$'.format(int(np.log10(t))) for t in tick_locations]
cbar.ax.set_xticks(tick_locations)
cbar.ax.set_xticklabels(tick_labels)
#ax1.legend(lns,labs,loc=0, fontsize=25)
#ax1.set_ylim([0,3.5e-8])
#ax2.set_ylim([0,0.023])
#ax.set_title('CMN - AME (72 h)', fontsize=35)
date_limit = datetime.strptime("2017-07-06", "%Y-%m-%d")
date_limit_1 = datetime.strptime("2017-07-29", "%Y-%m-%d")
ax.set_xlim(left=date_limit, right=date_limit_1)
#fig.savefig('../figures/SIZE_DIS_AME_COMBINED.png', dpi=100, bbox_inches='tight')
plt.show()
