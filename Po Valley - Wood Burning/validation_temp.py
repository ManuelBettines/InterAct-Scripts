import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
import pandas as pd
import matplotlib.ticker as ticker
import datetime

ds = xr.open_dataset('/scratch/project_2007083/GC/CHIMERE/chimere_out_online_bologna_vbs_homs/nest-BOLOGNA2/temperatura.nc')
valori =  pd.read_csv('/scratch/project_2007083/GC/CHIMERE/chimere_out_online_bologna_vbs_homs/nest-BOLOGNA2/t_em.txt')

def get_GridData(df):
    coordi = pd.read_csv('/scratch/project_2007083/GC/CHIMERE/chimere_out_online_bologna_vbs_homs/nest-BOLOGNA2/coordi_temp_em.txt', header=None)
    data = df.sel(west_east=coordi[0][0]).sel(south_north=coordi[1][0]) - 273.15
    for k in range(len(coordi)-1):
        lonX, latX = coordi[0][k+1], coordi[1][k+1] 
        data = data + df.sel(west_east=lonX).sel(south_north=latX) - 273.15
    return data/len(coordi)

temp_modello = get_GridData(ds.tem2)
temp_misure = 2*valori.value

start = pd.to_datetime('20221201', format='%Y%m%d', errors='ignore')
time = [start + datetime.timedelta(hours=x) for x in range(744)]

fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot()
ax.plot(time, temp_modello, linewidth=3,label="WRF-CHIMERE")
ax.plot(time, temp_misure,'ko', markersize=3,label="Observation")
ax.legend(loc="upper left", fontsize=18)
ax.set_ylim(0,12)
ax.set_ylabel("2 meter temperature (°C)", fontsize=18)
fig.autofmt_xdate(rotation=45)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
fig.savefig("validazione.png", dpi=500)

def profili_giornalieri(arr):
    avg = []
    a = int(len(arr)/24 - 1)
    b = int(len(arr)/24)
    for i in range(24):
        somma = 0
        for j in range(a):
            somma = somma + arr[i + j*24]
        avg.append(somma/b)
    return avg

def dev_std(arr):
    avg = []
    a = int(len(arr)/24 - 1)
    b = int(len(arr)/24)
    for i in range(24):
        somma = []
        for j in range(a):
            somma.append(arr[i + j*24])
        avg.append(np.std(somma))
    return avg


daily_mis = profili_giornalieri(temp_misure)
daily_mod = profili_giornalieri(temp_modello)

std_mis = dev_std(temp_misure)
std_mod = dev_std(temp_modello)

high_mod = []
low_mod = []
high_mis = []
low_mis = []
for i in range(24):
    high_mod.append(daily_mod[i] + std_mod[i])
    low_mod.append(daily_mod[i] - std_mod[i])
    high_mis.append(daily_mis[i] + std_mis[i])
    low_mis.append(daily_mis[i] - std_mis[i])

x = list(range(0,24))

fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot()
ax.plot(daily_mod, color='red', linewidth=3,label="WRF-CHIMERE")
ax.plot(daily_mis, 'ko', markersize=10,label="Observation")
ax.fill_between(x,low_mod, high_mod, color='red', alpha=0.1)
ax.fill_between(x,low_mis, high_mis, color='black', alpha=0.1)
ax.legend(loc="upper left", fontsize=15,frameon=False)
ax.set_ylim(0,12)
ax.set_ylabel("2 meter temperature (°C)", fontsize=18)
ax.set_xlabel("Datetime", fontsize=18)
#fig.autofmt_xdate(rotation=45)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
fig.savefig("validazione_daily.png", dpi=500)





