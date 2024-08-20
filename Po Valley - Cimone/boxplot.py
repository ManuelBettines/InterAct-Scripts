import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import os
from matplotlib.colors import LogNorm
import math
import xarray as xr
from matplotlib.ticker import ScalarFormatter

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

ss = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_ssoff/nest-POVALLEY4/out_total.PV4.nossalt.nc').sel(Time=slice(4*24, 672))
ship = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_shipoff/nest-POVALLEY4/out_total.PV4.noships.nc').sel(Time=slice(4*24, 672))
dms = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_dmsoff/nest-POVALLEY4/out_total.PV4.nodms.nc').sel(Time=slice(4*24, 672))
indus = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley_indusoff/nest-POVALLEY4/out_total.PV4.noindus.nc').sel(Time=slice(4*24, 672))
base = xr.open_dataset('/scratch/project_2005956/GC/CHIMERE/chimere_out_online_povalley/nest-POVALLEY4/out_total_SO4.PV4_new.nc').sel(Time=slice(11*24, 840))

# Get chimere lat-lon indexes
idx_lats = find_nearest(indus.lat[:, 0], 44.1938)
idx_lons = find_nearest(indus.lon[0, :], 10.7015)

BASE = base.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)
IND = BASE - indus.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)
SS = BASE - ss.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)
DMS = BASE - dms.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)
SHIP = BASE - ship.pH2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)

sea_side = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 373, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622]


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
altezza = []
for i in range(47,622):
    height = df[0].loc[df[0]['Release'] == i, 'Height']
    height = np.min(height)
    altezza.append(height)

offset = - 48
sea_side = [i + offset for i in sea_side]
altezza = np.array(altezza)
#altezza_sea = altezza[sea_side]
altezza_sea = altezza[~np.isin(range(len(altezza)), sea_side)]
indus_sea = IND[sea_side]
indus_po = IND[~np.isin(range(len(IND)), sea_side)]
dms_sea = DMS[sea_side]
dms_po = DMS[~np.isin(range(len(IND)), sea_side)]
ship_sea = SHIP[sea_side]
ship_po = SHIP[~np.isin(range(len(IND)), sea_side)]
ss_sea = SS[sea_side]
ss_po = SS[~np.isin(range(len(IND)), sea_side)]

# Load model data
ds = xr.open_dataset('../../../nest-POVALLEY4/PV4_DMS.nc')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Get chimere lat-lon indexes
idx_lats = find_nearest(ds.lat[:, 0], 44.1938)
idx_lons = find_nearest(ds.lon[0, :], 10.7015)

conv_msa = 2.46e10

model = ds.H2SO4.sel(bottom_top=0).sel(south_north=idx_lats).sel(west_east=idx_lons)
model = model.values[sea_side]*conv_msa

ame = pd.read_csv('ame_results_cimone_OH.txt', sep=',')
ame_values = ame.AME
ame_po = ame_values[~np.isin(range(len(ame_values)), sea_side)]
ame_sea = ame_values[sea_side]

# Carica osservazioni Cimone
dati = pd.read_csv('ACSM.csv')
dati['UTC end time'] = pd.to_datetime(dati['UTC end time'])
start_date = '2017-07-05'
end_date = '2017-07-28 22:00:00'
filtered_data = dati[(dati['UTC end time'] >= start_date) & (dati['UTC end time'] < end_date)]
filtered_data.set_index('UTC end time', inplace=True)
filtered_data = filtered_data.select_dtypes(include=['number'])
hourly_average = filtered_data.resample('H').mean()
full_range = pd.date_range(start=start_date, end=end_date, freq='H')
hourly_average = hourly_average.reindex(full_range)
#hourly_average = hourly_average.reset_index()
hourly_SO4_sea = hourly_average.SO4.values[sea_side]
hourly_SO4_po = hourly_average.SO4.values[~np.isin(range(len(hourly_average.SO4.values)), sea_side)]
#time_SO4 = hourly_average.index.values[sea_side]
time_SO4 = hourly_average.index.values[~np.isin(range(len(hourly_average.index.values)), sea_side)]

time_SO4 = pd.to_datetime(time_SO4)

print(time_SO4)

ft = []
bl_day = []
bl_night = []
# Divisione per tipo di traiettoria
for i in range(len(altezza_sea)):
    if altezza_sea[i] >= 840:
        ft.append(model[i])
    else:
        if time_SO4[i].hour in [7,8,9,10,11,12,13,14,15,16,17,18]:
            bl_day.append(model[i])
        else:
            bl_night.append(model[i])

ft = [x for x in ft if not math.isnan(x)]
bl_day = [x for x in bl_day if not math.isnan(x)]
bl_night = [x for x in bl_night if not math.isnan(x)]

continental = hourly_SO4_po
mediterraneo = hourly_SO4_sea

continental = [x for x in continental if not math.isnan(x)]
mediterraneo = [x for x in mediterraneo if not math.isnan(x)]

data_to_plot = [indus_po, indus_sea]
data_to_plot_2 = [ame_po, ame_sea]


print(ft)
print(bl_day)
print(bl_night)

# Plot
import matplotlib.pyplot as plt

# Data preparation (replace 'data_to_plot' and 'data_to_plot_2' with your actual data)
data_to_plot_continental = [indus_po, dms_po + ship_po + ss_po]  # Replace with your actual data for Continental
data_to_plot_mediterranean = [indus_sea, dms_sea + ship_sea + ss_sea]  # Replace with your actual data for Mediterranean

# Define figure and axis
fig, ax = plt.subplots(figsize=(9, 9))

#ax1 = ax.twinx()
# Plot the boxplots for Continental
boxprops_red = dict(color='red', linewidth=2)
boxprops_green = dict(color='green', linewidth=2)
boxprops_purple = dict(color='purple', linewidth=2)
boxprops_gray = dict(color='gray', linewidth=2)
whiskerprops = dict(color='black', linewidth=1)
capprops = dict(color='black', linewidth=1)
medianprops = dict(color='black', linewidth=2)
meanprops = dict(markerfacecolor='green', markeredgecolor='black', markersize=10)

# Boxplots for Continental data
bp1 = ax.boxplot(data_to_plot_continental[0], positions=[1.2], widths=0.35, showmeans=True,
                 boxprops=boxprops_purple, whiskerprops=whiskerprops, capprops=capprops,
                 medianprops=medianprops, meanprops=meanprops, showfliers=False)
bp2 = ax.boxplot(data_to_plot_continental[1], positions=[1.8], widths=0.35, showmeans=True,
                 boxprops=boxprops_red, whiskerprops=whiskerprops, capprops=capprops,
                 medianprops=medianprops, meanprops=meanprops, showfliers=False)

#bp3 = ax.boxplot(data_to_plot_continental[2], positions=[2.4], widths=0.35, showmeans=True,
#                 boxprops=boxprops_red, whiskerprops=whiskerprops, capprops=capprops,
#                 medianprops=medianprops, meanprops=meanprops, showfliers=False)
#bp4 = ax.boxplot(data_to_plot_continental[3], positions=[3], widths=0.35, showmeans=True,
#                 boxprops=boxprops_gray, whiskerprops=whiskerprops, capprops=capprops,
#                 medianprops=medianprops, meanprops=meanprops, showfliers=False)

# Boxplots for Mediterranean data
bp5 = ax.boxplot(data_to_plot_mediterranean[0], positions=[3], widths=0.35, showmeans=True,
                 boxprops=boxprops_purple, whiskerprops=whiskerprops, capprops=capprops,
                 medianprops=medianprops, meanprops=meanprops, showfliers=False)
bp6 = ax.boxplot(data_to_plot_mediterranean[1], positions=[3.6], widths=0.35, showmeans=True,
                 boxprops=boxprops_red, whiskerprops=whiskerprops, capprops=capprops,
                 medianprops=medianprops, meanprops=meanprops, showfliers=False)

#bp7 = ax.boxplot(data_to_plot_mediterranean[2], positions=[5.2], widths=0.35, showmeans=True,
#                 boxprops=boxprops_red, whiskerprops=whiskerprops, capprops=capprops,
#                 medianprops=medianprops, meanprops=meanprops, showfliers=False)
#bp8 = ax.boxplot(data_to_plot_mediterranean[3], positions=[5.8], widths=0.35, showmeans=True,
#                 boxprops=boxprops_gray, whiskerprops=whiskerprops, capprops=capprops,
#                 medianprops=medianprops, meanprops=meanprops, showfliers=False)


# Customize the x-ticks
ax.set_xticks([1.5, 3.3])
ax.set_xticklabels(['Continental', 'Mediterranean'], fontsize=18)

# Grid, labels, and title
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=18)
#ax1.tick_params(axis='both', which='major', labelsize=18)
#ax.set_title('Industries SO$_4$ and AME to OH by Air Mass Direction', fontsize=21, fontweight='bold')
ax.set_ylabel('Contribution to SO$_4$ (µg m$^{-3}$)', fontsize=18)
#ax1.set_ylabel('AME to OH (ppbv s)', fontsize=18)

# Create a unified legend
purple_patch = plt.Line2D([0], [0], color='purple', lw=4, label='Industries')
#blue_patch = plt.Line2D([0], [0], color='green', lw=4, label='DMS')
red_patch = plt.Line2D([0], [0], color='red', lw=4, label='Marine')
#grey_patch = plt.Line2D([0], [0], color='gray', lw=4, label='Sea salt')
ax.legend(loc='upper center', handles=[purple_patch, red_patch], fontsize=15)

# Set y-axis limits if needed
ax.set_ylim(bottom=0)
#ax.set_yscale('log')

# Show the plot
plt.show()

