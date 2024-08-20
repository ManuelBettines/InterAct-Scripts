import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker


# Carica osservazioni Cimone
dati = pd.read_csv('ACSM.csv')
dati['UTC end time'] = pd.to_datetime(dati['UTC end time'], errors='coerce')
start_date = '2017-07-05'
end_date = '2017-07-28 22:00:00'
filtered_data = dati[(dati['UTC end time'] >= start_date) & (dati['UTC end time'] < end_date)]
filtered_data.set_index('UTC end time', inplace=True)
filtered_data = filtered_data.select_dtypes(include=['number'])
hourly_average = filtered_data.resample('H').mean()
full_range = pd.date_range(start=start_date, end=end_date, freq='H')
hourly_average = hourly_average.reindex(full_range)

# Carica AME
ame = pd.read_csv('ame_results_cimone_OH.txt', sep=',')
ame_values = ame.AME
ame_time = ame.Time

ame1 = pd.read_csv('ame_results_cimone_ToS.txt', sep=',')
tos = ame1.ToS
tos_time = ame1.Time

ame = pd.read_csv('ame_results_cimone_TOL.txt', sep=',')
tol = ame1.ToS
tol_time = ame.Time


# Convert to df
df_ame_tol = pd.DataFrame({'time': tol_time, 'tol': tol})
df_ame_tos = pd.DataFrame({'time': tos_time, 'tos': tos})
df_ame = pd.DataFrame({'time': ame_time, 'ame': ame_values})
df_so4 = pd.DataFrame({'time': ame_time, 'so4': hourly_average.SO4.values})

df_ame_tol['time'] = pd.to_datetime(df_ame_tol['time'])
df_ame_tos['time'] = pd.to_datetime(df_ame_tos['time'])
df_so4['time'] = pd.to_datetime(df_so4['time'])
df_ame['time'] = pd.to_datetime(df_ame['time'])

df_so4['hour'] = df_so4['time'].dt.floor('H')
df_ame['hour'] = df_ame['time'].dt.floor('H')
df_ame_tol['hour'] = df_ame_tol['time'].dt.floor('H')
df_ame_tos['hour'] = df_ame_tos['time'].dt.floor('H')

merged_df = pd.merge(df_ame_tol, df_so4, on='hour', how='inner')

merged_df['date'] = merged_df['hour'].dt.date
daily_avg = merged_df.groupby('date').mean().reset_index()

fig = plt.figure(figsize=(18,9))
ax = fig.add_subplot()
#ax.plot(merged_df.date, merged_df.so4, color='red')
#ax.plot(ame_time, merged_df.tol, marker='x',color='blue')
#ax1 = ax.twinx()
#ax1.plot(ame_time, ame_values, marker='s',color='orange')
c = ax.scatter(ame_values, merged_df.so4, c=ame_values, cmap='viridis', s=50,label='Hourly values')#, vmin=0.03,vmax=0.06)
#ax.scatter(daily_avg.tol, daily_avg.so4, s=50,marker='^', color='black', label='Daily Averages')
#ax.plot(x_range, y_range, color='black', linestyle='--')
ax.set_ylabel("SO$_4$ (µg m$^{-3}$)", fontsize=18)
ax.set_xlabel("Time over Sea (s)", fontsize=18)
#ax.set_xlabel("Temperature (°C)", fontsize=18)
cbar = plt.colorbar(c, fraction = 0.040, pad = 0.12,  extend="both")
cbar.set_label(label='AME to O3', fontsize=18, y=0.5)
cbar.ax.tick_params(labelsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title('SO$_4$ vs Time over Sea',fontsize=21)
ax.grid()
#ax1.xaxis.set_major_locator(ticker.MultipleLocator(12))
#fig.autofmt_xdate(rotation=45)
ax.legend(loc=0)
ax.set_ylim([0,3.1])
ax.set_xscale('log')
#ax.set_yscale('log')
#fig.savefig("../figures/SO4_time_series.png", dpi=500, bbox_inches='tight')
plt.show()


