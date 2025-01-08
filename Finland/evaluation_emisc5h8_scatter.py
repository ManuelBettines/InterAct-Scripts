import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime
import matplotlib.ticker as ticker
import pandas as pd
from datetime import timedelta
from pandas.tseries.offsets import DateOffset
from matplotlib.ticker import LogLocator, FuncFormatter


# Emissioni VOC
VOC_em = pd.read_csv("../data/VOC/VOC_fluxes/VOC_profileflux_all_data_2010_2023.txt", na_values=["NaN"])

for col in VOC_em.columns:
    VOC_em[col] = VOC_em[col].astype(float)

VOC_em = VOC_em[VOC_em.Year.isin([2017,2018,2019])]
VOC_em = VOC_em[VOC_em.Month.isin([6,7,8])]

datetime_data = {
    'Year': VOC_em['Year'],
    'Month': VOC_em['Month'],
    'Day': VOC_em['Day'],
    'Hour': VOC_em['Hour'],
}

time = pd.to_datetime(datetime_data)

VOC_em['datetime'] = time
VOC_em['date'] = VOC_em['datetime'].dt.date

# Load simulations output
#megan = xr.open_dataset("../data/FINLAND6-MEGAN3_isoprene.nc")
update = xr.open_dataset("../data/FINLAND6_emis_iso.nc")

# Account for time shift
times = update.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=3)
update['local_times'] = pd.DatetimeIndex(local_times)
update['C5H8_b'] = update.C5H8_b.swap_dims({'time_counter':'local_times'})

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(update.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(update.nav_lat[:,52], 61.8417)

# Select subset
model = update.C5H8_b.sel(x=idx_lon).sel(y=idx_lat)*1e6
#sub_update = np.roll(sub_update, shift=3, axis=1)

# Remove NaN data and align model and observation
VOC_em = VOC_em.sort_values(by='datetime')

VOC_em['datetime'] = pd.to_datetime(VOC_em['datetime'], errors='coerce')
model_times = pd.to_datetime(update['local_times'].values, errors='coerce')

model_times_trimmed = model_times[model_times <= VOC_em['datetime'].max()]
model_trimmed = model.sel(local_times=model_times_trimmed)
model_interpolated = pd.Series(model_trimmed.values, index=model_times_trimmed).reindex(VOC_em['datetime']).interpolate('time')

model_df = pd.DataFrame({
    'model': model_interpolated
}, index=VOC_em['datetime'])

model_daily_avg = model_df.resample('D').mean()
print(model_daily_avg)

VOC_em.set_index('datetime', inplace=True)
observational_temps = VOC_em['Isoprene'].resample('D').mean()

def compute_density(x, y):
    positions = np.vstack([x, y])
    kde = gaussian_kde(positions)
    densities = kde(positions)
    return densities

observational_temps_array = np.asarray(observational_temps).flatten()
model_daily_avg_array = np.asarray(model_daily_avg).flatten()

valid_indices = ~np.isnan(observational_temps_array) & ~np.isinf(observational_temps_array) & \
                ~np.isnan(model_daily_avg_array) & ~np.isinf(model_daily_avg_array)

observational_temps_clean = observational_temps_array[valid_indices]
model_daily_avg_clean = model_daily_avg_array[valid_indices]

densities = compute_density(observational_temps_clean, model_daily_avg_clean)

mbe = np.mean(model_daily_avg_clean - observational_temps_clean)

# Plot
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot()
c = ax.scatter(observational_temps_clean, model_daily_avg_clean, s=10, color='green', edgecolor='black', linewidth=0.2)# c=100*densities, cmap='viridis', s=1)
ax.set_xlabel("Measured emissions [μg m$^{-2}$ s$^{-1}$]")
ax.set_ylabel("Model emissions [μg m$^{-2}$ s$^{-1}$]")
r0 = np.linspace(0,2)
#cbar = plt.colorbar(c, fraction = 0.040, pad = 0.07,  extend="both")
#cbar.set_label(label='Data points density', y=0.5)
#plt.text(0.05, 0.95, f"MBE: {mbe:.2e} μg m$^{{-2}}$ s$^{{-1}}$", ha='left', va='top', transform=plt.gca().transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
mbe_str = "{:.2e}".format(mbe)
mbe_str = mbe_str.replace('e', r'x10$^{{') + '}}$'
plt.text(0.05, 0.95, f"MBE: {mbe_str} μg m$^{{-2}}$ s$^{{-1}}$", ha='left', va='top', 
         transform=plt.gca().transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
y0 = 10*r0
y1 = (1/10)*r0
ax.plot(r0,y0,'k--', alpha=0.8, linewidth=0.4)
ax.plot(r0,y1,'k--', alpha=0.8, linewidth=0.4)
ax.plot(r0,r0,'k--', alpha=0.8, linewidth=0.4)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.grid(alpha=0.3, linewidth=0.4, zorder=-10)
#ax.set_ylim(top=0.15)
#ax.set_xlim(right=0.15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([1e-7,2e0])
ax.set_xlim([1e-7,2e0])
x_major = LogLocator(base=10.0, subs=(1.0,), numticks=10)
x_minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
plt.gca().xaxis.set_major_locator(x_major)
plt.gca().xaxis.set_minor_locator(x_minor)
y_major = LogLocator(base=10.0, subs=(1.0,), numticks=10)
y_minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
plt.gca().yaxis.set_major_locator(y_major)
plt.gca().yaxis.set_minor_locator(y_minor)
tick_locations = [1e-7, 1e-5, 1e-3, 1e-1]
plt.xticks(tick_locations)
plt.yticks(tick_locations)
ax.set_title('MEG3-UPD', fontsize=9)
formatter = FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x != 0 else '0')
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().set_aspect('equal', adjustable='box')
fig.savefig("../figures/isoprene_emissions_scatter.png", dpi=350, bbox_inches='tight')
plt.show()

