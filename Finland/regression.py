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
import glob

# Load observation
file_path_pattern = "../data/Meteo/METEO*"

file_list = glob.glob(file_path_pattern)

all_data = []
for file in file_list:
    df = pd.read_csv(file, sep="\t", na_values=["NaN"])
    all_data.append(df)

observational_data_t = pd.concat(all_data, axis=0, ignore_index=True)

datetime_data = {
    'Year': observational_data_t['Year'],
    'Month': observational_data_t['Month'],
    'Day': observational_data_t['Day'],
    'Hour': observational_data_t['Hour'],
}

time = pd.to_datetime(datetime_data)

observational_data_t['datetime'] = time
observational_data_t['date'] = observational_data_t['datetime'].dt.date


# Load observation
file_path_pattern = "../data/Meteo/OZO*"

file_list = glob.glob(file_path_pattern)

all_data = []
for file in file_list:
    df = pd.read_csv(file, sep="\t", na_values=["NaN"])
    all_data.append(df)

observational_data_co2 = pd.concat(all_data, axis=0, ignore_index=True)

datetime_data = {
    'Year': observational_data_co2['Year'],
    'Month': observational_data_co2['Month'],
    'Day': observational_data_co2['Day'],
    'Hour': observational_data_co2['Hour'],
}

time = pd.to_datetime(datetime_data)

observational_data_co2['datetime'] = time
observational_data_co2['date'] = observational_data_co2['datetime'].dt.date

# Load observation
file_path_pattern = "../data/Meteo/RAD*"

file_list = glob.glob(file_path_pattern)

all_data = []
for file in file_list:
    df = pd.read_csv(file, sep="\t", na_values=["NaN"])
    all_data.append(df)

observational_data_rad = pd.concat(all_data, axis=0, ignore_index=True)

datetime_data = {
    'Year': observational_data_rad['Year'],
    'Month': observational_data_rad['Month'],
    'Day': observational_data_rad['Day'],
    'Hour': observational_data_rad['Hour'],
}

time = pd.to_datetime(datetime_data)

observational_data_rad['datetime'] = time
observational_data_rad['date'] = observational_data_rad['datetime'].dt.date


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
update = xr.open_dataset("../data/FINLAND6_emis_iso.nc")
temp = xr.open_dataset("../data/FIN6-temp.nc")

# Account for time shift
times = update.Times.astype(str)
times = np.core.defchararray.replace(times,'_',' ')
utc_times = pd.to_datetime(times, format='%Y%m%d%H%M%S.%f')
local_times = utc_times + DateOffset(hours=3)
update['local_times'] = pd.DatetimeIndex(local_times)
update['C5H8_b'] = update.C5H8_b.swap_dims({'time_counter':'local_times'})
temp['local_times'] = pd.DatetimeIndex(local_times)
temp['tem2'] = temp.tem2.swap_dims({'time_counter':'local_times'})

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

idx_lon = find_nearest(update.nav_lon[43,:], 24.2896)
idx_lat = find_nearest(update.nav_lat[:,52], 61.8417)

# Select subset
model = update.C5H8_b.sel(x=idx_lon).sel(y=idx_lat)*1e6
temp_model = temp.tem2.sel(x=idx_lon).sel(y=idx_lat) - 273.15

# Remove NaN data and align model and observation
VOC_em = VOC_em.sort_values(by='datetime')
observational_data_t = observational_data_t.sort_values(by='datetime')
observational_data_rad = observational_data_rad.sort_values(by='datetime')
observational_data_co2 = observational_data_co2.sort_values(by='datetime')

observational_data_t['datetime'] = pd.to_datetime(observational_data_t['datetime'], errors='coerce')
observational_data_rad['datetime'] = pd.to_datetime(observational_data_rad['datetime'], errors='coerce')
observational_data_co2['datetime'] = pd.to_datetime(observational_data_co2['datetime'], errors='coerce')

VOC_em = VOC_em[['datetime', 'Isoprene']]
observational_data_t = observational_data_t[['datetime', 'HYY_META.T42']]
observational_data_rad = observational_data_rad[['datetime', 'HYY_META.Glob']]
observational_data_co2 = observational_data_co2[['datetime', 'HYY_META.CO2icos168', 'HYY_META.O3168']]

merged_data = observational_data_co2.merge(observational_data_rad, on='datetime', how='outer')
tmp = merged_data[merged_data['HYY_META.Glob']>0]

tmp.set_index('datetime', inplace=True)
tmp = tmp.resample('h').first().reset_index()
#tmp.drop_duplicates(subset='datetime', inplace=True)
observational_data_t.drop_duplicates(subset='datetime', inplace=True)

tmp = tmp.merge(observational_data_t, on='datetime', how='outer')

merged_data = VOC_em.merge(tmp, on='datetime', how='outer')

VOC_em = merged_data

VOC_em['datetime'] = pd.to_datetime(VOC_em['datetime'], errors='coerce')
model_times = pd.to_datetime(update['local_times'].values, errors='coerce')

model_times_trimmed = model_times[model_times <= VOC_em['datetime'].max()]
model_trimmed = model.sel(local_times=model_times_trimmed)
model_interpolated = pd.Series(model_trimmed.values, index=model_times_trimmed).reindex(VOC_em['datetime']).interpolate('time')

temp_trimmed = temp_model.sel(local_times=model_times_trimmed)
temp_interpolated = pd.Series(temp_trimmed.values, index=model_times_trimmed).reindex(VOC_em['datetime']).interpolate('time')


model_df = pd.DataFrame({
    'model': model_interpolated
}, index=VOC_em['datetime'])

model_temp_df = pd.DataFrame({
    'model_temp': temp_interpolated
}, index=VOC_em['datetime'])


model_daily_avg = model_df.resample('h').mean()
temp_daily_avg = model_temp_df.resample('h').mean()

VOC_em.set_index('datetime', inplace=True)

observational_iso = VOC_em['Isoprene'].resample('h').mean()
observational_rad = VOC_em['HYY_META.Glob'].resample('h').mean()
observational_temp = VOC_em['HYY_META.T42'].resample('h').mean()
observational_co2 = VOC_em['HYY_META.CO2icos168'].resample('h').mean()
observational_o3 = VOC_em['HYY_META.O3168'].resample('h').mean()

observational_o3_array = np.asarray(observational_o3).flatten()
observational_co2_array = np.asarray(observational_co2).flatten()
observational_rad_array = np.asarray(observational_rad).flatten()
observational_temp_array = np.asarray(observational_temp).flatten()
observational_iso_array = np.asarray(observational_iso).flatten()
model_daily_avg_array = np.asarray(model_daily_avg).flatten()
temp_daily_avg_array = np.asarray(temp_daily_avg).flatten()

valid_indices = ~np.isnan(observational_iso_array) & ~np.isinf(observational_iso_array) & \
                ~np.isnan(model_daily_avg_array) & ~np.isinf(model_daily_avg_array)

temp_daily_avg_clean = temp_daily_avg_array[valid_indices]
observational_iso_clean = observational_iso_array[valid_indices]
model_daily_avg_clean = model_daily_avg_array[valid_indices]
observational_temp_clean = observational_temp_array[valid_indices]
observational_rad_clean = observational_rad_array[valid_indices]
observational_co2_clean = observational_co2_array[valid_indices]
observational_o3_clean = observational_o3_array[valid_indices]


bias = np.where(observational_iso_clean > 0, np.abs(np.log(model_daily_avg_clean) - np.log(observational_iso_clean)), np.nan)

rad_clean = np.where(observational_iso_clean > 0, observational_rad_clean, np.nan)
temp_clean = np.where(observational_iso_clean > 0, observational_temp_clean, np.nan)
co2_clean = np.where(observational_iso_clean > 0, observational_co2_clean, np.nan)
o3_clean = np.where(observational_iso_clean > 0, observational_o3_clean, np.nan)

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = pd.DataFrame({
    'Temp': temp_clean,
    'Rad': rad_clean,
    'Ozone': o3_clean,
    'CO2': co2_clean
})

Y = bias

mask = np.isfinite(X).all(axis=1) & np.isfinite(Y)
X = X[mask]
Y = Y[mask]

# Add a constant (intercept) to the independent variables
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(Y, X).fit()

# Print the summary of the regression model
print(model.summary())

# Check for multicollinearity using Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF) for each variable:")
print(vif_data)

y_pred = - 27.95 - 0.0515*X.Temp + 0.0003*X.Rad - 0.0288*X.Ozone + 0.0777*X.CO2

from sklearn.linear_model import Ridge

# Define and fit the Ridge model
ridge_model = Ridge(alpha=1.0)  # Adjust alpha as necessary
ridge_model.fit(X, Y)

# Print the coefficients
print("Ridge Coefficients:", ridge_model.coef_)
print("Intercept:", ridge_model.intercept_)
