import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Carica osservazioni Cimone
dati = pd.read_csv('ACSM.csv')
dati['UTC end time'] = pd.to_datetime(dati['UTC end time'])
start_date = '2017-07-03'
end_date = '2017-07-28 22:00:00'
filtered_data = dati[(dati['UTC end time'] >= start_date) & (dati['UTC end time'] < end_date)]
filtered_data.set_index('UTC end time', inplace=True)
hourly_average = filtered_data.resample('H').mean()
full_range = pd.date_range(start=start_date, end=end_date, freq='H')
hourly_average = hourly_average.reindex(full_range)

# Find indices where values are greater than 2
indices_greater_than_2 = hourly_average[hourly_average.SO4 > 2.5].dropna().index

print(indices_greater_than_2)

# Convert to numerical indices
numerical_indices = hourly_average.index.get_indexer(indices_greater_than_2)

# Print the numerical indices
print(','.join(map(str, numerical_indices)))
