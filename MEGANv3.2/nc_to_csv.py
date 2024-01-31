import xarray as xr
import pandas as pd
import numpy as np

# Apertura del file NetCDF
ds = xr.open_dataset('stand_age_final.nc')

# Calcolo delle dimensioni della griglia
x_dim, y_dim = ds.dims['x_dim'], ds.dims['y_dim']

# Inizializzazione del DataFrame per conservare i dati estratti
output_df = pd.DataFrame()

#
for variable in ds.variables:
    if variable not in ['x_dim', 'y_dim']:
        max_value = ds[variable].values.max()
        print(f'Max value for {variable}: {max_value}')

# Estrarre i dati per ogni variabile
for variable in ds.variables:
    if variable not in ['x_dim', 'y_dim']:
        # Estrazione dei dati dalla variabile
        data = ds[variable].values 
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0

        # Appiattimento dei dati in un array 1D
        #flat_data = data.ravel(order = 'C')
        flat_data = data.reshape(-1)

        # Aggiunta dei dati al DataFrame
        output_df[variable] = pd.to_numeric(flat_data, errors='coerce')

#
for variable in ds.variables:
    if variable not in ['x_dim', 'y_dim']:
        data = ds[variable].values
        flat_data = data.reshape(-1)
        print(f'Statistics for {variable}:')
        print(np.min(flat_data), np.max(flat_data), np.mean(flat_data), np.std(flat_data))


# Ricostruzione del gridID da x_idx e y_idx
gridID = np.array([y * x_dim + x + 1 for y in range(y_dim) for x in range(x_dim)])
output_df['gridID'] = gridID

# Riorganizzazione delle colonne per mettere 'gridID' per primo
cols = ['gridID'] + [col for col in output_df.columns if col != 'gridID']
output_df = output_df[cols]

# Salvataggio del DataFrame in un file CSV
output_df.to_csv('grid_age.csv', index=False)

