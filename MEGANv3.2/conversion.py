import pandas as pd
import numpy as np
from netCDF4 import Dataset
import xarray as xr

# Carica il file CSV con EF1, EF2, etc. e gridID
output_grid_df = pd.read_csv('OutputGridEF.v1.0_test.csv')

x_dim = 73
y_dim = 91

# Creazione del file NetCDF
ncfile = Dataset('grid_data.nc', mode='w', format='NETCDF4')

# Definizione delle dimensioni
ncfile.createDimension('y_dim', y_dim)
ncfile.createDimension('x_dim', x_dim)

# Creazione di variabili per ogni colonna (escluso gridID)
for column in output_grid_df.columns:
    if column != 'gridID':
        data_type = output_grid_df[column].dtype
        if data_type == 'object':
            data_type = str
        variable = ncfile.createVariable(column, data_type, ('x_dim', 'y_dim'))

        # Inizializzazione della matrice con NaN
        data_matrix = np.full((x_dim, y_dim), np.nan)
        ncfile.variables[column][:, :] = data_matrix

# Riempimento delle variabili con i dati
for _, row in output_grid_df.iterrows():
    grid_id = row['gridID']
    # Calcola le coordinate X e Y da gridID
    x_idx = (grid_id - 1) % x_dim
    y_idx = (grid_id - 1) // x_dim
    for column in output_grid_df.columns:
        if column != 'gridID':
            ncfile.variables[column][x_idx, y_idx] = row[column]

# Chiusura del file
ncfile.close()

# Replace 'your_file.nc' with the path to your NetCDF file
input_file = 'grid_data.nc'
output_file = 'test.nc'

# Open the NetCDF file
ds = xr.open_dataset(input_file)

# This line will transpose all variables in the dataset
ds_transposed = ds.transpose('y_dim', 'x_dim')

# Save the modified dataset to a new file
ds_transposed.to_netcdf(output_file)

