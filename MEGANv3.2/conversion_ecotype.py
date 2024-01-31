import pandas as pd
import numpy as np
from netCDF4 import Dataset
import xarray as xr

# Carica il file CSV con EF1, EF2, etc. e gridID
output_grid_df = pd.read_csv('grid_ecotype_finland_new.csv')

# Convert 'gridID' to integers
output_grid_df['gridID'] = output_grid_df['gridID'].astype(int)

x_dim = 73
y_dim = 91

# Creazione del file NetCDF
ncfile = Dataset('tmp_1.nc', mode='w', format='NETCDF4')

# Definizione delle dimensioni
ncfile.createDimension('y_dim', y_dim)
ncfile.createDimension('x_dim', x_dim)

# Dictionary to keep track of created variables
created_variables = {}

# Create variables for each unique combination of gridID and column
for _, row in output_grid_df.iterrows():
    grid_id = row['gridID']
    
    for column in output_grid_df.columns:
        if column != 'gridID':
            # Determine the type of data
            data_type = 'f4' if output_grid_df[column].dtype.kind in 'fc' else 'i4'
            
            # Create a new variable for each combination of gridID and column
            variable_name = f'{column}_{grid_id}'
            if variable_name not in created_variables:
                ncfile.createVariable(variable_name, data_type, ('x_dim', 'y_dim'))
                created_variables[variable_name] = True

# Initialize the variables with default values
for variable_name in created_variables.keys():
    ncfile.variables[variable_name][:] = 0

# Riempimento delle variabili con i dati
for _, row in output_grid_df.iterrows():
    grid_id = row['gridID']
    x_idx = (grid_id - 1) % x_dim
    y_idx = (grid_id - 1) // x_dim
    for column in output_grid_df.columns:
        if column != 'gridID':
            variable_name = f'{column}_{grid_id}'
            ncfile.variables[variable_name][x_idx, y_idx] = row[column]

# Chiusura del file
ncfile.close()

# Replace 'your_file.nc' with the path to your NetCDF file
input_file = 'tmp_1.nc'
output_file = 'tmp_2.nc'

# Open the NetCDF file
ds = xr.open_dataset(input_file)

# This line will transpose all variables in the dataset
ds_transposed = ds.transpose('y_dim', 'x_dim')

# Save the modified dataset to a new file
ds_transposed.to_netcdf(output_file)

