import numpy as np
import pandas as pd
from netCDF4 import Dataset

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

# Specifica il percorso del file NetCDF originale e del nuovo file
original_file_path = 'pv1_conc_reduit.nc'
new_file_path = 'sources_so4_srr.nc'
array_values = hourly_average.SO4.values # Array di valori con lunghezza x

print(array_values)
print(len(array_values))

# Apri il file NetCDF in modalità lettura
with Dataset(original_file_path, 'r') as nc_file:
    # Ottieni la variabile CONC
    conc_var = nc_file.variables['CONC']

    # Stampa le dimensioni della variabile CONC
    print("Dimensioni di CONC:", conc_var.dimensions)

# Apri il file NetCDF originale in modalità lettura
with Dataset(original_file_path, 'r') as src:
    # Crea un nuovo file NetCDF per la scrittura
    with Dataset(new_file_path, 'w') as dst:
        # Copia le dimensioni dal file sorgente
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

        # Copia tutte le variabili dal file sorgente, ma senza copiare i dati
        for name, variable in src.variables.items():
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            # Copia gli attributi delle variabili
            x.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})

        # Modifica i valori della variabile CONC in base all'array di valori
        conc_var = dst.variables['CONC']
        release_dim_index = conc_var.dimensions.index('releases')
        release_dim_size = dst.dimensions['releases'].size
        print(release_dim_size)

        # Controlla che la lunghezza dell'array corrisponda alla dimensione releases
        if len(array_values) != release_dim_size:
            raise ValueError('La lunghezza dell\'array non corrisponde alla dimensione releases del file NetCDF.')

        # Copia e modifica i dati a pezzi
        for i in range(release_dim_size):
            # Costruire uno slice object per indicizzare correttamente
            indexer = [slice(None)] * conc_var.ndim
            indexer[release_dim_index] = i
    
            # Copia i dati pezzo per pezzo dal file sorgente al file destinazione
            dst.variables['CONC'][tuple(indexer)] = src.variables['CONC'][tuple(indexer)]
    
            # Modifica solo dove CONC è diverso da zero
            conc_data = dst.variables['CONC'][tuple(indexer)]
            conc_data[conc_data != 0] *= array_values[i]
            dst.variables['CONC'][tuple(indexer)] = conc_data

        

print("Modifica completata e salvata con successo nel nuovo file!")
