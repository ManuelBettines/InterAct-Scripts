import netCDF4 as nc

file1 = 'grid_ecotype_tmp.nc'  
file2 = 'stand_age.nc' 

with nc.Dataset(file1, 'r+') as ds1, nc.Dataset(file2) as ds2:
    var_name = 'EcotypeID'

    data1 = ds1.variables[var_name][:] 
    data2 = ds2.variables['data'][:]

    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            if data1[i, j] > 0:
                data1[i, j] = data2[90-i,j]
    
    ds1.variables[var_name][:] = data1
