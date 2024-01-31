import netCDF4 as nc

file1 = 'grid_gf_new.nc'  
file2 = 'tree_luke.nc' 

with nc.Dataset(file1, 'r+') as ds1, nc.Dataset(file2) as ds2:
    var_name = 'TreeFrac'

    data1 = ds1.variables[var_name][:]
    data2 = ds2.variables['data'][:]

    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            if data2[90-i, j] > 0:
                data1[i, j] = data2[90-i,j]/100

    
    ds1.variables[var_name][:] = data1

