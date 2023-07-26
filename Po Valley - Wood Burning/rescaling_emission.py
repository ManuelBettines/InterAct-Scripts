import xarray as xr
import pandas as pd
import sys

filename = str(sys.argv[1]) 
VAR = filename.split('.s')[0].split('12.')[-1]

print("Rescaling {}...".format(VAR))

emis_dic = xr.open_dataset('/scratch/project_2007083/GC/CHIMERE/EMISURFOUT/VBS/BOLOGNA2/EMI_2018/EMIS.BOLOGNA2.12.{}.s.nc'.format(VAR))

def rescaling_dic(ds):
    x = getattr(ds, VAR)
    y = x[1,:,:,:,:,:]
    y = y*0.85
    y = y.transpose("Time", "type_day", "nlevel_emep", "south_north", "west_east")
    x[1,:,:,:,:,:] = y
    ds.to_netcdf('/scratch/project_2007083/GC/CHIMERE/EMISURFOUT/VBS/BOLOGNA2/EMI_DECREASE/EMIS.BOLOGNA2.12.{}.s.nc'.format(VAR))

rescaling_dic(emis_dic)

print("{} rescaled succesfully".format(VAR))
