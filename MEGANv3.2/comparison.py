import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

def create_spatial_plot(nc_file, title="", label="", cmap='viridis', savefig=None):
    dataset = Dataset(nc_file)
    data = dataset.variables['EF1']
    data = data[:,:]
    levels = np.arange(0,3,0.01)

    plt.figure(figsize=(10, 8))
    plt.contourf(data, cmap=cmap, levels=levels, extend="both")
    plt.colorbar(label=label)
    plt.title(title, fontsize=21)
    
    if savefig:
        plt.savefig(savefig, dpi=200)
    else:
        plt.show()

def create_difference_plot(file1, file2, title="", label="", cmap='coolwarm', savefig=None):
    dataset1 = Dataset(file1)
    dataset2 = Dataset(file2)

    data1 = dataset1.variables['EF1'] 
    data2 = dataset2.variables['EF1']

    diff = 100*(data2[:,:] - data1[:,:])/data1[:,:]
    levels = np.arange(-100,100,1)

    plt.figure(figsize=(10, 8))
    plt.contourf(diff, cmap=cmap, levels=levels, extend="both")
    plt.colorbar(label=label)
    plt.title(title, fontsize=21)

    if savefig:
        plt.savefig(savefig, dpi=200)
    else:
        plt.show()

# Example usage
file1 = 'EF_standard.nc'
file2 = 'EF_tree_luke_new_spec_final_v1.0.nc'

create_spatial_plot(file1, title="Isoprene Emission Factor", label="Isoprene Emission Factor (nanomoles m$^{-2}$ s$^{-1}$)", cmap='viridis', savefig='isoprene_final.png')
create_spatial_plot(file2, title="Isoprene Emission Factor", label="Isoprene Emission Factor (nanomoles m$^{-2}$ s$^{-1}$)", cmap='viridis', savefig='isoprene_final_v1.0.png')
create_difference_plot(file1, file2, title="Isoprene EF - Relative Difference", label="Isopre EF relative difference (%)", cmap='coolwarm', savefig='difference_final_v1.0.png')

