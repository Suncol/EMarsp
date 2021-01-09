# -*- coding: utf-8 -*-
# by sun 2021

import os
import netCDF4 as nc
from scipy.io import savemat

# get all file with a certain suffix
def get_file_list(dir,suffix):
    file_list = os.listdir(dir) # get all files in the dir
    d_list = []
    for filename in file_list:
        if (os.path.splitext(filename)[1] == suffix):
            d_list.append(os.path.join(dir,filename))
    return d_list

# read dataname in the filename
def nc_reader(filepath, dataname):  
    ds = nc.Dataset(filepath)
    data = ds[dataname][:]
    return data

# write file to mat
def mat_saver(filepath,data_dict): 
    '''
    mind that the filepath include the filename
    input a dict contains all the data and the dataname
    '''
    print('saving to mat file!')
    savemat(filepath,data_dict)
    print('mat file saved!')


    