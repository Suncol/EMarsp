# -*- coding: utf-8 -*-
# by sun 2021

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy import interpolate
import os
import EMars_plotter.calculator as calculator
import EMars_plotter.dataio as dataio

# residual circulation plotter
def mean_plotter_rc(filepath):
    '''
    input a filepath and plot the mean state result of residual circulations 
    auto build the result file in the basename of the filepath
    '''
    data_path=os.path.dirname(filepath)
    dataname = 'res_cir' # for residual circulation, the name of the result dir

    # set some default constant for cal res_cir on Mars
    rgas = 287.058
    g = 3.71
    radius = 3.397e6

    # load data 
    Ls = dataio.nc_reader(filepath,'Ls')
    lon = dataio.nc_reader(filepath, 'lon')
    lat = dataio.nc_reader(filepath, 'lat') 
    pfull = dataio.nc_reader(filepath, 'pfull')
    phalf = dataio.nc_reader(filepath, 'phalf')
    
    v = dataio.nc_reader(filepath,'v').transpose((3,2,1,0))
    omega = dataio.nc_reader(filepath,'omega').transpose((3,2,1,0))
    t = dataio.nc_reader(filepath,'t').transpose((3,2,1,0))
    h = dataio.nc_reader(filepath,'h').transpose((3,2,1,0))
    ps = dataio.nc_reader(filepath,'ps').transpose((2,1,0))

    # change omega wind unit from Pa/s to m/s
    tempm = np.nanmean(np.nanmean(np.nanmean(t,axis=3),axis=0),axis=0)
    rho  = pfull * 100 / (rgas * tempm) # mind this unit trans
    w = np.zeros_like(omega)
    for ipres in range(pfull.shape[0]):
        w[:,:,ipres,:] = -omega[:,:,ipres,:]/(rho[ipres]*g)

    # calcualte altitude level from pfull, phalf and h
    alt = np.nanmean(np.nanmean(np.nanmean(h,axis=0),axis=0),axis=1)
    p2a = interpolate.interp1d(phalf,alt)
    pfull_nma = np.ma.array(pfull, \
        mask=np.zeros_like(pfull), fill_value=-999).filled()
    alt = p2a(pfull_nma)
    
    # calculate the residual circulation
    vres, wres = calculator.cal_rc(lon, lat, pfull, alt, Ls, ps, v, w, t, radius)

    # now plot the figure using no gui backend --- agg backend
    plt.switch_backend('agg')
    fig, ax = plt.subplots(figsize=(8,6))
    plt.yscale('log')
    plt.gca().invert_yaxis()
    
    vrestm = np.nanmean(vres,axis=2)
    wrestm = np.nanmean(wres,axis=2)
    #vrestm[vrestm>200] = np.nan
    ax.quiver(lat[2:-1], pfull, vrestm[2:-1,:].T, wrestm[2:-1,:].T * 100  )#,scale=1,scale_units='xy')
    # vzm = np.nanmean(v,axis=0)
    # wzm = np.nanmean(w,axis=0)
    # ax.quiver(lat[2:-1], pfull, vzm[2:-1,:,100].T, wzm[2:-1,:,100].T*200,scale=1,scale_units='xy')
    ax.set_title("res cir during "+ filepath.split('_')[5] \
                + ' ' +filepath.split('_')[-1][:-3])
    plt.xlabel('latitude')
    plt.ylabel('lev')

    # save figure to png file with 800 dpi
    save_dir = os.path.join(data_path,dataname)
    
    # if the save_dir is not found, make it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    savepath = os.path.join(save_dir, dataname+'_'+ 
                filepath.split('_')[5] \
                + '_' +filepath.split('_')[-1][:-3]+'.png')
    
    plt.savefig(savepath,dpi=800)
    plt.close(fig)
    print('ploted '+ filepath + ' for residual circulation!') 
    

# ep flux plotter
def mean_plotter_ep(filepath):
    '''
    input a filepath and plot the mean state result of ep flux 
    auto build the result file in the basename of the filepath
    '''
    data_path=os.path.dirname(filepath)
    dataname = 'ep_flux' # for ep flux, the name of the result dir
    
    # set some default constant for cal ep flux on Mars
    rgas = 287.058
    g = 3.71
    av = 2*np.pi / (24*3600+37*60) # angular rotation velocity of Mars
    radius = 3.397e6 # radius of Mars
    
    # load data
    Ls = dataio.nc_reader(filepath,'Ls')
    lon = dataio.nc_reader(filepath,'lon')
    lat = dataio.nc_reader(filepath,'lat')
    pfull = dataio.nc_reader(filepath,'pfull')
    phalf = dataio.nc_reader(filepath,'phalf')

    u = dataio.nc_reader(filepath,'u').transpose((3,2,1,0))
    v = dataio.nc_reader(filepath,'v').transpose((3,2,1,0))
    omega = dataio.nc_reader(filepath,'omega').transpose((3,2,1,0))
    t = dataio.nc_reader(filepath,'t').transpose((3,2,1,0))
    h = dataio.nc_reader(filepath,'h').transpose((3,2,1,0))
    ps = dataio.nc_reader(filepath,'ps').transpose((2,1,0))
    
    # change vertical wind unit from Pa/s to m/s
    tempm = np.nanmean(np.nanmean(np.nanmean(t,axis=3),axis=0),axis=0)
    rho = pfull * 100 / (rgas * tempm)
    w = np.zeros_like(omega)
    for ipres in range(pfull.shape[0]):
        w[:,:,ipres,:] = -omega[:,:,ipres,:] / (rho[ipres] * g)
    
    # calculate altitude level from pfull, phalf and h
    alt = np.nanmean(np.nanmean(np.nanmean(h,axis=0),axis=0),axis=1)
    p2a = interpolate.interp1d(phalf, alt)
    pfull_nma = np.ma.array(pfull, mask=np.zeros_like(pfull),fill_value=-999).filled()
    alt = p2a(pfull_nma)
        
        
    EPphi, EPz, EPdiv = calculator.cal_ep(lon, lat, pfull, alt, Ls, ps, u, v, w, t, radius, av)
    
    # plot figure
    fig, ax = plt.subplots(figsize=(8,6))
    plt.yscale('log')
    plt.gca().invert_yaxis()
    
    EPphitm = np.nanmean(EPphi,axis=2)
    EPztm = np.nanmean(EPz,axis=2)    
    
    ax.quiver(lat[2:-1], pfull, EPphitm[2:-1,:].T, EPztm[2:-1,:].T  )
    ax.set_title("ep flux during "+ filepath.split('_')[5] \
                + ' ' +filepath.split('_')[-1][:-3])
        
    plt.xlabel('latitude')
    plt.ylabel('lev')
    
    save_dir = os.path.join(data_path,dataname)
    
    # if the save_dir is not found, make it 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    savepath = os.path.join(save_dir, dataname+'_'+ \
                filepath.split('_')[5] \
                + '_' +filepath.split('_')[-1][:-3]+'.png')
    
    plt.savefig(savepath,dpi=800)
    plt.close(fig)
    print('ploted '+ filepath + 'for ep flux!') 
    
# plot simple zonal mean vars, like temp, zonal wind, dust etc.
def mean_plotter_sm(filepath, dataname, vmin, vmax, cmap_use, level_num,log_cb=False):
    '''
    input a filepath and plot the mean state result of some simple variables 
    auto build the result file in the basename of the filepath
    default using for the grid in non-anal dataset 
    '''
    data_path=os.path.dirname(filepath)
    
    # load data, note that i don't transpose the matrix here
    # Ls = dataio.nc_reader(filepath, 'Ls')
    # lon = dataio.nc_reader(filepath, 'lon')
    lat = dataio.nc_reader(filepath, 'lat') 
    pfull = dataio.nc_reader(filepath, 'pfull')
    
    try:
        data = dataio.nc_reader(filepath, dataname)
    except:
        print('Can not extract: ' + dataname + 'in the file: ' + filepath)

    # change it to zonal mean and time mean
    data_tm = np.nanmean(data, axis=0) # time mean
    data_tlm = np.nanmean(data_tm, axis=2) # zonal mean
    
    # plot the time mean and zonal mean results
    plt.switch_backend('agg')
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.yscale('log')
    plt.gca().invert_yaxis()
    
    # if we need to plot log colorbar, usually for water vapor, water ice
    if log_cb:
        levels = np.logspace(vmin,vmax, level_num,base=10) 
        CS = ax.contourf(lat, pfull, data_tlm, levels=levels,cmap=cmap_use, vmin=levels[0], vmax=levels[-1], norm=LogNorm())
    else:
        levels = np.linspace(vmin, vmax, level_num)
        CS = ax.contourf(lat, pfull, data_tlm, levels=levels,cmap=cmap_use, vmin=vmin, vmax=vmax)
    
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    
    ax.set_title('Mean of '+ dataname + " during " + \
                 filepath.split('_')[5] \
                 + ' ' +filepath.split('_')[-1][:-3])
    
    plt.xlabel('latitude')
    plt.ylabel('lev')
    
    # save the figure to png file
    save_dir = os.path.join(data_path, dataname)
    
    if not os.path.exists(save_dir): # check and build the result dir
        os.makedirs(save_dir)
    
    savepath = os.path.join(save_dir, dataname+'_Mean_'+ 
                 filepath.split('_')[5] \
                 + '_' +filepath.split('_')[-1][:-3]+'.png')
    plt.savefig(savepath,dpi=800)
    plt.close(fig)
    print('Ploted result of '+ filepath + ' for '+ dataname + '!')
    
# plot simple zonal mean dust, the sum of o1, o2 and o3
def mean_plotter_dust(filepath, dataname, vmin, vmax, cmap_use, level_num,log_cb=True):
    '''
    input a filepath and plot the mean state result of some simple variables 
    auto build the result file in the basename of the filepath
    default using for the grid in non-anal dataset 
    '''
    data_path=os.path.dirname(filepath)
    
    # load data, note that i don't transpose the matrix here
    # Ls = dataio.nc_reader(filepath, 'Ls')
    # lon = dataio.nc_reader(filepath, 'lon')
    lat = dataio.nc_reader(filepath, 'lat') 
    pfull = dataio.nc_reader(filepath, 'pfull')
    
    o1 = dataio.nc_reader(filepath, 'o1')
    o2 = dataio.nc_reader(filepath, 'o2')
    o3 = dataio.nc_reader(filepath, 'o3')
    
    data = o1 + o2 + o3
    
    # change it to zonal mean and time mean
    data_tm = np.nanmean(data, axis=0) # time mean
    data_tlm = np.nanmean(data_tm, axis=2) # zonal mean
    
    # plot the time mean and zonal mean results
    plt.switch_backend('agg')
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.yscale('log')
    plt.gca().invert_yaxis()
    
    # if we need to plot log colorbar, usually for water vapor, water ice
    if log_cb:
        levels = np.logspace(vmin,vmax, level_num,base=10) 
        CS = ax.contourf(lat, pfull, data_tlm, levels=levels,cmap=cmap_use, vmin=levels[0], vmax=levels[-1], norm=LogNorm())
    else:
        levels = np.linspace(vmin, vmax, level_num)
        CS = ax.contourf(lat, pfull, data_tlm, levels=levels,cmap=cmap_use, vmin=vmin, vmax=vmax)
    
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    
    ax.set_title('Mean of '+ dataname + " during " + \
                 filepath.split('_')[5] \
                 + ' ' +filepath.split('_')[-1][:-3])
    
    plt.xlabel('latitude')
    plt.ylabel('lev')
    
    # save the figure to png file
    save_dir = os.path.join(data_path, dataname)
    
    if not os.path.exists(save_dir): # check and build the result dir
        os.makedirs(save_dir)
    
    savepath = os.path.join(save_dir, dataname+'_Mean_'+ 
                 filepath.split('_')[5] \
                 + '_' +filepath.split('_')[-1][:-3]+'.png')
    plt.savefig(savepath,dpi=800)
    plt.close(fig)
    print('Ploted result of '+ filepath + ' for '+ dataname + '!')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

