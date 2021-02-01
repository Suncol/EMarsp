# -*- coding: utf-8 -*-
# by sun 2021

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter
import os
import EMars_plotter.calculator as calculator
import EMars_plotter.dataio as dataio

# for movie maker
import cv2 # conda install -c conda-forge opencv or pip3 things

# a 2D smoother
def smoother(datain, sigma=1):
    return gaussian_filter(datain,sigma)


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
    # vertical coordinate
    h = dataio.nc_reader(filepath, 'h') # shape is ntime x nlevel x nlat x nlon
    htm = np.nanmean(h, axis=0) # time mean
    htlm = np.nanmean(htm, axis=2) # zonal mean
    alt = np.nanmean(htlm, axis=1) #  
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
    # set to the x scale, which is the main component in residual circulation
    q = ax.quiver(lat[2:-1], pfull, vrestm[2:-1,:].T, wrestm[2:-1,:].T * 100  ,scale=1,scale_units='x')
    ax.quiverkey(q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')
    # vzm = np.nanmean(v,axis=0)
    # wzm = np.nanmean(w,axis=0)
    # ax.quiver(lat[2:-1], pfull, vzm[2:-1,:,100].T, wzm[2:-1,:,100].T*200,scale=1,scale_units='xy')
    ax.set_title("res cir during "+ filepath.split('_')[5] \
                + ' ' +filepath.split('_')[-1][:-3])
    plt.xlim((-90,90))
    plt.xlabel('latitude')
    ax.set_ylabel('pressure level / hPa')
    ax2 = ax.twinx()
    alt = alt / 1000.0
    ax2.set_yticks(np.arange(0,np.around(np.max(alt),decimals=-1)+np.max(alt)//10,np.max(alt)//10))
    ax2.set_ylabel('altitude / km')
    # plt.ylabel('lev')

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
    ## vertical coordinate
    h = dataio.nc_reader(filepath, 'h') # shape is ntime x nlevel x nlat x nlon
    htm = np.nanmean(h, axis=0) # time mean
    htlm = np.nanmean(htm, axis=2) # zonal mean
    alt = np.nanmean(htlm, axis=1) # 
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

    # 2d smooth, method : gaussian
    EPphitm = smoother(EPphitm,4)
    EPztm = smoother(EPztm,4)

    q = ax.quiver(lat[2:-1], pfull, EPphitm[2:-1,:].T, EPztm[2:-1,:].T, scale=1,scale_units='x')
    ax.quiverkey(q, 0.9, 0.9, 1, r'$1 \frac{m^2}{s}$', labelpos='E', coordinates='figure')
    # ax.quiver(lat[2:-1], pfull[:5], EPphitm[2:-1,:5].T, EPztm[2:-1,:5].T, scale=0.01,scale_units='x')
    ax.set_title("ep flux during "+ filepath.split('_')[5] \
                + ' ' +filepath.split('_')[-1][:-3])

    plt.xlim((-90,90))
    plt.xlabel('latitude')
    ax.set_ylabel('pressure level / hPa')
    ax2 = ax.twinx()
    alt = alt / 1000.0
    ax2.set_yticks(np.arange(0,np.around(np.max(alt),decimals=-1)+np.max(alt)//10,np.max(alt)//10))
    ax2.set_ylabel('altitude / km')
    # plt.ylabel('lev')
    
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
# usually, we use the default option in the func defination
def mean_plotter_sm(filepath, dataname, vmin, vmax,cmap_use, level_num, ticks=[],log_cb=False,isDgrid=False):
    '''
    input a filepath and plot the mean state result of some simple variables 
    auto build the result file in the basename of the filepath
    default using for the grid in non-anal dataset 
    '''
    data_path=os.path.dirname(filepath)
    
    # load data, note that i don't transpose the matrix here
    # Ls = dataio.nc_reader(filepath, 'Ls')
    # lon = dataio.nc_reader(filepath, 'lon')

    # check if use the Dgrid
    if isDgrid:
        if dataname == 'U': # if plot the reanalysis dataset U in Dgrid,latu is used
            lat = dataio.nc_reader(filepath,'latu')
            print('Using Dgrid: latu instead of lat!!!')
    else:        
        lat = dataio.nc_reader(filepath, 'lat') 

    # vertical coordinate
    h = dataio.nc_reader(filepath, 'h') # shape is ntime x nlevel x nlat x nlon
    htm = np.nanmean(h, axis=0) # time mean
    htlm = np.nanmean(htm, axis=2) # zonal mean
    alt = np.nanmean(htlm, axis=1) # 
    pfull = dataio.nc_reader(filepath, 'pfull')
    
    try:
        data = dataio.nc_reader(filepath, dataname)
    except Exception as Exc:
        print("error from dataIO, check is that ok latter, we will go on now!")
        return None

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
        CB = plt.colorbar(CS, shrink=0.8, extend='both',ticks=ticks,pad=0.15)
        CB.ax.set_ylabel(dataname, rotation=0)
    else:
        levels = np.linspace(vmin, vmax, level_num)
        CS = ax.contourf(lat, pfull, data_tlm, levels=levels,cmap=cmap_use, vmin=vmin, vmax=vmax)       
        CB = plt.colorbar(CS, shrink=0.8, extend='both',ticks=ticks,pad=0.15)
        CB.ax.set_ylabel(dataname, rotation=0)
    
    ax.set_title('Mean of '+ dataname + " during " + \
                 filepath.split('_')[5] \
                 + ' ' +filepath.split('_')[-1][:-3])
    plt.xlim((-90,90))
    plt.xlabel('latitude')
    ax.set_ylabel('pressure level / hPa')
    ax2 = ax.twinx()
    alt = alt / 1000.0
    ax2.set_yticks(np.arange(0,np.around(np.max(alt),decimals=-1)+np.max(alt)//10,np.max(alt)//10))
    ax2.set_ylabel('altitude / km')
    # plt.ylabel('lev')
    
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
def mean_plotter_dust(filepath, dataname, vmin, vmax,level_num,cmap_use, ticks=[], log_cb=True):
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
    # vertical coordinate
    h = dataio.nc_reader(filepath, 'h') # shape is ntime x nlevel x nlat x nlon
    htm = np.nanmean(h, axis=0) # time mean
    htlm = np.nanmean(htm, axis=2) # zonal mean
    alt = np.nanmean(htlm, axis=1) # 
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
    
    CB = plt.colorbar(CS, shrink=0.8, extend='both',ticks=ticks,pad=0.15)
    CB.ax.set_ylabel(dataname, rotation=0)
    
    ax.set_title('Mean of '+ dataname + " during " + \
                 filepath.split('_')[5] \
                 + ' ' +filepath.split('_')[-1][:-3])
    plt.xlim((-90,90))
    plt.xlabel('latitude')
    ax.set_ylabel('pressure level / hPa')
    ax2 = ax.twinx()
    alt = alt / 1000.0
    ax2.set_yticks(np.arange(0,np.around(np.max(alt),decimals=-1)+np.max(alt)//10,np.max(alt)//10))
    ax2.set_ylabel('altitude / km')
    # plt.ylabel('lev')
    
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
    

# plot simple 2D vars, usually some column dataset
def mean_plotter_sm2d(filepath, dataname, vmin, vmax, cmap_use, level_num, log_cb=False): 
    '''
    input a filepath and plot the mean state result of some simple variables 
    auto build the result file in the basename of the filepath
    default using for the grid in non-anal dataset 
    '''
    data_path=os.path.dirname(filepath)
    
    # load data, note that i don't transpose the matrix here
    # Ls = dataio.nc_reader(filepath, 'Ls')    
    lon = dataio.nc_reader(filepath, 'lon')
    lat = dataio.nc_reader(filepath, 'lat') 
    #pfull = dataio.nc_reader(filepath, 'pfull')
    
    try:
        data = dataio.nc_reader(filepath, dataname)
    except Exception as Exc:
        print("error from dataIO, check is that ok latter, we will go on now!")
        return None

    # change it to time mean
    data_tm = np.nanmean(data, axis=0) # time mean
    
    # plot the time mean and zonal mean results
    plt.switch_backend('agg')
    fig, ax = plt.subplots(figsize=(8, 6))
    #plt.yscale('log')
    #plt.gca().invert_yaxis()
    
    # if we need to plot log colorbar, usually for water vapor, water ice
    if log_cb:
       levels = np.logspace(vmin,vmax, level_num,base=10) 
       CS = ax.contourf(lon, lat, data_tm, levels=levels,cmap=cmap_use, vmin=levels[0], vmax=levels[-1], norm=LogNorm())
    else:
        levels = np.linspace(vmin, vmax, level_num)
        CS = ax.contourf(lon, lat, data_tm, levels=levels,cmap=cmap_use, vmin=vmin, vmax=vmax)
    
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    
    ax.set_title('Mean of '+ dataname + " during " + \
                 filepath.split('_')[5] \
                 + ' ' +filepath.split('_')[-1][:-3])
    
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    
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


# plot simple zonal mean vars, like temp, zonal wind, dust etc in the user chosen Ls range 
# usually, we use the default option in the func defination
# for nowadays, using the background dataset, the anal dataset don't have the h data component 
def plotter_sm(data_path, dataname, Mars_year, Ls_range,vmin, vmax,cmap_use, level_num, ticks=[],log_cb=False,isDgrid=False):
    '''
    input a filepath and dataname with the time range of the plot 
    plot the mean state result of some simple variables 
    auto build the result file in the basename of the filepath
    default using for the grid in non-anal dataset 
    '''
    
    # get the data file we will use in the plotter 
    file_list = dataio.get_file_list(data_path,'.nc')
    file_clist = [] # files that stay in the time range of the input
    for filename in file_list:
        fileMY = int(filename.split('_')[5][2:])
        filesLs = int(filename.split('_')[6][2:5])
        fileeLs = int(filename.split('_')[6][6:9])
        if (fileMY == Mars_year) and (Ls_range[0]<fileeLs) and (Ls_range[1]>filesLs):
            file_clist.append(filename)
    # sort the file_clist
    file_clist.sort(key=file_sort)
    
    # print filename info for plotting
    print('Files that will be used in the plot '+ dataname+' between '+str(Ls_range[0])+'-'+str(Ls_range[1])+' :')
    print(file_clist)

    # get the coordinate from the first file in the file_clist
    lon = dataio.nc_reader(file_clist[0],'lon')
    if isDgrid: # check if use the Dgrid
        if dataname == 'U': # if plot the reanalysis dataset U in Dgrid,latu is used
            lat = dataio.nc_reader(file_clist[0],'latu')
            print('Using Dgrid: latu instead of lat!!!')
    else:        
        lat = dataio.nc_reader(file_clist[0], 'lat')
    pfull = dataio.nc_reader(file_clist[0], 'pfull')

    # get the data slice
    data_tlm = np.zeros((len(pfull),len(lat)))
    alt = np.zeros((len(pfull)+1)) # dirty solve, mark it for update in the future
    for i in range(len(file_clist)):
        try:
            data = dataio.nc_reader(file_clist[i],dataname) # shape is ntime x nlevel x nlat x nlon
        except Exception as Exc:
            print("error from dataIO, check is that ok latter, we will go on now!")
            return None
        h = dataio.nc_reader(file_clist[i],'h') # same shape as data
        
        # change it to zonal mean and time mean
        Ls = dataio.nc_reader(file_clist[i],'Ls')
        begin_ls = np.max((float(file_clist[i].split('_')[6][2:5]),Ls_range[0]))
        end_ls = np.min((float(file_clist[i].split('_')[6][6:9]),Ls_range[1]))
        begin_ls = get_nearpos(Ls,begin_ls)
        end_ls = get_nearpos(Ls,end_ls)

        data = data[begin_ls:end_ls,:,:,:]
        h = h[begin_ls:end_ls,:,:,:]

        # add mean of the data to data_tlm 
        data_tm = np.nanmean(data, axis=0) # time mean
        data_tlm = data_tlm + np.nanmean(data_tm, axis=2) # zonal mean
        
        # read another vertical coordinate and add mean of altitude to alt 
        htm = np.nanmean(h, axis=0) # time mean
        htlm = np.nanmean(htm, axis=2) # zonal mean
        alt = alt + np.nanmean(htlm, axis=1)  

    # get the mean value of the data and h
    data_tlm = data_tlm / len(file_clist)
    alt = alt / len(file_clist)
    
    # plot the time mean and zonal mean results
    plt.switch_backend('agg')
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.yscale('log')
    plt.gca().invert_yaxis()
    
    # if we need to plot log colorbar, usually for water vapor, water ice
    if log_cb:
        levels = np.logspace(vmin,vmax, level_num,base=10) 
        CS = ax.contourf(lat, pfull, data_tlm, levels=levels,cmap=cmap_use, vmin=levels[0], vmax=levels[-1], norm=LogNorm())
        CB = plt.colorbar(CS, shrink=0.8, extend='both',ticks=ticks,pad=0.15)
        CB.ax.set_ylabel(dataname, rotation=0)
    else:
        levels = np.linspace(vmin, vmax, level_num)
        CS = ax.contourf(lat, pfull, data_tlm, levels=levels,cmap=cmap_use, vmin=vmin, vmax=vmax)       
        CB = plt.colorbar(CS, shrink=0.8, extend='both',ticks=ticks,pad=0.15)
        CB.ax.set_ylabel(dataname, rotation=0)

    ax.set_title('Mean of '+ dataname + " during " + "MY" + str(Mars_year)+ \
        " "+ str(Ls_range[0]) + "-" + str(Ls_range[1]))

    plt.xlim((-90,90))
    plt.xlabel('latitude')
    ax.set_ylabel('pressure level / hPa')
    ax2 = ax.twinx()
    alt = alt / 1000.0
    ax2.set_yticks(np.arange(0,np.around(np.max(alt),decimals=-1)+np.max(alt)//10,np.max(alt)//10))
    ax2.set_ylabel('altitude / km')
    # plt.ylabel('lev')
    
    # save the figure to png file
    save_dir = os.path.join(data_path, dataname)
    
    if not os.path.exists(save_dir): # check and build the result dir
        os.makedirs(save_dir)
    
    savepath = os.path.join(save_dir, dataname+'_Mean_'+'MY'+str(Mars_year)+\
        '_'+str(Ls_range[0])+'_'+str(Ls_range[1]))

    plt.savefig(savepath,dpi=800)
    plt.close(fig)
    print('Ploted result of '+ data_path + ' for '+ dataname + ' between MY'+str(Mars_year) \
        +str(Ls_range[0]) + "-" + str(Ls_range[1])+'!')

# get the nearest position in a array
def get_nearpos(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


# sort the filename
def file_sort(filename): # input a filename and sort the sequence
    mars_year = int(filename.split('_')[5][2:])
    begin_day = int(filename.split('_')[6][2:5])
    
    return mars_year*360+begin_day 

# sort the images
def image_sort(image): # input a image name and sort the sequence for movie maker
    '''
    It's funny to know, this func maybe is no need to use in the windows
    '''
    # images = [img for img in os.listdir(plot_path) if img.endswith('.png')]
    # for image in images:
    #     mars_year = int(image.split('_')[2][2:])
    #     begin_day = image.split('_')[-1][2:5] # end_day: image.split('_')[-1][6:9]
    #     print(mars_year,begin_day,int(mars_year)*360+int(begin_day))
    
    # return int(image.split('_')[2][2:])*360+int(image.split('_')[-1][2:5])
    
    mars_year = int(image.split('_')[2][2:])
    begin_day = int(image.split('_')[-1][2:5])
    
    return mars_year*360+begin_day 
    
# movie maker for plot files
def movie_maker(plot_path, plot_type='.png',video_name='results.avi', fps=15): # default plot type is png 
    # get the image file lists
    # usually no need to sort the image list, it is ok for normal use    
    images = [img for img in os.listdir(plot_path) if img.endswith(plot_type)]
    # sort the images list
    images.sort(key=image_sort)
    print('ploting the images below, check the sequence if you wanna make sure')
    print(images)
    image_sort(images[0])
    
    # get the images frame from the first image
    frame = cv2.imread(os.path.join(plot_path, images[0])) 
    height, width, layers = frame.shape
    
    # use the video writer from the cv2 python lib
    print('Begin to make video, video name: ' + video_name)
    video = cv2.VideoWriter(os.path.join(plot_path,video_name), \
                            cv2.VideoWriter_fourcc(*'DIVX'),\
                            fps, (width,height))
    
    for image in images:
        print('Processing %.2f' %((images.index(image)+1) / len(images) * 100) + '% of images')
        video.write(cv2.imread(os.path.join(plot_path, image)))
        
    cv2.destroyAllWindows()
    video.release()
    print('video done, please check the dir: ', os.path.join(plot_path))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

