# -*- coding: utf-8 -*-
# by sun 2021

import numpy as  np
import matplotlib.pyplot as plt
import EMars_plotter.plotter as plotter
import EMars_plotter.dataio as dataio
from joblib import Parallel, delayed

if __name__ == "__main__":

    data_path = '/home/data/Mars/EMARS/back_mean/'
    plot_path = '/home/data/Mars/EMARS/back_mean/u'
    file_list = dataio.get_file_list(data_path,'.nc')
    do_loop = False # do loop is false, not to do the loop
    do_plot = False
    do_joblib = False
    do_test = True

    # using joblib for parallel plot
    if do_joblib:
        # plot the residual circulation
        Parallel(n_jobs=5,verbose=100)(delayed(plotter.mean_plotter_rc)(filepath) for filepath in file_list)
        
        # plot the ep flux
        Parallel(n_jobs=5,verbose=100)(delayed(plotter.mean_plotter_ep)(filepath) for filepath in file_list)

    if do_loop: # just for convinent test
        # processing info
        file_index = 0
        total_filenum = len(file_list)
        print('total nc files num: ', total_filenum)

        # unit test, maybe you can also use it, no need to sort the file list, we add processing info here
        for filepath in file_list: # as i know, server take so long time to start the process in the joblib

            print('Processing %.2f' %((file_index+1) / total_filenum * 100) + '% of nc files')
            file_index += 1
            # # try plot residual circulation
            # plotter.mean_plotter_rc(filepath)

            # # try plot ep flux
            # plotter.mean_plotter_ep(filepath)

            # # try plot temperature
            plotter.mean_plotter_sm(filepath, 't', vmin=0, vmax=250, ticks=[0,50,100,150,200,250],cmap_use=plt.cm.jet, level_num=50)

            # # #try plot zonal wind
            plotter.mean_plotter_sm(filepath, 'u', vmin=-150, vmax=150, ticks=[-150,-100,-50,0,50,100,150],cmap_use=plt.cm.seismic,level_num=50,isDgrid=False)

            # # # try plot water ice
            # plotter.mean_plotter_sm(filepath, 'cld', vmin=-16, vmax=-2, ticks=np.logspace(-16,-2,15,base=10),cmap_use=plt.cm.gist_ncar,level_num=50,log_cb=True)

            # # # try plot water vapor
            # plotter.mean_plotter_sm(filepath, 'vap', vmin=-16, vmax=-2, ticks=np.logspace(-16,-2,15,base=10),cmap_use=plt.cm.gist_ncar,level_num=50,log_cb=True)

            # # # try plot dust o1+o2+o3
            # plotter.mean_plotter_dust(filepath, 'dust',vmin=-23, vmax=-4,ticks=np.logspace(-23,-4,20,base=10), cmap_use=plt.cm.YlOrBr,level_num=60,log_cb=True)

            ## some 2d variables

            # try plot water vapor column
            #plotter.mean_plotter_sm2d(filepath, 'wcol', vmin=-8, vmax=-3, cmap_use=plt.cm.cool, level_num=6, log_cb=True)

            # try plot total visible opacity from aerosols
            #plotter.mean_plotter_sm2d(filepath,'vod', vmin=-3, vmax=2, cmap_use=plt.cm.cool, level_num=6, log_cb=True)


    if do_plot:
        # try to make movie in certain plot path
        plotter.movie_maker(plot_path,fps=5)

    if do_test:
        # plot the simple vars, take temperature from MY Ls20 to Ls110 as an example
        #plotter.plotter_sm(data_path, 't', 26, (0,30),vmin=0, vmax=250,ticks=[0,50,100,150,200,250],cmap_use=plt.cm.jet, level_num=50)
        #plotter.plotter_dust(data_path,'dust',25,(20,110), vmin=-23, vmax=-4,ticks=np.logspace(-23,-4,20,base=10), cmap_use=plt.cm.YlOrBr,level_num=60,log_cb=True)
        for i in range(360*4):
            plotter.plotter_sm(data_path, 't', 26, (i/4,(i+1)/4),vmin=0, vmax=250,ticks=[0,50,100,150,200,250],cmap_use=plt.cm.jet, level_num=50)
            plotter.plotter_sm(data_path, 'u', 26, (i/4,(i+1)/4) , vmin=-150, vmax=150, ticks=[-150,-100,-50,0,50,100,150],cmap_use=plt.cm.seismic,level_num=50,isDgrid=False)
            plotter.plotter_sm(data_path, 'cld', 26, (i/4,(i+1)/4), vmin=-16, vmax=-2, ticks=np.logspace(-16,-2,15,base=10),cmap_use=plt.cm.gist_ncar,level_num=50,log_cb=True)
            plotter.plotter_sm(data_path, 'vap', 26, (i/4,(i+1)/4),vmin=-16, vmax=-2, ticks=np.logspace(-16,-2,15,base=10),cmap_use=plt.cm.gist_ncar,level_num=50,log_cb=True)