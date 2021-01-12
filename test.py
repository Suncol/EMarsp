# -*- coding: utf-8 -*-
# by sun 2021

import matplotlib.pyplot as plt
import EMars_plotter.plotter as plotter
import EMars_plotter.dataio as dataio
from joblib import Parallel, delayed

if __name__ == "__main__":
    
    data_path = 'D:\\Program\\Mars\\Mars_data\\EMARS'
    plot_path = 'D:\\Program\\Mars\\Mars_data\EMARS\\anal_mean\\T'
    file_list = dataio.get_file_list(data_path,'.nc')
    
    # using joblib for parallel plot 
    #Parallel(n_jobs=5,verbose=100)(delayed(plotter.mean_plotter_rc)(filepath) for filepath in file_list)
    
    # unit test, maybe you can also use it 
    #for filepath in file_list: # as i know, server take so long time to start the process in the joblib
        ## try plot residual circulation
        #plotter.mean_plotter_rc(filepath)
        
        ## try plot ep flux
        #plotter.mean_plotter_ep(filepath)
    
        # # try plot temperature 
        # plotter.mean_plotter_sm(filepath, 't', vmin=0, vmax=300, cmap_use=plt.cm.YlOrRd, level_num=301)
        
        # # try plot zonal wind
        # plotter.mean_plotter_sm(filepath, 'u', vmin=-150, vmax=150, cmap_use=plt.cm.seismic,level_num=301)
        
        # # try plot water ice
        # plotter.mean_plotter_sm(filepath, 'cld', vmin=-11, vmax=-2, cmap_use=plt.cm.cool,level_num=10,log_cb=True)
        
        # # try plot water vapor
        # plotter.mean_plotter_sm(filepath, 'vap', vmin=-11, vmax=-2, cmap_use=plt.cm.cool,level_num=10,log_cb=True)
        
        # # try plot dust o1
        # plotter.mean_plotter_dust(filepath, 'dust',vmin=-23, vmax=-4, cmap_use=plt.cm.YlOrBr,level_num=20,log_cb=True)

        ## some 2d variables
        
        # try plot water vapor column 
        #plotter.mean_plotter_sm2d(filepath, 'wcol', vmin=-8, vmax=-3, cmap_use=plt.cm.cool, level_num=6, log_cb=True)

        # try plot total visible opacity from aerosols
        #plotter.mean_plotter_sm2d(filepath,'vod', vmin=-3, vmax=2, cmap_use=plt.cm.cool, level_num=6, log_cb=True)
        
    # try to make movie in certain plot path
    plotter.movie_maker(plot_path,fps=5)
