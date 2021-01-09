# -*- coding: utf-8 -*-
# by sun 2021

import numpy as np
from scipy import interpolate
# numba not support for now, maybe check it latter

# zoanl anomalies func
def zonal_anom(x):
    nlon = x.shape[0] # mind this, check the input matrix dim
    anom = np.zeros_like(x)
    xzm = np.nanmean(x, axis=0)
    for ilon in range(nlon):
        anom[ilon,:,:,:] = x[ilon,:,:,:] - xzm
    return anom

# residual circulation calculator
# default for mars atmosphere
def cal_rc(lon, lat, pres, alt,Ls, ps, v, w, t, radius,pref=600.0): # cal rc (background and control) 
    nlon = lon.shape[0]
    nlat = lat.shape[0]
    npres = pres.shape[0]
    nLs = Ls.shape[0]
    
    phi = lat * np.pi / 180.0
    hp = np.zeros((nlon,nlat,npres,nLs))
    
    for ilon in range(nlon):
        for ilat in range(nlat):
            for ipres in range(npres):
                hp[ilon,ilat,ipres,:] = np.power(ps[ilon,ilat,:] / (pres[ipres]*100), 0.286)
   
    H = np.zeros((npres)) # define H by scale height
    #for ipres in range(npres):
    #    H[ipres] = (-1)*alt[ipres]/(np.log(pres[ipres]/600.0))
    H = (-1) * alt / np.log(pres / pref)

    #density = np.zeros((npres))
    #for ipres in range(npres):
    #    density[ipres] = np.exp(-alt[ipres]/H[ipres])
    density = np.exp(-alt / H)

    theta = np.zeros_like(v)
    theta = t * hp
    
    vzm = np.nanmean(v,axis=0)
    wzm = np.nanmean(w,axis=0)
    
    thetazm = np.nanmean(theta,axis=0)
    thetaz = np.zeros_like(thetazm)
    
    #for ipres in range(1,npres-1):
    #    thetaz[:,ipres,:] = (thetazm[:,ipres+1,:] \
    #    - thetazm[:,ipres-1,:]) / (alt[ipres+1] - alt[ipres-1])
    #thetaz[:,0,:] = (thetazm[:,1,:]-thetazm[:,0,:]) / \
    #    (alt[1]-alt[0])
    #thetaz[:,-1,:] = (thetazm[:,-1,:]-thetazm[:,-2,:]) / \
    #    (alt[-1]-alt[-2])
    thetaz = np.gradient(thetazm,alt,axis=1) # we get 2 order inner and 1 order boundarys

    #wza = zonal_anom(w)
    vza = zonal_anom(v)
    thetaza = zonal_anom(theta)
    
    vtheta = vza * thetaza
    vthetazm = np.nanmean(vtheta,axis=0)
    
    tmp = np.zeros_like(vthetazm) # density*vtheta/THETAZ
    for ipres in range(npres):
        tmp[:,ipres,:] = density[ipres] * vthetazm[:,ipres,:] / thetaz[:,ipres,:]
    
    tmpz = np.zeros_like(vthetazm) # vertical gradient of tmp, weight with density
    #for ipres in range(1,npres-1):
    #    tmpz[:,ipres,:] = (1/density[ipres]) * (tmp[:,ipres+1,:]-tmp[:,ipres-1,:]) \
    #        / (alt[ipres+1]-alt[ipres-1])
    #tmpz[:,0,:] = (1/density[0]) * (tmp[:,1,:]-tmp[:,0,:]) / (alt[1]-alt[0])
    #tmpz[:,-1,:] = (1/density[-1]) * (tmp[:,-1,:]-tmp[:,-2,:]) / (alt[-1]-alt[-2])
    tmpz = np.gradient(tmp,alt,axis=1)
    for ipres in range(npres):
        tmpz[:,ipres,:] = (1/density[ipres]) * tmpz[:,ipres,:]

    vres = vzm-tmpz # residual meridional wind
    
    tmpphi = np.zeros_like(vthetazm)
    #for ilat in range(1,nlat-1): # phi gradient of tmp, weight with latitude
    #    tmpphi[ilat,:,:] = (np.cos(phi[ilat+1]) * tmp[ilat+1,:,:]- np.cos(phi[ilat-1]) \
    #       * tmp[ilat-1,:,:]) / (phi[ilat+1]-phi[ilat-1])
    #tmpphi[0,:,:] = (np.cos(phi[1]) * tmp[1,:,:] - np.cos(phi[0])*tmp[0,:,:])\
    #    / (phi[1] - phi[0])
    #tmpphi[-1,:,:] = (np.cos(phi[-1]) * tmp[-1,:,:] - np.cos(phi[-2])*tmp[-2,:,:])\
    #    / (phi[-1] - phi[-2])
    
    tmpwc = np.zeros_like(vthetazm)
    for ilat in range(nlat):
        tmpwc[ilat,:,:] = np.cos(phi[ilat]) * tmp[ilat,:,:]
    tmpphi = np.gradient(tmpwc,phi,axis=0)

    for ilat in range(nlat):
        tmpphi[ilat,:,:] = (1/(radius*np.cos(phi[ilat]))) * tmpphi[ilat,:,:]
    
    wres = wzm + tmpphi # residual vertical wind
    
    return vres,wres

# ep flux and its divergence calculator
# also default for mars atmosphere
def cal_ep(lon,lat,pres,alt,Ls,ps,u,v,w,t,radius,omega,pref=600.0):
    nlon = lon.shape[0]
    nlat = lat.shape[0]
    npres = pres.shape[0]
    nLs = Ls.shape[0]
    
    hp = np.zeros((nlon,nlat,npres,nLs))
    
    for ilon in range(nlon):
        for ilat in range(nlat):
            for ipres in range(npres):
                hp[ilon,ilat,ipres,:] = np.power(ps[ilon,ilat,:] / (pres[ipres]*100), 0.286)
    
    # hp = np.power(pref / pres,0.286) 
    
    H = np.zeros((npres)) # define H by scale height
    H = (-1) * alt / np.log(pres / pref)
    
    density = np.exp(-alt / H)
    
    theta = np.zeros_like(v)
    theta = t * hp
    # for ipres in range(npres):
    #     theta[:,:,ipres,:] = t[:,:,ipres,:] * hp[ipres]
        
    phi = lat * np.pi / 180.0
    acphi = radius * np.cos(phi)
    #asphi = radius * np.sin(phi)
    
    f = 2 * omega * np.sin(phi)
    
    thetazm = np.nanmean(theta, axis=0)
    thetaz = np.zeros_like(thetazm)
    thetaz = np.gradient(thetazm, alt, axis=1) # 2 order inner and 1 order boundaries
    
    uzm = np.nanmean(u, axis=0)
    #vzm = np.nanmean(v, axis=0)
    #wzm = np.nanmean(w, axis=0)
    
    uzmz = np.zeros_like(uzm)
    uzmz = np.gradient(uzm, alt, axis=1)
    
    ucosphi = np.zeros_like(uzm)
    for ilat in range(nlat):
        ucosphi[ilat,:,:] = uzm[ilat,:,:] * np.cos(phi[ilat])
    
    ucosphi_d = np.zeros_like(uzm)
    ucosphi_d = np.gradient(ucosphi, phi, axis=0)
    
    uza = zonal_anom(u)
    vza = zonal_anom(v)
    thetaza = zonal_anom(theta)
    
    
    uv = uza * vza
    uvzm = np.nanmean(uv, axis=0)
    
    vtheta = vza * thetaza
    
    vthetazm = np.nanmean(vtheta, axis=0)
    #thetazzm = np.nanmean(thetaz, axis=0)
    vthetazm[0,:,:] = vthetazm[1,:,:]
    
    Fphi = np.zeros_like(uzm)
    for ilat in range(nlat):
        for ipres in range(npres):
            for ils in range(nLs):
                Fphi[ilat,ipres,ils] = acphi[ilat] * density[ipres] \
                    * (uzmz[ilat,ipres,ils] * vthetazm[ilat,ipres,ils] \
                    / thetaz[ilat,ipres,ils]-uvzm[ilat,ipres,ils])
    ## a simple version
    # for ils in range(nLs):
    #     for ilat in range(nlat):
    #         Fphi[ilat,:,ils] = -uvzm[ilat,:,ils] * acphi[ilat] * density
    
    Fpls = np.zeros_like(uzm)
    for ilat in range(nlat):
        for ipres in range(npres):
            for ils in range(nLs):
                Fpls[ilat,ipres,ils] = acphi[ilat] * density[ipres] * \
                    (vthetazm[ilat,ipres,ils] / thetaz[ilat,ipres,ils] * \
                    (f[ilat] - 1.0/(acphi[ilat]) * ucosphi_d[ilat,ipres,ils] ))
    ## a simple version
    # for ilat in range(nlat):
    #     for ipres in range(npres):
    #         for ils in range(nLs):
    #             Fpls[ilat,ipres,ils] = vthetazm[ilat,ipres,ils]*f[ilat]* \
    #                 acphi[ilat] * density[ipres] / thetaz[ilat,ipres,ils]
    
    # calculat the ep flux divergence
    Fphicp = np.zeros_like(Fphi)
    for ilat in range(nlat):
        Fphicp[ilat,:,:] = Fphi[ilat,:,:] * np.cos(phi[ilat])
    Fphicp_d = np.zeros_like(Fphi)
    Fphicp_d = np.gradient(Fphicp, phi, axis=0) 
    for ilat in range(nlat): # weight it
        Fphicp_d[ilat,:,:] = Fphicp_d[ilat,:,:] * 1.0/(radius*np.cos(phi[ilat]))
    
    div = np.zeros_like(Fphi)
    div = Fphicp_d + np.gradient(Fpls, alt, axis=1)
    
    
    # last processing for output
    F2 = np.zeros_like(Fpls)
    for ilat in range(nlat):
        F2[ilat,:,:] = Fpls[ilat,:,:] * np.cos(phi[ilat])
    
    F1 = Fphi / radius / np.pi
    F2 = F2 / (pref*100) 
    rhofac = np.sqrt(pref / pres)
    for ipres in range(npres):
        F1[:,ipres,:] = F1[:,ipres,:] * rhofac[ipres]
        F2[:,ipres,:] = F2[:,ipres,:] * rhofac[ipres]
    
    
    return F1, F2, div # ep flux phi, ep flux z, ep flux div
