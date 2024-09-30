import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import pyshtools as pysh
import oldsolver.RFmod as RF
from numpy import pi as pi
from scipy.interpolate import RegularGridInterpolator
sentinel = object()


if __name__ == "__main__":
    pass


#########################################################
# set some constants
b    = 6368000.          # mean radius of the solid Earth 
g    = 9.825652323       # mean surface gravity
G    = 6.6723e-11        # gravitational constant
rhow = 1000.0            # density of water
rhoi =  917.0            # density of ice
rhos = 2600.0            # surface density of solid Earth
CC = 8.038e37            # polar moment of inertia 
AA = 8.012e37            # equatorial moment of inertia
Om = 2.*pi/(24.*3600.)   # mean rotation rate
Me = 5.974e24            # mass of the Earth
##########################################################


##########################################################
# set some options
ep = 1.e-8             # tolerance for iterations
##########################################################



############################################################
# function to plot geographic data on GL grid.
def plot(fun,**kwargs):



    
    # deal with optional arguments
    cstring = kwargs.get('cmap',None)
    ofile = kwargs.get('ofile',None)
    contour = kwargs.get('contour',False)
    ncont = kwargs.get('ncont',6)
    label = kwargs.get('label','')
    marker = kwargs.get('marker',None)
    clim = kwargs.get('clim',None)
    clim_sym = kwargs.get('clim_sym',True)
    clim_pos = kwargs.get('clim_pos',False)
    clim_neg = kwargs.get('clim_neg',False)
    clim_scale = kwargs.get('clim_scale',None)
    xlim =  kwargs.get('xlim',None)
    ylim =  kwargs.get('ylim',None)
   
    if(cstring == None):
        if(clim_pos):
            cstring = "Blues"
        elif(clim_neg):
            cstring = "Reds_r"
        else:
            cstring = "RdBu"
    ax = plt.axes(projection=ccrs.PlateCarree())
     
    
    if(contour):
        plt.contourf(fun.lons()-180,fun.lats(),fun.data,cmap=cstring,levels = ncont)
    else:
        plt.pcolormesh(fun.lons()-180,fun.lats(),fun.data,shading = 'gouraud',cmap=cstring)

  
    
    cbar = plt.colorbar(location = "bottom",fraction=0.072, pad=0.04)
    ax.coastlines()
    cbar.set_label(label,labelpad = 10)
    if(marker != None):
        lat = marker[0]
        lon = marker[1]
        plt.plot([lon],[lat],marker='o', markersize=10, color="green")
    cm = np.nanmax(np.abs(fun.data))
    if(clim != None):
        plt.clim(clim)
    else:    
        if(clim_sym):
            clim = [-cm,cm]
        if(clim_pos):
            clim = [0,cm]
        if(clim_neg):
            clim = [-cm,0]        
        if(clim_scale != None):
            c1 = clim_scale*clim[0]
            c2 = clim_scale*clim[1]
            clim = [c1,c2]
        plt.clim(clim)

    if(xlim != None):
        plt.xlim(xlim)

    if(ylim != None):
        plt.ylim(ylim)



    
    if(ofile == None):
        plt.show()
    else:
        plt.savefig(ofile,bbox_inches='tight')
        plt.close()
    return 


######################################################################
# function to read in present sea level and ice thickness and
# interpolate onto GL-grid (from ice6g0.dat)
def get_sl_ice_data(L):

    # initialise the grids
    sl = pysh.SHGrid.from_zeros(lmax=L,grid = 'GLQ')
    ice = sl.copy()
    
    dat = np.loadtxt('data/ice6g0.dat')

    nlat = int(dat[0,0])
    nlon = int(dat[0,1])
    
    sld = np.zeros([nlat,nlon])
    lat = np.zeros([nlat])
    lon = np.zeros([nlon])
    icd = np.zeros([nlat,nlon])

    k = 0
    for ilon in range(nlon):
        for ilat in range(nlat):
            k = k + 1
            lat[ilat] =  dat[k,1]
            lon[ilon] =  dat[k,0]+180.
            icd[ilat,ilon]  =  dat[k,2]
            sld[ilat,ilon]  = -dat[k,3]
             
            
        
    fun = RegularGridInterpolator((lat, lon), sld,method = 'linear',
                                 bounds_error=False, fill_value=None)
       
    # interpolate onto sl grid
    for ilat,llat in enumerate(sl.lats()):
        for ilon,llon in enumerate(sl.lons()):
            sl.data[ilat,ilon] = fun((llat,llon))


    fun = RegularGridInterpolator((lat, lon), icd,method = 'linear',
                                 bounds_error=False, fill_value=None)
       
    # interpolate onto sl grid
    for ilat,llat in enumerate(ice.lats()):
        for ilon,llon in enumerate(ice.lons()):
            ice.data[ilat,ilon] = fun((llat,llon))



            
    # correct sl where there is land-based ice
    for ilat,llat in enumerate(sl.lats()):
        for ilon,llon in enumerate(sl.lons()):
            if(ice.data[ilat,ilon] > 0 and sl.data[ilat,ilon] < 0):
                sl.data[ilat,ilon] = sl.data[ilat,ilon] + ice.data[ilat,ilon]


    # remove Alaskan glaciers
    lat_G = 58.0
    lon_G = -136.
    th_G = (90-lat_G)*pi/180
    ph_G= (180+lon_G)*pi/180
    delta_G = 15*pi/180
    for ilat,lat in enumerate(sl.lats()):
        for ilon,lon in enumerate(sl.lons()):
            th = (90-lat)*pi/180
            ph = lon*pi/180            
            delta = np.cos(th)*np.cos(th_G) + np.sin(th)*np.sin(th_G)*np.cos(ph-ph_G)
            delta = np.arccos(delta)
            if(delta < delta_G):
                ice.data[ilat,ilon] = 0.0
                
    return sl,ice               


######################################################################
# function to format lat and lon for xarray data
def format_latlon(da):
    ## Function that flips lats and shifts lons to -180 to 180 d

    ## First rename latitude and longitude to lat and lon, if they are not already
    if 'latitude' in da.coords:
        da = da.rename({'latitude':'lat'})
    if 'longitude' in da.coords:
        da = da.rename({'longitude':'lon'})

    ## Now flip the lats
    da = da.reindex(lat=list(reversed(da.lat)))

    ## Now shift the lons
    da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby('lon')

    return da


######################################################################
# function to read in xarray data and interpolate onto an l=L grid
def interpolate_xarray(L, da, shift_lons=True):

    output = pysh.SHGrid.from_zeros(lmax=L,grid = 'GLQ')    

    ## First rename latitude and longitude to lat and lon, if they are not already
    if 'latitude' in da.coords:
        da = da.rename({'latitude':'lat'})
    if 'longitude' in da.coords:
        da = da.rename({'longitude':'lon'})

    if shift_lons == True:
        da = da.assign_coords(lon=(((da.lon + 180) % 360))).sortby('lon')

    lats = da.lat.values
    lons = da.lon.values  
    dat = da.values 

    fun = RegularGridInterpolator((lats, lons), dat, method = 'linear',
                                 bounds_error=False, fill_value=None)
    
    # interpolate onto sl grid
    for ilat,llat in enumerate(output.lats()):
        for ilon,llon in enumerate(output.lons()):
            output.data[ilat,ilon] = fun((llat,llon))   

    return output

########################################################
# returns thw value of a field at a given point via
# spherical harmonic expansion
def point_evaluation(fun,lat,lon):
    fun_lm = fun.expand()
    return pysh.expand.MakeGridPoint(fun_lm.coeffs,lat,lon+180)


###################################################################
# integrates a function over the surface
def surface_integral(fun):
    fun_lm = fun.expand(lmax_calc = 0,normalization = 'ortho')
    int = np.sqrt(4*pi)*b*b*fun_lm.coeffs[0,0,0]
    return int


###################################################################
# integrates a function over the oceans
def ocean_integral(C,fun):
    tmp    = C*fun
    tmp_lm = tmp.expand(lmax_calc = 0,normalization = 'ortho')
    int = np.sqrt(4*pi)*b*b*tmp_lm.coeffs[0,0,0]
    return int


###################################################################
# function that returns the ocean function and ocean area
def ocean_function(sl0,ice0):

    # data for Caspian sea
    lat_C = 42.2
    lon_C = 50.7
    th_C = (90-lat_C)*pi/180
    ph_C= (180+lon_C)*pi/180
    delta_C = 7.5*pi/180

    # data for Lake Eyre
    lat_E = -29.2
    lon_E = 137.3
    th_E = (90-lat_E)*pi/180
    ph_E= (180+lon_E)*pi/180
    delta_E = 2.0*pi/180
        
    C = sl0.copy()        
    for ilat,lat in enumerate(C.lats()):
        for ilon,lon in enumerate(C.lons()):            
            sll  = sl0.data[ilat,ilon] 
            icel = ice0.data[ilat,ilon]            
            if(rhow*sll - rhoi*icel >= 0.): 
                C.data[ilat,ilon] = 1.
            else:
                C.data[ilat,ilon] = 0.

            # remove the Caspian sea
            th = (90-lat)*pi/180
            ph = lon*pi/180
            delta = np.cos(th)*np.cos(th_C) + np.sin(th)*np.sin(th_C)*np.cos(ph-ph_C)
            delta = np.arccos(delta)
            if(delta < delta_C):
                C.data[ilat,ilon] = 0.0
                
            # remove the Lake Eyre
            delta = np.cos(th)*np.cos(th_E) + np.sin(th)*np.sin(th_E)*np.cos(ph-ph_E)
            delta = np.arccos(delta)
            if(delta < delta_E):
                C.data[ilat,ilon] = 0.0

                
    return C


#############################################################
# returns function equal to 1 in oceans and val on land
def ocean_mask(sl0,ice0,val = np.nan):
    mask = ocean_function(sl0,ice0)
    for ilat in range(mask.nlat):
        for ilon in range(mask.nlon):            
            if(mask.data[ilat,ilon] == 0.0): 
                mask.data[ilat,ilon] = val
    return mask

#############################################################
# returns function equal to 1 on land and val in oceans
def land_mask(sl0,ice0,val = np.nan):
    mask = ocean_function(sl0,ice0)
    for ilat in range(mask.nlat):
        for ilon in range(mask.nlon):            
            if(mask.data[ilat,ilon] == 1.0): 
                mask.data[ilat,ilon] = val
            else:
                mask.data[ilat,ilon] = 1.0
            
    return mask


#############################################################
# returns function equal to 1 where there is ice and val
# elsewhere
def ice_mask(sl0,ice0,val = np.nan):
    mask = ocean_function(sl0,ice0)
    for ilat in range(mask.nlat):
        for ilon in range(mask.nlon):            
            if(mask.data[ilat,ilon] == 1.0 or ice0.data[ilat,ilon] == 0): 
                mask.data[ilat,ilon] = val
            else:
                mask.data[ilat,ilon] = 1.0
    return mask


#############################################################
# returns function equal to 1 where over antarctica and val
# elsewhere
def antarctica_mask(sl0,ice0,val = np.nan):
    mask = ice_mask(sl0,ice0,val)
    for ilat,lat in enumerate(mask.lats()):
        if(lat > 0):
            mask.data[ilat,:] = val
    return mask


#############################################################
# returns function equal to 1 where over west antarctica and val
# elsewhere
def west_antarctica_mask(sl0,ice0,val = np.nan):
    mask = ice_mask(sl0,ice0,val)
    for ilat,lat in enumerate(mask.lats()):
        if(lat > 0):
            mask.data[ilat,:] = val
        else:
            for ilon,lon in enumerate(mask.lons()):
                if(lon > 125):
                    mask.data[ilat,ilon] = val
    return mask



#############################################################
# returns function equal to 1 where over Greenland and val
# elsewhere
def greenland_mask(sl0,ice0,val = np.nan):
    mask = ice_mask(sl0,ice0,val)
    for ilat,lat in enumerate(mask.lats()):
        if(lat < 0):
            mask.data[ilat,:] = val
    return mask


# returns function equal to 1 in oceans for lat in [lat1,lat2]
# and equal to val elsewhere
def altimetery_mask(sl0,ice0,lat1 = -66., lat2 = 66.,val = np.nan):
    mask = ocean_function(sl0,ice0)
    for ilat,lat in enumerate(mask.lats()):
        for ilon,lon in enumerate(mask.lons()):            
            if(lat <= lat1 or lat >= lat2):
                mask.data[ilat,ilon] = val
    return mask


#############################################################
# sets a spatial grid's values to zero in the southern
# hemisphere
def zero_southern_hemisphere(fin):
    fout = fin.copy()
    for ilat,lat in enumerate(fout.lats()):
        if(lat < 0.):
            fout.data[ilat,:] = 0.
    return fout


#############################################################
# sets a spatial grid's values to zero in the northern
# hemisphere
def zero_northern_hemisphere(fin):
    fout = fin.copy()
    for ilat,lat in enumerate(fout.lats()):
        if(lat > 0.):
            fout.data[ilat,:] = 0.
    return fout


#####################################################################
# returns the jx = Om*i_{zx} and jy = Om*i_{zy} components of the
# torque perturbation associated with the gravitational potential
def inertia_tensor_perturbation(phi_lm):
    j = Om*np.sqrt(5./(12*pi))*(b**3/G)*phi_lm.coeffs[:,2,1] 
    return j
 

#####################################################################
# returns the rotation vector perturbations given those for the
# torque vector, j.
def rotation_vector_perturbation(j):
    om = j/(CC-AA)
    return om


######################################################################
# returns the centrifugal potential perturbation in spherical harmonic
# domain given the rotation vector perturbation
def centrifugal_perturbation_coefficients(om):
    psi_2m = np.zeros([2,3])
    psi_2m[:,1] = b**2*Om*np.sqrt((4*pi)/15.)*om
    return psi_2m


########################################################################
# returns the point value of the centrifugal potential perturbation at
# a given location
def centrifugal_perturbation_value(om,lat,lon):
    th = (90-lat)*pi/180
    ph = lon*pi/180
    psi = Om*b*b*np.cos(th)*np.sin(th)*(om[0]*np.cos(ph) + om[1]*np.sin(ph))
    return psi

########################################################################
# returns a vector kk such that (kk,om) is equal to the
# centrifugal potential perturbation at the given location
def centrifugal_perturbation_vector(om,lat,lon):
    kk = np.zeros(2)
    th = (90-lat)*pi/180
    ph = lon*pi/180
    kk[0] = Om*b*b*np.cos(th)*np.sin(th)*np.cos(ph)
    kk[0] = Om*b*b*np.cos(th)*np.sin(th)*np.sin(ph)
    return kk

################################################
# reads in and returns love numbers from a file
def love_numbers(L):
    data = np.loadtxt('data/love.dat')
    h = data[:L+1,1] + data[:L+1,3]
    k = data[:L+1,2] + data[:L+1,4]
    ht = data[2,5]
    kt = data[2,6]
    return h,k,ht,kt


################################################
# reads in and returns love numbers from a file
def generalised_love_numbers(L):
    data = np.loadtxt('data/love.dat')
    h_u   = data[:L+1,1]
    k_u   = data[:L+1,2]
    h_phi = data[:L+1,3]
    k_phi = data[:L+1,4]
    h     = h_u + h_phi
    k     = k_u + k_phi
    ht = data[2,5]
    kt = data[2,6]
    return h_u,k_u,h_phi,k_phi,h,k,ht,kt

    


#####################################################################
# function to calculate uniform sea level rise from driect load
def bathtub(C,zeta):
    L = C.lmax
    A = surface_integral(C)
    return -surface_integral(zeta)/(rhow*A)


#####################################################################
# returns the solid earth deformation associated with a given load
def loading_response(sigma):
    L = sigma.lmax
    h,k,ht,kt = love_numbers(L)
    u_lm   = sigma.expand(normalization = 'ortho')
    phi_lm = u_lm.copy()
    for l in range(L+1):
        u_lm.coeffs[:,l,:]   *= h[l]
        phi_lm.coeffs[:,l,:] *= k[l]
    u   = u_lm.expand(grid = 'GLQ')
    phi = phi_lm.expand(grid = 'GLQ')    
    return u,phi

#####################################################################
# function to solve the fingerprint problem for a given direct load
def fingerprint(C,zeta,rotation=True,verbose=True,ep=ep):

    # get the maximum degree
    L = C.lmax

    # get the ocean area
    A = surface_integral(C)
    
    # get the love numbers
    h,k,ht,kt = love_numbers(L)
    
    # calculate the average change in sea level
    slu = -surface_integral(zeta)/(rhow*A)
    onegrid = pysh.SHGrid.from_zeros(lmax=L,grid = 'GLQ')
    onegrid.data[:,:] = 1.
    sl = slu*onegrid
    
    # initialise displacement and potential perturbations
    u_lm   = pysh.SHCoeffs.from_zeros(lmax=sl.lmax,normalization = 'ortho')
    phi_lm = u_lm.copy()
    psi_lm = u_lm.copy()
        
    # store the initial guess
    sl0 = sl.copy()
    err = 1.

    # start the iterations
    it = -1
    while(err > ep):
        
        # compute the current loads
        sigma = rhow*C*sl + zeta
        sigma_lm  = sigma.expand(normalization = 'ortho')
        
        # determine the response to the loading
        for l in range(L+1):
            u_lm.coeffs[:,l,:]   =  h[l]*sigma_lm.coeffs[:,l,:]    
            phi_lm.coeffs[:,l,:] =  k[l]*sigma_lm.coeffs[:,l,:]    
        

        # add in the centrifugal contribution
        u_lm.coeffs[:,2,:]   +=  ht*psi_lm.coeffs[:,2,:]
        phi_lm.coeffs[:,2,:] +=  kt*psi_lm.coeffs[:,2,:]

        # get the centrifugal potential perturbation
        if(rotation):
            j  = inertia_tensor_perturbation(phi_lm)
            om = rotation_vector_perturbation(j)
            psi_2m = centrifugal_perturbation_coefficients(om)
            psi_lm.coeffs[:,2,:3] = psi_2m

        # get the spatial fields
        u   =   u_lm.expand(grid = 'GLQ')
        phi = phi_lm.expand(grid = 'GLQ')
        psi = psi_lm.expand(grid = 'GLQ')


        # update the sea level
        fac = ocean_integral(C,g*u + phi + psi)/(g*A) + slu
        sl    = -1.*(g*u + phi + psi)/g + fac*onegrid

        it = it+1
        if(it == 0):
            slnorm = np.max(np.abs(sl.data))
        else:
            err = np.max(np.abs(sl.data - sl0.data))/np.abs(slnorm)
            if verbose == True:
                print('iteration = ',it,'relative change = ',err)

        
        # store the most recent solution
        sl0 = sl.copy()
        
    if(not rotation):
        om = np.zeros(2)
        
    return sl,u,phi,om,psi




#####################################################################
# function to solve the fingerprint problem for a given direct load
def generalised_fingerprint(C,zeta,zeta_u,zeta_phi,kk,rotation=True,verbose=True,ep=ep):

    # get the maximum degree
    L = C.lmax

    # get the ocean area
    A = surface_integral(C)
    
    # get the love numbers
    h_u,k_u,h_phi,k_phi,h,k,ht,kt = generalised_love_numbers(L)
        
    # calculate the average change in sea level
    slu = -surface_integral(zeta)/(rhow*A)
    onegrid = pysh.SHGrid.from_zeros(lmax=L,grid='GLQ')
    onegrid.data[:,:] = 1.
    sl = slu*onegrid
    
    # initialise displacement and potential perturbations
    u_lm   = pysh.SHCoeffs.from_zeros(lmax=sl.lmax,normalization = 'ortho')
    phi_lm = u_lm.copy()
    psi_lm = u_lm.copy()


    # expand the generalised loads
    zeta_u_lm   = zeta_u.expand(normalization = 'ortho')
    zeta_phi_lm = zeta_phi.expand(normalization = 'ortho')

    # start the iterations
    err = 1.
    it = -1
    while(err > ep):
        
        # compute the current loads
        sigma       = rhow*C*sl + zeta
        sigma_lm    = sigma.expand(normalization = 'ortho')
        
        # determine the response to the loading
        for l in range(L+1):
            u_lm.coeffs[:,l,:]   =    h[l]*sigma_lm.coeffs[:,l,:]        \
                                    + h_u[l]*zeta_u_lm.coeffs[:,l,:]     \
                                    + h_phi[l]*zeta_phi_lm.coeffs[:,l,:]
            phi_lm.coeffs[:,l,:] =    k[l]*sigma_lm.coeffs[:,l,:]        \
                                    + k_u[l]*zeta_u_lm.coeffs[:,l,:]     \
                                    + k_phi[l]*zeta_phi_lm.coeffs[:,l,:]


        # add in the centrifugal contribution
        u_lm.coeffs[:,2,:]   += ht*psi_lm.coeffs[:,2,:]
        phi_lm.coeffs[:,2,:] +=  kt*psi_lm.coeffs[:,2,:]

        # get the centrifugal potential perturbation
        if(rotation):
            j  = inertia_tensor_perturbation(phi_lm) - kk
            om = rotation_vector_perturbation(j)
            psi_2m = centrifugal_perturbation_coefficients(om)
            psi_lm.coeffs[:,2,:3] = psi_2m
            
        # get the spatial fields
        u   =   u_lm.expand(grid = 'GLQ')
        phi = phi_lm.expand(grid = 'GLQ')
        psi = psi_lm.expand(grid = 'GLQ')

        # update the sea level
        fac = ocean_integral(C,g*u + phi + psi)/(g*A) + slu
        sl    = -1.*(g*u + phi + psi)/g + fac*onegrid

        it = it+1
        if(it == 0):
            slnorm = np.max(np.abs(sl.data))
        else:
            err = np.max(np.abs(sl.data - sl0.data))/np.abs(slnorm)
            if verbose == True:
                print('iteration = ',it,'relative change = ',err)

        
        # store the most recent solution
        sl0 = sl.copy()
        
    if(not rotation):
        om = np.zeros(2)
        
    return sl,u,phi,om,psi


#####################################################################
# returns a point load at a given geographic location with optional
# inverse Laplacian smoothing
def point_load(L,lats,lons,grid = 'GLQ',angle = 0.,w=[sentinel]):


    if(len(lats)!=len(lons)):
        raise SystemExit('lats and lons are different sizes!')
        
    if(w[0]==sentinel):
        w = np.ones(len(lats))

    th = 0.4*angle*pi/180
    t  = th*th
        
    ths = 90.- lats
    phs = lons + 180.
               
    pl_lm = pysh.SHCoeffs.from_zeros(lmax=L,normalization = 'ortho')
               
    for isource in range(len(lats)):

        ylm = pysh.expand.spharm(pl_lm.lmax,ths[isource],phs[isource],normalization = 'ortho')
        
        for l in range(0,pl_lm.lmax+1):
            fac = np.exp(-l*(l+1)*t)
            pl_lm.coeffs[0,l,0] +=  w[isource]*ylm[0,l,0]*fac
            for m in range(1,l+1):
                pl_lm.coeffs[0,l,m] += w[isource]*ylm[0,l,m]*fac
                pl_lm.coeffs[1,l,m] += w[isource]*ylm[1,l,m]*fac


    pl_lm = (1/b**2)*pl_lm
    pl = pl_lm.expand(grid = 'GLQ')
               
    return pl



#########################################################################
# returns the vector -\int_{\partial M} [\mathbf{x} \times (\Bom \times
# \mathbf{x})] \zeta_{\phi} \dd S needed to deal with phi measurements
def rotation_vector_from_zeta_phi(zeta_phi):
    zeta_phi_lm = zeta_phi.expand(normalization='ortho')
    kk = np.zeros(2)
    for i in range(2):
        om = np.zeros(2)
        om[i] = 1.
        phi_2m = centrifugal_perturbation_coefficients(om)
        kk[i] = np.sum(phi_2m[:,:3]*zeta_phi_lm.coeffs[:,2,:3])*b*b
    return kk


        
##########################################################################
# returns the adjoint loads for a sea level measurement at 
def sea_level_load(L,lat,lon,grid = 'GLQ',angle = 1.):
    lats = np.full((1),lat)
    lons = np.full((1),lon)
    zeta     = point_load(L,lats,lons,angle = angle,grid = grid)
    zeta_u   = pysh.SHGrid.from_zeros(lmax=L,grid = grid)
    zeta_phi = pysh.SHGrid.from_zeros(lmax=L,grid = grid)
    kk       = np.zeros(2)
    return zeta,zeta_u,zeta_phi,kk


##########################################################################
# returns the adjoint loads for a sea level measurement at 
def displacement_load(L,lat,lon,grid = 'GLQ',angle = 1.):
    lats = np.full((1),lat)
    lons = np.full((1),lon)
    zeta     =  pysh.SHGrid.from_zeros(lmax=L,grid = grid)
    zeta_u   = -1*point_load(L,lats,lons,angle = angle,grid = grid)
    zeta_phi =  pysh.SHGrid.from_zeros(lmax=L,grid = grid)
    kk       = np.zeros(2)
    return zeta,zeta_u,zeta_phi,kk




##########################################################################
# returns the adjoint loads for a gravitational potential measurement
def absolute_gravity_load(L,lat,lon,grid = 'GLQ',angle = 1.,remove_psi = True):
    lats = np.full((1),lat)
    lons = np.full((1),lon)
    zeta     =  pysh.SHGrid.from_zeros(lmax=L,grid = grid)
    zeta_u   =  (g/b-4*pi*G*rhos)*point_load(L,lats,lons,angle = angle,grid = grid)
    zeta_phi = -g*point_load(L,lats,lons,angle = angle,grid = grid)
    zeta_phi_lm = zeta_phi.expand
    for l in range(L+1):
        zeta_phi_lm.coeffs[:,l,:] *= (l+1)/b;
    zeta_phi = zeta_phi_lm.expand(grid = grid)
    if(remove_psi):
        kk = -rotation_vector_from_zeta_phi(zeta_phi)
    else:
        kk = np.zeros(2)
    return zeta,zeta_u,zeta_phi,kk


    
##########################################################################
# returns the adjoint loads for a measurement of the (l,m)th real spherical
# harmonic coefficient of the gravitational potential perturbation
def potential_coefficient_load(L,l,m,grid = 'GLQ',remove_psi = True):
    zeta   = pysh.SHGrid.from_zeros(lmax=L,grid = grid)
    zeta_u = pysh.SHGrid.from_zeros(lmax=L,grid = grid)
    zeta_phi_lm =  pysh.SHCoeffs.from_zeros(lmax=L,normalization = 'ortho')
    if(m >= 0):
        zeta_phi_lm.coeffs[0,l,m]  = -g/b**2
    else:
        zeta_phi_lm.coeffs[1,l,-m] = -g/b**2
    zeta_phi = zeta_phi_lm.expand(grid = grid)
    if(remove_psi):
        kk = -rotation_vector_from_zeta_phi(zeta_phi)
    else:
        kk = np.zeros(2)
    return zeta,zeta_u,zeta_phi,kk




############################################################################
# returns the adjoint loads corresponding an altimetery estimate
def sea_altimetery_load(sl0,ice0,lat1 = -66,lat2 = 66,grid = 'GLQ',remove_psi = True):
    L = sl0.lmax
    zeta = altimetery_mask(sl0,ice0,lat1,lat2,val = 0.0)
    A = surface_integral(zeta)
    zeta = zeta/A
    zeta_u   = -1*zeta.copy()
    zeta_phi = pysh.SHGrid.from_zeros(lmax = L,grid=grid)
    if(remove_psi):
        kk = -rotation_vector_from_zeta_phi(zeta)
    else:
        kk = np.zeros(2)
    return zeta,zeta_u,zeta_phi,kk


############################################################################
# returns the load average corresponding to the averaging function, w,
# following the method of Wahr et al. (1998)
def GRACE_average_measurement(phi,w,LT = 0):
    L = phi.lmax
    if(LT == 0):
        LT = L
    _,k,_,_ = love_numbers(L)
    sigma_lm = phi.expand(normalization = 'ortho')
    for l in range(L+1):
        if(l >= 2 and l <= LT):
            sigma_lm.coeffs[:,l,:] *= 1./k[l]
        else:
            sigma_lm.coeffs[:,l,:] = 0.            
    sigma = sigma_lm.expand(grid='GLQ')
    return surface_integral(w*sigma)


############################################################################
# returns the adjoint loads corresponding to a GRACE load average defined
# with respec to the input function w. The truncation degree for the
# average can be input as an optional parameter
def GRACE_average_load(w,LT = 0):
    L = w.lmax
    if(LT  == 0):
        LT = L
    zeta   = pysh.SHGrid.from_zeros(lmax = L,grid='GLQ')
    zeta_u = pysh.SHGrid.from_zeros(lmax = L,grid='GLQ')
    _,k,_,_ = love_numbers(L)
    w_lm = w.expand(normalization = 'ortho')    
    for l in range(L+1):
        if(l >= 2 and l <= LT):
            w_lm.coeffs[:,l,:] *= -g/k[l]
        else:
            w_lm.coeffs[:,l,:] = 0.            
    zeta_phi = w_lm.expand(grid='GLQ')
    kk = -rotation_vector_from_zeta_phi(zeta_phi)
    return zeta,zeta_u,zeta_phi,kk


####################################################
# returns the averaging function of Jekeli (1981)
# as discussed in Wahr et al (1998). Note that
# the averaging length, r, is input in km
def gaussian_averaging_function(L,r,lat0,lon0,cut = False):
    th0 = (90-lat0)*pi/180
    ph0 = (lon0-180)*pi/180
    c = np.log(2)/(1-np.cos(1000*r/b))
    fac = 2*pi*(1-np.exp(-2*c))
    fac = c/(b*b*fac)
    w = pysh.SHGrid.from_zeros(lmax=L,grid = 'GLQ')
    for ilat,lat in enumerate(w.lats()):
        th = (90-lat)*pi/180
        fac1 = np.cos(th)*np.cos(th0)
        fac2 = np.sin(th)*np.sin(th0)
        for ilon,lon in enumerate(w.lons()):
            ph = lon*pi/180
            calpha = fac1 + fac2*np.cos(ph-ph0)
            w.data[ilat,ilon] = fac*np.exp(-c*(1-calpha))
    if(cut):
        w_lm = w.expand()
        w_lm.coeffs[:,:2,:]  =0.
        w = w_lm.expand(grid='GLQ')
    return w



###################################################################
# returns a random ice model derived from a zero-mean rotationally
# invariant Gassian random field with the prescribed covariance

def random_ice_model(sl0,ice0,Q,b = 1.):
    return RF.random_field(Q,b)*ice_mask(sl0,ice0,val = 0.)

###################################################################
# returns a random ocean surface mass from a zero-mean rotationally
# invariant Gassian random field with the prescribed covariance
# note that the integral of the field over the oceans is set equal to
# zero in accordance with conservation of mass
def random_ocean_load(C,Q,b = 1.):
    zeta = rhow*RF.random_field(Q,b=b)*C
    zeta -= C*surface_integral(zeta)/surface_integral(C)
    return zeta
    
###################################################################
# returns the relative amplitude of a function at a given degree
def amplitude_at_degree(fun,l):
    L = fun.lmax
    if(l > L):
        return 0.
    fun_lm = fun.expand(normalization = "ortho")
    p  = 0.
    for ll in range(L+1):
        for m in range(l+1):
            p += fun_lm.coeffs[0,l,m]**2 + fun_lm.coeffs[1,l,m]**2
    pl = 0.
    for m in range(ll+1):
        pl +=fun_lm.coeffs[0,ll,m]**2 + fun_lm.coeffs[1,ll,m]**2
    if(p == 0.):
        return 0
    return np.sqrt(pl/p)




