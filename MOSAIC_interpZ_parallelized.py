#!/usr/bin/env python
# coding: utf-8

# In[1]:


### THIS SCRIPT USES THE wrfout* FILES FROM A VORTEX-FOLLOWING GRID
### AND INTERPOLATES THEM ONTO A STATIC DOMAIN THAT ENCOMPASSES THE 
### DESIRED LOCATION OF A HIGH-RESOLUTION SMALLER LES DOMAIN

import numpy as np
import scipy
import xarray as xr
import sys
from calendar import monthrange
import os
import matplotlib.pyplot as plt
from scipy import stats,signal
from copy import deepcopy
import shutil
import netCDF4
import stat
try:
    from mpi4py import MPI
except:
    print("Error loading mpi4py")



# In[2]:


#%% Get MPI stuff
try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except:
    rank = 0
    size = 1


# In[3]:


#%% File locations
# Location of files with static mesh
dir_LES = "/projects/storm/mgomez/realSims/hurrDelta/LES/"
# Location of precursor mesoscale simulation
dir_wrfout_mvng = dir_LES + "wrfoutFiles_prec/" 
# Location of new interpolated boundary conditions
dir_wrfout_save = dir_LES + "wrfoutFiles_save/"


# Choose domain
domain_static = 'd01'
domain_mvng = 'd02'
# File name root
fName = 'wrfout'

# Base file with static mesh to interpolate the moving domain data
fName_wrfout_static = "wrfout_staticMesh_dx1500m"

# Start time of static simulation
desired_time = np.datetime64('2020-10-09T07:30:00') 
timeAfter_desiredTime = np.timedelta64(90,'m')


# In[4]:


#%% Load static mesh

ds = xr.open_dataset(dir_LES+fName_wrfout_static,decode_times=False)
# List of variables that need to be re-written
list_vars = list(ds.variables.keys())
## Extract domain dimensions
# Mass points
XLAT_static = np.array(ds['XLAT'])[0,:,:]
XLONG_static = np.array(ds['XLONG'])[0,:,:]
# u-points
XLAT_U_static = np.array(ds['XLAT_U'])[0,:,:]
XLONG_U_static = np.array(ds['XLONG_U'])[0,:,:]
# v-points
XLAT_V_static = np.array(ds['XLAT_V'])[0,:,:]
XLONG_V_static = np.array(ds['XLONG_V'])[0,:,:]
# Height arrays
PH_static = np.array(ds['PH'])[0,:,:,:]
PHB_static = np.array(ds['PHB'])[0,:,:,:]
hgt_stag_static = (PH_static+PHB_static)/9.81
hgt_unstag_static = 0.5*(hgt_stag_static[1:,:,:] + hgt_stag_static[0:-1,:,:])
ds.close()

print(list_vars)


# In[5]:


#%% Location of interest for vertical interpolation

# Lat,Lon of location of interest
lon_desired,lat_desired = np.mean(XLONG_static),np.mean(XLAT_static) # Choose central location of LES domain

# Find i,j indices of desired location
i_desired_static,j_desired_static = 0,0
for i in np.arange(5):
    i_desired_static = np.nanargmin(np.abs(XLONG_static[j_desired_static,:] - lon_desired))
    j_desired_static = np.nanargmin(np.abs(XLAT_static[:,i_desired_static] - lat_desired))
print([XLONG_static[j_desired_static,i_desired_static],lon_desired])
print([XLAT_static[j_desired_static,i_desired_static],lat_desired])

print("")

print("i_desired_static = %i" % i_desired_static)
print("j_desired_static = %i" % j_desired_static)


# In[6]:


#%% Function to calculate distance between lat,lon coordinates
def dist_lat_lon(lat1,lon1,lat2,lon2):
    lat1 = lat1*np.pi/180
    lon1 = lon1*np.pi/180
    lat2 = lat2*np.pi/180
    lon2 = lon2*np.pi/180
    
    return np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))*6371*1000 # [m]

# In[7]:


#%% Define interpolation function

def func_interp_xy(var_mvng,lat_u_mvng,lon_u_mvng,lat_v_mvng,lon_v_mvng,lat_m_mvng,lon_m_mvng,
               var_static,lat_u_static,lon_u_static,lat_v_static,lon_v_static,lat_m_static,lon_m_static,
               stag_X,stag_Y,
               i_desired_mvng,j_desired_mvng):
    ## Use correct grid for current variable
    if stag_Y:
        use_lat_mvng = lat_v_mvng[:,i_desired_mvng]
        use_lon_mvng = lon_v_mvng[j_desired_mvng,:]
        use_lat_static = lat_v_static
        use_lon_static = lon_v_static
    elif stag_X:
        use_lat_mvng = lat_u_mvng[:,i_desired_mvng]
        use_lon_mvng = lon_u_mvng[j_desired_mvng,:]
        use_lat_static = lat_u_static
        use_lon_static = lon_u_static
    else:
        use_lat_mvng = lat_m_mvng[:,i_desired_mvng]
        use_lon_mvng = lon_m_mvng[j_desired_mvng,:]
        use_lat_static = lat_m_static
        use_lon_static = lon_m_static

    ## Function that interpolates
    f = scipy.interpolate.RectBivariateSpline(use_lat_mvng,use_lon_mvng,var_mvng)
    # Interpolate to static grid
    for iy in np.arange(np.shape(use_lon_static)[0]):
        for ix in np.arange(np.shape(use_lon_static)[1]):
            x = use_lon_static[iy,ix]
            y = use_lat_static[iy,ix]
            ww = f(y,x)
            var_static[iy,ix] = ww[0][0]
    return var_static
    


# In[8]:


#%% Variables that should not have negative values

vas_no_neg = ['QRAIN','QCLOUD','QVAPOR']


# In[9]:


#%% Variables that don't need to be overwritten

vars_not_overwrtie = ['LON','LAT','LAKEMASK','LANDMASK','XLAND','HGT','COSALPHA','SINALPHA','MAPFAC','VEGFRA','LU_INDEX',
                     'RDY','RDX','DZS','C4F','C3F','C4H','C3H','C2F','C1F','C2H','C1H','MF_VX_INV','EL_PBL','VAR','COSZEN',
                     'ISLTYP','IVGTYP','UDROFF','SFROFF','XICEM','SEAICE','CF3','CF2','CF1','RESM','CFN1','CFN','DN','DNW',
                     'RDN','RDNW','FNP','FNM','VAR_SSO','ZNU','M_PBLH','P_TOP','ZETATOP','ZS','LAI','SHDMAX', 'SHDMIN',
                     'SHDMAX', 'SHDMIN', 'SNOALB', 'TSLB', 'SMOIS', 'SH2O', 'GRDFLX', 'ACGRDFLX','SNOW', 'SNOWH','CANWAT',
                     'NUPDRAFT','SNOWC','SR','SMCREL','EDMF_A','EDMF_W', 'EDMF_W', 'EDMF_THL', 'EDMF_QT', 'EDMF_ENT',
                     'EDMF_QC', 'MAX_MSTFX', 'MAX_MSTFY', 'PERT_T', 'X_INFLOW', 'Y_INFLOW', 'CUTIN', 'CUTOUT', 'TURB_TYPE', 
                     'HS_WV', 'DIR_WV', 'TPEAK_WV', 'WLEN_WV']

orig_vars = deepcopy(list_vars)
for vari in orig_vars:
    if vari in vars_not_overwrtie:
        list_vars.remove(vari)
        print(f"remove {vari}")


# In[11]:


#%% Load data from wrfout* files and save the modified versions

# Find files matching description for current domain within directory
entries = os.listdir(dir_wrfout_mvng)
entries.sort()
toRead = []
for ii in np.arange(len(entries)):
    # Check if file name matches
    if domain_mvng in entries[ii]:
        if fName in entries[ii]:
            toRead.append(entries[ii])

# Split file list between ranks
files_per_rank = np.array_split(toRead, size)
my_files_toRead = files_per_rank[rank]
print(f"{len(my_files_toRead)} being processed in rank {rank}")


# Load file, modify lat,lon and save to new file
for i_f in np.arange(0,len(my_files_toRead)):
    ## Get time for current wrfout file
    ds = xr.open_dataset(dir_wrfout_mvng+my_files_toRead[i_f],decode_times=False)
    ## Extract time data
    Time = np.array(ds['Times'])
    a = str(Time)
    ttime = np.datetime64(a[3:13] + str(' ')+ a[14:-2])
    ds.close()

    ## Make sure time is within desired range
    if (ttime >= desired_time) & (ttime <= desired_time + timeAfter_desiredTime):
        print("Continue with %s" % str(ttime))
        ## Copy wrfout to new file to save
        tmp_str = str(my_files_toRead[i_f])
        i_change = tmp_str.find(domain_mvng)
        fName_static = tmp_str[0:i_change] + domain_static + tmp_str[i_change+3:]

        if not(os.path.exists(dir_wrfout_save + fName_static)):
            ## Extract domain dimensions
            ds = xr.open_dataset(dir_wrfout_mvng+str(my_files_toRead[i_f]),decode_times=False)
            # Mass points
            XLAT_mvng = np.array(ds['XLAT'])[0,:,:]
            XLONG_mvng = np.array(ds['XLONG'])[0,:,:]
            # u-points
            XLAT_U_mvng = np.array(ds['XLAT_U'])[0,:,:]
            XLONG_U_mvng = np.array(ds['XLONG_U'])[0,:,:]
            # v-points
            XLAT_V_mvng = np.array(ds['XLAT_V'])[0,:,:]
            XLONG_V_mvng = np.array(ds['XLONG_V'])[0,:,:]
            # Height arrays
            PH_mvng = np.array(ds['PH'])[0,:,:,:]
            PHB_mvng = np.array(ds['PHB'])[0,:,:,:]
            hgt_stag_mvng = (PH_mvng+PHB_mvng)/9.81
            hgt_unstag_mvng = 0.5*(hgt_stag_mvng[1:,:,:] + hgt_stag_mvng[0:-1,:,:])
            ds.close()

            # Find i,j indices of desired location for vertical interpolation
            i_desired_mvng,j_desired_mvng = 0,0
            for i in np.arange(5):
                i_desired_mvng = np.nanargmin(np.abs(XLONG_mvng[j_desired_mvng,:] - lon_desired))
                j_desired_mvng = np.nanargmin(np.abs(XLAT_mvng[:,i_desired_mvng] - lat_desired))
            print([XLONG_mvng[j_desired_mvng,i_desired_mvng],lon_desired])
            print([XLAT_mvng[j_desired_mvng,i_desired_mvng],lat_desired])
            print("i_desired_mvng = %i" % i_desired_mvng)
            print("j_desired_mvng = %i" % j_desired_mvng)
            
            # Re-write all variables of the static domain
            ds_save = xr.open_dataset(dir_LES+fName_wrfout_static,decode_times=False)
            for vari in list_vars: 
                ## Extract current variable from each data structure
                ds_mvng = xr.open_dataset(dir_wrfout_mvng+str(my_files_toRead[i_f]),decode_times=False)
                curr_var_mvng = ds_mvng[vari]
                ds_mvng.close()
                ds_sttc = xr.open_dataset(dir_LES+fName_wrfout_static,decode_times=False)
                curr_var_static = ds_sttc[vari]
                ds_sttc.close()
    
                ## Find dimensions of current variable
                dim_var = np.shape(curr_var_static)
    
                ## Interpolate field along each dimension from moving grid to static grid
                if len(dim_var)>2:
                    stag_x = False
                    stag_y = False
                    # Find correct combination of staggered/non-staggered fields for current variable
                    if ('west_east_stag' in curr_var_static.dims):
                        stag_x = True
                    if ('south_north_stag' in curr_var_static.dims):
                        stag_y = True
                    # Interpolate field for each vertical level
                    if ('bottom_top' in curr_var_static.dims):
                        ### Interpolate fields vertically, then horizontally
                        for iz in np.arange(dim_var[1]):
                            # Find levels in between current height
                            i_z0_mvng = np.nanargmin(np.abs(hgt_unstag_static[iz,j_desired_static,i_desired_static] - hgt_unstag_mvng[:,j_desired_mvng,i_desired_mvng]))
                            if i_z0_mvng==0:
                                i_z1_mvng = 1
                            elif i_z0_mvng==len(hgt_unstag_mvng[:,j_desired_mvng,i_desired_mvng])-1:
                                i_z0_mvng = i_z0_mvng -1
                                i_z1_mvng = i_z0_mvng + 1
                            else:
                                i_z1_mvng = i_z0_mvng + 1
                            # Interpolate precursor array vertically to current height
                            wsp_var = (vari=='U') or (vari=='V')
                            if (wsp_var==True) & (i_z0_mvng==0) & (hgt_unstag_mvng[i_z0_mvng,j_desired_mvng,i_desired_mvng]>hgt_unstag_static[iz,j_desired_static,i_desired_static]):
                                # Assume power-law wind profile to interpolate below lowest precursor model level for WIND SPEED ONLY to avoid gravity waves in domain
                                alpha_ = 0.14
                                print("Using power-law wind profile to extrapolate " + vari + " near the surface")
                                temp_var_mvng = np.array(curr_var_mvng[0,i_z0_mvng,:,:])*((hgt_unstag_static[iz,j_desired_static,i_desired_static]/hgt_unstag_mvng[i_z0_mvng,j_desired_mvng,i_desired_mvng])**alpha_)
                            else:
                                # Linear interpolation
                                dz_mvng = np.array(hgt_unstag_mvng[i_z1_mvng,j_desired_mvng,i_desired_mvng]) - np.array(hgt_unstag_mvng[i_z0_mvng,j_desired_mvng,i_desired_mvng])
                                dvar_mvng = np.array(curr_var_mvng[0,i_z1_mvng,:,:]) - np.array(curr_var_mvng[0,i_z0_mvng,:,:])
                                temp_var_mvng = np.array(curr_var_mvng[0,i_z0_mvng,:,:]) + (dvar_mvng/dz_mvng)*(np.array(hgt_unstag_static[iz,j_desired_static,i_desired_static]) - np.array(hgt_unstag_mvng[i_z0_mvng,j_desired_mvng,i_desired_mvng]))
                            # Interpolate horizontally
                            temp_var_static = np.array(curr_var_static[0,iz,:,:]) 
                            temp_var = func_interp_xy(temp_var_mvng,XLAT_U_mvng,XLONG_U_mvng,XLAT_V_mvng,XLONG_V_mvng,XLAT_mvng,XLONG_mvng,
                                                   temp_var_static,XLAT_U_static,XLONG_U_static,XLAT_V_static,XLONG_V_static,XLAT_static,XLONG_static,
                                                   stag_x,stag_y,i_desired_mvng,j_desired_mvng)
                            curr_var_static[0,iz,:,:] = temp_var
                    elif ('bottom_top_stag' in curr_var_static.dims):
                        ### Interpolate fields vertically, then horizontally
                        for iz in np.arange(dim_var[1]):
                            # Find levels in between current height
                            i_z0_mvng = np.nanargmin(np.abs(hgt_stag_static[iz,j_desired_static,i_desired_static] - hgt_stag_mvng[:,j_desired_mvng,i_desired_mvng]))
                            if i_z0_mvng==0:
                                i_z1_mvng = 1
                            elif i_z0_mvng==len(hgt_stag_mvng[:,j_desired_mvng,i_desired_mvng])-1:
                                i_z0_mvng = i_z0_mvng -1
                                i_z1_mvng = i_z0_mvng + 1
                            else:
                                i_z1_mvng = i_z0_mvng + 1
                            # Interpolate precursor array vertically to current height
                            dz_mvng = np.array(hgt_stag_mvng[i_z1_mvng,j_desired_mvng,i_desired_mvng]) - np.array(hgt_stag_mvng[i_z0_mvng,j_desired_mvng,i_desired_mvng])
                            dvar_mvng = np.array(curr_var_mvng[0,i_z1_mvng,:,:]) - np.array(curr_var_mvng[0,i_z0_mvng,:,:])
                            temp_var_mvng = np.array(curr_var_mvng[0,i_z0_mvng,:,:]) + (dvar_mvng/dz_mvng)*(np.array(hgt_stag_static[iz,j_desired_static,i_desired_static]) - np.array(hgt_stag_mvng[i_z0_mvng,j_desired_mvng,i_desired_mvng]))
                            # Interpolate horizontally
                            temp_var_static = np.array(curr_var_static[0,iz,:,:]) 
                            temp_var = func_interp_xy(temp_var_mvng,XLAT_U_mvng,XLONG_U_mvng,XLAT_V_mvng,XLONG_V_mvng,XLAT_mvng,XLONG_mvng,
                                                   temp_var_static,XLAT_U_static,XLONG_U_static,XLAT_V_static,XLONG_V_static,XLAT_static,XLONG_static,
                                                   stag_x,stag_y,i_desired_mvng,j_desired_mvng)
                            curr_var_static[0,iz,:,:] = temp_var
                    elif ('soil_layers_stag' in curr_var_static.dims):
                        for iz in np.arange(dim_var[1]):
                            temp_var_mvng = np.array(curr_var_mvng[0,iz,:,:])
                            temp_var_static = np.array(curr_var_static[0,iz,:,:]) 
                            temp_var = func_interp_xy(temp_var_mvng,XLAT_U_mvng,XLONG_U_mvng,XLAT_V_mvng,XLONG_V_mvng,XLAT_mvng,XLONG_mvng,
                                                   temp_var_static,XLAT_U_static,XLONG_U_static,XLAT_V_static,XLONG_V_static,XLAT_static,XLONG_static,
                                                   stag_x,stag_y,i_desired_mvng,j_desired_mvng)
                            curr_var_static[0,iz,:,:] = temp_var
                    else:
                        temp_var_mvng = np.array(curr_var_mvng[0,:,:])
                        temp_var_static = np.array(curr_var_static[0,:,:]) 
                        temp_var = func_interp_xy(temp_var_mvng,XLAT_U_mvng,XLONG_U_mvng,XLAT_V_mvng,XLONG_V_mvng,XLAT_mvng,XLONG_mvng,
                                               temp_var_static,XLAT_U_static,XLONG_U_static,XLAT_V_static,XLONG_V_static,XLAT_static,XLONG_static,
                                               stag_x,stag_y,i_desired_mvng,j_desired_mvng)
                        curr_var_static[0,:,:] = temp_var
                elif len(dim_var)==2:
                    if np.shape(curr_var_static)[-1]==np.shape(curr_var_mvng)[-1]:
                        curr_var_static[:,:] = np.array(curr_var_mvng[:,:])
                    else:
                        # Interpolate field for each vertical level
                        if ('bottom_top' in curr_var_static.dims):
                            ### Interpolate fields vertically, then horizontally
                            for iz in np.arange(dim_var[1]):
                                # Find levels in between current height
                                i_z0_mvng = np.nanargmin(np.abs(hgt_unstag_static[iz,j_desired_static,i_desired_static] - hgt_unstag_mvng[:,j_desired_mvng,i_desired_mvng]))
                                if i_z0_mvng==0:
                                    i_z1_mvng = 1
                                elif i_z0_mvng==len(hgt_unstag_mvng[:,j_desired_mvng,i_desired_mvng])-1:
                                    i_z0_mvng = i_z0_mvng -1
                                    i_z1_mvng = i_z0_mvng + 1
                                else:
                                    i_z1_mvng = i_z0_mvng + 1
                                # Interpolate precursor array to current height
                                dz_mvng = np.array(hgt_unstag_mvng[i_z1_mvng,j_desired_mvng,i_desired_mvng]) - np.array(hgt_unstag_mvng[i_z0_mvng,j_desired_mvng,i_desired_mvng])
                                dvar_mvng = np.array(curr_var_mvng[0,i_z1_mvng]) - np.array(curr_var_mvng[0,i_z0_mvng])
                                temp_var = np.array(curr_var_mvng[0,i_z0_mvng]) + (dvar_mvng/dz_mvng)*(np.array(hgt_unstag_static[iz,j_desired_static,i_desired_static]) - np.array(hgt_unstag_mvng[i_z0_mvng,j_desired_mvng,i_desired_mvng]))
                                curr_var_static[0,iz] = temp_var
                        elif ('bottom_top_stag' in curr_var_static.dims):
                            ### Interpolate fields vertically, then horizontally
                            for iz in np.arange(dim_var[1]):
                                # Find levels in between current height
                                i_z0_mvng = np.nanargmin(np.abs(hgt_stag_static[iz,j_desired_static,i_desired_static] - hgt_stag_mvng[:,j_desired_mvng,i_desired_mvng]))
                                if i_z0_mvng==0:
                                    i_z1_mvng = 1
                                elif i_z0_mvng==len(hgt_stag_mvng[:,j_desired_mvng,i_desired_mvng])-1:
                                    i_z0_mvng = i_z0_mvng -1
                                    i_z1_mvng = i_z0_mvng + 1
                                else:
                                    i_z1_mvng = i_z0_mvng + 1
                                # Interpolate precursor array to current height
                                dz_mvng = np.array(hgt_stag_mvng[i_z1_mvng,j_desired_mvng,i_desired_mvng]) - np.array(hgt_stag_mvng[i_z0_mvng,j_desired_mvng,i_desired_mvng])
                                dvar_mvng = np.array(curr_var_mvng[0,i_z1_mvng]) - np.array(curr_var_mvng[0,i_z0_mvng])
                                temp_var = np.array(curr_var_mvng[0,i_z0_mvng]) + (dvar_mvng/dz_mvng)*(np.array(hgt_stag_static[iz,j_desired_static,i_desired_static]) - np.array(hgt_stag_mvng[i_z0_mvng,j_desired_mvng,i_desired_mvng]))
                                curr_var_static[0,iz] = temp_var
                elif len(dim_var)==1:
                    curr_var_static[:] = curr_var_mvng[:]

                ## Make sure there are no negative values if variable should not have them
                if vari in vas_no_neg:
                    temp_arr_noNeg = np.array(curr_var_static)
                    if np.sum(temp_arr_noNeg<0)>0:
                        temp_arr_noNeg[temp_arr_noNeg<0] = 0.0
                        curr_var_static[:] = temp_arr_noNeg
                        print("Removing negative values from %s caused by interpolation" % vari)
                            
                ## Re-write in file
                rewrite = True
                if vari in vars_not_overwrtie:
                    rewrite = False
                    print('Not re-writing '+ vari)
                if rewrite:
                    ds_save.drop_vars(vari)
                    ds_save = ds_save.assign(**{vari: curr_var_static})
            # Save to new file and close
            ds_save.to_netcdf(dir_wrfout_save + fName_static)
            ds_save.close()
        else:
            print("%s already exists" % fName_static)
    else:
        print("Time out of desired range")

    print("Done with %i out of %i" % (i_f+1,len(my_files_toRead)))


# In[12]:


#%% Check that files have been modified
vari = 'U'

## Old file
ds_old = xr.open_dataset(dir_LES+fName_wrfout_static,decode_times=False)
U_old = ds_old[vari]
ds_old.close()

## Modified file
ds_mod = xr.open_dataset(dir_wrfout_save+fName_static,decode_times=False)
U_mod = ds_mod[vari]
ds_mod.close()

## Visualize planview of field
plt.figure(figsize=(10,4))
# Old field
plt.subplot(1,2,1)
plt.title("Old field")
plt.pcolormesh(U_old[0,0,:,:])
# Modified field
plt.subplot(1,2,2)
plt.title("Modified field")
plt.pcolormesh(U_mod[0,0,:,:])
plt.show()
plt.close()

## Visualize profile
plt.figure(figsize=(3,4))
plt.plot(U_old[0,:,j_desired_mvng,i_desired_mvng],hgt_unstag_static[:,j_desired_static,i_desired_static])
plt.plot(U_mod[0,:,j_desired_static,i_desired_static],hgt_unstag_static[:,j_desired_static,i_desired_static])
plt.show()
plt.close()


# In[13]:


try:
    MPI.Finalize()
    print("Now, run ndown.exe")
except:
    print("Now, run ndown.exe")


# In[ ]:





# In[ ]:




