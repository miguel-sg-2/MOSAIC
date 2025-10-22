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
#import cmocean
from scipy import stats,signal
from copy import deepcopy
#from geographiclib.geodesic import Geodesic
import shutil
#import netCDF4
import stat


# In[2]:


#%% File locations
# Location of files with static mesh
dir_LES = "/projects/storm/mgomez/realSims/hurrLaura/newLES/staticLES_dxSmaller_BCs_from_mvngNest/"

# dir_wrfout_prec =  dir_LES + "wrfoutFiles_prec/"
# dir_wrfout_save =  dir_LES + "wrfoutFiles_save/"
dir_wrfout_prec =  dir_LES + "wrfoutFiles_prec_1min/"
dir_wrfout_save =  dir_LES + "wrfoutFiles_save_1min/"

# Create directory if it does not exist
if not(os.path.exists(dir_wrfout_save)):
    STOP

# Choose domain
domain_static = 'd01'
domain_mvng = 'd02'
# File name root
fName = 'wrfout'

# Base file with static mesh to interpolate the moving domain data
fName_wrfout_static = "orig_wrfout_d01_2020-08-27_00:00:00"


# In[3]:


#%% Load hurricane best-track estimates
##### From IBTracs
dir_data = "/projects/storm/mgomez/ibtracs/"
ds = xr.open_dataset(dir_data+"IBTrACS.NA.v04r00.nc")
storm_name = np.array(ds['name']) # [storm]
lat_track = np.array(ds['usa_lat']) # [storm, time]
lon_track = np.array(ds['usa_lon']) # [storm, time]
ttime_track = np.array(ds['usa_lat'].time) # [storm, time]
max_1min_ws = np.array(ds['usa_wind']) # [storm, time]
ds.close()
#%% Get data for one hurricane in particular
hurrThisOne = b'LAURA'
i_hurr = np.where(storm_name==b'LAURA')[0][-1]
# Save only data for this storm
ttime_track = ttime_track[i_hurr,:]
lat_track = lat_track[i_hurr,:]
lon_track = lon_track[i_hurr,:]
max_1min_ws = max_1min_ws[i_hurr,:]
# Only keep good data
max_1min_ws = max_1min_ws[np.logical_not(np.isnan(lat_track))]
ttime_track = ttime_track[np.logical_not(np.isnan(lat_track))]
lon_track = lon_track[np.logical_not(np.isnan(lat_track))]
lat_track = lat_track[np.logical_not(np.isnan(lat_track))]
# Convert from kt to m/s
max_1min_ws = max_1min_ws*0.514444/1




# In[4]:


#%% Find center of hurricane for a given time

# i_desired = np.argmax(max_1min_ws)
desired_time = np.datetime64('2020-08-26T22:00:00') # ttime_track[i_desired] #
i_desired = np.argmin(np.abs(ttime_track - desired_time))
timeAfter_desiredTime = np.timedelta64(4,'h')

print("Hurricane center at %s is at [lat,lon] = [%.5f,%.5f]" % (str(desired_time),lat_track[i_desired],lon_track[i_desired]))


# In[5]:


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
ds.close()

# list_vars = list_vars[1:]
print(list_vars)


# In[6]:




#%% Define interpolation function

def func_interp(var_mvng,lat_u_mvng,lon_u_mvng,lat_v_mvng,lon_v_mvng,lat_m_mvng,lon_m_mvng,
               var_static,lat_u_static,lon_u_static,lat_v_static,lon_v_static,lat_m_static,lon_m_static,
               stag_X,stag_Y):
    ## Use correct grid for current variable
    if stag_Y:
        use_lat_mvng = lat_v_mvng[:,int(0.5*np.shape(lat_v_mvng)[1])]
        use_lon_mvng = lon_v_mvng[int(0.5*np.shape(lon_v_mvng)[0]),:]
        use_lat_static = lat_v_static
        use_lon_static = lon_v_static
    elif stag_X:
        use_lat_mvng = lat_u_mvng[:,int(0.5*np.shape(lat_u_mvng)[1])]
        use_lon_mvng = lon_u_mvng[int(0.5*np.shape(lon_u_mvng)[0]),:]
        use_lat_static = lat_u_static
        use_lon_static = lon_u_static
    else:
        use_lat_mvng = lat_m_mvng[:,int(0.5*np.shape(lat_m_mvng)[1])]
        use_lon_mvng = lon_m_mvng[int(0.5*np.shape(lon_m_mvng)[0]),:]
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
            var_static[iy,ix] = ww[0,0]
    return var_static
    


# In[ ]:


#%% Load data from wrfout* files and save the modified versions

# Find files matching description for current domain within directory
entries = os.listdir(dir_wrfout_prec)
ofInterest = []
for ii in np.arange(len(entries)):
    # Check if file name matches
    if domain_mvng in entries[ii]:
        if fName in entries[ii]:
            ofInterest = np.append(ofInterest,ii)
            # print(entries[ii])
ofInterest = ofInterest.astype('int')
rndmFromInterest = np.random.choice(ofInterest)

# Load file, modify lat,lon and save to new file
for i_f in np.arange(0,len(ofInterest)):
    ## Get time for current wrfout file
    ds = xr.open_dataset(dir_wrfout_prec+str(entries[ofInterest[i_f]]),decode_times=False)
    ## Extract time data
    Time = np.array(ds['Times'])
    a = str(Time)
    ttime = np.datetime64(a[3:13] + str(' ')+ a[14:-2])
    ds.close()

    ## Make sure time is within desired range
    if (ttime >= desired_time) & (ttime <= desired_time + timeAfter_desiredTime):
        print("Continue with %s" % str(ttime))
        ## Copy wrfout to new file to save
        tmp_str = str(entries[ofInterest[i_f]])
        i_change = tmp_str.find('d02')
        fName_new = tmp_str[0:i_change] + 'd01' + tmp_str[i_change+3:]
        # shutil.copyfile(dir_LES+fName_wrfout_static, dir_wrfout_save + fName_new)
        # Change file permissions
        # os.chmod(dir_wrfout_save + fName_new, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)

        if not(os.path.exists(dir_wrfout_save + fName_new)):
            ## Extract domain dimensions
            ds = xr.open_dataset(dir_wrfout_prec+str(entries[ofInterest[i_f]]),decode_times=False)
            # Mass points
            XLAT_mvng = np.array(ds['XLAT'])[0,:,:]
            XLONG_mvng = np.array(ds['XLONG'])[0,:,:]
            # u-points
            XLAT_U_mvng = np.array(ds['XLAT_U'])[0,:,:]
            XLONG_U_mvng = np.array(ds['XLONG_U'])[0,:,:]
            # v-points
            XLAT_V_mvng = np.array(ds['XLAT_V'])[0,:,:]
            XLONG_V_mvng = np.array(ds['XLONG_V'])[0,:,:]
            ds.close()
            
            # Re-write all variables of the static domain
            ds_save = xr.open_dataset(dir_LES+fName_wrfout_static,decode_times=False)
            for vari in list_vars: 
                ## Extract current variable from each data structure
                ds_mvng = xr.open_dataset(dir_wrfout_prec+str(entries[ofInterest[i_f]]),decode_times=False)
                curr_var_mvng = ds_mvng[vari]
                ds_mvng.close()
                ds_sttc = xr.open_dataset(dir_LES+fName_wrfout_static,decode_times=False)
                curr_var_static = ds_sttc[vari]
                ds_sttc.close()
                # ds_mvng = netCDF4.Dataset(dir_wrfout_prec+str(entries[ofInterest[i_f]]))#, 'r+')
                # curr_var_mvng = ds_mvng[vari]
                # ds_sttc = netCDF4.Dataset(dir_LES+fName_wrfout_static)#, 'r+')
                # curr_var_static = ds_sttc[vari]
    
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
                    if ('bottom_top' in curr_var_static.dims) | ('bottom_top_stag' in curr_var_static.dims) | ('soil_layers_stag' in curr_var_static.dims):
                        for iz in np.arange(dim_var[1]):
                            temp_var_mvng = np.array(curr_var_mvng[0,iz,:,:])
                            temp_var_static = np.array(curr_var_static[0,iz,:,:]) 
                            temp_var = func_interp(temp_var_mvng,XLAT_U_mvng,XLONG_U_mvng,XLAT_V_mvng,XLONG_V_mvng,XLAT_mvng,XLONG_mvng,
                                                   temp_var_static,XLAT_U_static,XLONG_U_static,XLAT_V_static,XLONG_V_static,XLAT_static,XLONG_static,
                                                   stag_x,stag_y)
                            curr_var_static[0,iz,:,:] = temp_var
                    else:
                        temp_var_mvng = np.array(curr_var_mvng[0,:,:])
                        temp_var_static = np.array(curr_var_static[0,:,:]) 
                        temp_var = func_interp(temp_var_mvng,XLAT_U_mvng,XLONG_U_mvng,XLAT_V_mvng,XLONG_V_mvng,XLAT_mvng,XLONG_mvng,
                                               temp_var_static,XLAT_U_static,XLONG_U_static,XLAT_V_static,XLONG_V_static,XLAT_static,XLONG_static,
                                               stag_x,stag_y)
                        curr_var_static[0,:,:] = temp_var
                elif len(dim_var)==2:
                    curr_var_static[:,:] = np.array(curr_var_mvng[:,:])
                elif len(dim_var)==1:
                    curr_var_static[:] = curr_var_mvng[:]
    
                ## Re-write in file
                if 'LON' in vari:
                    print('Not re-writing '+ vari)
                elif 'LAT' in vari:
                    print('Not re-writing '+ vari)
                else:
                    # Save modified met_em files
                    ds_save.drop_vars(vari)
                    ds_save = ds_save.assign(**{vari: curr_var_static}) # ds = ds.assign(**{replace_var_name: replace_var_value})
            # Save to new file and close
            ds_save.to_netcdf(dir_wrfout_save + fName_new)
            ds_save.close()
        else:
            print("%s already exists" % fName_new)
    else:
        print("Time out of desired range")

    print("Done with %i out of %i" % (i_f+1,len(ofInterest)))


# In[ ]:



#%% Check that files have been modified
vari = 'U'

## Old file
ds_old = xr.open_dataset(dir_LES+fName_wrfout_static,decode_times=False)
U_old = ds_old[vari]
ds_old.close()

## Modified file
ds_mod = xr.open_dataset(dir_wrfout_save+fName_new,decode_times=False)
U_mod = ds_mod[vari]
ds_mod.close()

## Visualize field
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


# In[ ]:


print("Now, run ndown.exe")


# In[ ]:


STOP


# In[29]:


#%% Now visualize the new initial conditions from ndown

dir_static = "/scratch/mgomez/STORM/realSims/hurrLaura/newLES/test_massive_LES_1/"
# dir_static = "/projects/storm/mgomez/realSims/hurrLaura/newLES/staticLES_highRes_BCs_from_mvngNest_2/"

ds = xr.open_dataset(dir_static+'wrfinput_d02_from_ndown',decode_times=False)
U_init = np.array(ds['U'])
U_init = 0.5*(U_init[:,:,:,1:] + U_init[:,:,:,0:-1])
V_init = np.array(ds['V'])
V_init = 0.5*(V_init[:,:,1:,:] + V_init[:,:,0:-1,:])
UV_init = (U_init**2 + V_init**2)**0.5
otherVar = np.array(ds['V10'])
theta_init = ds['T']
ph_init = ds['PH']
ds.close()


# In[30]:


plt.pcolormesh(ph_init[0,50,:,:])
plt.colorbar()


# In[31]:


plt.pcolormesh(otherVar[0,:,:])


# In[32]:


plt.pcolormesh(UV_init[0,50,:,:])


# In[33]:


plt.pcolormesh(theta_init[0,10,:,:])


# In[34]:


#%% Check for NaNs in wrfinput file
ds = xr.open_dataset(dir_static+'wrfinput_d02_from_ndown',decode_times=False)
# List of variables that need to be re-written
list_vars = list(ds.variables.keys())
ds.close()

print(list_vars)

ds = xr.open_dataset(dir_static+'wrfinput_d02_from_ndown',decode_times=False)
for vari in list_vars: 
    temp_var = ds[vari]
    if not('S' in str(temp_var.dtype)):
        i_nans = np.isnan(temp_var)
        if np.sum(i_nans)>0:
            STOP
        else:
            print("All good with %s with min,max = [%.2f,%.2f]" % (vari,np.min(temp_var),np.max(temp_var)))
ds.close()




# In[ ]:


#%% Check for NaNs in wrfbdy file
ds = xr.open_dataset(dir_static+'wrfbdy_d02_from_ndown',decode_times=False)
# List of variables that need to be re-written
list_vars = list(ds.variables.keys())
ds.close()

print(list_vars)

ds = xr.open_dataset(dir_static+'wrfbdy_d02_from_ndown',decode_times=False)
for vari in list_vars: 
    temp_var = ds[vari]
    if not('S' in str(temp_var.dtype)):
        i_nans = np.isnan(temp_var)
        if np.sum(i_nans)>0:
            STOP
        else:
            print("All good with %s with min,max = [%.2f,%.2f]" % (vari,np.min(temp_var),np.max(temp_var)))
ds.close()


# In[35]:


ds = xr.open_dataset(dir_static+'wrfbdy_d02_from_ndown',decode_times=False)
var_PH = ds['PH_BXS']
ds.close()


# In[36]:


var_PH


# In[37]:


plt.pcolormesh(var_PH[0,0,:,:])
plt.colorbar()


# In[43]:


plt.pcolormesh(var_PH[:,0,50,:])
plt.colorbar()


# In[39]:


var_PH


# In[ ]:





