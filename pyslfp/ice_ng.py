import xarray as xr
import pyshtools as sh
import numpy as np
import bisect
from scipy.interpolate import RegularGridInterpolator
from . import DATADIR

if __name__ == "__main__":
    pass

class IceNG:

    def __init__(self, /, *, version = 7):
        
        # Check the version choice is okay. 
        if version in [5, 6, 7]:
            self._version = version
        else:
            raise ValueError("chosen version not implemented")

    # Convert a date into the appropriate file name.     
    def _date_to_file(self,date):
        
        if self._version in [6, 7]:
            date_string =  f"{int(date):d}" if date.is_integer() else f"{date:3.1f}" 
        else:
            date_string = f"{date:04.1f}"
            
        if self._version == 7:
            return DATADIR + "/ice7g/I7G_NA.VM7_1deg." + date_string + ".nc"
        elif self._version == 6:
            return DATADIR + "/ice6g/I6_C.VM5a_1deg." + date_string + ".nc"
        else:
            return DATADIR + "/ice5g/ice5g_v1.2_" + date_string + "k_1deg.nc"


    # Given a date, find the data files that bound it for linear interpolation. 
    # The fraction needed in interpolation is also returned. 
    def _find_files(self,date):        
        
        if self._version in [6, 7]:
            dates = np.append(np.linspace(0,21,43),np.linspace(22,26,5))
        else:
            dates = np.append(np.linspace(0,17,35),np.linspace(18,21,4))

        i = bisect.bisect_left(dates,date)    
        if i == 0:
            date1 = dates[0]
            date2 = dates[0]        
        elif i == len(dates):
            date1 = dates[i-1]
            date2 = dates[i-1]        
        else:
            date1 = dates[i-1]
            date2 = dates[i]

        fraction = (date2 - date)/(date2-date1) if date1 != date2 else 0
            
        return self._date_to_file(date1),self._date_to_file(date2), fraction

    
    # Reads in a date file and interpolates ice thickness and sea level onto a SHGrid. 
    def _get_time_slice(self,file, lmax, /, *,  grid="DH", sampling=1, extend=True):        

        data = xr.open_dataset(file)                        
        ice_thickness = sh.SHGrid.from_zeros(lmax, grid=grid, sampling=sampling, extend=extend)  
        topography = sh.SHGrid.from_zeros(lmax, grid=grid, sampling=sampling, extend=extend)      

        if self._version == 5:
            ice_thickness_function = RegularGridInterpolator((data.lat.values, data.long.values), data.sftgit.values, bounds_error= False, fill_value=None) 
            topography_function = RegularGridInterpolator((data.lat.values, data.long.values), data.orog.values, bounds_error= False, fill_value=None)  
        else:            
            ice_thickness_function = RegularGridInterpolator((data.lat.values, data.lon.values), data.stgit.values, bounds_error= False, fill_value=None) 
            topography_function = RegularGridInterpolator((data.lat.values, data.lon.values), data.Topo.values, bounds_error= False, fill_value=None) 

        lats,lons = np.meshgrid(ice_thickness.lats(), ice_thickness.lons(), indexing="ij")
        ice_thickness.data = ice_thickness_function((lats,lons))
        topography.data = topography_function((lats,lons))        

        return ice_thickness, topography

    # Returns the ice thickness and sea level at a given date interpolated onto a SHGrid. If the date 
    # does not exist within the data set, linear interpolation is used between the two closest times. 
    # If the date is out of range, constant extrapolation is applied from the boundary values. 
    def get_time_slice(self, date, lmax, /, *,  grid = "DH", sampling=1, extend = True):
        file1, file2, fraction = self._find_files(date)        
        if file1 == file2:
            ice_thickness, topography = self._get_time_slice(file1, lmax, grid = grid, sampling=sampling, extend = extend)
        else:
            ice_thickness1, topography1 = self._get_time_slice(file1, lmax, grid = grid, sampling=sampling, extend = extend)
            ice_thickness2, topography2 = self._get_time_slice(file2, lmax, grid = grid, sampling=sampling, extend = extend)
            ice_thickness = fraction * ice_thickness1 + (1 - fraction) * ice_thickness2
            topography = fraction * topography1 + (1 - fraction) * topography2        
        return ice_thickness, topography

   




       