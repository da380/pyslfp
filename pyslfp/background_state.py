import xarray as xr
import pyshtools as sh
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyslfp.plotting import plot_SHGrid
from . import DATADIR

if __name__ == "__main__":
    pass


class BackgroundState(Units): 
                        
    def __init__(self, /, *, maximum_degree = 128, date = None, grid = "DH", model = "ice7g", units = None):

        # Set up the units. 
        if units is None:            
            super(BackgroundState,self).__init__()
        else:
            super(BackgroundState, self).__init__(length_scale = units.length_scale, mass_scale = units.mass_scale, time_scale = units.mass_scale)
        
        # Check the model choice is okay.
        if model in ["ice5g", "ice6g", "ice7g"]:
            self._model = model
        else:
            raise ValueError("chosen model not implemented")

        # If date not set, use the present. 
        if date is None:
            date = self._dates[0]                    

        # Set up the pyshtools grid functions. 
        self._maximum_degree = maximum_degree
        self._topography = sh.SHGrid.from_zeros(maximum_degree, grid = grid)
        self._sea_level = sh.SHGrid.from_zeros(maximum_degree, grid = grid)
        self._ice_thickness = sh.SHGrid.from_zeros(maximum_degree, grid = grid)        

        # Load in the data file. 
        if date in self._dates:
            file = self._file_name(date)
            data = xr.open_dataset(file)            
        else:
            raise ValueError("Date not present within the chosen model")

        # Interpolate onto the grid functions. 
        
        if self._model == "ice5g":
            ice = RegularGridInterpolator((data.lat.values, data.long.values), data.sftgit.values / self.length_scale, bounds_error= False, fill_value=None) 
            topo = RegularGridInterpolator((data.lat.values, data.long.values), data.orog.values / self.length_scale, bounds_error= False, fill_value=None)  
        else:            
            ice = RegularGridInterpolator((data.lat.values, data.lon.values), data.stgit.values / self.length_scale, bounds_error= False, fill_value=None) 
            topo = RegularGridInterpolator((data.lat.values, data.lon.values), data.Topo.values / self.length_scale, bounds_error= False, fill_value=None) 

        lats,lons = np.meshgrid(self.latitudes, self.longitudes, indexing="ij")
        self._ice_thickness.data = ice((lats,lons))
        self._topography.data = topo((lats,lons))

        # Set the sea level values. 
        self.sea_level.data = np.where(np.logical_and(self.topography.data > 0, self.ice_thickness.data > 0 ),
                                       -self.topography.data + self.ice_thickness.data, -self.topography.data)


        # Set the ocean function.         
        self._ocean_function = sh.SHGrid.from_zeros(maximum_degree, grid = grid)
        self._ocean_function.data = np.where(self.water_density * self.sea_level.data - self.ice_density * self.ice_thickness.data > 0, 1, 0)

    # Return a list of the times (as stings).
    @property
    def _dates(self):
        if self._model == "ice5g":
            return ["00.0", "00.5", "01.0", "01.5", "02.0", "02.5", 
                    "03.0", "03.5", "04.0", "04.5", "05.0", "05.5",
                    "06.0", "06.5", "07.0", "07.5", "08.0", "08.5", 
                    "09.0", "09.5", "10.0", "10.5", "11.0", "11.5",
                    "12.0", "12.5", "13.0", "13.5", "14.0", "14.5", 
                    "15.0", "15.5", "16.0", "16.5", "17.0", "18.0", 
                    "19.0", "20.0", "21.0"]
        else:
            return ["0", "0.5", "1", "1.5", "2", "2.5", 
                    "3", "3.5", "4", "4.5", "5", "5.5",
                    "6", "6.5", "7", "7.5", "8", "8.5", 
                    "9", "9.5", "10", "10.5", "11", "11.5",
                    "12", "12.5", "13", "13.5", "14", "14.5", 
                    "15", "15.5", "16", "16.5", "17", "17.5", 
                    "18", "18.5", "19", "19.5", "20", "21", 
                    "22", "23", "24", "25", "26"]

    # Return the file name for a given date. 
    def _file_name(self,date):
        if self._model == "ice5g":
            file = DATADIR + "/ice5g/ice5g_v1.2_" + date + "k_1deg.nc"
        elif self._model == "ice6g":
            file = DATADIR + "/ice6g/I6_C.VM5a_1deg." + date + ".nc"
        else:
            file = DATADIR + "/ice7g/I7G_NA.VM7_1deg." + date + ".nc"
        return file        

    @property
    def water_density(self):
        return 1000 / self.density_scale

    @property
    def ice_density(self):
        return 917 / self.density_scale

    @property
    def latitudes(self):
        return self._sea_level.lats()

    @property
    def longitudes(self):
        return self._sea_level.lons()

    @property
    def ice_thickness(self):
        return self._ice_thickness

    @property
    def topography(self):
        return self._topography

    @property
    def sea_level(self):
        return self._sea_level

    @property
    def ocean_function(self):
        return self._ocean_function

    


    
    
        

    
    