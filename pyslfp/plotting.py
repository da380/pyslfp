import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 


if __name__ == "__main__":
    pass

def plot_SHGrid(grid, /, *, cmap="RdBu", symmetric=True,
                colorbar=False, show=True,  file=None,
                clim=None, contour=False, levels=10):

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    if contour:
        plt.contourf(grid.lons(), grid.lats(), grid.data,cmap = cmap, levels=levels)
    else:
        plt.pcolor(grid.lons(), grid.lats(), grid.data,cmap = cmap)

    # Symmetrise the clim values if requested. 
    if symmetric:
        max = np.nanmax(np.abs(grid.data))
        plt.clim([-max, max])
    
    if clim is not None:
        plt.clim(clim)

    # Add a colour bar if requested. 
    if colorbar:
        plt.colorbar(orientation="horizontal")

    # Show the figure if requested. 
    if show:
        plt.show()        

    # Save the figure if requested. 
    if file is not None:
        pass