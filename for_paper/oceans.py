import matplotlib.pyplot as plt
import pyslfp as sl


def test_iho_seas():
    print("Initializing FingerPrint (using lmax=64 for speed)...")
    fp = sl.FingerPrint(lmax=64)

    print("Creating map figure for IHO Seas...")
    fig1, ax1 = sl.create_map_figure(figsize=(14, 7))

    print("Loading and plotting IHO Sea Areas (from shapefile)...")
    fp.plot_iho_sea_boundaries(ax1)
    ax1.set_title("Global IHO Sea Areas (Version 3)")
    plt.show()


def test_ne_oceans():
    print("\nInitializing FingerPrint again...")
    fp = sl.FingerPrint(lmax=64)

    print("Creating map figure for Natural Earth Oceans...")
    fig2, ax2 = sl.create_map_figure(figsize=(14, 7))

    print("Loading and plotting Natural Earth Ocean Basins (built-in)...")
    fp.plot_ne_ocean_boundaries(ax2)
    ax2.set_title("Global Natural Earth Ocean Basins")
    plt.show()


if __name__ == "__main__":
    #    test_iho_seas()
    test_ne_oceans()
