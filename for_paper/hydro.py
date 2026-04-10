import pyslfp as sl
import matplotlib.pyplot as plt


def test_hydrobasins():
    print("Initializing FingerPrint (using lmax=64 for speed)...")
    # We use a low lmax here just to get the class instantiated quickly
    fp = sl.FingerPrint(lmax=64)

    print("Creating map figure...")
    fig, ax = sl.create_map_figure(figsize=(14, 7))

    print("Loading and plotting HydroBASINS...")

    # This triggers the lazy-loading property and plots the vectors
    fp.plot_hydrobasin_boundaries(ax)

    ax.set_title("Global HydroBASINS (Level 3 Catchments)")

    print("Rendering plot...")
    plt.show()


if __name__ == "__main__":
    test_hydrobasins()
