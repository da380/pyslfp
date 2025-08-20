import numpy as np
import matplotlib.pyplot as plt
import pyshtools as sh
from pyslfp import SHVectorConverter


lmax = 4
u = sh.SHGrid.from_zeros(8)

for i, lat in enumerate(u.lats()):
    th = (90 - lat) * np.pi / 180
    for j, lon in enumerate(u.lons()):
        ph = lon * np.pi / 180
        u.data[i, j] = np.sin(th) * np.cos(ph)


# u.plot()
# plt.show()

ulm = u.expand()

coeffs_in = ulm.coeffs

converter = SHVectorConverter(8, lmin=2)

vec = converter.to_vector(coeffs_in)

coeffs_out = converter.from_vector(vec)

print(coeffs_in - coeffs_out)
