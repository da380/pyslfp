import matplotlib.pyplot as plt
import numpy as np
import pyslfp as sl


fp = sl.FingerPrint(
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation()
)
fp.set_state_from_ice_ng()

A = fp.as_sobolev_linear_operator(2, 0.1)

P = A.codomain.subspace_projection(0)

B = P @ A

u = fp.northern_hemisphere_load()

v = A.domain.dirac_representation((42.36, -71.06))

w = B.adjoint(v)

sl.plot(w * fp.ice_projection())
plt.show()
