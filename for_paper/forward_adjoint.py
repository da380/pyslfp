"""
Minimal example with pyslfp
"""

import numpy as np
import matplotlib.pyplot as plt
import pyslfp as sl
from pyslfp.linear_operators import grace_observation_operator, WMBMethod

F = sl.linear_operators.FingerPrintOperator.from_defaults()
zeta = F.state.west_antarctic_load(fraction=0.01)
slc, _, _, _ = F(zeta)

_, (ax1, ax2) = sl.subplots(2, 1, figsize=(8, 10))
sl.plot(
    zeta
    * F.state.ice_projection()
    * F.parameters.length_scale
    / F.parameters.water_density,
    ax=ax1,
    colorbar_kwargs={"label": "Direct load ETW (m)"},
)
sl.plot(
    slc * F.state.ocean_projection() * F.parameters.length_scale,
    ax=ax2,
    colorbar_kwargs={"label": "Sea level change (m)"},
)


kernel_dim = (
    F.parameters.length_scale**2
    * F.parameters.time_scale ** (-2)
    * F.parameters.mass_scale ** (-1)
)


OBS_DEGREE = 100
load_to_grace_wmb = WMBMethod(
    F.model, OBS_DEGREE
).load_to_potential_coefficient_operator(F.domain)
load_to_grace = grace_observation_operator(F.codomain, OBS_DEGREE) @ F

L = 5
M = 3
index = F.domain.index_to_integer((L, M)) - 4
data = load_to_grace.codomain.basis_vector(index)
kernel = load_to_grace.adjoint(data)
kernel_wmb = load_to_grace_wmb.adjoint(data)


kernel_norm = np.max(np.abs(kernel.data))
_, (ax1, ax2) = sl.subplots(2, 1, figsize=(8, 10))
sl.plot(
    kernel * kernel_dim,
    ax=ax1,
    colorbar_kwargs={
        "label": rf"$\phi_{{{L},{M}}}$ kernel (m$^{{2}}$ s$^{{-2}}$ kg$^{{-1}}$)"
    },
)
sl.plot(
    (kernel - kernel_wmb) * 100 / kernel_norm,
    ax=ax2,
    colorbar_kwargs={"label": rf"$\phi_{{{L},{M}}}$ kernel difference (%))"},
)

plt.show()
