import matplotlib.pyplot as plt
import pyslfp as sl


import pyslfp.linear_operators as op

A = op.physics.FingerPrintOperator.for_sobolev_testing(128, 2, 0.1, regularity=True)

B = op.physics.centrifugal_potential_operator(A.state.model)


A.ice_projection_operator().check(
    domain_measure=A.load_measure_for_testing(),
)

u = A.domain.project_function(lambda p: 1)


sl.plot(u, symmetric=True)


sl.plot(A.land_projection_operator()(u), symmetric=True)

plt.show()
