import pyslfp.linear_operators as op

A = op.physics.FingerPrintOperator.for_testing(
    128, load_parameters=(2, 0.1), response_parameters=(2, 0.1)
)

tide_gauges_obs = op.tide_gauges.TideGaugeObservationModel.from_gloss_network(A)

tide_gauges_obs.forward_operator.check(domain_measure=A.load_measure_for_testing())
