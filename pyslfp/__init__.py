from os.path import dirname, join as joinpath

DATADIR = joinpath(dirname(__file__), "data")


from pyslfp.ice_ng import IceNG, IceModel
from pyslfp.physical_parameters import EarthModelParameters
from pyslfp.finger_print import FingerPrint

from pyslfp.operators import (
    tide_gauge_operator,
    grace_operator,
    field_to_sh_coefficient_operator,
    sh_coefficient_to_field_operator,
    averaging_operator,
    WahrMolenaarByranMethod,
)

# from pyslfp.operators import (
#    FingerPrintOperator,
#    ObservationOperator,
#    PropertyOperator,
#    GraceObservationOperator,
#    TideGaugeObservationOperator,
#    LoadAveragingOperator,
# )


# from pyslfp.operators import (
#    SeaLevelOperator,
#    GraceObservationOperator,
#    TideGaugeObservationOperator,
#    AveragingOperator,
#    WahrOperator,
# )
from pyslfp.plotting import plot

from pyslfp.utils import SHVectorConverter
