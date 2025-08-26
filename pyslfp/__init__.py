from os.path import dirname, join as joinpath

DATADIR = joinpath(dirname(__file__), "data")


from pyslfp.ice_ng import IceNG, IceModel
from pyslfp.physical_parameters import EarthModelParameters
from pyslfp.finger_print import FingerPrint

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
