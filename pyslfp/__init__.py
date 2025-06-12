from os.path import dirname, join as joinpath

DATADIR = joinpath(dirname(__file__), "data")


from pyslfp.ice_ng import IceNG
from pyslfp.physical_parameters import EarthModelParamters
from pyslfp.finger_print import FingerPrint
