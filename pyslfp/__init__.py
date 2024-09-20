from os.path import dirname, join as joinpath
DATADIR = joinpath(dirname(__file__), 'data')

from pyslfp.fields import ResponseFields, ResponseCoefficients
from pyslfp.ice_ng import IceNG
from pyslfp.plotting import plot_SHGrid
from pyslfp.solver import Solver


