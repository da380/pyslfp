import matplotlib.pyplot as plt
from pyslfp import FingerPrint
from pygeoinf import LinearOperator
from pygeoinf.homogeneous_space.sphere import Lebesgue


class SeaLevelOperator(LinearOperator):
    """
    LinearOperator that maps a direct load to the resulting sea level change.
    """

    def __init__(self, fingerprint):
        self._fingerprint = fingerprint

        domain = Lebesgue(
            self._fingerprint.lmax, radius=self._fingerprint.mean_sea_floor_radius
        )
        codomain = domain

        super().__init__(domain, codomain, self._mapping, adjoint_mapping=self._mapping)

    def ice_sheet_projection(self):
        """
        Returns a LinearOperator that multiplies fields by one over the
        ice sheets and zero elsewhere.
        """
        return LinearOperator.self_adjoint(
            self.domain, lambda u: u * self._fingerprint.ice_projection(0)
        )

    def _mapping(self, zeta):
        sea_level_change, _, _, _ = self._fingerprint(direct_load=zeta)
        return sea_level_change


# Set up FingerPrint instance.
fingerprint = FingerPrint()
fingerprint.set_state_from_ice_ng()

# Set up the sea level operator.
A = SeaLevelOperator(fingerprint)
P = A.ice_sheet_projection()

# Generate a Gaussian measure on the operators domain.
E = A.domain
mu = E.sobolev_gaussian_measure(2, 0.4, 1)


# Check the adjoint identity.
u = mu.sample()
v = P(u)

fig, ax, im = fingerprint.plot(u)
fig.colorbar(im, ax=ax, orientation="horizontal")

fig, ax, im = fingerprint.plot(v)
fig.colorbar(im, ax=ax, orientation="horizontal")


plt.show()
