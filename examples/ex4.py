from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev


from pyslfp.operators import (
    field_to_sh_coefficient_operator,
    sh_coefficient_to_field_operator,
)

# field_space = Lebesgue(128, radius=4)
field_space = Sobolev(16, 2, 0.5, radius=3)


A = sh_coefficient_to_field_operator(field_space, lmax=8, lmin=2)
B = field_to_sh_coefficient_operator(field_space, lmax=8, lmin=2)

u = A.domain.random()

v = (B @ A)(u)

print(A.domain.norm(u - v) / A.domain.norm(u))
