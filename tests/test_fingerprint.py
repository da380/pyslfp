import pytest
import numpy as np
from scipy.stats import norm
from pyslfp.finger_print import FingerPrint
from pyslfp.fields import ResponseFields

@pytest.mark.parametrize("lmax", [(8), (32), (128)])
class TestFingerPrint:
    
    def fingerprint(self, lmax):
        fp = FingerPrint(lmax)
        fp.set_background_state_from_ice_ng()
        return fp
    
    def inner_product(self, fp, f1, f2):
        g = fp.gravitational_acceleration
        return fp.integrate(f1.sl * f2.sl)  \
            - (1 /g) * fp.integrate((g * f1.u * f2.u + f1.phi * f2.phi))  \
            - (1 /g) * np.inner(f1.omega, f2.omega)
    
    def test_solver(self, lmax):
        fp = self.fingerprint(lmax)
        zeta = fp.northern_hemisphere_load()
        response = fp.solver(zeta)
        assert isinstance(response, ResponseFields)
        assert response.u.lmax == lmax
        assert response.phi.lmax == lmax
        assert response.omega.shape == (2,)
        assert response.sl.lmax == lmax

    def test_generalised_solver(self, lmax):
        fp = self.fingerprint(lmax)
        zeta1 = fp.northern_hemisphere_load()
        zeta2 = fp.southern_hemisphere_load()
        zero = fp.zero_grid()
        # Invent some adjoint loads
        zeta1_u = zeta1 
        zeta1_phi = zeta1 
        kk1 = np.array([1,5]) 
        zeta2_u = zeta2 
        zeta2_phi = zeta2
        kk2 = np.array([5,10]) 
        # Initialise the generalised loads
        generalised_load1 = ResponseFields(zeta1_u, zeta1_phi, kk1, zeta1)
        generalised_load2 = ResponseFields(zeta2_u, zeta2_phi, kk2, zeta2)
        # Compute the response. 
        response1 = fp.generalised_solver(generalised_load1, rotational_feedbacks=True)
        response2 = fp.generalised_solver(generalised_load2, rotational_feedbacks=True)
        # Check to see if the two inner products are equal
        lhs = self.inner_product(fp, generalised_load1, response2)
        rhs = self.inner_product(fp, generalised_load2, response1)
        print(lhs)
        print(rhs)
        assert np.isclose(lhs, rhs, rtol=1e-3)

    def test_solver_no_rotational_feedbacks(self, lmax):
        fp = self.fingerprint(lmax)
        zeta = fp.northern_hemisphere_load()
        response = fp.solver(zeta, rotational_feedbacks=False)
        assert isinstance(response, ResponseFields)
        assert response.u.lmax == lmax
        assert response.phi.lmax == lmax
        assert response.omega.shape == (2,)
        assert response.sl.lmax == lmax

    def test_generalised_solver_no_rotational_feedbacks(self, lmax):
        fp = self.fingerprint(lmax)
        zeta1 = fp.northern_hemisphere_load()
        zeta2 = fp.southern_hemisphere_load()
        zero = fp.zero_grid()
        # Invent some adjoint loads
        zeta1_u = zeta1 
        zeta1_phi = zeta1 
        kk1 = np.array([1,5]) 
        zeta2_u = zeta2 
        zeta2_phi = zeta2
        kk2 = np.array([5,10]) 
        # Initialise the generalised loads
        generalised_load1 = ResponseFields(zeta1_u, zeta1_phi, kk1, zeta1)
        generalised_load2 = ResponseFields(zeta2_u, zeta2_phi, kk2, zeta2)
        # Compute the response. 
        response1 = fp.generalised_solver(generalised_load1, rotational_feedbacks=False)
        response2 = fp.generalised_solver(generalised_load2, rotational_feedbacks=False)
        # Check to see if the two inner products are equal
        lhs = self.inner_product(fp, generalised_load1, response2)
        rhs = self.inner_product(fp, generalised_load2, response1)
        print(lhs)
        print(rhs)
        assert np.isclose(lhs, rhs, rtol=1e-3)

    def test_integrate(self, lmax):
        fp = self.fingerprint(lmax)
        zeta1 = fp.northern_hemisphere_load()
        response1 = fp.solver(zeta1)
        assert isinstance(fp.integrate(response1.sl * zeta1), float)
        
    def test_zeros(self, lmax):
        fp = self.fingerprint(lmax)
        zero = fp.zero_grid()
        assert np.all(zero.data == 0)

    def test_addition(self, lmax):
        fp = self.fingerprint(lmax)
        zeta1 = fp.northern_hemisphere_load()
        zeta2 = fp.southern_hemisphere_load()
        response1 = fp.solver(zeta1)
        response2 = fp.solver(zeta2)
        response = response1 + response2
        assert isinstance(response, ResponseFields)
        assert response.u.lmax == lmax
        assert response.phi.lmax == lmax
        assert response.omega.shape == (2,)
        assert response.sl.lmax == lmax

    def test_subtraction(self, lmax):
        fp = self.fingerprint(lmax)
        zeta1 = fp.northern_hemisphere_load()
        zeta2 = fp.southern_hemisphere_load()
        response1 = fp.solver(zeta1)
        response2 = fp.solver(zeta2)
        response = response1 - response2
        assert isinstance(response, ResponseFields)
        assert response.u.lmax == lmax
        assert response.phi.lmax == lmax
        assert response.omega.shape == (2,)
        assert response.sl.lmax == lmax

    def test_multiplication(self, lmax):
        fp = self.fingerprint(lmax)
        zeta1 = fp.northern_hemisphere_load()
        response1 = fp.solver(zeta1)
        s = 2
        response = response1 * s
        assert isinstance(response, ResponseFields)
        assert response.u.lmax == lmax
        assert response.phi.lmax == lmax
        assert response.omega.shape == (2,)
        assert response.sl.lmax == lmax

    def test_division(self, lmax):
        fp = self.fingerprint(lmax)
        zeta1 = fp.northern_hemisphere_load()
        response1 = fp.solver(zeta1)
        s = 2
        response = response1 / s
        assert isinstance(response, ResponseFields)
        assert response.u.lmax == lmax
        assert response.phi.lmax == lmax
        assert response.omega.shape == (2,)
        assert response.sl.lmax == lmax



    

    