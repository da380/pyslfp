from pyshtools import SHCoeffs, SHGrid
import numpy as np

if __name__ == "__main__":
    pass


class ResponseBase:

    def __init__(self):        
        pass            

    @property
    def u(self):
        return self._u
    
    @u.setter
    def u(self, value):
        self._u = value

    @property
    def phi(self):
        return self._phi
    
    @phi.setter
    def phi(self, value):
        self._phi = value        

    @property
    def omega(self):
        return self._omega
    
    @omega.setter
    def omega(self, value):
        self._omega = value

    @property
    def sl(self):
        return self._sl

    @sl.setter
    def sl(self,value):
        self._sl = value

    def __add__(self, other):
        u = self.u + other.u
        phi = self.phi + other.phi        
        omega = self.omega + other.omega
        sl = self.sl + other.sl
        return type(self)(u, phi, omega, sl)

    def __sub__(self, other):
        u = self.u - other.u
        phi = self.phi - other.phi
        omega = self.omega - other.omega
        sl = self.sl - other.sl
        return type(self)(u, phi, omega, sl)

    def __mul__(self, s):
        u = s * self.u 
        phi = s * self.phi 
        omega = s * self.omega
        sl = s* self.sl
        return type(self)(u, phi, omega, sl)        

    def __rmul__(self,s):
        return self * s

    def __div__(self, s):
        return self * (1/s)


class ResponseFields(ResponseBase):

    def __init__(self, u, phi, omega, sl):
        self._u = u
        self._phi = phi
        self._omega = omega
        self._sl = sl
        


    @staticmethod
    def from_zeros(lmax, /, *, grid = "DH", extend=True):
        u = SHGrid.from_zeros(lmax, grid=grid, extend=extend)
        phi = u.copy()
        sl = u.copy()
        omega = np.zeros(2)
        return ResponseFields(u, phi, omega, sl)    


    def _expand(self, f, /, *, normalization="ortho", csphase=1):
        return f.expand(normalization=normalization, csphase=csphase)

    def expand(self, /, *, normalization="ortho", csphase=1):
        u = self._expand(self.u, normalization=normalization, csphase=csphase)
        phi = self._expand(self.phi, normalization=normalization, csphase=csphase)
        sl = self._expand(self.sl, normalization=normalization, csphase=csphase)
        return ResponseCoefficients(u, phi, self.omega, sl)


class ResponseCoefficients(ResponseBase):

    def __init__(self, u, phi, omega, sl):                            
        self._u = u
        self._phi = phi
        self._omega = omega
        self._sl = sl
        
    
    @staticmethod
    def from_zeros(lmax, /, *, normalization="ortho", csphase=1):
        u = SHCoeffs.from_zeros(lmax, normalization=normalization, csphase=csphase)
        phi = u.copy()
        sl = u.copy()
        omega = np.zeros(2)
        return ResponseCoefficients(u, phi, omega, sl)    

    def _expand(self, f, /, *, grid="DH", extend="True"):
        return f.expand(grid=grid, extend=extend)

    def expand(self, /, *, grid="DH", extend="True"):
        u = self._expand(self.u, grid=grid, extend=extend)
        phi = self._expand(self.phi, grid=grid, extend=extend)
        sl = self._expand(self.sl, grid=grid, extend=extend)
        return ResponseFields(u, phi, self.omega, sl)

    
