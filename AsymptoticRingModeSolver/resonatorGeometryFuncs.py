import numpy as np
import math

class Clothoid3:
    def __init__(self, clothoidPars):
        """ Class describing a 3-bend clothoid triangle resonator """
        self.resonatorLength = clothoidPars["resonatorLength"]                      # Length of the resonator (m)
        self.clothoidOrder = clothoidPars["bendOrder"]                              # Order of the clothoid bend
        self.curveFraction = clothoidPars["curveFraction"]                          # Fraction of the resonator length comprized of curved section (remaining length correspond to straight sections between the curves)
        self.couplingLengthFraction = clothoidPars["couplingLengthFraction"]        # Fractional length of one triangle side coupled to the input/output waveguide
        
        #self.couplingLength = self.resonatorLength / 3
        self.curvatureRef = 0                                                       # Reference curvature when calculating mode properties in the structure
        
    @property
    def a_val(self):
        """ Single bend clothoid a parameter """
        return ((math.pi / 3)**(1/(self.clothoidOrder + 1))) / (self.curveFraction * self.resonatorLength / 2)
    
    def IsCoupled(self, z):
        """ Returns True is the given z point is found within the ring/waveguide coupling region """
        if (z < (self.resonatorLength * (1 - self.couplingLengthFraction) / 6) or z > (self.resonatorLength * (1 + self.couplingLengthFraction) / 6)):
            """ Point is not in the coupling region """
            return False
        
        """ Point is within the coupling region """
        return True
        
    def GetRelativeZ(self, z):
        """ Return the position of the z point relative to the zero curvature point of the nearest clothoid spiral """
        z_ref = z % (self.resonatorLength / 3)
        
        if (z_ref <= self.curveFraction*self.resonatorLength/6):
            """ z point is in an 'out curve' """
            return self.curveFraction*self.resonatorLength/6 - z_ref
        elif (z_ref > (1 - self.curveFraction/2)*self.resonatorLength/3):
            """ z point is in an 'in curve' """
            return z_ref - (1 - self.curveFraction/2)*self.resonatorLength/3
        
        """ z point is in a straight section """
        return 0
    
    def Curvature(self, z):
        """ Instantaneous curvature at the provided z point """
        z_ref = self.GetRelativeZ(z)
        
        return (self.clothoidOrder + 1) * self.a_val * ((self.a_val * z_ref)**self.clothoidOrder)
        
    def CurveDeriv(self, z):
        """ Value of the curvature spatial derivative at the provided z point """
        z_temp = z % (self.resonatorLength / 3)
        z_ref = self.GetRelativeZ(z)
        if (z_ref == 0 and self.curveFraction < 1): return 0
        
        derivMagnitude = self.clothoidOrder * (self.clothoidOrder + 1) * (self.a_val**2) * ((self.a_val * z_ref)**(self.clothoidOrder-1))
        
        # Add a minus sign if the point corresponds to an 'out bend'
        curveDeriv = -derivMagnitude if (z_temp < self.curveFraction*self.resonatorLength/6) else derivMagnitude
        
        return curveDeriv
    
    def Separation(self, z):
        """ Get the hight of the clothoid spiral relative to the tangent at the zero curvature point """
        z_ref = self.GetRelativeZ(z)
        l_0 = self.a_val * z_ref
        
        y = 0
        expansionOrder = 4      # Maximum order in the Taylor series expansion for the curve height
        for index in range(expansionOrder): 
            y += (1 / self.a_val) * ((-1)**index) * ((l_0)**((2*index+1)*(self.clothoidOrder+1)+1)) / (((2*index+1)*(self.clothoidOrder+1)+1) * math.factorial(2*index+1))
            
        return y
        
    

class Ring:
    def __init__(self, ringPars):
        """ Class describing a ring resonator of constant radius """
        self.resonatorLength = ringPars["resonatorLength"]                      # Length of the resonator (m)
        self.couplingLengthFraction = ringPars["couplingLengthFraction"]        # Length of the coupling region relative to the length of the ring
        #self.theta_min = ringPars["theta_min"]                                  # Angle corresponding to the starting point of the coupling region relative to the the point of nearest separtion to the waveguide
        #self.theta_max = ringPars["theta_max"]                                  # Angle corresponding to the ending point of the coupling region relative to the the point of nearest separtion to the waveguide
        
        #self.ringRadius = self.resonatorLength / (2*math.pi)
        self.curvatureRef = self.Curvature(0)                                   # Reference curvature when calculating mode properties in the structure
        #self.couplingLength = (self.theta_max - self.theta_min) * self.ringRadius
        
    @property
    def ringRadius(self):
        """ Radius of the ring """
        return self.resonatorLength / (2*math.pi)
        
    @property
    def theta_min(self):
        """ Angle corresponding to the starting point of the coupling region relative to the the point of nearest separtion to the waveguide """
        return -self.resonatorLength * self.couplingLengthFraction / (2 * self.ringRadius)
        
    @property
    def theta_max(self):
        """ Angle corresponding to the ending point of the coupling region relative to the the point of nearest separtion to the waveguide """
        return self.resonatorLength * self.couplingLengthFraction / (2 * self.ringRadius)
        
    @property
    def couplingLength(self):
        """ Length of the coupling region shared between the ring and the input/output waveguide """
        return (self.theta_max - self.theta_min) * self.ringRadius
        
    def IsCoupled(self, z):
        """ Returns True is the given z point is found within the ring/waveguide coupling region """
        
        if ((z < 0) or (z > self.couplingLength)):
            """ z is outside the coupling region """
            return False
        
        """ z is within the coupling region """
        return True
        
    def Curvature(self, z):
        """ Instantaneous curvature at the given z point (for this structure, it is independent of z) """
        return 1 / self.ringRadius
    
    def CurveDeriv(self, z):
        """ Value of the curvature spatial derivative at the provided z point (for this structure, it is independent of z) """
        return 0
    
    def Separation(self, z):
        """ Get the hight of the ring relative to the tangent at the point of closest aproach to the input/output waveguide """
        z_ref = z % self.resonatorLength
        theta = self.theta_min + z_ref / self.ringRadius
        
        return (1 - math.cos(theta)) * self.ringRadius