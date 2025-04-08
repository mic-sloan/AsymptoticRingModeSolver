import numpy as np
import math 
import json
from matplotlib import pyplot as plt
import AsymptoticRingModeSolver.resonatorGeometryFuncs

# Define some global constants
C_CONST = 2.998e8                   # Speed of light in vacuum (m/s)
hbar = 1.054e-34                    # Reduced Plank's constant (Js)

class SystemParameters:
    def __init__(self, parDict):
        """ Class which describes the properties of the coupled resonator system """
        self.resonator = parDict["resonator"]                                                   # Resonator structure class
        self.lambda_ref = parDict["lambda"]                                                     # Reference wavelength (used to identify the n=0 resonance) (m)
        self.meshSize = parDict["meshSize"]                                                     # Number of points in the resonator position mesh
        self.omega_c = parDict["omega_c"]                                                       # Waveguide/resonator coupling parameter (TAKEN TO BE CONSTANT, WILL BE GENERALIZED IN FUTUR VERSION)
        self.scatteringLoss_dB = np.array(parDict["scatteringLoss_dB"], dtype=float)            # Phenomenological scattering attenuation factor (dB/cm)
        self.modePars_raw = parDict["modePars_raw"]                                             # 'Raw' mode property fitting parameters
        self.waveguideWidth = parDict["waveguideWidth"]                                         # Cross-sectional width of the waveguide/resonator
        
        self.numRails = 2                                                                       # Number of photonic elements (rails) (Here: input/output waveguide, resonator)
        self.Ni = 2                                                                             # Number of spatial modes (Here fixed to TE00 and TE01)
        
        
    @property
    def waveguideWidth(self):
        """ Cross-sectional width of the waveguide/resonator """
        return self._waveguideWidth
    
    @waveguideWidth.setter
    def waveguideWidth(self, width):
        """ 
        Sets the waveguide with, then uses the value to derive the Taylor series coefficients of the mode properties. 
        
        Input:
            - width : Width of the waveguide/resonator cross-section (um)
        """
        self._waveguideWidth = width
        self.GenerateFittedParameters()
        self.GenerateResonatorMesh()
        
    def GenerateFittedParameters(self):
        """ Construct the matrix of Taylor series coefficients for the effective refractive index, radiation loss, and the overlap strength for the given cross-sectional width """
        self.neffMat = np.array(np.real([self.ConstructParameterMatrix(self.modePars_raw["neffPars_mode1_raw"]), 
                                         self.ConstructParameterMatrix(self.modePars_raw["neffPars_mode2_raw"])]), dtype=float)
        
        self.radLossMat = np.array(np.real([self.ConstructParameterMatrix(self.modePars_raw["radLossPars_mode1_raw"]), 
                                            self.ConstructParameterMatrix(self.modePars_raw["radLossPars_mode2_raw"])]), dtype=float)
        
        self.HOMcouplingMat = np.array([[self.ConstructParameterMatrix(self.modePars_raw["selfHOMcoupling_mode1_raw"]), self.ConstructParameterMatrix(self.modePars_raw["crossHOMcoupling_mode1_raw"])], 
                                        [self.ConstructParameterMatrix(self.modePars_raw["crossHOMcoupling_mode2_raw"]), self.ConstructParameterMatrix(self.modePars_raw["selfHOMcoupling_mode2_raw"])]], dtype=complex)
        
    def ConstructParameterMatrix(self, pars_raw):
        """ 
        Converts the raw fitting parameters (pars_raw) into Taylor seris coefficients.
        
        Input:
            - pars_raw (nx, ny, nz) : Matrix corrsponding to the fitting parameters for the wavelength (nx), curvature (ny), and waveguide cross-sectional width (nz) 
            
        Output:
            - parsMat (mx, my) : Matrix containing the Taylor series coefficients in terms of the wavelength (mx) and curvature (my), for the pre-set value of waveguide width (in um)
        """
        pars = np.array(pars_raw, dtype=complex)
        nx, ny, nz = pars.shape
        
        parsMat = np.zeros((nx, ny), dtype=complex)
        for widthIndex in range(nz): parsMat += pars[:, :, widthIndex] * (self.waveguideWidth**widthIndex)
        
        return parsMat
    
    def GenerateResonatorMesh(self):
        """ Constructs a 'mesh' of mode and structure properties at each position point with the resonator """
        self.meshPositions = np.linspace(0, self.resonator.resonatorLength, self.meshSize+1)
        self.curvatureMesh = np.zeros(self.meshSize, dtype=np.float32)
        self.curveDerivMesh = np.zeros(self.meshSize, dtype=np.float32)
        self.separationMesh = np.zeros(self.meshSize, dtype=np.float32)
        
        for meshIndex in range(self.meshSize):
            z_temp = (self.meshPositions[meshIndex] + self.meshPositions[meshIndex + 1]) / 2
            self.curvatureMesh[meshIndex] = self.resonator.Curvature(z_temp)
            self.curveDerivMesh[meshIndex] = self.resonator.CurveDeriv(z_temp)
            self.separationMesh[meshIndex] = self.resonator.Separation(z_temp)
    
    def GetFittedValue(self, parMat, wavelength, curvature):
        """ 
        Extracts the fitted function values from the given Taylor series coefficients (parMat) and the given wavelength and curvature 
        
        Input:
            - parMat (nx, ny) : Matrix containing the Taylor series coefficients for the given mode property in terms of wavelength (nx) and curvature (ny).
            - wavelength : Wavelength of the field (in m).
            - curvature : Instaneous curvature of the structure (in m^-1).
            
        Output:
            - Value of the function described by (parMat) at the provided wavelength and curvature. 
        """
        n_wl, n_cv = parMat.shape
        wl = wavelength * (1e6)                 # Convert from (m) --> (um)
        cv = curvature * (1e-6)                 # Convert from (m^-1) --> (um^-1)
        
        wl_vec = np.power(wl * np.ones(n_wl), np.arange(n_wl))
        cv_vec = np.power(cv * np.ones(n_cv), np.arange(n_cv))
        
        return wl_vec @ parMat @ cv_vec
    
    def couplingStrength(self, z):
        """ 
        Coupling strength between the waveguide/resonator 
        
        Input:
            - z : Position along the waveguide/resonator structure (in m)
            
        Output:
            - Coupling coefficient between the waveguide and resonator (in Hz)
        """
        return self.omega_c if (self.resonator.IsCoupled(z)) else 0
    
    def Neff(self, modeIndex, wavelength, curvature):
        """ 
        Effective index for the given mode index, wavelength, and bend curvature
        
        Input:
            - modeIndex : Spatial mode (0 = TE00, 1 = TE01).
            - wavelength : Wavelength of the field (in m)
            - curvature : Instantaneous curvature of the structure (in m^-1)
            
        Output:
            - Effective refractive index for the provided mode index, wavelength, and curvature 
        """
        return self.GetFittedValue(self.neffMat[modeIndex, :, :], wavelength, curvature)
    
    def Ng(self, modeIndex, wavelength, curvature):
        """ 
        Computes the group index from the effective refractive index fitting parameters 
        
        Input:
            - modeIndex : Spatial mode (0 = TE00, 1 = TE01).
            - wavelength : Wavelength of the field (in m)
            - curvature : Instantaneous curvature of the structure (in m^-1)
            
        Output:
            - Effective group index for the provided mode index, wavelength, and curvature 
        """
        n_wl, n_cv = (self.neffMat[modeIndex, :, :]).shape
        wl = wavelength * (1e6)             # Convert from (m) --> (um)
        cv = curvature * (1e-6)               # Convert from (m^-1) --> (um^-1)
        
        dwl_vec = np.arange(1, n_wl) * np.power(wl * np.ones(n_wl - 1), np.arange(1, n_wl))
        cv_vec = np.power(cv * np.ones(n_cv), np.arange(n_cv))
        dn = dwl_vec @ self.neffMat[modeIndex, 1:, :] @ cv_vec
        
        return (self.Neff(modeIndex, wavelength, curvature) - wavelength * dn)
    
    def Attenuation_dB(self, modeIndex, wavelength, curvature):
        """ 
        Net amplitude attenuation coefficient from radiation and scattering loss (in dB/cm) 
        
        Input:
            - modeIndex : Spatial mode (0 = TE00, 1 = TE01).
            - wavelength : Wavelength of the field (in m)
            - curvature : Instantaneous curvature of the structure (in m^-1)
            
        Output:
            - Effective amplitude attenuation coefficient for the provided mode index, wavelength, and curvature (in dB/cm) 
        """
        return self.scatteringLoss_dB[modeIndex] + self.GetFittedValue(self.radLossMat[modeIndex, :, :], wavelength, curvature)
    
    def OverlapStrength(self, modeIndex1, modeIndex2, wavelength, curvature):
        """ 
        Overlap coupling strength between the modes with mode index (modeIndex1) and (modeIndex2) for the given wavelength ans curvature 
        
        Input:
            - modeIndex : Spatial mode (0 = TE00, 1 = TE01).
            - wavelength : Wavelength of the field (in m)
            - curvature : Instantaneous curvature of the structure (in m^-1)
            
        Output:
            - Overlap strength between the modes with the provided mode indices, wavelength, and curvature (in m)
        """
        return self.GetFittedValue(self.HOMcouplingMat[modeIndex1, modeIndex2, :, :], wavelength, curvature) * (1e-6)
    
    def GetEffectiveRingLength(self, lambda_r):
        """ 
        Compute the effective optical path length dus to variation in the refractive index and group velocity in the resonator bends 
        
        Input:
            - lambda_r : Wavelength of the field (m)
            
        Output:
            - Leff : Effective optical pathlength of the resonator at the given wavelength (in m)
        """
        dz = self.resonator.resonatorLength / self.meshSize
        
        phase_tot = 0
        for zIndex in range(self.meshSize): 
            phase_tot += self.Neff(0, lambda_r, self.curvatureMesh[zIndex]) * (2*math.pi) * (3e8) * dz / lambda_r
        
        centralNeff = self.Neff(0, lambda_r, self.resonator.curvatureRef)
        k_ref = centralNeff * (2*math.pi) * (3e8) / lambda_r
        Leff = phase_tot * (1e6) / (k_ref)          # in (um)
        
        return Leff
    
    def FindResonantLambda(self, modeIndex, curvature, n):
        """ 
        Finds the wavelength correpsonding to a coupled waveguide and resonator resonance 
        
        Input:
            - modeIndex : Spatial mode (0 = TE00, 1 = TE01)
            - curvature : Instantaneous curvature of the structure (in m^-1)
            - n : Resonance index relative to the system zero index (resonance closest to self.lambda_ref)
            
        Output:
            - lambda_est : Estimated lambda corresponding to the resonance n away from the system zero resonance index (in m)
        """
        # Make an inital guess of the resonant lambda
        lambda_0 = self.lambda_ref
        neff_0 = self.Neff(modeIndex, lambda_0, curvature)
        L_0 = self.GetEffectiveRingLength(lambda_0) * (1e-6)
        
        m_0 = np.floor(neff_0 * L_0 / lambda_0)
        resIndex = m_0 + n
        
        # Iteratively update the guess of the resonance lambda until a given tolerence is achieved
        lambda_est = lambda_0
        flag = False
        tol = 1e-3
        counter = 0
        while (flag == False):
            lambda_est = self.Neff(modeIndex, lambda_est, curvature) * self.GetEffectiveRingLength(lambda_est) * (1e-6) / resIndex
            k_est = 2 * np.pi * (self.Neff(modeIndex, lambda_est, curvature) / lambda_est)
            err = 2 * np.pi * resIndex - k_est * self.GetEffectiveRingLength(lambda_est) * (1e-6)
            
            counter = counter + 1
            
            if (abs(err) < tol):
                flag = True
                
            if (counter > 1000):
                print("Couldn't find the resonant lambda. Returning first lambda guess...")
                return self.Neff(modeIndex, lambda_0, curvature) * self.GetEffectiveRingLength(lambda_0) * (1e-6) / resIndex
            
        return lambda_est
    
    def PlotNeffDiff(self, wavelength=-1):
        """ 
        Plots the difference in the effective refractive index between the TE00 and TE01 mode 
        
        Input:
            - (Optional) wavelength : Wavelength used when evaluating the refractive index difference (defaults to system reference lambda) (in m)
        """
        if (wavelength == -1): wavelength = self.lambda_ref
        
        curvatureVec = np.linspace(0, 1/(20e-6))
        NeffDiffVec = [(self.Neff(0, wavelength, curvature) - self.Neff(1, wavelength, curvature)) for curvature in curvatureVec] 
        
        plt.plot(curvatureVec * (1e-6), NeffDiffVec)
        plt.xlabel("Curvature (um^-1)")
        plt.ylabel("Neff Difference")
        plt.title("Effective refractive Index Difference (TE00 - TE01)")
        plt.show()
        
    def PlotOverlapStrength(self, wavelength=-1):
        """ 
        Plots the overlap strength between the TE00 and TE01 mode 
        
        Input:
            - (Optional) wavelength : Wavelength used when evaluating the refractive index difference (defaults to system reference lambda) (in m)
        """
        if (wavelength == -1): wavelength = self.lambda_ref 
        
        curvatureVec = np.linspace(0, 1/(20e-6))
        OverlapDiffVec = [self.OverlapStrength(0, 1, wavelength, curvature) for curvature in curvatureVec] 
        
        plt.plot(curvatureVec * (1e-6), np.absolute(OverlapDiffVec))
        plt.xlabel("Curvature (um^-1)")
        plt.ylabel("abs(Overlap Strength) (m)")
        plt.title("TE00 --> TE01 Overlap Strength")
        plt.show()
        
    def PlotResonatorCurvature(self):
        """ Plot the resonator curvature and curvature derivative as a function of relartive resonator position """
        fig, ax = plt.subplots(2, 1, constrained_layout=True)
        positionAxis = (self.meshPositions[1:] + self.meshPositions[:self.meshSize]) / (2 * self.resonator.resonatorLength)
        
        ax[0].plot(positionAxis, self.curvatureMesh * (1e-6))
        ax[0].set_title("Resonator Curvature")
        ax[0].set_ylabel("Curvature (um^-1)")
        ax[0].set_xlabel("Relative Resonator Position")
        
        ax[1].plot(positionAxis, self.curveDerivMesh * (1e-6)**2)
        ax[1].set_title("Curvature Derivative")
        ax[1].set_ylabel("Curvature Derivative (um^-2)")
        ax[1].set_xlabel("Relative Resonator Position")
        
        plt.show()