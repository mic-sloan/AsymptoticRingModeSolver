import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from matplotlib import animation as ani
from tqdm import tqdm
from AsymptoticRingModeSolver.systemParameters import SystemParameters
from scipy.linalg import expm

C_CONST = 3e8                 # Speed of light in vacuum (m/s)
hbar = 1.054e-34            # Reduced Plank's constant (Js)

class Resonance:
    def __init__(self, resPars):
        """ Class describing a given resonance (or set of resonances) """
        self.systemPars = resPars["systemPars"]                                                                                             # System parameters class
        self.lambda0 = resPars["lambda0"] if ("lambda0" in resPars) else self.systemPars.FindResonantLambda(0, self.systemPars.resonator.curvatureRef, resPars["Nres"])     # Central wavelength (m)
        self.omegaRange = resPars["omegaRange"]                                                                                             # Width of the range of frequency values (2pi Hz)
        self.Nf = resPars["Nf"]                                                                                                             # Number of frequency points
        self.generateAsyFields = True if ("generateAsyFields" not in resPars) else resPars["generateAsyFields"]                             # If False, skips the generation of the full asymptotic field distributions
        self.generateSinglePassFields = True if ("generateSinglePassFields" not in resPars) else resPars["generateSinglePassFields"]        # If False, skips the computation of the single pass resonator fields
        
        self.omegaJ = (2 * math.pi * C_CONST) / self.lambda0                                                                # Central frequency value (2pi Hz)
        self.omega_vec = np.linspace(self.omegaJ - self.omegaRange/2, self.omegaJ + self.omegaRange/2, self.Nf)             # Vector of frequency values (2pi Hz)
        
        self.InitializeFields()
        
    @property
    def lambda_vec(self):
        """ Vector of wavelength values for the resonance (in m) """
        return (2 * math.pi * C_CONST) / self.omega_vec
    
    def InitializeFields(self, hideProgress=False):
        """ 
        Computes the mode properties at the central resonance frequency, then solves for the asymptotic and single pass field amplitudes 
        
        Input:
            - hideProgress : If true, hides the progress bar displayed during the computation.
        """
        self.SetReferenceValues()
        self.GenerateResonanceMesh()
        if (self.generateAsyFields == True): self.GenerateAsymptoticFieldDistributions(hideProgress=hideProgress)
        if (self.generateSinglePassFields == True): self.GenerateSinglePassRingAmplitudes(hideProgress=hideProgress)
        
    def ResetWaveguideWidth(self, newWidth):
        """ 
        Resets the waveguide with within the systemPars class, then recomputes the derived values
        
        Input:
            - newWidth : New values of the waveguide/resonator cross-sectional width (in um)
        """
        self.systemPars.waveguideWidth = newWidth
        self.InitializeFields(hideProgress=True)
        
    def SetReferenceValues(self):
        """ Set the mode properties at the central frequency value of the resonance """
        self.neff_ref = np.array([[self.systemPars.Neff(0, self.lambda0, 0), self.systemPars.Neff(0, self.lambda0, self.systemPars.resonator.curvatureRef)],
                                  [self.systemPars.Neff(1, self.lambda0, 0), self.systemPars.Neff(1, self.lambda0, self.systemPars.resonator.curvatureRef)]], dtype=np.float32)
        self.kJ = np.array([[self.neff_ref[0,0] * self.omegaJ / C_CONST, self.neff_ref[0,1] * self.omegaJ / C_CONST],
                            [self.neff_ref[1,0] * self.omegaJ / C_CONST, self.neff_ref[1,1] * self.omegaJ / C_CONST]], dtype=np.float32)
        self.ng_wg = np.array([self.systemPars.Ng(0, self.lambda0, 0), self.systemPars.Ng(1, self.lambda0, 0)], dtype=np.float32)
        self.loss_wg = np.array([self.systemPars.Attenuation_dB(0, self.lambda0, 0), self.systemPars.Attenuation_dB(1, self.lambda0, 0)], dtype=np.float32)
        
    def GenerateResonanceMesh(self):
        """ Generate a 'mesh' of mode properties at each of the positions along the length of the resonantor """
        self.neffMesh = np.zeros((self.systemPars.Ni, self.systemPars.meshSize), dtype=np.float32)
        self.ngMesh = np.zeros((self.systemPars.Ni, self.systemPars.meshSize), dtype=np.float32)
        self.lossMesh = np.zeros((self.systemPars.Ni, self.systemPars.meshSize), dtype=np.float32)
        self.HOMcouplingMesh = np.zeros(self.systemPars.meshSize, dtype=np.complex128)
        self.couplingStrengthMesh = np.zeros(self.systemPars.meshSize, dtype=np.complex128)
        
        for meshIndex in range(self.systemPars.meshSize):
            self.HOMcouplingMesh[meshIndex] = 1j * self.systemPars.OverlapStrength(0, 1, self.lambda0, self.systemPars.curvatureMesh[meshIndex]) * self.systemPars.curveDerivMesh[meshIndex]
            self.couplingStrengthMesh[meshIndex] = self.systemPars.couplingStrength((self.systemPars.meshPositions[meshIndex] + self.systemPars.meshPositions[meshIndex+1])/2)
            
            for modeIndex in range(self.systemPars.Ni):
                self.neffMesh[modeIndex, meshIndex] = self.systemPars.Neff(modeIndex, self.lambda0, self.systemPars.curvatureMesh[meshIndex])
                self.ngMesh[modeIndex, meshIndex] = self.systemPars.Ng(modeIndex, self.lambda0, self.systemPars.curvatureMesh[meshIndex])
                self.lossMesh[modeIndex, meshIndex] = 100 * (self.systemPars.Attenuation_dB(modeIndex, self.lambda0, self.systemPars.curvatureMesh[meshIndex]) / 20) * math.log(10)
    
    def PropagateFieldAmplitude(self, delOmega, noHOMcoupling=False, noWaveguideCoupling=False):
        """ 
        Perform the field propagation along the full length of the coupled waveguide and resonator system 
        
        Input:
            - delOmega : Detuning of the field frequency from the resonance center (2pi Hz)
            - noHOMcoupling : If True, turns off the methods which implement the higher order mode coupling in the resonator
            - noWaveguideCoupling : If True, turns off the methods which implement the coupling between the input/output waveguide and the resonator
            
        Output:
            - propagationMats (nx, ny, nz) : Net transfer matrix for an input in mode (nz) and output in mode (ny) at the position index (nx). Ordering of the mode indices are [TE00 waveguide, TE01 waveguide, TE00 resonator, TE01 resonator]
        """
        Ni = self.systemPars.Ni; Nr = self.systemPars.numRails; Nm = self.systemPars.meshSize
        
        vg_00 = C_CONST / self.ng_wg[0]; vg_10 = C_CONST / self.ng_wg[1]
        neff_00 = self.neff_ref[0,0]; neff_10 = self.neff_ref[1,0]
        gamma_00 = self.loss_wg[0]; gamma_10 = self.loss_wg[1] 
        propagationMats = np.zeros((Nm, Ni * Nr, Ni * Nr), dtype=np.complex128)   
        for stepIndex in range(Nm):
            z_i = self.systemPars.meshPositions[stepIndex]; z_f = self.systemPars.meshPositions[stepIndex+1]
            dz = z_f - z_i
            tempPropMat = np.zeros((Ni * Nr, Ni * Nr), dtype=np.complex128)
            
            # Fill in the linear terms
            kJ_ave = (self.kJ[0,0] + self.kJ[0,1] + self.kJ[1,0] + self.kJ[1,1]) / 4
            vg_01 = C_CONST / self.ngMesh[0, stepIndex]; vg_11 = C_CONST / self.ngMesh[1, stepIndex]
            neff_01 = self.neffMesh[0, stepIndex]; neff_11 = self.neffMesh[1, stepIndex]
            gamma_01 = self.lossMesh[0, stepIndex]; gamma_11 = self.lossMesh[1, stepIndex]
            tempPropMat[0, 0] = -gamma_00 + 1j*(delOmega / vg_00) - 1j*(1 - neff_00 / self.neff_ref[0,0]) * self.kJ[0,0] + 1j*(self.kJ[0,0] - kJ_ave)
            tempPropMat[1, 1] = -gamma_10 + 1j*(delOmega / vg_10) - 1j*(1 - neff_10 / self.neff_ref[1,0]) * self.kJ[1,0] + 1j*(self.kJ[1,0] - kJ_ave)
            tempPropMat[2, 2] = -gamma_01 + 1j*(delOmega / vg_01) - 1j*(1 - neff_01 / self.neff_ref[0,1]) * self.kJ[0,1] + 1j*(self.kJ[0,1] - kJ_ave)
            tempPropMat[3, 3] = -gamma_11 + 1j*(delOmega / vg_11) - 1j*(1 - neff_11 / self.neff_ref[1,1]) * self.kJ[1,1] + 1j*(self.kJ[1,1] - kJ_ave) 
            
            # Fill in the terms for the HOM coupling
            alphaHOM_wg = 0         # Waveguide is straight and thus does not result in HOM coupling from bends
            alphaHOM_ring = 0 if (noHOMcoupling == True) else self.HOMcouplingMesh[stepIndex]
            tempPropMat[0, 1] = -1j * alphaHOM_wg
            tempPropMat[1, 0] = -1j * np.conj(alphaHOM_wg)
            tempPropMat[2, 3] = -1j * alphaHOM_ring
            tempPropMat[3, 2] = -1j * np.conj(alphaHOM_ring)
            
            # Fill in the terms for the waveguide / ring coupling
            omegaCoup = 0 if (noWaveguideCoupling == True) else self.couplingStrengthMesh[stepIndex]          # Assume the coupling between modes 0 and 1 are equal (Should be updated)
            tempPropMat[0, 2] = -1j * omegaCoup / vg_00
            tempPropMat[2, 0] = -1j * np.conj(omegaCoup) / vg_10
            tempPropMat[1, 3] = -1j * omegaCoup / vg_01
            tempPropMat[3, 1] = -1j * np.conj(omegaCoup) / vg_11
                
            # Perform the evolution
            tempPropMat = tempPropMat * dz
            tempPropMat_exp = expm(tempPropMat)
            
            phaseMat_left = np.diag(np.array([np.exp(-1j*(self.kJ[0,0] - kJ_ave)*z_f), np.exp(-1j*(self.kJ[1,0] - kJ_ave)*z_f), np.exp(-1j*(self.kJ[0,1] - kJ_ave)*z_f), np.exp(-1j*(self.kJ[1,1] - kJ_ave)*z_f)], dtype=np.complex128))
            phaseMat_right = np.diag(np.array([np.exp(1j*(self.kJ[0,0] - kJ_ave)*z_i), np.exp(1j*(self.kJ[1,0] - kJ_ave)*z_i), np.exp(1j*(self.kJ[0,1] - kJ_ave)*z_i), np.exp(1j*(self.kJ[1,1] - kJ_ave)*z_i)], dtype=np.complex128))
            if (stepIndex == 0):
                propagationMats[stepIndex, :, :] = phaseMat_left @ tempPropMat_exp @ phaseMat_right
            else:
                propagationMats[stepIndex, :, :] = phaseMat_left @ tempPropMat_exp @ phaseMat_right @ propagationMats[stepIndex - 1, :, :]
        
        return propagationMats
    
    def GenerateAsymptoticFieldDistributions(self, hideProgress=False):
        """ 
        Generate the full asymptotic field distirubutions corresponding to the coupled resonator and waveguide 
        
        Input:
            - (Optional) hideProgress : If True, hides the progress bar shown when performing the calculation.
        """
        Ni = self.systemPars.Ni; Nr = self.systemPars.numRails; Nf = self.Nf; Nm = self.systemPars.meshSize; Lr = self.systemPars.resonator.resonatorLength
        
        self.fieldDist_asyIn = np.zeros((Ni, Ni, Nr, Nf, Nm), dtype=np.complex128)
        self.fieldDist_asyOut = np.zeros((Ni, Ni, Nr, Nf, Nm), dtype=np.complex128)
        
        for omegaIndex in tqdm(range(Nf), disable=hideProgress, desc="Generating asymptotic fields..."):
            DelOmega = self.omega_vec[omegaIndex] - self.omegaJ
            propagationsMatrices_temp = self.PropagateFieldAmplitude(DelOmega)
            
            # Extract the full evolution matrix and add the rapidly varying phase
            RTpropMat = propagationsMatrices_temp[-1, :, :]
            for railIndex in range(Nr):
                for iIndex in range(Ni):
                    RTpropMat[railIndex*Ni + iIndex, :] = RTpropMat[railIndex*Ni + iIndex, :] * np.exp(1j * self.kJ[iIndex, railIndex] * Lr)
            
            # Apply the circulation condition and solve for the ring field amplitudes
            for inMode in range(Ni):
                Mvec = RTpropMat[Ni:, inMode]
                Mmat = np.eye(Ni, dtype=np.complex128) - RTpropMat[Ni:, Ni:]
                bVec = np.linalg.inv(Mmat) @ Mvec
                    
                Svec = np.zeros(Nr * Ni, dtype=np.complex128)
                Svec[inMode] = 1; Svec[Ni:] = bVec[:]
                    
                for stepIndex in range(Nm):
                    tempFieldAmp = propagationsMatrices_temp[stepIndex, :, :] @ Svec
                        
                    for railIndex in range(Nr):
                        for outMode in range(Ni):
                            self.fieldDist_asyIn[inMode, outMode, railIndex, omegaIndex, stepIndex] = tempFieldAmp[railIndex*Ni + outMode]
                            
    def GenerateSinglePassRingAmplitudes(self, hideProgress=False):
        """ 
        Generate the field amplitudes for a single pass of the resonator 
        
        Input:
            - (Optional) hideProgress : If True, hides the progress bar shown when performing the calculation.
        """
        Ni = self.systemPars.Ni; Nr = self.systemPars.numRails; Nf = self.Nf; Nm = self.systemPars.meshSize; Lr = self.systemPars.resonator.resonatorLength
        
        self.singlePassFields = np.zeros((Ni, Ni, Nf, Nm), dtype=complex)
        for omegaIndex in tqdm(range(Nf), disable=hideProgress, desc="Generating single pass fields..."):
            DelOmega = self.omega_vec[omegaIndex] - self.omegaJ
            propagationsMatrices_temp = self.PropagateFieldAmplitude(DelOmega, noWaveguideCoupling=True)
            
            for inMode in range(Ni):
                for outMode in range(Ni):
                    self.singlePassFields[inMode, outMode, omegaIndex, :] = propagationsMatrices_temp[:, Ni + outMode, Ni + inMode]
                    
    def GetSinglePassField(self, freqIndex, posIndexShift=0, maxSteps=-1):
        """ 
        Comopute that field transfer matrices for a single pass of the resonator (including the rapidly varying phase) 
        
        Input:
            - freqIndex : Index of the desired frequency bin.
            - posIndexShift : Shifts the starting point of the evolution by (posIndexShift) steps along the position mesh of the resonator
            - maxSteps : truncates the output to include only (maxSteps) number of propagation steps.
            
        Output:
            - fieldAmp (nx, ny, nz) : Net transfer matrix for an input in mode (nx) and output in mode (ny) at the position index (nz) in the resonator. Ordering of the mode indices are [TE00 resonator, TE01 resonator]
        """
        if (maxSteps < 0): maxSteps = self.systemPars.meshSize
        
        fieldAmp = self.singlePassFields[:, :, freqIndex, :]
        if (posIndexShift > 0):
            Lr = self.systemPars.resonator.resonatorLength
            kJ_ave = np.sum(self.kJ)
            phaseMat = np.diag(np.array([np.exp(1j * (self.kJ[0, 1] - kJ_ave) * Lr), np.exp(1j * (self.kJ[1, 1] - kJ_ave) * Lr)], dtype=complex)) #np.diag(np.array([np.exp(1j * (self.kJ[0, 0] - kJ_ave) * Lr), np.exp(1j * (self.kJ[1, 0] - kJ_ave) * Lr), np.exp(1j * (self.kJ[0, 1] - kJ_ave) * Lr), np.exp(1j * (self.kJ[1, 1] - kJ_ave) * Lr)], dtype=complex))
            
            tempPropMats = np.transpose(self.singlePassFields[:, :, freqIndex, :], (2, 1, 0))
            tempSinglePass = tempPropMats[posIndexShift:, :, :] @ np.linalg.inv(tempPropMats[posIndexShift - 1, :, :])
            tempSinglePass = np.concatenate((tempSinglePass, np.conj(phaseMat) @ tempPropMats[:posIndexShift, :, :] @ phaseMat @ tempSinglePass[-1, :, :]))
            fieldAmp = np.transpose(tempSinglePass, (2, 1, 0)) #tempSinglePass[:maxSteps, outMode, inMode]
            
        for meshIndex in range(self.systemPars.meshSize):
            z = self.systemPars.meshPositions[meshIndex + 1]
            fieldAmp[:, :, meshIndex] = np.diag(np.array([np.exp(1j * self.kJ[0, 1] * z), np.exp(1j * self.kJ[1, 1] * z)])) @ fieldAmp[:, :, meshIndex]
            
        return fieldAmp[:, :, :maxSteps]
            
                    
    def TransmissionSpectrum(self, inMode, outMode):
        """ 
        Return the transmission spectrum for the asymptotic field with a given input mode (inMode) into a given output mode (outMode) 
        
        Input:
            - inMode : Spatial mode input (0 = TE00, 1 = TE01)
            - outMode : Spatial mode output (0 = TE00, 1 = TE01)
            
        Output:
            - Magnitude squared of the output field spectrum at each frequency point.
        """
        return np.absolute(self.fieldDist_asyIn[inMode, outMode, 0, :, -1])**2
    
    def PlotTransmissionSpectrum(self):
        """ Plot the transmission spectras of the asymptotic fields for each input type """
        fig, ax = plt.subplots(2,2)
        fig.set_size_inches(10, 10)
        fig.suptitle(f"Resonant Lambda = {self.lambda0 * 1e9 : .3f} nm")

        for modeIndex1 in range(2):
            for modeIndex2 in range(2):
                ax[modeIndex1, modeIndex2].plot(self.omega_vec - self.omegaJ, self.TransmissionSpectrum(modeIndex1, modeIndex2))
                ax[modeIndex1, modeIndex2].set_title(f"TE0{modeIndex1} --> TE0{modeIndex2}")
                ax[modeIndex1, modeIndex2].set_xlabel("Detuning (2pi Hz)")
                ax[modeIndex1, modeIndex2].set_ylabel("Transmission")

        plt.show()
        
    def RingField(self, inMode, outMode, omegaIndex):
        """ 
        Return the ring field for the given input mode (inMode) into a given output mode (outMode) 
        
        Input:
            - inMode : Spatial mode input (0 = TE00, 1 = TE01)
            - outMode : Spatial mode output (0 = TE00, 1 = TE01)
            - omegaIndex : Frequency bin of the reosonance
            
        Output:
            - Field amplitude of the asymptotic mode in the resonator
        """
        return self.fieldDist_asyIn[inMode, outMode, 1, omegaIndex, :]
    
    def RingFieldAnimation(self, outfileName, interval=100):
        """ 
        Construct an animation of the ring field amplitudes for each input type 
        
        Input:
            - outfileName : Name of the file the animation is to be saved in
            - (Optional) interval : Time between frames.
        """
        fig, ax_amp = plt.subplots(4, 1, constrained_layout=True)
        fig.set_size_inches(10, 10)
        fig.suptitle(f"Resonant Lambda = {self.lambda0 * 1e9 : .3f} nm, \t Delta Omega = {(self.omega_vec[0] - self.omegaJ) / 1e9 : .3f} Hz")
        
        # Plot the lines corresponding to the field amplitude
        lines_amp = []
        positionAxis = self.systemPars.meshPositions[1:] / self.systemPars.resonator.resonatorLength
        for modeIndex_in in range(2):
            for modeIndex_out in range(2):
                y_max = np.max(20 * np.log10(np.absolute(self.fieldDist_asyIn[modeIndex_in, modeIndex_out, 1, :, :])))
                y_min = np.min(20 * np.log10(np.absolute(self.fieldDist_asyIn[modeIndex_in, modeIndex_out, 1, :, :])))
                
                initialField_amp = 20 * np.log10(np.absolute(self.RingField(modeIndex_in, modeIndex_out, 0)))
                lines_amp.append(ax_amp[2*modeIndex_in + modeIndex_out].plot(positionAxis, initialField_amp, color='tab:blue')[0])
                
                ax_amp[2*modeIndex_in + modeIndex_out].set_xlabel("Relative Ring Position")
                ax_amp[2*modeIndex_in + modeIndex_out].set_ylabel("Field Magnitude (dB)", color='tab:blue')
                ax_amp[2*modeIndex_in + modeIndex_out].set_title(f"TE0{modeIndex_out} from TE0{modeIndex_in} input")
                ax_amp[2*modeIndex_in + modeIndex_out].set_ylim([y_min, y_max])
        
        # Plot the lines corresponding to the field phase   
        ax_phase = [0, 0, 0, 0]  
        lines_phase = []
        for modeIndex_in in range(2):
            for modeIndex_out in range(2):
                ax_phase[2*modeIndex_in + modeIndex_out] = ax_amp[2*modeIndex_in + modeIndex_out].twinx()
                
                y_max = np.max(np.unwrap(np.angle(self.fieldDist_asyIn[modeIndex_in, modeIndex_out, 1, :, :])))
                y_min = np.min(np.unwrap(np.angle(self.fieldDist_asyIn[modeIndex_in, modeIndex_out, 1, :, :]))) 
                
                initialField_phase = np.unwrap(np.angle(self.RingField(modeIndex_in, modeIndex_out, 0)))
                lines_phase.append(ax_phase[2*modeIndex_in + modeIndex_out].plot(positionAxis, initialField_phase, color='tab:red')[0])
                
                ax_phase[2*modeIndex_in + modeIndex_out].set_ylabel("Field Phase", color='tab:red')
                ax_phase[2*modeIndex_in + modeIndex_out].set_ylim([y_min, y_max])
            
        def update(frame):
            for tempModeIn in range(2):
                for tempModeOut in range(2):
                    tempField = self.RingField(tempModeIn, tempModeOut, frame)
                    lines_amp[2*tempModeIn + tempModeOut].set_ydata(20 * np.log10(np.absolute(tempField)))
                    lines_phase[2*tempModeIn + tempModeOut].set_ydata(np.unwrap(np.angle(tempField)))
                    
                    fig.suptitle(f"Resonant Lambda = {self.lambda0 * 1e9 : .3f} nm, Delta Omega = {(self.omega_vec[frame] - self.omegaJ) / 1e9 : .3f} Hz")
                
            return (lines_amp, lines_phase)
        
        newAni = ani.FuncAnimation(fig=fig, func=update, frames=self.Nf, interval=interval)
        newAni.save(filename=outfileName)
        plt.close()