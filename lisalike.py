# Functions relevant for LISA-like orbits
#
# A LISA-like orbit has 3 satellites in a triangular configuration.
# Eccentricity and inclination combine to make the triangle 
# tumble with a minimal amount of variation in the arm lengths

# See, e.g., K Rajesh Nayak et al, Class. Quantum Grav. 23, 1763 (2006).
import math
from constants import constants as k

import numpy as np
from scipy.optimize import newton

class lisalike:
    
    # Defaults: orbit average radius = 1.0 AU
    #           orbit angualr velocity average = omegaEarthPerDay
    #           delta = 0.0
    def __init__(self, triangleSideLength=k.rLISAm/k.mPerAU, orbitRadius=1.0, orbitFrequency=k.omegaEarthPerDay, delta=0.0):
        self.triangleSideLength = triangleSideLength
        self.orbitRadius = orbitRadius
        self.orbitFrequency = orbitFrequency
        
        # delta is a (small) correction to the orbital plane tilt, as in 
        # Fig. 1 of Nayak (2006).
        self.delta = delta

        self.alpha = self.triangleSideLength / (2.0 * orbitRadius)
        
        # Eq. (1). of Nayak (2006)
        tanInclinationNumerator = (2.0*self.alpha/math.sqrt(3)) * math.sin(math.pi/3.0 + self.delta)
        tanInclinationDenominator = (1.0 + (2.0 * self.alpha / math.sqrt(3)) * math.cos(math.pi/3.0 + self.delta))
        self.inclination = math.atan2(tanInclinationNumerator, tanInclinationDenominator)
        
        self.eccentricity = math.sqrt(1+(4.0/3.0)*self.alpha*self.alpha + (4.0/math.sqrt(3.0))*self.alpha * math.cos(math.pi/3.0 + self.delta)) - 1.0
        
        self.lastEccentricAnomaly = 0.0
    
    # Sigma is a phase that depends on whichSatellite = 1,2,3
    def sigma(self, whichSatellite):
        return float(whichSatellite - 1) * 2.0 * math.pi / 3.0
        
    # Returns a function of eccentric anomaly
    # Numerically find the root of this function to get the eccentric anomaly
    def getEccentricAnomalyFunction(self, time, whichSatellite):
        def eccentricAnomalyFunction(eccentricAnomaly):
            return eccentricAnomaly + self.eccentricity * math.sin(eccentricAnomaly) - self.orbitFrequency * time + self.sigma(whichSatellite)
        return eccentricAnomalyFunction
    
    def getEccentricAnomaly(self, time, whichSatellite, x0=0.0):
        return newton(self.getEccentricAnomalyFunction(time, whichSatellite), x0, tol=1.e-15, maxiter=int(1e9))
    
    # $\Omega t - \sigma_k - \psi_k = e \sin\psi_k$
    # $\Omega - \frac{d\psi_k}{dt} = e \cos\psi_k \frac{d\psi_k}{dt}$
    # $\frac{d\psi_k}{dt} = \Omega / (1 + e \cos\psi_k)$
    def dtEccentricAnomaly(self, time, whichSatellite, x0=0.0):
        eccentricAnomaly = self.getEccentricAnomaly(time, whichSatellite, x0)
        return self.orbitFrequency / (1.0 + self.eccentricity * math.cos(eccentricAnomaly))
        
    def position1(self, time, whichSatellite, x0=0.0):
        eccentricAnomaly = self.getEccentricAnomaly(time, whichSatellite, x0)
        temp = (math.cos(eccentricAnomaly)+self.eccentricity)
        position1 = np.array([self.orbitRadius * temp * math.cos(self.inclination),
                              self.orbitRadius * math.sin(eccentricAnomaly)*math.sqrt(1.0-self.eccentricity**2),
                              self.orbitRadius * temp * math.sin(self.inclination)])
        return position1
    
    # Note: whichSatellite = 1,2,3 for the 3 satellites
    def position(self, time, whichSatellite, x0=0.0):
        position1 = self.position1(time, whichSatellite, x0)
        sigma = self.sigma(whichSatellite)
        position = np.array([position1[0] * math.cos(sigma) - position1[1] * math.sin(sigma),
                            position1[0] * math.sin(sigma) + position1[1] * math.cos(sigma),
                            position1[2]])
        return position
    
    def relativePosition(self, time, whichSatellite, x0=0.0):
        position = self.position(time, whichSatellite, x0)
        return position - self.orbitRadius * np.array([math.cos(self.orbitFrequency * time),
                                                       math.sin(self.orbitFrequency * time),
                                                       0.0])
    
    def velocity1(self, time, whichSatellite, x0=0.0):
        eccentricAnomaly = self.getEccentricAnomaly(time, whichSatellite, x0)
        velocity1 = self.orbitRadius * np.array([-1.0 * math.sin(eccentricAnomaly) * math.cos(self.inclination),
                                                 math.cos(eccentricAnomaly)*math.sqrt(1.0-self.eccentricity**2),
                                                 -1.0 * math.sin(eccentricAnomaly) * math.sin(self.inclination)
                                                 ])
        velocity1 *= self.dtEccentricAnomaly(time, 1, x0)
        return velocity1
    
    def velocity(self, time, whichSatellite, x0=0.0):
        velocity1 = self.velocity1(time, whichSatellite, x0)
        sigma = self.sigma(whichSatellite)
        velocity = np.array([velocity1[0] * math.cos(sigma) - velocity1[1] * math.sin(sigma),
                            velocity1[0] * math.sin(sigma) + velocity1[1] * math.cos(sigma),
                            velocity1[2]])
        return velocity
    
    def relativeVelocity(self, time, whichSatellite, x0=0.0):
        velocity = self.velocity(time, whichSatellite, x0=0.0)
        return velocity - self.orbitRadius * self.orbitFrequency * np.array([-1.0* math.sin(self.orbitFrequency * time),
                                                                             math.cos(self.orbitFrequency * time),
                                                                             0.0])
    
        
        
        
