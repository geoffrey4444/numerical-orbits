    #0.b Right hand side related functions.

    #Note: units are (AU, days, solar masses).

    # First, let's program in the circular orbits of our test solar system model.

import math
from constants import constants as k
from timestepping import timestepping
from jpl import jpl as jpl

import scipy
import scipy.optimize as optimize
import scipy.integrate
from scipy.integrate import ode
import numpy as np


class orbit:
    
    ##########################################################################
    # Constructor
    ##########################################################################
    
    def __init__(self,
                 system = "L1_2Body",                 
                 massesSatellites = [k.msat1kg/k.msunkg, 
                                 k.msat2kg/k.msunkg, 
                                 k.msat3kg/k.msunkg],
                 earthTrailingDays = 20,
                 gravSoftenEps = 1.e-13,
                 firstTimeToCountFuel = 0.0):
        # What type of system are we solving?
        # Possible values: "L1_2Body", "EarthTrailing_2Body", "EarthTrailing_1Body", "L1", "EarthTrailing"
        self.system = system
        
        # Initialize the satellites        
        self.massesSatellites = massesSatellites
        
        # For earth trailing orbits, how many days behind in earth's orbit are we?
        self.earthTrailingDays = earthTrailingDays 
        
        # Only count fuel accumulation after a certain time
        self.firstTimeToCountFuel = firstTimeToCountFuel
        
        #Length scale below which 
        #neglect gravity to avoid
        #singularities on close
        #encounters
        #UNITS: AU
        self.gravSoftenEps = gravSoftenEps
        
        self.theJPL = jpl() # so we have access to JPL data
        
        # Set up constants
        if system == "L1_2Body":
            self.bodiesAstro = ['earth','sun']
            self.massesAstro = [k.mearthkg/k.msunkg,1.0]
            self.L1Rad2Body = optimize.fsolve(self.lagrangeRootFindFuncTimeZero,
                                              [0.99], xtol=1.e-14)[0]
        elif system == "EarthTrailing_2Body":
            self.bodiesAstro = ['earth','sun']
            self.massesAstro = [k.mearthkg/k.msunkg,1.0]
            
        elif system == "EarthTrailing_1Body":
            self.bodiesAstro = ['sun']
            self.massesAstro = [1.0]
       
        elif system == "L1":
            self.bodiesAstro = self.theJPL.bodies
            self.massesAstro = self.theJPL.bodyMasses
            
        elif system == "EarthTrailing":
            self.bodiesAstro = self.theJPL.bodies
            self.massesAstro = self.theJPL.bodyMasses
        
        else:
            print("WARNING! Defaulting to L1_2Body system in orbit.__init__.")
            system = "L1_2Body"
            self.bodiesAstro = ['earth','sun']
            self.massesAstro = [k.mearthkg/k.msunkg,1.0]
            self.L1Rad2Body = optimize.fsolve(self.lagrangeRootFindFuncTimeZero,
                                              [0.99], xtol=1.e-14)[0]
            self.system = system
        
        
    ##########################################################################
    # Gravitational functions
    ##########################################################################
        
    # Newton's law of gravity
    # At time t and position vector x, what is the gravitational acceleration?
    def FgravPerMass(self, t, x):
        accelVecList = [0.0,0.0,0.0]
        xAstro = self.getAstroPositions(t)
        for N,MN in enumerate(self.massesAstro):
            xN = np.array(xAstro[N])
            rN = np.array(x - xN)
            rSqN = 0.0
            for rjN in rN:
                rSqN += rjN*rjN
            rSqN += self.gravSoftenEps * self.gravSoftenEps
            accelVecList -= np.array((k.GinAU3dm2*MN/(rSqN**1.5))*rN)
        return np.array(accelVecList)

    def FgravPerMassOtherSats(self, nThisSat, fj):
        #How many satellites?
        nSatellites = fj.shape[0] / 6
        ns = 0
        accelVecList = [0.0,0.0,0.0]
        while ns < nSatellites:
            if ns == nThisSat:
                ns += 1
                continue
            else:
                rN = np.array(np.array([fj[6*nThisSat+0],fj[6*nThisSat+1],fj[6*nThisSat+2]])
                              -np.array([fj[6*ns+0],fj[6*ns+1],fj[6*ns+2]]))
                rSqN = 0.0
                for rjN in rN:
                    rSqN += rjN*rjN
                rSqN += self.gravSoftenEps * self.gravSoftenEps  
                accelVecList -= np.array((k.GinAU3dm2*self.massesSatellites[ns]/(rSqN**1.5))*rN)
                ns += 1
        return np.array(accelVecList)

    ##########################################################################
    # 2-body astronomical position functions
    ##########################################################################
    
    # Function to return astronomical positions at a given time
    # Can use global numerical constants defined in constants library, such as masses
    # Time t is in days. Returns array of position vectors in AU. Masses should be in solar masses.
    # Must return same number of position vectors as masses in massesAstro
    
    # First, define 2-body functions
    # Helper function computes separation between sun and Earth for two-body problem
    # Also make helper functions to compute the first and second time derivatives
    def getSeparationVector(self, t, omega=k.omegaEarthPerDay2Body):
        startAngle = 0.0 #rad
        startTime = 0.0 #day
        rad = 1.0 #AU
        return rad * np.array([math.cos(startAngle + omega*(t-startTime)),
                         math.sin(startAngle + omega*(t-startTime)),
                         0.0])    
    def getSeparationVelocityVector(self, t, omega=k.omegaEarthPerDay2Body):
        startAngle = 0.0 #rad
        startTime = 0.0 #day
        rad = 1.0 #AU
        return rad * np.array([-1.0*omega*math.sin(startAngle + omega*(t-startTime)),
                         omega*math.cos(startAngle + omega*(t-startTime)),
                         0.0])    
    def getSeparationAccelerationVector(self, t, omega=k.omegaEarthPerDay2Body):
        startAngle = 0.0 #rad
        startTime = 0.0 #day
        rad = 1.0 #AU
        return rad * np.array([-1.0*omega*omega
                               *math.cos(startAngle + omega*(t-startTime)),
                         -1.0*omega*omega*math.sin(startAngle + omega*(t-startTime)),
                         0.0])

    # Return current positions of all astronomical bodies, for gravitation equations
    def getAstroPositions2Body(self, t):
        astroPositionsList = []
        # Position of smaller mass (Earth)
        astroPositionsList.append(np.array((self.massesAstro[1]/(self.massesAstro[0]+self.massesAstro[1])) 
                                           * self.getSeparationVector(t)))
        # Position of larger mass (sun)
        astroPositionsList.append(np.array(-1.0 * (self.massesAstro[0]/(self.massesAstro[0]+self.massesAstro[1])) 
                                           * self.getSeparationVector(t)))
        return np.array(astroPositionsList)
    
    def getAstroPositions1Body(self, t):
        astroPositionsList = []
        # Position of only mass (sun)
        astroPositionsList.append(np.zeros(3))
        return np.array(astroPositionsList)
    
    # Find the radial Lagrange points at time t=0.
    # At this time, put a particle on the x axis at some position x. The acceleration, if the 
    # particle is in a circular orbit, should be - x Omega^2.
    def lagrangeRootFindFuncTimeZero(self, x, omega=k.omegaEarthPerDay2Body):
        return omega*omega*x + self.FgravPerMass(0.0, [x,0.0,0.0])[0]
    
    # Functions that know the position of L1 vs. time.
    def getL1Position2body(self, t, omega=k.omegaEarthPerDay2Body):
        return np.array([self.L1Rad2Body*math.cos(omega*t), 
                         self.L1Rad2Body*math.sin(omega*t),
                         0.0])
    def getL1Velocity2body(self, t, omega=k.omegaEarthPerDay2Body):
        return np.array([-1.0*self.L1Rad2Body*omega*math.sin(omega*t), 
                         self.L1Rad2Body*omega*math.cos(omega*t),
                         0.0])
    def getL1Acceleration2body(self, t, omega=k.omegaEarthPerDay2Body):
        return np.array([-1.0*self.L1Rad2Body*omega*omega
                         *math.cos(omega*t), 
                         -1.0*self.L1Rad2Body*omega*omega
                         *math.sin(omega*t),
                         0.0])
    # Functions that know the earth trailing position vs time
    def getEarthTrailingPosition2body(self, t):
        return np.array((self.massesAstro[1]/(self.massesAstro[0]+self.massesAstro[1])) 
                         * self.getSeparationVector(t - self.earthTrailingDays))
    def getEarthTrailingVelocity2body(self, t):
        return np.array((self.massesAstro[1]/(self.massesAstro[0]+self.massesAstro[1])) 
                         * self.getSeparationVelocityVector(t - self.earthTrailingDays))
    def getEarthTrailingAcceleration2body(self, t):
        return np.array((self.massesAstro[1]/(self.massesAstro[0]+self.massesAstro[1])) 
                         * self.getSeparationAccelerationVector(t - self.earthTrailingDays))
    
    # Functions that know the earth trailing position vs time (sun only)
    def getEarthTrailingPosition1body(self, t):
        return np.array(self.getSeparationVector(t - self.earthTrailingDays, omega=k.omegaEarthPerDay1Body))
    def getEarthTrailingVelocity1body(self, t, omega=k.omegaEarthPerDay1Body):
        return np.array(self.getSeparationVelocityVector(t - self.earthTrailingDays, omega=k.omegaEarthPerDay1Body))
    def getEarthTrailingAcceleration1body(self, t):
        return np.array(self.getSeparationAccelerationVector(t - self.earthTrailingDays, omega=k.omegaEarthPerDay1Body))
    # fj = [deltaxsat1,deltaysat1,deltazsat1,deltavxsat1,deltavysat1,deltavzsat1,deltaxsat2,...]
    # ak = [ax1,ay1,az1,ax2,...]
    
    ##########################################################################
    # JPL horizons astronomical position functions
    ##########################################################################
    
    # Return current positions of all objects for gravitation functions
    def getAstroPositionsJPL(self, t):

        # Each element in astroPositionList should be a numpy array of the [x,y,z] positions at time t
        # The order should follow the order of self.bodiesAstro
        astroPositionsList = []
        for body in self.bodiesAstro:
            astroPositionsList.append(np.array([
                                        self.theJPL.bodyPositions[body][0](t),
                                        self.theJPL.bodyPositions[body][1](t),
                                        self.theJPL.bodyPositions[body][2](t)]))            
        return np.array(astroPositionsList)
    
    # L1 position, velocity, acceleration
    def getL1PositionJPL(self, t):
        return np.array([self.theJPL.dynamicPointPositions["L1"][0](t),
                         self.theJPL.dynamicPointPositions["L1"][1](t),
                         self.theJPL.dynamicPointPositions["L1"][2](t)])
    def getL1VelocityJPL(self, t):
        return np.array([self.theJPL.dynamicPointPositions["L1"][3](t),
                         self.theJPL.dynamicPointPositions["L1"][4](t),
                         self.theJPL.dynamicPointPositions["L1"][5](t)])
    def getL1AccelerationJPL(self, t):
        return np.array([self.theJPL.dynamicPointPositions["L1"][6](t),
                         self.theJPL.dynamicPointPositions["L1"][7](t),
                         self.theJPL.dynamicPointPositions["L1"][8](t)])
    
    # Earth-trailing position, velocity, acceleration
    def getEarthTrailingPositionJPL(self, t):
        return np.array([self.theJPL.dynamicPointPositions["emb"][0](t - self.earthTrailingDays),
                         self.theJPL.dynamicPointPositions["emb"][1](t - self.earthTrailingDays),
                         self.theJPL.dynamicPointPositions["emb"][2](t - self.earthTrailingDays)])
    def getEarthTrailingVelocityJPL(self, t):
        return np.array([self.theJPL.dynamicPointPositions["emb"][3](t - self.earthTrailingDays),
                         self.theJPL.dynamicPointPositions["emb"][4](t - self.earthTrailingDays),
                         self.theJPL.dynamicPointPositions["emb"][5](t - self.earthTrailingDays)])
    def getEarthTrailingAccelerationJPL(self, t):
        return np.array([self.theJPL.dynamicPointPositions["emb"][6](t - self.earthTrailingDays),
                         self.theJPL.dynamicPointPositions["emb"][7](t - self.earthTrailingDays),
                         self.theJPL.dynamicPointPositions["emb"][8](t - self.earthTrailingDays)])
    
    def rightHandSideNBody(self, t, fj, ak):
        # Make RHS vector with same shape as fj
        gi = np.zeros(np.array(fj).shape)
        
        #fj should be a vector with positions and velocities of each satellite. 
        #How many satellites?
        nSatellites = gi.shape[0] / 6
        
        #Where is the "base" position? e.g. L1, earth trailing location
        if self.system == "L1_2Body":
            x0 = self.getL1Position2body(t)
            dvdt0 = self.getL1Acceleration2body(t)     
        elif self.system == "EarthTrailing_2Body":
            x0 = self.getEarthTrailingPosition2body(t)
            dvdt0 = self.getEarthTrailingAcceleration2body(t)
        elif self.system == "EarthTrailing_1Body":
            x0 = self.getEarthTrailingPosition1body(t)
            dvdt0 = self.getEarthTrailingAcceleration1body(t)
        elif self.system == "L1":
            x0 = self.getL1PositionJPL(t)
            dvdt0 = self.getL1AccelerationJPL(t)
        elif self.system == "EarthTrailing":
            x0 = self.getEarthTrailingPositionJPL(t)
            dvdt0 = self.getEarthTrailingAccelerationJPL(t)
        else:
            print("WARNING! Defaulting to L1_2Body in rightHandSide2Body().")
            x0 = self.getL1Position2body(t)
            dvdt0 = self.getL1Acceleration2body(t) 
            
        
        #Loop over satellites
        ns = 0
        while ns < nSatellites:
            # Gravitational acceleration from astronomical bodies
            aGrav = np.array(self.FgravPerMass(t,np.array([x0[0]+fj[6*ns+0],
                                                          x0[1]+fj[6*ns+1],
                                                          x0[2]+fj[6*ns+2]])))
            # Gravitational influence of other satellites
            # Negligible because of their small mass (I can check this later)
            aGravSat = np.array(self.FgravPerMassOtherSats(ns,fj))
            
            # Position RHSs = velocities
            gi[6*ns+0] = fj[6*ns+3]
            gi[6*ns+1] = fj[6*ns+4]
            gi[6*ns+2] = fj[6*ns+5]
            
            # Velocity RHSs have 4 terms:
            # 1. -dv/dt of the base position (e.g. L1 or earth-trailing point)
            # 2. control acceleration ak
            # 3. gravitational acceleration from astronomical bodies
            # 4. gravitational acceleration from other satellities
            gi[6*ns+3] = -1.0*dvdt0[0] + ak[3*ns+0] + aGrav[0] + aGravSat[0]
            gi[6*ns+4] = -1.0*dvdt0[1] + ak[3*ns+1] + aGrav[1] + aGravSat[1]
            gi[6*ns+5] = -1.0*dvdt0[2] + ak[3*ns+2] + aGrav[2] + aGravSat[2]
            
            ns += 1
            
        return np.array(gi)
    
    # General calls for positions: for now, just call the 2-body functions
    # TO-DO: generalize to n bodies, reading in JPL data, optionally
    def getAstroPositions(self, t):
        if self.system == "L1_2Body":
            return self.getAstroPositions2Body(t)            
        elif self.system == "EarthTrailing_2Body":
            return self.getAstroPositions2Body(t)
        elif self.system == "EarthTrailing_1Body":
            return self.getAstroPositions1Body(t)
        elif self.system == "L1":
            return self.getAstroPositionsJPL(t)
        elif self.system == "EarthTrailing":
            return self.getAstroPositionsJPL(t)
        else:
            print("WARNING! Defaulting to L1_2Body in getAstroPositions().")
            return self.getAstroPositions2Body(t)

    def getL1Position(self, t):
        return self.getL1Position2body(t)
    def getL1Velocity(self, t):
        return self.getL1Velocity2body(t)
    def getL1Acceleration(self, t):
        return self.getL1Acceleration2body(t)
    
    def rightHandSide(self,t,fj,ak):
        #fj is an array fj[0] = xsat1, fj[1] = ysat1, fj[2] = zsat1, fj[3] = vxsat1, ... fj[6] = xsat2, ...
        #ak has 3 elements per satellite, ak[3*ns+j] which is a control acceleration in the j (x/y/z) direction
        #the control system will adjust ak to hold the satellite at a fixed position
        #Returns right hand sides g_i

        # Include only the sun, Earth, and other satellites, with sun and Earth in circular orbits
        return np.array(self.rightHandSideNBody(t,fj,ak)) #numpy arrays let you multiply the array by a scalar, which I use later

    def solveOde(self,tstart,tend,dt0,eps,f0,a0,useControl=True, pidParams=[[-1.0,-0.4,-1.0]], stepper='dopri5'):
        # Initial conditions
        t = tstart
        fj = f0
        dt = dt0
        
        #How many satellites?
        nSatellites = len(fj) / 6
        
        # Initialize arrays that will hold the results
        fjList = [f0]
        timesList = [tstart]
        
        # Initialize errors list, which tracks control system error
        # Also track a cumulative error for the I part of the controller
        errors0 = []
        errorsSum0 = []
        derrors0 = []
        for ns in np.arange(nSatellites):
            for j in np.arange(3): #x,y,z control for this sattelite               
                errors0.append(f0[int(6*ns+j)]-pidParams[int(3*ns+j)][0])
                errorsSum0.append(0.0)
                derrors0.append(0.0)
        errorsList = [errors0]
        errorsSumList = [errorsSum0]
        derrorsList = [derrors0]
        
        # Initialize the thruster accelerations
        ak = a0
        akList = [a0]
        
        # Set up the time stepper
        solver = ode(self.rightHandSide)
        solver.set_integrator(stepper, rtol=eps*10.0, atol=eps)
        solver.set_f_params(a0)
        solver.set_initial_value(f0, tstart)
        
        # Counters to track x,y,z thruster fuel for each satellite
        fuel=[[0.0, 0.0, 0.0] for x in range(0,int(nSatellites),1)]
        
        # Loop over time steps
        while solver.successful() and solver.t < tend:
            # Step to time t + dt0
            # The stepper will internally use adaptive stepping to take
            # many smaller steps until, using dense output, the result at t + dt0
            # is returned. So I can pretend I'm doing fixed stepping for my purposes,
            # and the solver will get me there with the error tolerances I request.
            tNewRequested = t + dt0 

            # Take a time step (many internal steps might be taken)
            solver.integrate(tNewRequested)
            
            # Get new values of variables fj
            # Reset tNew, in case the stepper reached a slightly different time
            # than the time I requested
            fjNew = solver.y
            tNew = solver.t
            
            if useControl: #update control function
                #Update to adjust ak with PID control
                # Treat error as a vector, each element is error for one controller
                errors = []
                errorsSum = []
                derrors = []
                akNew = []
                for ns in np.arange(nSatellites):
                    for j in np.arange(3): #x,y,z control for this sattelite
                        error = fj[int(6*ns+j)] - pidParams[int(3*ns+j)][0]
                        errors.append(error)
                        errorsSum.append(errorsSumList[-1][int(3*ns+j)] + error * dt)
                        derrors.append((error - errorsList[-1][int(3*ns+j)])/dt)
                
                        kp = pidParams[int(3*ns+j)][1]
                        ki = pidParams[int(3*ns+j)][2]
                        kd = pidParams[int(3*ns+j)][3]   
                
                        akNew0 = kp*errors[-1]+ki*errorsSum[-1]+kd*derrors[-1]
                        akNew.append(akNew0)
                
                errorsList.append(errors)
                errorsSumList.append(errorsSum)
                derrorsList.append(derrors)
            else:
                #Just leave ak at its initial value.
                #Initialize to zero for an uncontrolled evolution of the base system.
                akNew = ak
            
            # Append our lists with the updated values
            fjList.append(fjNew)
            timesList.append(tNew)
            akList.append(akNew)
            
            # Loop over satellites, and then over x,y,z
            # Append to running totals how much fuel was consumed, using 
            # the trapezoid rule
            n = 0
            while n < nSatellites:
               i = 0
               while i<3: 
#                fuel[n][i] += 0.1
#                  fuel[n][i]+=0.0001
                  if (tNew > self.firstTimeToCountFuel):
                      fuel[n][i] += k.msat1kg*(((k.sPerDay*(tNew-t))*(k.accelToSI*(abs(akNew[3*n+i])+abs(ak[3*n+i])))/2))/(k.impulsePerMassFuelSI)
                  i+=1
               n+=1
            
            # New values become the old values for next time in the loop
            t = tNew
            fj = fjNew
            ak = akNew
            
            # Reset the control values
            solver.set_f_params(ak)
            
            
        # Make numpy arrays, which have fancy indexing vs. regular arrays, to store the output
        timesToReturn = np.array(timesList)
        fjToReturn = np.array(fjList)
        akToReturn = np.array(akList)
        
        if useControl:
            return (fuel, timesToReturn, fjToReturn, akToReturn, 
                    (np.array(errorsList),np.array(errorsSumList),np.array(derrorsList)))
        else:
            return (timesToReturn, fjToReturn, akToReturn)    
        

    
    def solve(self,tstart,tend,dt0,eps,dtmin,f0,tableau,useControl,a0,pidParams=[[0.0,0.0,0.0,0.0]],
              dtmax=-1.0, useAdaptiveTimeStep=True):
        
        n = 0 #loop over satellite when computing fuel
        i = 0 #loop over x,y,z rockets when computing fuel
       
        
        # Initial conditions
        t = tstart
        fj = f0
        dt = dt0
    
        #How many satellites?
        nSatellites = len(fj) / 6
    
        # Initialize arrays that will hold the results
        fjList = [f0]
        stepErrjList = [0.0]
        timesList = [tstart]
    
        # Initialize errors list, which tracks control system error
        # Also track a cumulative error for the I part of the controller
        errors0 = []
        errorsSum0 = []
        derrors0 = []
        for ns in np.arange(nSatellites):
            for j in np.arange(3): #x,y,z control for this sattelite
                errors0.append(f0[int(6*ns+j)]-pidParams[int(3*ns+j)][0])
                errorsSum0.append(0.0)
                derrors0.append(0.0)
        errorsList = [errors0]
        errorsSumList = [errorsSum0]
        derrorsList = [derrors0]
        
        ak = a0
        akList = [a0]
        
        # Make steps a bit smaller than you might estimate, to account for our 
        # imperfect knowledge of errors (we only estimate the errors)
        safetyFactor = 0.9
        
        # Counters to track x,y,z thruster fuel for each satellite
        fuel=[[0.0, 0.0, 0.0] for x in range(0,int(nSatellites),1)]
    
        # Loop over times
        stepper = timestepping()
        
        while t <= tend:
            # Update time and variables fj
            tNew = t+dt
    
            fjNew, stepErrjNew = stepper.takeTimeStep(fj,t,dt,tableau,ak,self.rightHandSide)
            dfjNew = np.array((fjNew - fj))/dt
            
            # Follow numerical recipes Eq. (16.2.10)
            # delta0 = target accuracy
            # delta1 = accuracy of current step
            # These are pragmatic steps to increase or decrease the step size by
            # If current step error less than target error (delta1 <= delta0)
            # then increase the step next time. Otherwise, discard this step and 
            # try again with a smaller step.

            delta0j = eps * dt * (np.abs(fj) + np.abs(np.array(dfjNew))) 
            delta0max = np.max(np.array(delta0j)) #target accuracy
            delta1max = np.max(np.abs(np.array(stepErrjNew))) #actual accuracy

            # If using adaptive time stepping, estimate the error and increase or 
            # decrease the time step
            if useAdaptiveTimeStep:
                if delta0max >= delta1max:
                    # Step OK, try larger step next time
                    dtNew = safetyFactor * dt * abs(delta0max/delta1max)**0.2
                    if (dtmax > 0.0) and (dtNew > dtmax):
                        dtNew = dtmax
                else:
                    # Failed step: discard it and try again with smaller step
                    dtNew = safetyFactor * dt * abs(delta0max/delta1max)**0.25
                    dt = dtNew
                    if dt < dtmin: #step size smaller than minimum
                        print("Step size smaller than minimum! Exiting.")
                        break #step sizes got too small. Bail so we don't take tiny steps forever.
                    else:
                        continue
            else:
                dtNew = dt
                
            
            if useControl: #update control function
                #Update to adjust ak with PID control
                # Treat error as a vector, each element is error for one controller
                errors = []
                errorsSum = []
                derrors = []
                akNew = []
                for ns in np.arange(nSatellites):
                    for j in np.arange(3): #x,y,z control for this sattelite
                        error = fj[int(6*ns+j)] - pidParams[int(3*ns+j)][0]
                        errors.append(error)
                        errorsSum.append(errorsSumList[-1][int(3*ns+j)] + error * dt)
                        derrors.append((error - errorsList[-1][int(3*ns+j)])/dt)
                
                        kp = pidParams[int(3*ns+j)][1]
                        ki = pidParams[int(3*ns+j)][2]
                        kd = pidParams[int(3*ns+j)][3]   
                
                        akNew0 = kp*errors[-1]+ki*errorsSum[-1]+kd*derrors[-1]
                        akNew.append(akNew0)
                
                errorsList.append(errors)
                errorsSumList.append(errorsSum)
                derrorsList.append(derrors)
            else:
                #Just leave ak at its initial value.
                #Initialize to zero for an uncontrolled evolution of the base system.
                akNew = ak
            
        
            # Append our lists with the updated values
            fjList.append(fjNew)    
            timesList.append(tNew)
            akList.append(akNew)
            
            #change to i's
        #fuel=[[0.0, 0.0, 0.0] for x in range(0,nSatellites,1)] put in while
            
            # Loop over satellites, and then over x,y,z
            # Append to running totals how much fuel was consumed, using 
            # the trapezoid rule
            n = 0
            while n < nSatellites:
               i = 0
               while i<3: 
#                fuel[n][i] += 0.1
#                  fuel[n][i]+=0.0001
                  if tNew > self.firstTimeToCountFuel:
                      fuel[n][i] += k.msat1kg*(((k.sPerDay*(tNew-t))*(k.accelToSI*(abs(akNew[3*n+i])+abs(ak[3*n+i])))/2))/(k.impulsePerMassFuelSI)
                  i+=1
               n+=1

         
            
                
            # New values become the old values for next time in the loop
            t = tNew
            fj = fjNew
            ak = akNew
            dt = dtNew
    
        # Make numpy arrays, which have fancy indexing vs. regular arrays, to store the output
        timesToReturn = np.array(timesList)
        fjToReturn = np.array(fjList)
        akToReturn = np.array(akList)
        
        if useControl:   #addfuel
            return (fuel,timesToReturn, fjToReturn, akToReturn, 
                    (np.array(errorsList),np.array(errorsSumList),np.array(derrorsList)))
        else:
            return (timesToReturn, fjToReturn, akToReturn)
    
    