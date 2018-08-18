import numpy as np
import csv
import scipy.interpolate
from constants import constants as k
import os

class jpl:
    def readJPLFile(self, file, readVelocities=True, interpolate=True, computeAccel=True):
        with open(file, 'rU') as file:
            reader = csv.reader(file, delimiter=',')
            readNextRow = False
            rowsRead = 0
            timesList = []
            xList = []
            yList = []
            zList = []
            vxList = []
            vyList = []
            vzList = []
            axList = []
            ayList = []
            azList = []
            for row in reader:
                if len(row) > 0:
                    if row[0] == '$$SOE':
                        readNextRow = True
                        continue
                    if row[0] == '$$EOE':
                        readNextRow = False
                        continue
                if readNextRow:
                    xList.append(float(row[2]))
                    yList.append(float(row[3]))
                    zList.append(float(row[4]))
                    timesList.append(float(row[0]))

                    if readVelocities:
                        vxList.append(float(row[5]))
                        vyList.append(float(row[6]))
                        vzList.append(float(row[7]))
                        
                    rowsRead += 1
            
            # Define time t=0 the be 365 days after the 
            # first time in timesList
            # this allows negative times for earth-trailing
            # orbits
            times = np.array(timesList)-timesList[0]-365.0
            
            x = np.array(xList)
            y = np.array(yList)
            z = np.array(zList)
            vx = np.array(vxList)
            vy = np.array(vyList)
            vz = np.array(vzList) 
            
            if computeAccel:
                if readVelocities:
                    # Take one finite difference derivative of \vec{v}(t)
                    dvx = np.diff(vx)
                    dvy = np.diff(vy)
                    dvz = np.diff(vz)
                    dt = np.diff(times)
                    dvxdt = dvx/dt
                    dvydt = dvy/dt
                    dvzdt = dvz/dt
                    dvdtTimes = 0.5*(times[:-1]+times[1:])
                else:
                    # Take two finite difference derivatives of \vec{x}(t)
                    dx = np.diff(x)
                    dy = np.diff(y)
                    dz = np.diff(z)
                    dt = np.diff(times)
                    dxdt = dx/dt
                    dydt = dy/dt
                    dzdt = dz/dt
                    dxdtTimes = 0.5*(times[:-1]+times[1:])
                    dvx = np.diff(dxdt)
                    dvy = np.diff(dydt)
                    dvz = np.diff(dzdt)
                    dt2 = np.diff(dxdtTimes)
                    dvxdt = dvx/dt2
                    dvydt = dvy/dt2
                    dvzdt = dvz/dt2
                    dvdtTimes = 0.5*(dxdtTimes[:-1]+dxdtTimes[1:])
            
            if interpolate:
                if computeAccel:
                    if readVelocities:   
                        return (scipy.interpolate.interp1d(times,x),
                                scipy.interpolate.interp1d(times,y),
                                scipy.interpolate.interp1d(times,z),
                                scipy.interpolate.interp1d(times,vx),
                                scipy.interpolate.interp1d(times,vy),
                                scipy.interpolate.interp1d(times,vz),
                                scipy.interpolate.interp1d(dvdtTimes,dvxdt),
                                scipy.interpolate.interp1d(dvdtTimes,dvydt),
                                scipy.interpolate.interp1d(dvdtTimes,dvzdt),
                                dvdtTimes[0], dvdtTimes[-1],
                                times[0],times[-1])
                    else:
                        return (scipy.interpolate.interp1d(times,x),
                                scipy.interpolate.interp1d(times,y),
                                scipy.interpolate.interp1d(times,z),
                                scipy.interpolate.interp1d(dxdtTimes,dxdt),
                                scipy.interpolate.interp1d(dxdtTimes,dydt),
                                scipy.interpolate.interp1d(dxdtTimes,dzdt),
                                scipy.interpolate.interp1d(dvdtTimes,dvxdt),
                                scipy.interpolate.interp1d(dvdtTimes,dvydt),
                                scipy.interpolate.interp1d(dvdtTimes,dvzdt),
                                dxdtTimes[0], dxdtTimes[-1],
                                dvdtTimes[0], dvdtTimes[-1],
                                times[0],times[-1])
                else:
                    if readVelocities:   
                        return (scipy.interpolate.interp1d(times,x),
                                scipy.interpolate.interp1d(times,y),
                                scipy.interpolate.interp1d(times,z),
                                scipy.interpolate.interp1d(times,vx),
                                scipy.interpolate.interp1d(times,vy),
                                scipy.interpolate.interp1d(times,vz),
                                times[0],times[-1])
                    else:
                        return (scipy.interpolate.interp1d(times,x),
                                scipy.interpolate.interp1d(times,y),
                                scipy.interpolate.interp1d(times,z),
                                times[0],times[-1])
            else:
                if computeAccel:
                    if readVelocities:
                        return (x,y,z,vx,vy,vz,dvxdt,dvydt,dvzdt,
                                dvdtTimes,dvdtTimes[0],dvdtTimes[-1],
                                times,times[0],times[-1])
                    else:
                        return (x,y,z,dxdt,dydt,dzdt,dvxdt,dvydt,dvzdt,
                                dxdtTimes,dxdtTimes[0],dxdtTimes[-1],
                                dvdtTimes,dvdtTimes[0],dvdtTimes[-1],  
                                times,times[0],times[-1])
                else:
                    if readVelocities:
                        return (x,y,z,vx,vy,vz,times,times[0],times[-1])
                    else:
                        return (x,y,z,times,times[0],times[-1])

    def readJPLFiles(self, bodiesOrDynamicalPoints, root='./',
                     readVelocities=True, interpolate=True, computeAccel=True):
        jplData = {}
        for bodyOrPoint in bodiesOrDynamicalPoints:
            jplData[bodyOrPoint] = self.readJPLFile(root+bodyOrPoint+'.txt',
                                                   readVelocities=readVelocities,
                                                   interpolate=interpolate,
                                                   computeAccel=computeAccel)
        return jplData
    
    
    def __init__(self,
                 jplRoot="./JPLOrbitData/",
                 theBodies=['sun','mercury','venus','earth','moon','mars','jupiter','saturn','uranus','neptune'],
                 theMasses= np.array([k.msunkg,k.mmercurykg,k.mvenuskg,k.mearthkg,k.mmoonkg,k.mmarskg,k.mjupiterkg,
                            k.msaturnkg,k.muranuskg,k.mneptunekg])/k.msunkg,
                 theDynamicalPoints=['ssb', 'L1', 'L2', 'emb']
                ):
        self.bodies = theBodies
        self.bodyMasses = theMasses
        self.dynamicalPoints = theDynamicalPoints
        self.bodyPositions = self.readJPLFiles(self.bodies, jplRoot)
        self.dynamicPointPositions = self.readJPLFiles(self.dynamicalPoints, jplRoot)
        self.pathToFiles = jplRoot
    

