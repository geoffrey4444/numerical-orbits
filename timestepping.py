# 3. Functions that take time steps
import numpy as np
class timestepping:

    # Let's start by setting up the Butcher tableaus for the different steppers
    #|   |   |   |
    #|---|---|---|
    #| 0 |   |   |
    #|   |   | 1 |

    tableauEuler = {"aIJ": [[]],
                    "cI": [0.0],
                    "bJ": [[1.0]]}


    #|     |     |   |
    #|-----|-----|---|
    #| 0   |     |   |
    #| 1/2 | 1/2 |   |
    #|     | 0   | 1 |

    tableauRK2 = {"aIJ": [[1.0/2.0]],
                  "cI":  [0,1.0/2.0],
                  "bJ": [[0.0,1.0]]}

    #|     |     |     |     |   |
    #|-----|-----|-----|-----|---|
    #| 0   |     |     |     |   |
    #| 1/2 | 1/2 |     |     |   |
    #| 1/2 | 0   | 1/2 |     |   | 
    #| 1   | 0   | 0   | 1   |   |
    #|     | 1/6 | 1/3 | 1/3 | 1/6 |
    tableauRK4 = {"aIJ": [[1.0/2.0], 
                          [0.0, 1.0/2.0], 
                          [0.0, 0.0, 1.0]],
                  "cI": [0, 1.0/2.0, 1.0/2.0, 1.0],
                  "bJ": [[1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]]}
    
    #|       |           |                |          |         |              |        |    |
    #|-------|-----------|----------------|----------|---------|--------------|--------|----|
    #|0      |           |                |          |         |              |        |    |
    #|1/5    |1/5        |                |          |         |              |        |    |
    #|3/10   |3/40       | 9/40           |          |         |              |        |    |
    #|4/5    |44/45      | -56/15         |32/9      |         |              |        |    |
    #|8/9    |19372/6561 | -25360/2187    |64448/6561| -212/729|              |        |    |
    #|1      |9017/3168  | -355/33        |46732/5247| 49/176  | -5103/18656  |        |    |
    #|1      |35/384     | 0              |500/1113  | 125/192 | -2187/6784   |11/84   |    | 
    #|       |35/384     | 0              |500/1113  | 125/192 | -2187/6784   |11/84   |0   |
    #|       |5179/57600 | 0              |7571/16695| 393/640 | -92097/339200|187/2100|1/40|
    tableauDP5 = {"aIJ": [[1.0/5.0], 
                          [3.0/40.0, 9.0/40.0], 
                          [44.0/45.0, -56.0/15.0, 32.0/9.0], 
                          [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0], 
                          [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0], 
                          [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0]],
                  "cI": [0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0],
                  "bJ": [[35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0],
                         [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0]]}

    # Take a time step given a tableau. This is a generic function that can 
    # take a step using any stepper you like, defined by its tableau.
    # Tableau is a dictionary with arrays aIJ, cI, bJ, as defined above.
    # ak is a control input passed into the RHS, updated elsewhere with a control system.
    def takeTimeStep(self,fj,t,dt,tableau,ak,rightHandSide):
        # new array k stores the "stage" values k1, k2, ... (but note we number from 0, not 1 as in the literature)
        k = []
    
        # Loop over stage I. Each stage evaluates the RHS at time t + cI*dt
        # All stages evaluate the RHS at variables that start out at the current values fj
        # Stages beyond the 0th stage also add aIJ*kJ to fj; i.e., they evaluate the RHS with partially updated variables
        for I,cI in enumerate(tableau["cI"]):
            tRHS = t + cI * dt
            fjRHS = np.array(fj)
            if I>0:
                for J,aIJ in enumerate(tableau["aIJ"][I-1]):
                    fjRHS += dt * aIJ * k[J]
            k.append(rightHandSide(tRHS,fjRHS,ak))
        
        # Combine results from the stages to get update values
        # The updated variables start out at the current values fj.
        # Then add bJ*k[j], i.e., bJ * stage J.
        fjNew = []
        for bJrow in tableau["bJ"]:
            fjRowNew = np.array(fj)
            for J,bJ in enumerate(bJrow):
                fjRowNew += dt * bJ * k[J]
            fjNew.append(fjRowNew)
     
        # Decide what to return
        if len(fjNew) > 1: # there is an error estimate
            # Return (updated variables, error estimate)
            return (fjNew[0], fjNew[0]-fjNew[1])
        else:
            # No error estimate, just return the updated variables
            return (fjNew[0], 0.0)
        