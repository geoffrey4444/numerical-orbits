# Define physical constants
# We will work in units of AU for length, solar mass for mass, and days for time.

#Let's put rsun - rearth = 1 AU. The analytic solution for Earth-Sun motion is then given by the 
#effective one body problem in Lagrangian mechanics.


# 0. Physical constants
# Note: The astronomical masses numbers all come from  the headers in the JPL horizons data files.
# Note: I need to double-check where I got G*Msun in units of AU**3/day**2.
import math

class constants:
    # Masses and distances for astronomical bodies
    mearthkg = 5.97219e24
    mmoonkg = 734.9e20
    mjupiterkg = 1898.13e24
    msunkg = 1.988544e30
    msaturnkg = 5.68319e26
    muranuskg = 86.8103e24
    mneptunekg = 102.41e24
    mmercurykg = 3.302e23
    mvenuskg = 48.683e23
    mmarskg = 6.4185e23
    rsunm = 6.963e8 #radius of sun surface
    mPerAU = 149597870700
    #rsunau = rsunm/149597870700 #(* m/AU *)
    rsunau = rsunm/mPerAU #(* m/AU *)

    GinAU3dm2 = 0.000295912208  #(* msun=1, AU3dm2 = AU**3 day**-2 *) #(*msun=1*)
    omegaEarthPerDay = math.sqrt(GinAU3dm2*(1.0+mearthkg/msunkg))

    rearthau = 1.0 #(* AU *) #radius of earth orbit

    # satellite masses in kg
    # lisa pathfinder beginning of operational life mass is 480 kg
    # https://en.wikipedia.org/wiki/LISA_Pathfinder
    msat1kg = 500
    msat2kg = 500
    msat3kg = 500
    
    sPerDay = 86400.0
    accelToSI = mPerAU / (sPerDay * sPerDay) # converts AU/day/day to m/s/s
    
    # micronewton thruster efficiency: specific impulse
    # specific impulse in m/s is impulse per ejected mass
    # specific impulse in s is impulse per ejected force
    # https://trs.jpl.nasa.gov/bitstream/handle/2014/45399/08-2220_A1b.pdf?sequence=1&isAllowed=y
    # says something like 200 s is reasonable for specific impulse
    # that means roughly 2000 m/s = 200 s * g = 200 s * 10 m/s/s impulse per kg of fuel
    #
    # This says your typical ion thruster carries 113 g of propellant 
    # Our satellites have 6 such thrusters, but let's assume they have a common pool of 
    # 0.6 kg of fuel
    mFuelPerSatkg = 0.6
    
    # https://directory.eoportal.org/web/eoportal/satellite-missions/content/-/article/lisa-pathfinder
    # says LISA Pathfinder specific impulse is something like 150s to 300 s
    # I think 200 s is reasonable for now
    impulsePerMassFuelSI = 2000
    