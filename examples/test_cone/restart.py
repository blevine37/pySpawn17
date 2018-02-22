# this script restarts the simulation using data from sim.json (which 
# contains the entire current state of the simulation) and sim.hdf5 (which 
# contains a selected history of the simulation
import numpy as np
import pyspawn         

pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.rk2)
pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)
pyspawn.import_methods.into_traj(pyspawn.potential.test_cone)
pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)
    
tfinal = 160.0

sim = pyspawn.simulation()

sim.restart_from_file("sim.json","sim.hdf5")

sim.set_maxtime_all(tfinal)

sim.propagate()







