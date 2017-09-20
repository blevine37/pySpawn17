import numpy as np
import pyspawn         

pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.rk2)
pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)

pyspawn.import_methods.into_traj(pyspawn.potential.terachem_cas)
pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)
    
tfinal = 800.0

sim = pyspawn.simulation()

sim.read_from_file("sim.json")

sim.set_maxtime_all(tfinal)

sim.propagate()







