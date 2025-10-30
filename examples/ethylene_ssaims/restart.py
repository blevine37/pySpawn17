# this script restarts the simulation using data from sim.json (which 
# contains the entire current state of the simulation) and sim.hdf5 (which 
# contains a selected history of the simulation
import numpy as np
import pyspawn         

pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.fulldiag)
pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)
pyspawn.import_methods.into_traj(pyspawn.potential.terachem_cas)
pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)
    
tfinal = 30000.0

sim = pyspawn.simulation()

sim.restart_from_file("sim.json","sim.hdf5")

sim.set_maxtime_all(tfinal)

#SSAIMS control
sim.enable_ssaims(
   epsilon=1e-10,             # tune for your system
   ss_seed=527518,            # optional
   suspend_during_spawn=True, # suspend SSAIMS during spawn process
   spawn_delay_steps=10,      # wait N time steps after spawn to avoide premature killing
   min_tbf_to_start=2,        # require at least M TBFs to begin selection
   verbose=True               # detailed output for SSAIM. Good for debugging
)

sim.propagate()







