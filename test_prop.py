import numpy as np
import pyspawn        

pyspawn.import_methods.into_simulation(pyspawn.qm_integrator_rk2)
pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian_adiabatic)
    
traj1 = pyspawn.traj()

t0 = 0.0

timestep = 0.02

tfinal = 4.0

prop = "vv"

ndims = 2

nstates = 2

istate = 1

pos = np.random.normal(0.0,1.0,ndims)

mom = np.random.normal(0.0,0.1,ndims)

wid = np.ones(ndims)

m = np.ones(ndims)

traj1.init_traj(t0,ndims,pos,mom,wid,m,nstates,istate,"00")

traj1.set_spawnthresh(1.0)

sim = pyspawn.simulation()

sim.add_traj(traj1)

sim.set_timestep_all(timestep)

sim.set_mintime_all(t0)

sim.set_maxtime_all(tfinal)

sim.set_propagator_all(prop)

sim.init_amplitudes_one()

sim.propagate()







