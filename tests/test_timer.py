import numpy as np
import pyspawn        

pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.rk2)
pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)

pyspawn.import_methods.into_traj(pyspawn.potential.test_cone)
pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)
    
traj1 = pyspawn.traj()

t0 = 0.0

timestep = 0.02

tfinal = 40.0

ndims = 2

nstates = 2

istate = 1

pos = np.random.normal(0.0,1.0,ndims)

mom = np.random.normal(0.0,0.1,ndims)

wid = np.ones(ndims)

m = np.ones(ndims)

max_walltime = '00:01:04'

traj1.init_traj(t0,ndims,pos,mom,wid,m,nstates,istate,"00")

traj1.set_spawnthresh(1.0)

sim = pyspawn.simulation()

sim.set_max_walltime_formatted(max_walltime)

sim.add_traj(traj1)

sim.set_timestep_all(timestep)

sim.set_mintime_all(t0)

sim.set_maxtime_all(tfinal)

sim.init_amplitudes_one()

sim.propagate()







