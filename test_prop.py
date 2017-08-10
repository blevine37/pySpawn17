import numpy as np
import fms         

traj1 = fms.traj()

t0 = 0.0

timestep = 0.1

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

sim = fms.simulation()

sim.add_traj(traj1)

sim.set_timestep_all(timestep)

sim.set_mintime_all(t0)

sim.set_maxtime_all(tfinal)

sim.set_propagator_all(prop)

sim.propagate()







