import numpy as np
import fms         

traj1 = fms.traj()

t0 = 0.0

timestep = 0.1

tfinal = 0.3

prop = "vv"

ndims = 2

pos = np.random.normal(0.0,1.0,ndims)

mom = np.random.normal(0.0,0.1,ndims)

wid = np.ones(ndims)

m = np.ones(ndims)

traj1.init_traj(t0,ndims,pos,mom,wid,m)

sim = fms.simulation()

sim.add_traj(traj1,'0')

sim.set_timestep_all(timestep)

sim.set_mintime_all(t0)

sim.set_maxtime_all(tfinal)

sim.set_propagator_all(prop)

sim.propagate()







