import numpy as np
import fms         

t1 = fms.traj()

t = 0.0

ndims = 2

pos = np.random.normal(0.0,1.0,ndims)

mom = np.random.normal(0.0,0.1,ndims)

wid = np.ones(ndims)

m = np.ones(ndims)

t1.init_traj(t,ndims,pos,mom,wid,m)

t1.set_timestep(0.1)

t1.propagate()
t1.propagate()

t1.write_to_file("t1.json")

t2 = fms.traj()

t2.read_from_file("t1.json")

t3 = fms.traj()

t3.read_from_file("t1.json")

t3.write_to_file("t3.json")

t1.propagate()
t1.propagate()

t2.propagate()
t2.propagate()

t2.write_to_file("t1-2.json")
t2.write_to_file("t2.json")

sim = fms.simulation()

#sim.set_spawntraj(t2)

sim.write_to_file("sim.json")

sim2 = fms.simulation()

sim2.read_from_file("sim.json")

sim2.add_traj(t1,'0')

print sim2.traj['0']

sim2.write_to_file("sim2.json")

print sim2.traj['0']

sim3 = fms.simulation()

sim3.read_from_file("sim2.json")

print sim2.traj['0']
print sim3.traj['0']
print sim3.traj['0'].positions

sim3.add_traj(t3,'0>0')

sim3.write_to_file("sim3.json")

sim4 = fms.simulation()

sim4.read_from_file("sim3.json")

sim4.write_to_file("sim4.json")

print sim4.get_numtasks()

sim4.add_task("abc")

sim4.add_task("def")

print sim4.get_numtasks()

print type(sim4.traj["0"].positions).__module__








