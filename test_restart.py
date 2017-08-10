import numpy as np
import fms         

tfinal = 6.0

sim = fms.simulation()

sim.read_from_file("sim.json")

sim.set_maxtime_all(tfinal)

sim.propagate()







