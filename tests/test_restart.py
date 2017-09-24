import numpy as np
import pyspawn         

tfinal = 6.0

sim = pyspawn.simulation()

sim.read_from_file("sim.json")

sim.set_maxtime_all(tfinal)

sim.propagate()







