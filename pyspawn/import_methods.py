from pyspawn.traj import traj
from pyspawn.simulation import simulation

def into_traj(x):
    for method in x.__dict__:
        if method[0] != "_":
            exec("traj." + method + " = x." + method)
            
def into_simulation(x):
    for method in x.__dict__:
        if method[0] != "_":
            exec("simulation." + method + " = x." + method)
            
