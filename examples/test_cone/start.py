# this script starts a new FMS calculation on a model cone potential
import numpy as np
import pyspawn        
import pyspawn.general

# Velocity Verlet classical propagator
clas_prop = "vv"

# adapative 2nd-order Runge-Kutta quantum propagator
qm_prop = "fulldiag"

# diabatic ehrenfest Hamiltonian
qm_ham = "ehrenfest"

# use TeraChem CASSCF or CASCI to compute potentials
potential = "test_cone_td"

# initial time
t0 = 0.0

# time step
ts = 0.05

# final simulation time
tfinal = 30.0

# number of dimensions                                                                                           
numdims = 2

# number of electronic states                                                                                                                    
numstates = 2

# trajectory parameters
traj_params = {
    # initial time
    "time": t0,
    # time step
    "timestep": ts,
    # final simulation time
    "maxtime": tfinal,
    # coupling threshhold
    "clonethresh": 0.10,
    # initial electronic state (indexed such that 0 is the ground state)
    "istate": 1,
    # Gaussian widths
    "widths": np.asarray([6.0, 6.0]),
    # nuclear masses (in a.u)    
    "masses": np.asarray([1822.0, 1822.0]),
    # initial positions
    "positions": np.asarray([-0.2, -0.2]),
    # inition momenta
    "momenta": np.asarray([1.0, 1.0]),
    #
    "numstates": numstates,
    }

sim_params = {
    # initial time   
    "quantum_time": traj_params["time"],
    # time step
    "timestep": traj_params["timestep"],
    # final simulation time
    "max_quantum_time": traj_params["maxtime"],
    # initial qm amplitudes
    "qm_amplitudes": np.ones(1, dtype=np.complex128),
    # energy shift used in quantum propagation
    "qm_energy_shift": -2.9413,
    # cloning probability threshold
    "p_threshold": 0.018,
    # cloning minimum population parameter
    "pop_threshold": 0.15,
}

# import routines needed for propagation
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")

# check for the existence of files from a past run
pyspawn.general.check_files()    

# set up first trajectory
traj1 = pyspawn.traj()
# traj1.set_numstates(numstates)
# traj1.set_numdims(numdims)
traj1.set_parameters(traj_params)

# set up simulation 
sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_parameters(sim_params)
# begin propagation
sim.propagate()
print sim.num_traj_qm







