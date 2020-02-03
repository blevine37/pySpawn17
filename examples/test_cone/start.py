# this script starts a new FMS calculation on a model cone potential
import numpy as np
import pyspawn
import pyspawn.general

# Velocity Verlet classical propagator
clas_prop = "vv"

# adaptive 2nd-order Runge-Kutta quantum propagator
qm_prop = "fulldiag"

# adiabatic NPI quantum Hamiltonian
qm_ham = "adiabatic"

# use TeraChem CASSCF or CASCI to compute potentials
potential = "test_cone"

# initial time
t0 = 0.0

# time step
ts = 0.1

# final simulation time
tfinal = 200.0

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
    # coupling threshold
    "spawnthresh": (0.5 * np.pi) / ts / 20.0,
    # initial electronic state (indexed such that 0 is the ground state)
    "istate": 1,
    # Gaussian widths
    "widths": np.asarray([6.0, 6.0]),
    # nuclear masses (in a.u)
    "masses": np.asarray([1822.0, 1822.0]),
    # initial positions
    "positions": np.asarray([0.45, 0.1]),
    # inition momenta
    "momenta": np.asarray([-5.0, 0.0]),
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
    "qm_energy_shift": -5.18,
}

# import routines needed for propagation
exec ("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
exec ("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
exec ("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
exec ("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")

# check for the existence of files from a past run
pyspawn.general.check_files()

# set up first trajectory
traj1 = pyspawn.traj(numdims, numstates)
# traj1.set_numstates(numstates)
# traj1.set_numdims(numdims)
traj1.set_parameters(traj_params)

# set up simulation
sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_parameters(sim_params)

# begin propagation
sim.propagate()
