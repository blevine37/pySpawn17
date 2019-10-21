# this script starts a new AIMS calculation.  Ethylene, SA2-CASSCF(2/2).
import numpy as np
import pyspawn        
import pyspawn.general

# terachemserver port 
port = 54322

# random number seed
seed=87062

# Velocity Verlet classical propagator
clas_prop = "vv"

# adapative 2nd-order Runge-Kutta quantum propagator
qm_prop = "fulldiag"

# adiabtic NPI quantum Hamiltonian
qm_ham = "adiabatic"

# use TeraChem CASSCF or CASCI to compute potentials
potential = "terachem_cas"

# initial time
t0 = 0.0

# time step
ts = 10.0

# final simulation time
tfinal = 2500.0

# number of dimensions                                                                                           
numdims = 18

# number of electronic states                                                                                                                    
numstates = 2

# TeraChem job options                                                                                    
tc_options = {
    "method":       'hf',
    "basis":        '6-31g',
    "atoms":        ["C", "C", "H", "H", "H", "H"],
    "charge":       0,
    "spinmult":     1,
    "closed_shell": True,
    "restricted":   True,

    "precision":    "double",
    "threall":      1.0e-20,

    "casscf":        "yes",
    "closed":       7,
    "active":       2,
    "cassinglets":  numstates,
    "castargetmult": 1,
    "cas_energy_states": [0, 1],
    "cas_energy_mults": [1, 1],
    }

# trajectory parameters
traj_params = {
    # terachem port
    "tc_port": port,
    # initial time
    "time": t0,
    # time step
    "timestep": ts,
    # final simulation time
    "maxtime": tfinal,
    # coupling threshhold
    "spawnthresh": (0.5 * np.pi) / ts / 20.0,
    # initial electronic state (indexed such that 0 is the ground state)
    "istate": 1,
    # Gaussian widths
    "widths": np.asarray([30.0, 30.0, 30.0,
                        30.0, 30.0, 30.0,
                        6.0, 6.0, 6.0,
                        6.0, 6.0, 6.0,
                        6.0, 6.0, 6.0,
                        6.0, 6.0, 6.0]),
    # atom labels
    "atoms": tc_options["atoms"],
    # nuclear masses (in a.u)    
    "masses": np.asarray([21864.0, 21864.0, 21864.0,
                    21864.0, 21864.0, 21864.0,
                    1822.0, 1822.0, 1822.0,
                    1822.0, 1822.0, 1822.0,
                    1822.0, 1822.0, 1822.0,
                    1822.0, 1822.0, 1822.0]),
    # terachem options (above)
    "tc_options": tc_options
    }

sim_params = {
    # initial time   
    "quantum_time": traj_params["time"],
    # time step
    "timestep": traj_params["timestep"],
    # final simulation time
    "max_quantum_time": traj_params["maxtime"],
    # initial qm amplitudes
    "qm_amplitudes": np.ones(1,dtype=np.complex128),
    # energy shift used in quantum propagation
    "qm_energy_shift": 77.6,
}

# import routines needed for propagation
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")

# check for the existence of files from a past run
pyspawn.general.check_files()    

# set up first trajectory
traj1 = pyspawn.traj(numdims, numstates)
traj1.set_numstates(numstates)
traj1.set_numdims(numdims)
traj1.set_parameters(traj_params)

# sample initial position and momentum from Wigner distribution (requires hessian.hdf5)
traj1.initial_wigner(seed)

# set up simulation 
sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_parameters(sim_params)

# begin propagation
sim.propagate()







