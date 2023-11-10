# this script starts a new AIMS calculation for 696-Pristine CDot at FON-0.15-camB3LYP-CAS(2,2)CI/6-31gss.
import numpy as np
import pyspawn
import pyspawn.general
import pyspawn.process_geometry as pg
import pyspawn.dictionaries as dicts
import sys

# terachemserver port
if sys.argv[1]:
    port = int(sys.argv[1])
else:
    "Please provide a port number as a command line argument"
    sys.exit()

# Processing geometry.xyz file (positions are in hessian.hdf5 so we don't need them)
natoms, atoms, _, comment = pg.process_geometry('geometry.xyz')

# Getting atomic masses from the dictionary and converting to atomic units
# If specific isotopes are needed masses array can be set manually
mass_dict = dicts.get_atomic_masses()
masses = np.asarray([mass_dict[atom]*1822.0 for atom in atoms for i in range(3)])

widths_dict = {'C': 22.7, 'H': 4.7} # Obtained from A L Thompson et al Chemical Physics 370 (2010) 70-77.
widths = np.asarray([widths_dict[atom] for atom in atoms for i in range(3)])

# finite wigner temperature
wigner_temp = 0

# Simulation temperature (K) 
sim_temp = 300.0

# Langevin damping factor in unit of time (a.u) 
damp = 10.0

# random number seed
seed=87010

# Velocity Verlet classical propagator with Langevin thermostat 
clas_prop = "langevin"

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
tfinal = 30000.0

# number of dimensions                                                                                           
numdims = natoms*3

# number of electronic states                                                                                                                    
numstates = 3

# TeraChem job options                                                                                    
tc_options = {
    "method":       'hf',
    "basis":        '6-31gs',
    "atoms":        atoms,
    "charge":       0,
    "spinmult":     1,
    "closed_shell": True,
    "restricted":   True,
    "scf":          "diis",
    "precision":    "double",
    "threall":      1.0e-20,
    "thregr":       1.0e-20,
    "convthre":     1.0e-8,
    "xtol":         1.0e-4,
    "casscf":       "yes",
    "casguess":     "c0_old.casscf",

    "casscfmaxiter": 500,
    "dcimaxiter":   500,
    "closed":       7,
    "active":       2,
    "cassinglets":  numstates,
    "castargetmult": 1,
    "cas_energy_states": "[0,1]",
    "cas_energy_mults": "[1,1]",
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
    "widths": widths,
    # Temperature for Thermostat
    "sim_temp": sim_temp,
    # Legvine damping 
    "lan_damp": damp,
    # atom labels
    "atoms": tc_options["atoms"],
    # nuclear masses (in a.u)    
    "masses": masses,
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
    "qm_energy_shift": 78.04834682,
    
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

