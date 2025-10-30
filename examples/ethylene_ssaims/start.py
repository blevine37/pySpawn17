# this script starts a new AIMS calculation.  PTF Optimized D0 cation, CASSCF(5,6).
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

widths_dict = {'C': 22.7, 'H': 4.7, 'O': 12.2, 'S': 16.7, 'F': 8.5} # Obtained from A L Thompson et al Chemical Physics 370 (2010) 70-77.
widths = np.asarray([widths_dict[atom] for atom in atoms for i in range(3)])

# finite wigner temperature
wigner_temp = 0

# random number seed
seed=73647

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
tfinal = 8000.0

# number of dimensions                                                                                           
numdims = natoms*3

# number of electronic states                                                                                                                    
numstates = 3

# TeraChem job options                                                                                    
tc_options = {
    "method":       'hf',
    "basis":        'cc-pvdz',
    "atoms":        atoms,
    "charge":       0,
    "spinmult":     1,
    "closed_shell": True,
    "restricted":   True,
    "scf":          "diis+a",
    "precision":    "double",
    "threall":      1.0e-20,
    "thregr":       1.0e-20,
    "convthre":     1.0e-8,
    "fon":          "yes",
    "fon_method":   "gaussian",
    "fon_temperature": 0.15,
    "cphfiter":     80,
    "casci":        "yes",
	"CASDFT":       "yes",
	"CASDFTEmbedding": "yes",
	"CASFunctional": "wPBEh",
    "maxit":        500,
    "dcimaxiter":   300,
    "closed":       56,
    "active":       3,
    "cassinglets":  numstates,
    "castargetmult": 1,
    "cas_energy_states": [0,1],
    "cas_energy_mults": [1,1],
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
    "istate": 2,
    # Gaussian widths
    "widths": widths,
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
    "qm_energy_shift": 0.0e0,
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
#traj1.initial_wigner(seed)
traj1.read_initial_conds()

# set up simulation 
sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_parameters(sim_params)

#SSAIMS control
sim.enable_ssaims(
   epsilon=1e-10,             # tune for your system
   ss_seed=527518,            # optional
   suspend_during_spawn=True, # suspend SSAIMS during spawn process
   spawn_delay_steps=10,      # wait N time steps after spawn to avoide premature killing
   min_tbf_to_start=2,        # require at least M TBFs to begin selection
   verbose=True               # detailed output for SSAIM. Good for debugging
)

# if you want to turn it off:
#sim.disable_ssaims()
# begin propagation
sim.propagate()

