# this script starts a new AIMS calculation.  Ethylene, SA3-CASSCF(2/2)/6-31G*
import numpy as np
import pyspawn
import pyspawn.general
import pyspawn.process_geometry as pg
import pyspawn.dictionaries as dicts
import sys


# Processing geometry.xyz file (positions are in hessian.hdf5 so we don't need them)
natoms, atoms, _, comment = pg.process_geometry('geometry.xyz')

# Getting atomic masses from the dictionary and converting to atomic units
# If specific isotopes are needed masses array can be set manually
mass_dict = dicts.get_atomic_masses()
masses = np.asarray([mass_dict[atom]*1822.0 for atom in atoms for i in range(3)])

widths_dict = {'C': 22.7, 'H': 4.7}
widths = np.asarray([widths_dict[atom] for atom in atoms for i in range(3)])

# finite wigner temperature
wigner_temp = 0

# random number seed
seed=12345

# Velocity Verlet classical propagator
clas_prop = "vv"

# adapative 2nd-order Runge-Kutta quantum propagator
qm_prop = "fulldiag"

# adiabtic NPI quantum Hamiltonian
qm_ham = "adiabatic"

# use TeraChem CASSCF or CASCI to compute potentials
potential = "molcas_cas"

# initial time
t0 = 0.0

# time step
ts = 5.0

# final simulation time
tfinal = 1800.00 

# number of dimensions                                                                                           
numdims = natoms*3

# number of electronic states                                                                                                                    
numstates = 2

# TeraChem job options                                                                                    
molcas_options = {
    "method":       'casscf',
    "pt2":          'xms',
    "basis":        '6-31g*',
    "atoms":        atoms,
    "charge":       0,
    "spinmult":     1,
    "closed_shell": True,
    "restricted":   True,

    "precision":    "double",
    "threall":      1.0e-20,
    "convthre":     1.0e-08,
    "casscf":        "yes",
    "closed":       7,
    "nactel":       6,
    "actorb":       6,
    "inactive":     18,
    "cassinglets":  numstates,
    "castargetmult": 1,
    "cas_energy_states": [0, 1],
    "cas_energy_mults": [1, 1],
    "python3" : ## ADD the path to python 3 
    "project": 'fulvene_test'
    }

# trajectory parameters
traj_params = {
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
    # atom labels
    "atoms": molcas_options["atoms"],
    # nuclear masses (in a.u)    
    "masses": masses,
    # molcas options (above)
    "molcas_options": molcas_options
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
    "qm_energy_shift": 0.000,
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

# set momentum by reading the file velocities.xyz
traj1.read_initial_conds()

## set up simulation 
sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_parameters(sim_params)

## begin propagation
sim.propagate()

