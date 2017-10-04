# this script starts a new AIMS calculations.  Ethylene, SA2-CASSCF(2/2).
import numpy as np
import pyspawn        

# random number seed
seed=28396

# Velocity Verlet classical propagator
clas_prop = "vv"

# adapative 2nd-order Runge-Kutta quantum propagator
qm_prop = "rk2"

# adiabtic NPI quantum Hamiltonian
qm_ham = "adiabatic"

# use TeraChem CASSCF or CASCI to compute potentials
potential = "terachem_cas"

# initial time
t0 = 0.0

# time step
timestep = 10.0

# final simulation time
tfinal = 8000.0

# coupling threshhold
spawnthresh = (0.5 * np.pi) / timestep / 20.0

# number of dimensions
ndims = 18

# number of electronic states
nstates = 2

# initial electronic state (indexed such that 0 is the ground state)
istate = 1

# energy shift for QM propagation
exshift = 78.0

# initial positions and momenta are all set to zero, later replaced by Wigner initial conditions
pos = mom = np.zeros(ndims)

# Gaussian widths
wid = np.asarray([30.0, 30.0, 30.0,
                  30.0, 30.0, 30.0,
                  6.0, 6.0, 6.0,
                  6.0, 6.0, 6.0,
                  6.0, 6.0, 6.0,
                  6.0, 6.0, 6.0])

# atom labels
atoms = ['C', 'C', 'H', 'H', 'H', 'H']

# nuclear masses (in a.u)    
m = np.asarray([21864.0, 21864.0, 21864.0,
                21864.0, 21864.0, 21864.0,
                1822.0, 1822.0, 1822.0,
                1822.0, 1822.0, 1822.0,
                1822.0, 1822.0, 1822.0,
                1822.0, 1822.0, 1822.0])

# TeraChem job options
tc_options = {
    "method":       'hf',
    "basis":        '6-31g',
    "atoms":        atoms,
    "charge":       0,
    "spinmult":     1,
    "closed_shell": True,
    "restricted":   True,
    
    "precision":    "double",
    "threall":      1.0e-20,
    
    "casscf":        "yes",
    "closed":       7,
    "active":       2,
    "cassinglets":  2,
    "castargetmult": 1,
    "cas_energy_labels":    [(0, 1), (1, 1)]
    }

# and off to the races...
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")
    
traj1 = pyspawn.traj()
traj1.init_traj(t0,ndims,pos,mom,wid,m,nstates,istate,"00")
traj1.set_spawnthresh(spawnthresh)
traj1.set_tc_options(tc_options)
traj1.set_atoms(atoms)
traj1.initial_wigner(seed)
sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_timestep_all(timestep)
sim.set_mintime_all(t0)
sim.set_maxtime_all(tfinal)
sim.set_qm_energy_shift(exshift)
sim.init_amplitudes_one()
sim.propagate()







