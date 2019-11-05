import numpy as np
import pyspawn        
import sys
import pyspawn.process_geometry as pg

# terachemserver port
if sys.argv[1]: 
    port = int(sys.argv[1])
else:
    "Please provide a port number as a command line argument"
    sys.exit()

# Processing geometry.xyz file
natoms, atoms, pos, comment = pg.process_geometry('geometry.xyz')

# if geometry file is in Angstrom converting to Bohr!
pos *= 1.889725989

print "Number of atoms =", natoms
print "Atom labels:", atoms
print "Positions in Bohr:\n", pos

# choose TeraChem potential
pyspawn.import_methods.into_hessian(pyspawn.potential.terachem_cas)

# number of dimensions (3 * number of atoms)
ndims = natoms * 3

# number of electronic states
numstates = 1 # state-average casscf hessian can be very bad, so doing state-specific

# create a Hessian object
hess = pyspawn.hessian(ndims, numstates)

# select the ground state
istate = 0

# the step size for numerical Hessian calculation
dr = 0.001

# TeraChem options for SA2-CAS(2/2)/6-31G calculation
tc_options = {
    "method":            'hf',
    "basis":             '6-31g',
    "atoms":             atoms,
    "charge":            0,
    "spinmult":          1,
    "closed_shell":      True,
    "restricted":        True,
    "precision":         "double",
    "threall":           1.0e-20,
    "thregr":            1.0e-20,
    "convthre":          1.0e-08,
    "dcimaxiter":        500,
    "casscfmaxiter":     200,
    "cpsacasscfmaxiter": 200,
    "casscf":            "yes",
    "closed":            7,
    "active":            2,
    "cassinglets":       1, # doing state specific GS hessian since SA gives wild results
    "castargetmult":     1,
    "cas_energy_states": [0, 1],
    "cas_energy_mults":  [1, 1],
    }

# build a dictionary containing all options
hess_options = {
    # terachem port
    "tc_port":    port,
    'istate':     istate,
    'positions':  pos,
    'tc_options': tc_options,
    }

# set number of dimensions
hess.set_numdims(ndims)

# pass options to hess object
hess.set_parameters(hess_options)

# compute Hessian semianalytically (using analytic first derivatives)
hess.build_hessian_hdf5_semianalytical(dr)

