import numpy as np
import pyspawn        

# choose TeraChem potential
pyspawn.import_methods.into_hessian(pyspawn.potential.terachem_cas)

# create a Hessian object
hess = pyspawn.hessian()

# number of dimensions (3 * number of atoms)
ndims = 18

# select the ground state
istate = 0

# this is the ground state minimum energy structure of ethylene in Bohr.  It has previously been optimized!
pos =  np.asarray([  0.000000000,    0.000000000,    0.101944554,
                     0.000000000,    0.000000000,    2.598055446,
                     0.000000000,    1.743557978,    3.672987826,
                     0.000000000,   -1.743557978,    3.672987826,
                     0.000000000,    1.743557978,   -0.972987826,
                     0.000000000,   -1.743557978,   -0.972987826])

# the step size for numerical Hessian calculation
dr = 0.001

# atom labels
atoms = ['C', 'C', 'H', 'H', 'H', 'H']    

# TeraChem options for SA2-CAS(2/2)/6-31G calculation
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

    "casci":        "yes",
    "fon":          "yes",
    "closed":       7,
    "active":       2,
    "cassinglets":  2,
    "castargetmult": 1,
    "cas_energy_states": [0, 1],
    "cas_energy_mults": [1, 1],
    }

# build a dictionary containing all options
hess_options = {
    'istate': istate,
    'positions': pos,
    'tc_options':tc_options,
    }

# set number of dimensions
hess.set_numdims(ndims)

# pass options to hess object
hess.set_parameters(hess_options)

# compute Hessian semianalytically (using analytic first derivatives)
hess.build_hessian_hdf5_semianalytical(dr)



