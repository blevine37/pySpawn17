import numpy as np
import pyspawn        

pyspawn.import_methods.into_hessian(pyspawn.potential.terachem_cas)

hess = pyspawn.hessian()

ndims = 18

istate = 0

pos =  np.asarray([  0.000000000,    0.000000000,    0.101944554,
                     0.000000000,    0.000000000,    2.598055446,
                     0.000000000,    1.743557978,    3.672987826,
                     0.000000000,   -1.743557978,    3.672987826,
                     0.000000000,    1.743557978,   -0.972987826,
                     0.000000000,   -1.743557978,   -0.972987826])


dr = 0.001

atoms = ['C', 'C', 'H', 'H', 'H', 'H']    

tc_options = {
    "method":       'hf',
    "basis":        '6-31g**',
    "atoms":        atoms,
    "charge":       0,
    "spinmult":     1,
    "closed_shell": True,
    "restricted":   True,
    
    "precision":    "double",
    "threall":      1.0e-20,
    }

hess.set_numdims(ndims)
hess.set_istate(istate)
hess.set_positions(pos)
hess.set_tc_options(tc_options)
hess.build_hessian_hdf5_semianalytical(dr)



