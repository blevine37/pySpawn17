import numpy as np
import pyspawn        

pyspawn.import_methods.into_hessian(pyspawn.potential.terachem_cas)

hess = pyspawn.hessian()

ndims = 18

istate = 0

pos =  np.asarray([ 0.0, 0.0, 0.0,
                    0.0, 0.0, 2.7,
                    0.0, 1.5, 3.8,
                    0.0, 1.5, -1.1,
                    0.0, -1.5, 3.8,
                    0.0, -1.5, -1.1])

dr = 0.01

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
    
    "casci":        "yes",
    "fon":          "yes",
    "closed":       7,
    "active":       2,
    "cassinglets":  2
    }

hess.set_numdims(ndims)
hess.set_istate(istate)
hess.set_positions(pos)
hess.set_tc_options(tc_options)
hess.build_hessian_hdf5_semianalytical(dr)



