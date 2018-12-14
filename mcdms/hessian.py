import numpy as np
import math
from pyspawn.fmsobj import fmsobj
from pyspawn.traj import traj
import h5py
import os

class hessian(traj):
    def build_hessian_hdf5_semianalytical(self,dr):
        ndims = self.get_numdims()
        istate = self.get_istate()
        self.set_timestep(1.0)
        self.compute_elec_struct(False)
        g = -1.0 * self.get_forces_i()

        filename = "hessian.hdf5"

        if not os.path.isfile(filename):
            h5f = h5py.File(filename, "a")
            dsetname = "geometry"
            dset = h5f.create_dataset(dsetname,(1,ndims))
            pos = self.get_positions().reshape(1,ndims)
            dset[:,:] = pos
            
            dsetname = "hessian"
            dset = h5f.create_dataset(dsetname,(ndims,ndims))           
            dset[:,:] = -1000.0 * np.ones((ndims,ndims))
            mindim = 0
        else:
            h5f = h5py.File(filename, "a")
            mindim = -1
            dsetname = "geometry"
            dset = h5f.get(dsetname)
            pos = dset[:,:].flatten()
            self.set_positions(pos)
            dsetname = "hessian"
            dset = h5f.get(dsetname)           
            for idim in range(ndims):
                if mindim < 0:
                    tmp = dset[idim,0]
                    if tmp < -999.0 and tmp > -1001.0:
                        mindim = idim

        h5f.close()

        for idim in range(mindim,ndims):
            pos = self.get_positions()
            pos[idim] += dr
            self.set_positions(pos)
            self.compute_elec_struct(False)
            gp = -1.0 * self.get_forces_i()

            pos[idim] -= 2.0*dr
            self.set_positions(pos)
            self.compute_elec_struct(False)
            gm = -1.0 * self.get_forces_i()
            
            pos[idim] += dr
            self.set_positions(pos)

            de2dr2 = (gp - gm) / (2.0 * dr)

            h5f = h5py.File(filename, "a")
            mindim = -1
            dsetname = "hessian"
            dset = h5f.get(dsetname)           

            h5f = h5py.File(filename, "a")
            mindim = -1
            dsetname = "hessian"
            dset = h5f.get(dsetname)           
            dset[idim,:] = de2dr2
            h5f.close()

        print "Done building hessian.hdf5!"

            

            
            
