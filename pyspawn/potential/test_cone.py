import math
import numpy as np

#################################################
### electronic structure routines go here #######
#################################################

#each electronic structure method requires at least two routines:
#1) compute_elec_struct_, which computes energies, forces, and wfs
#2) init_h5_datasets_, which defines the datasets to be output to hdf5
#3) potential_specific_traj_copy, which copies data that is potential specific 
#   from one traj data structure to another 
#other ancillary routines may be included as well

### pyspawn_cone electronic structure ###
def compute_elec_struct(self,zbackprop):
    if not zbackprop:
        cbackprop = ""
    else:
        cbackprop = "backprop_"

    exec("self.set_" + cbackprop + "prev_wf(self.get_" + cbackprop + "wf())")

    exec("x = self.get_" + cbackprop + "positions()[0]")
    exec("y = self.get_" + cbackprop + "positions()[1]")
    r = math.sqrt( x * x + y * y )
    theta = (math.atan2(y,x)) / 2.0

    e = np.zeros(self.numstates)
    e[0] = ( r - 1.0 ) * ( r - 1.0 ) - 1.0
    e[1] = ( r + 1.0 ) * ( r + 1.0 ) - 1.0
    exec("self.set_" + cbackprop + "energies(e)")

    f = np.zeros((self.numstates,self.numdims))
    ftmp = -2.0 * ( r - 1.0 )
    f[0,0] = ( x / r ) * ftmp
    f[0,1] = ( y / r ) * ftmp
    ftmp = -2.0 * ( r + 1.0 )
    f[1,0] = ( x / r ) * ftmp
    f[1,1] = ( y / r ) * ftmp
    exec("self.set_" + cbackprop + "forces(f)")

    wf = np.zeros((self.numstates,self.length_wf))
    wf[0,0] = math.sin(theta)
    wf[0,1] = math.cos(theta)
    wf[1,0] = math.cos(theta)
    wf[1,1] = -math.sin(theta)
    exec("prev_wf = self.get_" + cbackprop + "prev_wf()")
    # phasing wave function to match previous time step
    W = np.matmul(prev_wf,wf.T)
    if W[0,0] < 0.0:
        wf[0,:] = -1.0 * wf[0,:]
        W[:,0] = -1.0 * W[:,0]
    if W[1,1] < 0.0:
        wf[1,:] = -1.0 * wf[1,:]
        W[:,1] = -1.0 * W[:,1]
    
    # computing NPI derivative coupling
    tmp = self.compute_tdc(W)
    tdc = np.zeros(self.numstates)
    if self.istate == 1:
        jstate = 0
    else:
        jstate = 1
    tdc[jstate] = tmp
    exec("self.set_" + cbackprop + "timederivcoups(tdc)")
    print "\nwf = \n", type(wf) 
    exec("self.set_" + cbackprop + "wf(wf)")

def init_h5_datasets(self):
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["forces_i"] = self.numdims
    self.h5_datasets["wf0"] = self.numstates
    self.h5_datasets["wf1"] = self.numstates
    self.h5_datasets_half_step["time_half_step"] = 1
    self.h5_datasets_half_step["timederivcoups"] = self.numstates

def potential_specific_traj_copy(self,from_traj):
    return

def get_wf0(self):
    return self.wf[0,:].copy()

def get_wf1(self):
    return self.wf[1,:].copy()

def get_backprop_wf0(self):
    return self.backprop_wf[0,:].copy()

def get_backprop_wf1(self):
    return self.backprop_wf[1,:].copy()

###end pyspawn_cone electronic structure section###
