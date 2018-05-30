import math
import numpy as np
from scipy import linalg as lin
from Cython.Compiler.PyrexTypes import c_ref_type
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
    
    n_el_steps = 2000
    time = self.get_time()
    el_timestep = self.timestep / n_el_steps
    a = 6
    k = 3
    wf = np.zeros((self.numstates,self.length_wf), dtype = np.complex128)

    # Constructing Hamiltonian
    H_elec = np.zeros((self.numstates, self.numstates))
    H_elec[0, 0] = 0.5 * (x + a/2)**2 + 0.5 * (y)**2
    H_elec[1, 1] = 0.5 * (x - a/2)**2 + 0.5 * (y)**2
    H_elec[0, 1] = k * y
    H_elec[1, 0] = k * y
    print "H_elec = ", H_elec 
    energies, eigenvectors = lin.eigh(H_elec)
    print "\neigenvalues = ", energies
      
    r = math.sqrt( x * x + y * y )
    theta = (math.atan2(y,x)) / 2.0

    exec("self.set_" + cbackprop + "energies(energies)")    
    
    # Computing forces
    f = np.zeros((self.numstates,self.numdims))
    ftmp = - (r - a/2)
    f[0,0] = (x/r) * ftmp
    f[0,1] = (y/r) * ftmp
    ftmp = - (r + a/2)
    f[1,0] = (x/r) * ftmp
    f[1,1] = (y/r) * ftmp
    exec("self.set_" + cbackprop + "forces(f)")

    # This part performs the propagation of the electronic wave function for ehrenfest dynamics
    if time < 1e-8: # first time step electronic wave function propagated only by dt/2:
        print "\nPropagating first iteration:"
        # Constructing electronic wave function for the first timestep
        prev_wf = np.zeros((self.numstates, self.length_wf), dtype=np.complex128)
        prev_wf[0,0] = math.sin(theta) / np.sqrt(2)
        prev_wf[0,1] = math.cos(theta) / np.sqrt(2)
        prev_wf[1,0] = math.cos(theta) / np.sqrt(2)
        prev_wf[1,1] = -math.sin(theta) / np.sqrt(2)
#         print "\nprev_wf =", prev_wf
#         print "\nprev_norm = ", np.dot(np.transpose(np.conjugate(prev_wf[:, 0])), prev_wf[:, 0]) + np.dot(np.transpose(np.conjugate(prev_wf[:, 1])), prev_wf[:, 1])
        
        wf = propagate_symplectic(self, H_elec, prev_wf, el_timestep/2, n_el_steps/2) 
           
    else: # propagation for not the first step
        
        exec("prev_wf = self.get_" + cbackprop + "prev_wf()")
        wf = propagate_symplectic(self, H_elec, prev_wf, el_timestep, n_el_steps)
    
    eigenvectors_t = np.transpose(np.conjugate(eigenvectors))    
    amp_0 = np.dot(eigenvectors_t[0, :], wf)
    pop_0 = np.dot(np.transpose(np.conjugate(amp_0)), amp_0)
    print "\nwf =", wf
    print "\n0th state population:", pop_0
    print "\nnorm = ", np.dot(np.transpose(np.conjugate(wf[:, 0])), wf[:, 0]) + np.dot(np.transpose(np.conjugate(wf[:, 1])), wf[:, 1])

#     phasing wave function to match previous time step
    W = np.matmul(prev_wf,wf.T)
     
    if W[0,0] < 0.0:
        wf[0,:] = -1.0 * wf[0,:]
        W[:,0] = -1.0 * W[:,0]
     
    if W[1,1] < 0.0:
        wf[1,:] = -1.0 * wf[1,:]
        W[:,1] = -1.0 * W[:,1]
        
    exec("self.set_" + cbackprop + "wf(wf)")

def propagate_symplectic(self, H, wf, timestep, nsteps):

    c_r = np.real(wf)
    c_i = np.imag(wf)
            
    for i in range(nsteps):
        
        c_r_dot = np.dot(H, c_i)
        c_r = c_r + 0.5 * timestep * c_r_dot
        c_i_dot = -1.0 * np.dot(H, c_r)
        c_i = c_i + timestep * c_i_dot  
        c_r_dot = np.dot(H, c_i)
        c_r = c_r + 0.5 * timestep * c_r_dot  
    
    wf = c_r + 1j * c_i
    return wf
    
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
