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

#     exec("self.set_" + cbackprop + "prev_wf(self.get_" + cbackprop + "wf())")
    prev_wf = self.td_wf_real + 1j*self.td_wf_imag
    exec("x = self.get_" + cbackprop + "positions()[0]")
    exec("y = self.get_" + cbackprop + "positions()[1]")
    
    n_el_steps = 2000
    time = self.get_time()
    el_timestep = self.timestep / n_el_steps
    a = 6
    k = 3

    # Constructing Hamiltonian
    H_elec = np.zeros((self.numstates, self.numstates))
    H_elec[0, 0] = 0.5 * (x + a/2)**2 + 0.5 * (y)**2
    H_elec[1, 1] = 0.5 * (x - a/2)**2 + 0.5 * (y)**2
    H_elec[0, 1] = k * y
    H_elec[1, 0] = k * y
#     print "\nH_elec = ", H_elec 
    energies, eigenvectors = lin.eigh(H_elec)
#     print "\neigenvalues = ", energies
 
    r = math.sqrt( x * x + y * y )
    theta = (math.atan2(y,x)) / 2.0

    self.set_energies(energies)      

    # This part performs the propagation of the electronic wave function for ehrenfest dynamics
    if time < 1e-8 or prev_wf.all() < 1e-8: # first time step electronic wave function propagated only by dt/2:
        print "\nPropagating first iteration:"

        # Constructing electronic wave function for the first timestep
        prev_wf[0] = math.sin(theta) 
        prev_wf[1] = math.cos(theta) 

        wf = propagate_symplectic(self, H_elec, prev_wf, el_timestep/2, n_el_steps/2) 
           
    else: # propagation for not the first step
        print "\nPropagating NOT first iteration:"
        wf = propagate_symplectic(self, H_elec, prev_wf, el_timestep, n_el_steps)
    
    # Computing forces
    Hx = np.zeros((self.numstates, self.numstates))
    Hx[0, 0] = x + a/2
    Hx[1, 1] = x - a/2
    
    Hy = np.zeros((self.numstates, self.numstates))
    Hy[0, 0] = y
    Hy[1, 1] = y
    Hy[0, 1] = k
    Hy[1, 0] = k
    
    forces_av = np.zeros((self.numdims))    
    prev_wf_T = np.transpose(np.conjugate(prev_wf))
#     print "F=", np.dot(np.dot(prev_wf_T, Hx), prev_wf), np.dot(np.dot(prev_wf_T, Hy), prev_wf)
    forces_av[0] = np.real(np.dot(np.dot(prev_wf_T, Hx), prev_wf))
    forces_av[1] = np.real(np.dot(np.dot(prev_wf_T, Hy), prev_wf))
      
    pop = np.zeros(self.numstates)
    amp = np.zeros((self.numstates), dtype=np.complex128) 
    eigenvectors_t = np.transpose(np.conjugate(eigenvectors))    
    
    for k in range(self.numstates):
        amp[k] = np.dot(eigenvectors_t[k, :], wf)
        pop[k] = np.real(np.dot(np.transpose(np.conjugate(amp[k])), amp[k]))
    
    av_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(wf)), H_elec), wf))
    self.set_av_energy(float(av_energy))
    av_energy2 = pop[0]*energies[0] + pop[1]*energies[1]
    print "eigenvecs = ", eigenvectors     
    print "average energy =", self.av_energy
    print "average energy 2 =", av_energy2
    print "energies = ", energies
    print "prev_wf =", prev_wf
    print "\nwf =", wf
    print "\namp_0 = ", amp[0]
    print "\n1st state population:", pop[0]
    print "2nd state population:", pop[1]
    print "norm = ", pop[0] + pop[1]
    print "istate =", self.istate
    print ""

#     phasing wave function to match previous time step
#     W = np.matmul(prev_wf,wf.T)
#      
#     if W[0,0] < 0.0:
#         wf[0,:] = -1.0 * wf[0,:]
#         W[:,0] = -1.0 * W[:,0]
#      
#     if W[1,1] < 0.0:
#         wf[1,:] = -1.0 * wf[1,:]
#         W[:,1] = -1.0 * W[:,1]

    self.set_av_force = forces_av
    self.set_approx_eigenvecs(eigenvectors)
    self.set_mce_amps_real(np.real(amp))
    self.set_mce_amps_imag(np.imag(amp))
    self.td_wf_real = np.real(wf)
    self.td_wf_imag = np.imag(wf)
    self.populations = pop

    
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
    self.h5_datasets["av_energy"] = 1
    self.h5_datasets["av_force"] = self.numdims
    self.h5_datasets["td_wf_real"] = self.numstates
    self.h5_datasets["td_wf_imag"] = self.numstates
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["forces_i"] = self.numdims
    self.h5_datasets["populations"] = self.numstates
    self.h5_datasets_half_step["time_half_step"] = 1

def potential_specific_traj_copy(self,from_traj):
    return

# 
# def get_backprop_wf0(self):
#     return self.backprop_wf[0,:].copy()
# 
# def get_backprop_wf1(self):
#     return self.backprop_wf[1,:].copy()

###end pyspawn_cone electronic structure section###
