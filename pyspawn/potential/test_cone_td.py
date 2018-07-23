import math
import sys
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


def compute_elec_struct(self):
    """electronic structure model for a 2d Jahn Teller model for MCE
    This subroutine solves the electronic SE and propagates electronic 
    ehrenfest wave function one nuclear time step that's split into 
    smaller electronic timesteps"""
    wf = self.td_wf
    prev_wf = wf
    print "\nPerforming electronic structure calculations:"
#     print "Norm of the wf before propagation = ", np.dot(np.transpose(np.conjugate(prev_wf)), prev_wf)
    x = self.positions[0]
    y = self.positions[1]
    
    n_el_steps = 1000
    eshift = -5.9891 + 0.645
    time = self.time
    el_timestep = self.timestep / n_el_steps
    a = 6
    k = 3

    # Constructing Hamiltonian, for now solving the eigenvalue problem to get adiabatic states
    # for real systems it will be replaced by approximate eigenstates
    
    H_elec = np.zeros((self.numstates, self.numstates))
    H_elec[0, 0] = 0.5 * (x + a/2)**2 + 0.5 * (y)**2 + eshift
    H_elec[1, 1] = 0.5 * (x - a/2)**2 + 0.5 * (y)**2 + eshift
    H_elec[0, 1] = k * y
    H_elec[1, 0] = k * y
#     H_elec += eshift    
    print "\nH_elec = ", H_elec 
    ss_energies, eigenvectors = lin.eigh(H_elec)
    eigenvectors_T = np.transpose(np.conjugate(eigenvectors))
 
    r = math.sqrt( x * x + y * y )
    theta = (math.atan2(y, x)) / 2.0      
    
    # Computing forces
    Hx = np.zeros((self.numstates, self.numstates))
    Hy = np.zeros((self.numstates, self.numstates))
    Hx[0, 0] = x + a/2
    Hx[1, 1] = x - a/2
    Hy[0, 0] = y
    Hy[1, 1] = y
    Hy[0, 1] = k
    Hy[1, 0] = k   
    
    if wf.all() < 1e-8:
        print "constructing electronic wf for the first timestep"
        # Constructing electronic wave function for the first timestep
#         wf[0] = math.sin(theta) 
#         wf[1] = math.cos(theta)
        wf = eigenvectors[:, 1]
    else:
        print "\nPropagating electronic wave function first half of timestep to compute forces, energies"
        wf = propagate_symplectic(self, H_elec, wf, self.timestep/2, n_el_steps/2)
    
    wf_T = np.transpose(np.conjugate(wf))
    av_energy = np.real(np.dot(np.dot(wf_T, H_elec), wf))    
    forces_av = np.zeros((self.numdims))    
    
    forces_av[0] = -np.real(np.dot(np.dot(wf_T, Hx), wf))
    forces_av[1] = -np.real(np.dot(np.dot(wf_T, Hy), wf))
      
    pop = np.zeros(self.numstates)
    amp = np.zeros((self.numstates), dtype=np.complex128) 
    for j in range(self.numstates):
        amp[j] = np.dot(eigenvectors_T[:, j], wf)
        pop[j] = np.real(np.dot(np.transpose(np.conjugate(amp[j])), amp[j]))
    
    self.av_energy = float(av_energy)
    self.energies = ss_energies
    self.av_force = forces_av
    self.approx_eigenvecs = eigenvectors
    self.mce_amps = amp
    self.populations = pop

    # DEBUG
#     mdx, pdx = -1e-6, 1e-6
#     pdx_H_elec = construct_el_H(self, self.positions[0] , self.positions[1]+ pdx)
#     mdx_H_elec = construct_el_H(self, self.positions[0] , self.positions[1]+ mdx)
#     pdx_energies, pdx_eigenvecs = lin.eigh(pdx_H_elec)
#     mdx_energies, mdx_eigenvecs = lin.eigh(mdx_H_elec)
#     pdx_wf = pdx_eigenvecs[:, 1]
#     mdx_wf = mdx_eigenvecs[:, 1]
#     pdx_wf_T = np.transpose(np.conjugate(pdx_wf))
#     mdx_wf_T = np.transpose(np.conjugate(mdx_wf))
#     pdx_av_energy = np.real(np.dot(np.dot(pdx_wf_T, pdx_H_elec), pdx_wf))
#     mdx_av_energy = np.real(np.dot(np.dot(mdx_wf_T, mdx_H_elec), mdx_wf)) 
#     num_force = (pdx_av_energy - mdx_av_energy) / (pdx - mdx)
#     print "num_force =", num_force
#     print "real force =", self.av_force
#     sys.exit()
    # DEBUG

    print "norm =", np.dot(wf_T, wf)
    print "time =", self.time     
    print "average energy =", self.av_energy
    print "total E =", self.calc_kin_en(self.momenta_tpdt, self.masses) + self.av_energy
    print "wf =", wf
    print "forces_av =", forces_av
    print "pop_total = ", pop[0] + pop[1]
    print "\n1st state population:", pop[0]
    print "2nd state population:", pop[1]

    # This part performs the propagation of the electronic wave function 
    # for ehrenfest dynamics at a half step and save it
        
    print "\nPropagating electronic wf second half of timestep"
    wf = propagate_symplectic(self, H_elec, wf, self.timestep/2, n_el_steps/2)
    self.td_wf = wf
    
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
  
def construct_el_H(self, x, y):
    
    a = 6
    k = 3
    
    H_elec = np.zeros((self.numstates, self.numstates))
    H_elec[0, 0] = 0.5 * (x + a/2)**2 + 0.5 * (y)**2
    H_elec[1, 1] = 0.5 * (x - a/2)**2 + 0.5 * (y)**2
    H_elec[0, 1] = k * y
    H_elec[1, 0] = k * y
    
    return H_elec    

def propagate_symplectic(self, H, wf, timestep, nsteps):
    
    el_timestep = timestep / nsteps
    c_r = np.real(wf)
    c_i = np.imag(wf)
            
    for i in range(nsteps):
        
        c_r_dot = np.dot(H, c_i)
        c_r = c_r + 0.5 * el_timestep * c_r_dot
        c_i_dot = -1.0 * np.dot(H, c_r)
        c_i = c_i + el_timestep * c_i_dot  
        c_r_dot = np.dot(H, c_i)
        c_r = c_r + 0.5 * el_timestep * c_r_dot  
    
    wf = c_r + 1j * c_i
    return wf
    
def init_h5_datasets(self):
    self.h5_datasets["av_energy"] = 1
    self.h5_datasets["av_force"] = self.numdims
    self.h5_datasets["td_wf"] = self.numstates
    self.h5_datasets["mce_amps"] = self.numstates    
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["forces_i"] = self.numdims
    self.h5_datasets["populations"] = self.numstates
    self.h5_datasets_half_step["time_half_step"] = 1

def potential_specific_traj_copy(self,from_traj):
    return

###end pyspawn_cone electronic structure section###
