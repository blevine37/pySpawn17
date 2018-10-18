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
    """Electronic structure model for a 1d system with multiple parallel states 
    that intersect a single state with a different slope 
    This subroutine solves the electronic SE and propagates electronic 
    ehrenfest wave function one nuclear time step that's split into 
    smaller electronic timesteps"""
    
    wf = self.td_wf
    print "Performing electronic structure calculations:"
    x = self.positions[0]

    n_el_steps = self.n_el_steps
    time = self.time
    el_timestep = self.timestep / n_el_steps

    # Constructing Hamiltonian, computing derivatives, for now solving the eigenvalue problem to get adiabatic states
    # for real systems it will be replaced by approximate eigenstates
    H_elec, Force = self.construct_el_H(x) 
    ss_energies, eigenvectors = lin.eigh(H_elec)
    eigenvectors_T = np.transpose(np.conjugate(eigenvectors))
    
    pop = np.zeros(self.numstates)
    amp = np.zeros((self.numstates), dtype=np.complex128) 
        
    if np.dot(np.transpose(np.conjugate(wf)), wf)  < 1e-8:
        print "WF = 0, constructing electronic wf for the first timestep", wf
        wf = eigenvectors[:, 4]
    else:
        if not self.first_step:
            print "\nPropagating electronic wave function not first step"
            wf = propagate_symplectic(self, (H_elec), wf, self.timestep/2, n_el_steps/2)
        if self.first_step:
            print "\n first step, skipping electronic wave function propagation"
    
    wf_T = np.transpose(np.conjugate(wf))
    av_energy = np.real(np.dot(np.dot(wf_T, H_elec), wf))    
    
    av_force = np.zeros((self.numdims))    
    for n in range(self.numdims):
        av_force[n] = -np.real(np.dot(np.dot(wf_T, Force[n]), wf))
      
    for j in range(self.numstates):
        amp[j] = np.dot(np.conjugate(np.transpose(eigenvectors[:, j])), wf)
        pop[j] = np.real(np.dot(np.transpose(np.conjugate(amp[j])), amp[j]))
    norm = np.dot(wf_T, wf)
    
    self.av_energy = float(av_energy)
    self.energies = np.real(ss_energies)
    self.av_force = av_force
    self.approx_eigenvecs = eigenvectors
    self.mce_amps_prev = self.mce_amps
    self.mce_amps = amp
    self.td_wf_full_ts = np.complex128(wf)
    self.populations = pop

    if abs(norm - 1.0) > 1e-6:
        print "WARNING: Norm is not conserved!!! N =", norm  
    
    # DEBUGGING
    
    def print_stuff():
#         print "ES Time =", self.time
#         print "Position =", self.positions
#         print "Hamiltonian =\n", H_elec
#         print "positions =", self.positions
#         print "momentum =", self.momenta
#         print "H_elec =\n", H_elec
#         print "Average energy =", self.av_energy
#         print "Energies =", ss_energies
#         print "Force =", av_force
#         print "Wave function =\n", wf
#         print "Eigenvectors =\n", eigenvectors
        print "Population = ", pop 
#         print "norm =", sum(pop)
#         print "amps =", amp
    print_stuff()
    # DEBUGGING
    
    # This part performs the propagation of the electronic wave function 
    # for ehrenfest dynamics at a half step and save it
    wf = propagate_symplectic(self, H_elec, wf, self.timestep/2, n_el_steps/2)
    self.td_wf = wf
    self.H_elec = H_elec
    
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
  
def construct_el_H(self, x):
    """Constructing the 2D potential (Jahn-Teller model) and computing d/dx, d/dy for
    force computation. Later will be replaced with the electronic structure program call"""
    
    a = 6
    k = 0.2
    w = 1
    delta = 0.1
    
    H_elec = np.zeros((self.numstates, self.numstates))
    H_elec[0, 0] = w * (-x) 
    H_elec[1, 1] = w * x 
    H_elec[2, 2] = w * x - delta
    H_elec[3, 3] = w * x - delta*2
    H_elec[4, 4] = w * x - delta*3
    
    H_elec[0, 1] = k * x
    H_elec[0, 2] = k * x
    H_elec[0, 3] = k * x
    H_elec[0, 4] = k * x
    
    H_elec[1, 0] = k * x
    H_elec[2, 0] = k * x
    H_elec[3, 0] = k * x
    H_elec[4, 0] = k * x
    
    Hx = np.zeros((self.numstates, self.numstates))
#     Hy = np.zeros((self.numstates, self.numstates))
#     Hz = np.zeros((self.numstates, self.numstates))
    Hx[0, 0] = -w
    Hx[1, 1] = w
    Hx[2, 2] = w
    Hx[3, 3] = w
    Hx[4, 4] = w
    
    Hx[0, 1] = k
    Hx[1, 0] = k
    Hx[0, 2] = k
    Hx[2, 0] = k
    Hx[0, 3] = k
    Hx[3, 0] = k
    Hx[0, 4] = k
    Hx[4, 0] = k

    
    Force = [Hx]    
    return H_elec, Force

def propagate_symplectic(self, H, wf, timestep, nsteps):
    """Symplectic split propagator, similar to classical Velocity-Verlet"""
    
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
    self.h5_datasets["populations"] = self.numstates
    self.h5_datasets["td_wf_full_ts"] = self.numstates
    
    self.h5_datasets_half_step["time_half_step"] = 1

def potential_specific_traj_copy(self,from_traj):
    return

###end pyspawn_cone electronic structure section###
