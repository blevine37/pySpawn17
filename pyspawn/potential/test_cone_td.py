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
    """Electronic structure model for a 2d Jahn Teller model for MCE
    This subroutine solves the electronic SE and propagates electronic 
    ehrenfest wave function one nuclear time step that's split into 
    smaller electronic timesteps"""

    n_krylov = self.krylov_sub_n
    
    wf = self.td_wf
    print "Performing electronic structure calculations:"
    x = self.positions[0]
    y = self.positions[1]
    r = math.sqrt( x * x + y * y )
    theta = (math.atan2(y, x)) / 2.0 
    n_el_steps = self.n_el_steps
    time = self.time
    el_timestep = self.timestep / n_el_steps

    # Constructing Hamiltonian, computing derivatives, for now solving the eigenvalue problem to get adiabatic states
    # for real systems it will be replaced by approximate eigenstates
    H_elec, Force = self.construct_el_H((x, y)) 
    ss_energies, eigenvectors = lin.eigh(H_elec)
    eigenvectors_T = np.transpose(np.conjugate(eigenvectors))
    
    pop = np.zeros(self.numstates)
    amp = np.zeros((self.numstates), dtype=np.complex128) 
        
    if np.dot(np.transpose(np.conjugate(wf)), wf)  < 1e-8:
        print "WF = 0, constructing electronic wf for the first timestep", wf
        wf = eigenvectors[:, 4]
#         wf = 1 /np.sqrt(self.numstates) * sum(eigenvectors[:, n] for n in range(self.numstates)) 
    else:
        if not self.first_step:
            print "\nPropagating electronic wave function not first step"
            wf = propagate_symplectic(self, (H_elec), wf, self.timestep/2, n_el_steps/2,\
                                      n_krylov)
            self.wf_store_full_ts = self.wf_store.copy()
            
        if self.first_step:
            print "\nFirst step, skipping electronic wave function propagation"
            symplectic_backprop(self, H_elec, wf, el_timestep, n_krylov, n_krylov)
    
    wf_T = np.transpose(np.conjugate(wf))
    av_energy = np.real(np.dot(np.dot(wf_T, H_elec), wf))    

    ########### Approximate eigenstates ########################
    q, r = np.linalg.qr(self.wf_store_full_ts)
    Hk = np.dot(np.transpose(np.conjugate(q)), np.dot(H_elec, q))
    Fk = np.dot(np.transpose(np.conjugate(q)), np.dot(Force, q))
    
    approx_e, approx_eigenvecs = np.linalg.eigh(Hk)
    self.approx_energies = approx_e
    self.approx_eigenvecs = approx_eigenvecs
    
    approx_wf = np.dot(np.conjugate(np.transpose(q)), wf)
    approx_pop = np.zeros(self.krylov_sub_n)
    approx_amp = np.zeros((self.krylov_sub_n), dtype=np.complex128)
        
    for j in range(self.krylov_sub_n):
        approx_amp[j] = np.dot(approx_eigenvecs[:, j], approx_wf)
        approx_pop[j] = np.real(np.dot(np.transpose(np.conjugate(approx_amp[j])), approx_amp[j]))

    self.approx_amp = approx_amp
    self.approx_pop = approx_pop    
    self.approx_wf_full_ts = np.complex128(approx_wf)

    ############################################################
    
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
    self.eigenvecs = eigenvectors
    self.mce_amps_prev = self.mce_amps
    self.mce_amps = amp
    self.td_wf_full_ts = np.complex128(wf)
    self.populations = pop

    if abs(norm - 1.0) > 1e-6:
        print "WARNING: Norm is not conserved!!! N =", norm  
    
    # DEBUGGING
    
    def print_stuff():
#         print "ES Time =", self.time
        print "Position =", self.positions
#         print "Hamiltonian =\n", H_elec
#         print "positions =", self.positions
#         print "momentum =", self.momenta
#         print "H_elec =\n", H_elec
        print "Average energy =", self.av_energy
        print "Energies =", ss_energies
#         print "Force =", av_force
        print "Wave function =\n", wf
        print "Eigenvectors =\n", eigenvectors
        print "Population = ", pop 
        print "norm =", sum(pop)
#         print "amps =", amp
    print_stuff()
    # DEBUGGING
    
    # This part performs the propagation of the electronic wave function 
    # for ehrenfest dynamics at a half step and save it
    wf = propagate_symplectic(self, H_elec, wf, self.timestep/2, n_el_steps/2, n_krylov)
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
  
def construct_el_H(self, pos):
    """Constructing the 2D potential (Jahn-Teller model) and computing d/dx, d/dy for
    force computation. Later will be replaced with the electronic structure program call"""
    
    x = pos[0]
    y = pos[1]
    a = 1
    k = 1
    w = 0.4
    delta = 0.3
    
    H_elec = np.zeros((self.numstates, self.numstates))
    H_elec[0, 0] = w * (x)**2 + w * (y)**2
    H_elec[1, 1] = w * (x - a/2)**2 + w * (y)**2 - delta
    H_elec[2, 2] = w * (x - 2*a/2)**2 + w * (y)**2 - 2*delta
    H_elec[3, 3] = w * (x - 3*a/2)**2 + w * (y)**2 - 3*delta
    H_elec[4, 4] = w * (x - 4*a/2)**2 + w * (y)**2 - 4*delta
    H_elec[0, 1] = k * y
    H_elec[1, 0] = k * y
    H_elec[0, 2] = k * y
    H_elec[2, 0] = k * y
    H_elec[0, 3] = k * y
    H_elec[3, 0] = k * y
    H_elec[0, 4] = k * y
    H_elec[4, 0] = k * y

    Hx = np.zeros((self.numstates, self.numstates))
    Hy = np.zeros((self.numstates, self.numstates))
#     Hz = np.zeros((self.numstates, self.numstates))
    Hx[0, 0] = 2 * w * (x)
    Hx[1, 1] = 2 * w * (x - a/2)
    Hx[2, 2] = 2 * w * (x - 2*a/2)
    Hx[3, 3] = 2 * w * (x - 3*a/2)
    Hx[4, 4] = 2 * w * (x - 4*a/2)
    
    Hy[0, 0] = 2 * w * y
    Hy[1, 1] = 2 * w * y
    Hy[2, 2] = 2 * w * y
    Hy[3, 3] = 2 * w * y
    Hy[4, 4] = 2 * w * y
    
    Hy[0, 1] = k
    Hy[1, 0] = k
    Hy[0, 2] = k
    Hy[2, 0] = k
    Hy[0, 3] = k
    Hy[3, 0] = k
    Hy[0, 4] = k
    Hy[4, 0] = k
    
    Force = [Hx, Hy]    
    return H_elec, Force

def propagate_symplectic(self, H, wf, timestep, nsteps, n_krylov):
    """Symplectic split propagator, similar to classical Velocity-Verlet"""
    
#     approx_eig = np.zeros((self.numstates, n_krylov), dtype = np.complex128)
    el_timestep = timestep / nsteps
    c_r = np.real(wf)
    c_i = np.imag(wf)
    n = 0        
    for i in range(nsteps):
        
        c_r_dot = np.dot(H, c_i)
        c_r = c_r + 0.5 * el_timestep * c_r_dot
        c_i_dot = -1.0 * np.dot(H, c_r)
        c_i = c_i + el_timestep * c_i_dot  
        c_r_dot = np.dot(H, c_i)
        c_r = c_r + 0.5 * el_timestep * c_r_dot  
        
        if nsteps - i <= n_krylov:
            self.wf_store[:, n] = c_r + 1j * c_i 
            n += 1
#     print "wf_store =\n", self.wf_store   
#     print "wf_store_full_ts =", self.wf_store_full_ts
    wf = c_r + 1j * c_i
#     print "wf =", wf
    return wf

def symplectic_backprop(self, H, wf, el_timestep, nsteps, n_krylov):
    """Symplectic split propagator, similar to classical Velocity-Verlet"""
    
#     approx_eig = np.zeros((self.numstates, n_krylov), dtype = np.complex128)
    c_r = np.real(wf)
    c_i = np.imag(wf)
    n = 0        
    self.wf_store_full_ts = np.zeros((self.numstates, self.krylov_sub_n), dtype = np.complex128)
    for n in range(nsteps):
        
        c_r_dot = np.dot(H, c_i)
        c_r = c_r - 0.5 * el_timestep * c_r_dot
        c_i_dot = -1.0 * np.dot(H, c_r)
        c_i = c_i - el_timestep * c_i_dot  
        c_r_dot = np.dot(H, c_i)
        c_r = c_r - 0.5 * el_timestep * c_r_dot  
    
        self.wf_store_full_ts[:, n_krylov-n-1] = c_r + 1j * c_i 
    
#     print "wf_store =\n", self.wf_store   
#     print "wf_store_full_ts =\n", self.wf_store_full_ts
    self.wf_store = self.wf_store_full_ts.copy()
#     print "wf =", wf
    return
   
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

def potential_specific_traj_copy(self,from_traj):
    return

###end pyspawn_cone electronic structure section###
