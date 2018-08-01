######################################################
# Ehrenfest Hamiltonian (diabatic basis)
######################################################
import sys
import numpy as np
import pyspawn.complexgaussian as cg

def build_Heff_first_half(self):
    """build Heff for the first half of the time step in the diabatic rep"""
    
    self.get_qm_data_from_h5()
    
    qm_time = self.quantum_time
    dt = self.timestep
    t_half = qm_time + 0.5 * dt
    self.quantum_time_half_step = t_half
    self.get_qm_data_from_h5_half_step()        
    
    self.build_S_elec_MCMDS()
    self.build_S_MCMDS()
    self.invert_S()
    self.build_Sdot()
    
    self.build_H()
    
    self.build_Heff()
        
def build_Heff_second_half(self):
    """build Heff for the second half of the time step in the dibatic rep"""
    
    self.get_qm_data_from_h5()
    
    qm_time = self.quantum_time
    dt = self.timestep
    t_half = qm_time - 0.5 * dt
    self.quantum_time_half_step = t_half
    self.get_qm_data_from_h5_half_step()        

    self.build_S_elec_MCMDS()    
    self.build_S_MCMDS()
    self.invert_S()
    self.build_Sdot()

    self.build_H_MCMDS()
    
    self.build_Heff()

def build_S_elec_MCMDS(self):
    """Build matrix of electronic overlaps"""
    ntraj = self.num_traj_qm
    self.S_elec = np.zeros((ntraj,ntraj))
    for keyi in self.traj:
        i = self.traj_map[keyi]
        if i < ntraj:
            for keyj in self.traj:
                j = self.traj_map[keyj]
                if j < ntraj:
                    if i == j:
                        self.S_elec[i,j] = 1.0
                    else:
                        Stmp = np.dot(np.transpose(np.conjugate(self.traj[keyi].td_wf)),\
                                      self.traj[keyj].td_wf)
                        self.S_elec[i,j] = Stmp
#     print "S_elec = ", self.S_elec

def build_S_MCMDS(self):
    """Build the overlap matrix, S"""
    
    ntraj = self.num_traj_qm
    self.S = np.zeros((ntraj,ntraj), dtype=np.complex128)
    self.S_nuc = np.zeros((ntraj,ntraj), dtype=np.complex128)
    for keyi in self.traj:
        i = self.traj_map[keyi]
        if i < ntraj:
            for keyj in self.traj:
                j = self.traj_map[keyj]
                if j < ntraj:
                    self.S_nuc[i,j] = cg.overlap_nuc(self.traj[keyi],\
                                                     self.traj[keyj],\
                                                     positions_i="positions_qm",\
                                                     positions_j="positions_qm",\
                                                     momenta_i="momenta_qm",\
                                                     momenta_j="momenta_qm") 
                    
                    self.S[i,j] = self.S_nuc[i,j] * self.S_elec[i,j]
                    
def build_H_MCMDS(self):
    """Building the Hamiltonian"""
    
#     print "Building potential energy matrix"
    self.build_V_MCMDS()
#     print "Building kinetic energy matrix"
    self.build_T_MCMDS()
    ntraj = self.num_traj_qm
    shift = self.qm_energy_shift * np.identity(ntraj)
    self.H = self.T + self.V + shift
#     print "Hamiltonian built"
    
def build_V_MCMDS(self):
    """build the potential energy matrix, V
    This routine assumes that S is already built"""
    c1i = (complex(0.0, 1.0))
    cm1i = (complex(0.0, -1.0))
    ntraj = self.num_traj_qm
    self.V = np.zeros((ntraj, ntraj), dtype=np.complex128)
    for keyi in self.traj:
        i = self.traj_map[keyi]
        if i < ntraj:
            for keyj in self.traj:
                j = self.traj_map[keyj]
                if j < ntraj:
                    if i == j:
                        self.V[i, j] = self.traj[keyi].av_energy_qm
                    else:
                        nuc_overlap = cg.overlap_nuc(self.traj[keyi], self.traj[keyj],\
                                                     positions_i="positions_qm",\
                                                     positions_j="positions_qm",\
                                                     momenta_i="momenta_qm",\
                                                     momenta_j="momenta_qm")
#                         print "nuc_overlap =", i, j, nuc_overlap 

                        self.V[i, j] = self.S_elec[i, j] * nuc_overlap
#     print "V =", self.V
    
def build_T_MCMDS(self):
    "Building kinetic energy, needs electronic overlap S_elec"
    
    ntraj = self.num_traj_qm
    self.T = np.zeros((ntraj,ntraj), dtype=np.complex128)
    for keyi in self.traj:
        i = self.traj_map[keyi]
        if i < ntraj:
            for keyj in self.traj:
                j = self.traj_map[keyj]
                if j < ntraj:
                    self.T[i,j] = cg.kinetic_nuc(self.traj[keyi], self.traj[keyj],\
                                                 positions_i="positions_qm",\
                                                 positions_j="positions_qm",\
                                                 momenta_i="momenta_qm",\
                                                 momenta_j="momenta_qm") * self.S_elec[i,j]
#     print "T =", self.T
                        