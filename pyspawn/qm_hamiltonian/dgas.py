import numpy as np
import pyspawn.complexgaussian as cg

######################################################
# adiabatic Hamiltonian
######################################################

# build Heff for the first half of the time step in the adibatic rep
# (with NPI)
def build_Heff_first_half(self):
    self.get_qm_data_from_h5()
    self.get_qm_data_from_h5_next_time()
    
    qm_time = self.get_quantum_time()
    dt = self.get_timestep()
    t_half = qm_time + 0.5 * dt
    self.set_quantum_time_half_step(t_half)
    self.get_qm_data_from_h5_half_step()        

    self.build_DGAS_coeffs()
    self.build_S_elec_DGAS()
    
    self.build_S_DGAS()
    self.invert_S()
    self.build_Sdot_nuc_DGAS()
    self.build_Sdot_elec_DGAS()
    self.build_Sdot_DGAS()
    self.build_H_DGAS()
    
    self.build_Heff()
        
# build Heff for the second half of the time step in the adibatic rep
# (with NPI)
def build_Heff_second_half(self):
    self.get_qm_data_from_h5()
    
    qm_time = self.get_quantum_time()
    dt = self.get_timestep()
    t_half = qm_time - 0.5 * dt
    self.set_quantum_time_half_step(t_half)
    self.get_qm_data_from_h5_half_step()        

    # don't need to do this - fix soon
    self.build_DGAS_coeffs()
    self.build_S_elec_DGAS()
    self.build_Sdot_DGAS()
    self.build_S_DGAS()
    self.invert_S()
    self.build_Sdot_DGAS()
    self.build_H_DGAS()
    
    self.build_Heff()

# get the position at the next time step
def get_qm_data_from_h5_next_time(self):
    qm_time = self.get_quantum_time() + self.get_timestep()
    ntraj = self.get_num_traj_qm()
    for key in self.traj:
        if self.traj_map[key] < ntraj:
            self.traj[key].get_all_qm_data_at_time_from_h5(qm_time,suffix="_next")
    for key in self.centroids:
        key1, key2 = str.split(key,"_a_")
        if self.traj_map[key1] < ntraj and self.traj_map[key2] < ntraj:
            self.centroids[key].get_all_qm_data_at_time_from_h5(qm_time,suffix="_next")

# build DGAS coefficients
def build_DGAS_coeffs(self):
    ntraj = self.get_num_traj_qm()
    dc = dict()
    for keyi in self.traj:
        i = self.traj_map[keyi]
        if i < ntraj:
            nstat = self.traj[keyi].get_numstates()
            ist = self.traj[keyi].get_istate()
            dc[keyi] = np.zeros(nstat)
            dc[keyi][ist] = 1.0
    self.dgas_coeffs = np.zeros((ntraj,nstat))
    for keyi in self.traj:
        i = self.traj_map[keyi]
        if i < ntraj:
            self.dgas_coeffs[i,:] = dc[keyi]
    

# build matrix of electronic overlaps
def build_S_elec_DGAS(self):
    ntraj = self.get_num_traj_qm()
    self.S_elec = np.zeros((ntraj,ntraj))
    for keyi in self.traj:
        i = self.traj_map[keyi]
        if i < ntraj:
            for keyj in self.traj:
                j = self.traj_map[keyj]
                if j < ntraj:
                    Stmp = np.dot(self.dgas_coeffs[i,:],self.dgas_coeffs[j,:])
                    self.S_elec[i,j] = Stmp

# build the overlap matrix, S
def build_S_DGAS(self):
    ntraj = self.get_num_traj_qm()
    self.S = np.zeros((ntraj,ntraj), dtype=np.complex128)
    self.S_nuc = np.zeros((ntraj,ntraj), dtype=np.complex128)
    for keyi in self.traj:
        i = self.traj_map[keyi]
        if i < ntraj:
            for keyj in self.traj:
                j = self.traj_map[keyj]
                if j < ntraj:
                    self.S_nuc[i,j] = cg.overlap_nuc(self.traj[keyi], self.traj[keyj],positions_i="positions_qm",positions_j="positions_qm",momenta_i="momenta_qm",momenta_j="momenta_qm") 
                    self.S[i,j] = self.S_nuc[i,j] * self.S_elec[i,j]

# build the right-acting time derivative operator
def build_Sdot_nuc_DGAS(self):
    ntraj = self.get_num_traj_qm()
    self.Sdot_nuc = np.zeros((ntraj,ntraj), dtype=np.complex128)
    for keyi in self.traj:
        i = self.traj_map[keyi]
        if i < ntraj:
            for keyj in self.traj:
                j = self.traj_map[keyj]
                if j < ntraj:
                    self.Sdot_nuc[i,j] = cg.Sdot_nuc(self.traj[keyi], self.traj[keyj],positions_i="positions_qm",positions_j="positions_qm",momenta_i="momenta_qm",momenta_j="momenta_qm",forces_j="forces_i_qm") * self.S_elec[i,j]

def build_Sdot_elec_DGAS(self):
    ntraj = self.get_num_traj_qm()
    self.Sdot_elect = np.zeros((ntraj,ntraj))
    

def build_Sdot_DGAS(self):
    self.Sdot = self.Sdot_nuc + self.Sdot_elec

# build the Hamiltonian matrix, H
# This routine assumes that S is already built
def build_H_DGAS(self):
    print "# building potential energy matrix"
    self.build_V_DGAS()
    print "# building NAC matrix"
    #self.build_tau_DGAS()
    print "# building kinetic energy matrix"
    self.build_T_DGAS()
    ntraj = self.get_num_traj_qm()
    shift = self.get_qm_energy_shift() * np.identity(ntraj)
    print "# summing Hamiltonian"
    #self.H = self.T + self.V + self.tau + shift
    self.H = self.T + self.V + shift

# build the potential energy matrix, V
# This routine assumes that S is already built
def build_V_DGAS(self):
    c1i = (complex(0.0,1.0))
    cm1i = (complex(0.0,-1.0))
    ntraj = self.get_num_traj_qm()
    self.V = np.zeros((ntraj,ntraj),dtype=np.complex128)
    for key in self.traj:
        i = self.traj_map[key]
        istate = self.traj[key].get_istate()
        if i < ntraj:
            self.V[i,i] = self.traj[key].get_energies_qm()[istate]
    for key in self.centroids:
        keyi, keyj = str.split(key,"_a_")
        i = self.traj_map[keyi]
        j = self.traj_map[keyj]
        if i < ntraj and j < ntraj:
            istate = self.centroids[key].get_istate()
            jstate = self.centroids[key].get_jstate()
            nstates = self.centroids[key].get_numstates()
            #if istate == jstate:
            #BGL this is not correct and must be fixed later
            E = self.centroids[key].get_energies_qm()
            for ist in range(nstates):
                Etmp = self.dgas_coeffs[i,ist] * self.dgas_coeffs[j,ist] * E[ist]
                self.V[i,j] += Etmp * self.S_nuc[i,j]
                self.V[j,i] += Etmp * self.S_nuc[j,i]

                
# build the nonadiabatic coupling matrix, tau
# This routine assumes that S is already built
#def build_tau_DGAS(self):
#    c1i = (complex(0.0,1.0))
#    cm1i = (complex(0.0,-1.0))
#    ntraj = self.get_num_traj_qm()
#    self.tau = np.zeros((ntraj,ntraj),dtype=np.complex128)
#    for key in self.centroids:
#        keyi, keyj = str.split(key,"_a_")
#        i = self.traj_map[keyi]
#        j = self.traj_map[keyj]
#        if i < ntraj and j < ntraj:
#            istate = self.centroids[key].get_istate()
#            jstate = self.centroids[key].get_jstate()
#            if istate != jstate:
#                Sij = cg.overlap_nuc(self.traj[keyi], self.traj[keyj],positions_i="positions_qm",positions_j="positions_qm",momenta_i="momenta_qm",momenta_j="momenta_qm")
#                tdc = self.centroids[key].get_timederivcoups_qm()[jstate]
#                self.tau[i,j] = Sij * cm1i * tdc
#                self.tau[j,i] = Sij.conjugate() * c1i * tdc

                
# build the kinetic energy matrix, T
def build_T_DGAS(self):
    ntraj = self.get_num_traj_qm()
    self.T = np.zeros((ntraj,ntraj), dtype=np.complex128)
    for keyi in self.traj:
        i = self.traj_map[keyi]
        if i < ntraj:
            for keyj in self.traj:
                j = self.traj_map[keyj]
                if j < ntraj:
                    self.T[i,j] = cg.kinetic_nuc(self.traj[keyi], self.traj[keyj],positions_i="positions_qm",positions_j="positions_qm",momenta_i="momenta_qm",momenta_j="momenta_qm") * self.S_elec[i,j]

