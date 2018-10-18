# trajectory objects contain individual trajectory basis functions
import numpy as np
import sys
import math
from pyspawn.fmsobj import fmsobj
import os
import shutil
import h5py
from scipy import linalg as lin
from numpy import dtype
from pyspawn.potential.linear_slope import propagate_symplectic
import cmath
from scipy.optimize import fsolve, root, broyden1

class traj(fmsobj):
        
    numdims = 1
    t_decoherence_par = 0.1
        
    def __init__(self):
        self.time = 0.0
        self.time_half_step = 0.0
        self.maxtime = -1.0
        self.mintime = 0.0
        self.firsttime = 0.0
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.widths = np.zeros(self.numdims)
        self.masses = np.zeros(self.numdims)
        self.label = "00"
        self.h5_datasets = dict()
        self.h5_datasets_half_step = dict()

        self.timestep = 0.0
        
        self.numstates = 5
        
        self.length_wf = self.numstates
        self.wf = np.zeros((self.numstates, self.length_wf))
        self.prev_wf = np.zeros((self.numstates, self.length_wf))
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates, self.numdims))
        self.S_elec_flat = np.zeros(self.numstates*self.numstates)

        self.positions_tpdt = np.zeros(self.numdims)
        self.positions_t = np.zeros(self.numdims)
        self.momenta_tpdt = np.zeros(self.numdims)
        self.momenta_t = np.zeros(self.numdims)
        
        self.energies_tpdt = np.zeros(self.numstates)
        self.energies_t = np.zeros(self.numstates)
        
        self.av_energy_tpdt = 0.0
        self.av_energy_t = 0.0
        
        self.av_force_t = np.zeros(self.numdims)
        self.av_force_tpdt = np.zeros(self.numdims)
        
        self.td_wf_full_ts_t = np.zeros((self.numstates), dtype = np.complex128)
        self.td_wf_full_ts_tpdt = np.zeros((self.numstates), dtype = np.complex128)
        
        self.spawnlastcoup = np.zeros(self.numstates)
        self.numchildren = 0
        
        self.positions_qm = np.zeros(self.numdims)
        self.momenta_qm = np.zeros(self.numdims)
        self.energies_qm = np.zeros(self.numstates)
        self.forces_i_qm = np.zeros(self.numdims)

        #In the following block there are variables needed for ehrenfest
        self.H_elec = np.zeros((self.numstates, self.numstates), dtype = np.complex128)
        self.first_step = False
        self.new_amp = np.zeros((1), dtype = np.complex128)
        self.rescale_amp = np.zeros((1), dtype = np.complex128)
        self.n_el_steps = 1000
        self.td_wf_full_ts = np.zeros((self.numstates), dtype = np.complex128)
        self.td_wf = np.zeros((self.numstates), dtype = np.complex128)
        self.mce_amps = np.zeros((self.numstates), dtype = np.complex128)
        self.populations = np.zeros(self.numstates)
        self.av_energy = 0.0
        self.av_force = np.zeros(self.numdims)
        self.approx_eigenvecs = np.zeros((self.numstates, self.numstates))
        self.z_clone_now = np.zeros(self.numstates)        
        self.clonethresh = 0.0
        self.clone_p = np.zeros((self.numstates, self.numstates))
    
    def solve_nonlin_system(self, S12nuc, norm_abk, norm_abi, norm_abj, guess):
    
        def Jacobian(X, *data):
            S12nuc, norm_abk, norm_abi, norm_abj = data
            J = np.zeros((6, 6))
#             print (X)
#             print "data=\n", data
            c1 = X[0]
            c2 = X[1]
            b1k = X[2]
            b2k = X[3]
            b1j = X[4]
            b2i = X[5]
            
            J[0, 0] = 0.0                   # d/dc1
            J[0, 1] = 0.0                   # d/dc2
            J[0, 2] = 2 * b1k * norm_abk    # d/db1k
            J[0, 3] = 0.0                   # d/db2k
            J[0, 4] = 2 * b1j * norm_abj    # d/b1j
            J[0, 5] = 0.0                   # d/b2i
            
            J[1, 0] = 0.0                   # d/dc1
            J[1, 1] = 0.0                   # d/dc2
            J[1, 2] = 0.0                   # d/db1k
            J[1, 3] = 2 * b2k * norm_abk    # d/db2k
            J[1, 4] = 0.0                   # d/b1j
            J[1, 5] = 2 * b2i * norm_abi    # d/b2i
            
            J[2, 0] = 2 * c1 * b1j**2 * (norm_abj - 1)\
                    + norm_abk * (2 * c1 * b1k**2 + 2 * c2 * b1k * b2k)     # d/dc1
            J[2, 1] = 2 * c2 * b2i**2 * norm_abi\
                    + norm_abk * (2 * c2 * b2k**2 + 2 * c1 * b1k * b2k)     # d/dc2
            J[2, 2] = norm_abk * (c1**2 * 2 * b1k + 2 * c1 * c2 * b2k)      # d/db1k
            J[2, 3] = norm_abk * (c2**2 * 2 * b2k + 2 * c1 * c2 * b1k)      # d/db2k
            J[2, 4] = (norm_abj - 1) * c1**2 * 2 * b1j                      # d/b1j
            J[2, 5] = norm_abi * c2**2 * 2 * b2i                            # d/b2i
            
            J[3, 0] = 2 * c1 * b1j**2 * (norm_abj)\
                    + norm_abk * (2 * c1 * b1k**2 + 2 * c2 * b1k * b2k)     # d/dc1
            J[3, 1] = 2 * c2 * b2i**2 * (norm_abi-1)\
                    + norm_abk * (2 * c2 * b2k**2 + 2 * c1 * b1k * b2k)     # d/dc2
            J[3, 2] = norm_abk * (c1**2 * 2 * b1k + 2 * c1 * c2 * b2k)      # d/db1k
            J[3, 3] = norm_abk * (c2**2 * 2 * b2k + 2 * c1 * c2 * b1k)      # d/db2k
            J[3, 4] = (norm_abj) * c1**2 * 2 * b1j                          # d/b1j
            J[3, 5] = (norm_abi-1) * c2**2 * 2 * b2i                        # d/b2i
            
            J[4, 0] = 2 * c1 * b1j**2 * (norm_abj)\
                    + (norm_abk-1) * (2 * c1 * b1k**2 + 2 * c2 * b1k * b2k)     # d/dc1
            J[4, 1] = 2 * c2 * b2i**2 * (norm_abi)\
                    + (norm_abk-1) * (2 * c2 * b2k**2 + 2 * c1 * b1k * b2k)     # d/dc2
            J[4, 2] = (norm_abk-1) * (c1**2 * 2 * b1k + 2 * c1 * c2 * b2k)      # d/db1k
            J[4, 3] = (norm_abk-1) * (c2**2 * 2 * b2k + 2 * c1 * c2 * b1k)      # d/db2k
            J[4, 4] = (norm_abj) * c1**2 * 2 * b1j                              # d/b1j
            J[4, 5] = (norm_abi) * c2**2 * 2 * b2i                              # d/b2i
            
            J[5, 0] = 2 * c1 + 2 * S12nuc * c2 * b1k * b2k * norm_abk   # d/dc1
            J[5, 1] = 2 * c2 + 2 * S12nuc * c1 * b1k * b2k * norm_abk   # d/dc2
            J[5, 2] = 2 * S12nuc * c1 * c2 * b2k * norm_abk             # d/db1k
            J[5, 3] = 2 * S12nuc * c1 * c2 * b1k * norm_abk             # d/db2k
            J[5, 4] = 0.0                                               # d/b1j
            J[5, 5] = 0.0                                               # d/b2i
            return J
        
        def equations(p, *data):
        
            c1, c2, b1k, b2k, b1j, b2i  = p
#             print "p =\n", p
            S12nuc, norm_abk, norm_abi, norm_abj = data
            tot_el_norm = norm_abk + norm_abi + norm_abj
            
            return (b1j**2 * norm_abj + b1k**2 * norm_abk - 1,\
                b2i**2 * norm_abi + b2k**2 * norm_abk - 1,\
                c1 * b1j - np.sqrt(c1**2 * b1j**2 * norm_abj + c2**2 * b2i**2 * norm_abi +\
                                  (c1 * b1k + c2 * b2k)**2 * norm_abk),\
                c2 * b2i - np.sqrt(c1**2 * b1j**2 * norm_abj + c2**2 * b2i**2 * norm_abi +\
                                  (c1 * b1k + c2 * b2k)**2 * norm_abk),\
                (c1*b1k + c2 * b2k) - np.sqrt(c1**2 * b1j**2 * norm_abj + c2**2 * b2i**2 * norm_abi +\
                                  (c1 * b1k + c2 * b2k)**2 * norm_abk),\
                c1**2 + c2**2 + 2 * S12nuc * c1 * c2 * b1k * b2k * norm_abk-1)
        
        data = (S12nuc, norm_abk, norm_abi, norm_abj)
#         print "data =\n", data
        q =  root(equations, guess, method="hybr", tol=None, jac=Jacobian,\
                                     args=data)
        print (q["message"])
        c1 = q["x"][0]
        c2 = q["x"][1]
        b1k = q["x"][2]
        b2k = q["x"][3]
        b1j = q["x"][4]
        b2i = q["x"][5]
        success = q["success"]
        print("C1 = {}\nC2 = {}\nb1k = {}\nb2k = {}\nb1j = {}\nb2i = {}".format(c1, c2, b1k, b2k, b1j, b2i))
        print("check:")
        check_el_norm_j = b1j**2 * norm_abj + b1k**2 * norm_abk - 1
        check_el_norm_i = b2i**2 * norm_abi + b2k**2 * norm_abk - 1
        check_wf_j = c1**2 * b1j**2 - (c1**2 * b1j**2 * norm_abj + c2**2 * b2i**2 * norm_abi +\
                                      (c1 * b1k + c2 * b2k)**2 * norm_abk)
        check_wf_i = c2**2 * b2i**2 - (c1**2 * b1j**2 * norm_abj + c2**2 * b2i**2 * norm_abi +\
                                      (c1 * b1k + c2 * b2k)**2 * norm_abk)
        check_wf_k = (c1*b1k + c2 * b2k)**2 - (c1**2 * b1j**2 * norm_abj + c2**2 * b2i**2 * norm_abi +\
                                      (c1 * b1k + c2 * b2k)**2 * norm_abk)
        check_tot_norm = c1**2 + c2**2 + 2 * S12nuc * c1 * c2 * b1k * b2k * norm_abk - 1
        
        check = (check_el_norm_j, check_el_norm_i, check_wf_j, check_wf_i, check_wf_k,\
                 check_tot_norm)
        tol = 1e-8
        err = sum(abs(el) > tol for el in check)
        if err > 0: print "WARNING: accuracy not achieved"
        
        return c1, c2, b1k, b2k, b1j, b2i, success  
           
    def calc_kin_en(self, p, m):
        ke = 0.0
        for idim in range(self.numdims):
            ke += 0.5 * p[idim] * p[idim] / m[idim]
        
        return ke
            
    def init_traj(self, t, ndims, pos, mom, wid, m, nstates, istat, lab):

        self.time = t
        self.positions = pos
        self.momenta = mom
        self.widths = wid
        self.masses = m
        self.label = lab
        self.numstates = nstates
        self.firsttime = t

    def propagate_half_step(self):
        """This is needed to obtain ES properties for the basis functions that are not cloned
        but have overlap with the cloning functions"""
        
        pos_t = self.positions
        mom_tmhdt = self.momenta 
    
        H_elec, Force = self.construct_el_H(pos_t)
        tmp_wf_tmhdt = self.td_wf
        tmp_wf_t = self.propagate_symplectic(H_elec, tmp_wf_tmhdt,\
                                             self.timestep/2, self.n_el_steps/2)
        eigenvals, eigenvectors = lin.eigh(H_elec)
        tmp_wf_t_T = np.transpose(np.conjugate(tmp_wf_t))
        tmp_energy = np.real(np.dot(np.dot(tmp_wf_t_T, H_elec), tmp_wf_t))    
    
        tmp_force = np.zeros((self.numdims))    
        for n in range(self.numdims):
            tmp_force[n] = -np.real(np.dot(np.dot(tmp_wf_t_T, Force[n]), tmp_wf_t))

        tmp_pop = np.zeros(self.numstates)
        tmp_amp = np.zeros((self.numstates), dtype=np.complex128)       
        
        for j in range(self.numstates):
            tmp_amp[j] = np.dot(np.conjugate(np.transpose(eigenvectors[:, j])), tmp_wf_t)
            tmp_pop[j] = np.real(np.dot(np.transpose(np.conjugate(tmp_amp[j])), tmp_amp[j]))

        a_t = tmp_force / self.masses
        mom_t = mom_tmhdt + a_t * self.timestep / 2 * self.masses
        
        return tmp_wf_t, mom_t
        
    def init_clone_traj(self, parent, istate, jstate, label, nuc_norm):

        self.numstates = parent.numstates
        self.timestep = parent.timestep
        self.maxtime = parent.maxtime
        
        self.widths = parent.widths
        self.masses = parent.masses

        if hasattr(parent,'atoms'):
            self.atoms = parent.atoms
        if hasattr(parent,'civecs'):
            self.civecs = parent.civecs
            self.ncivecs = parent.ncivecs
        if hasattr(parent,'orbs'):
            self.orbs = parent.orbs
            self.norbs = parent.norbs
        if hasattr(parent,'prev_wf_positions'):
            self.prev_wf_positions = parent.prev_wf_positions
        if hasattr(parent,'electronic_phases'):
            self.electronic_phases = parent.electronic_phases
        
        self.clonethresh = parent.clonethresh
        self.potential_specific_traj_copy(parent)
        
        time = parent.time
        self.time = time
        self.label = label
        pos_t = parent.positions
        mom_t = parent.momenta_tpdt
        tmp_pop = parent.populations
        tmp_amp = parent.mce_amps
        tmp_force = parent.av_force
        tmp_energy = parent.av_energy
        print "time =", time
        print "av_energy =", tmp_energy
        print "kin energy =", self.calc_kin_en(mom_t, parent.masses)
        print "total E =", self.calc_kin_en(mom_t, parent.masses) + tmp_energy

        tmp_wf = self.td_wf_full_ts
        H_elec, Force = self.construct_el_H(pos_t)
        eigenvals, eigenvectors = lin.eigh(H_elec)
        print "FORCE= ", Force 
#        During the cloning procedure we look at pairwise decoherence times, all population is going from
#        istate to jstate. The rest of the amplitudes change too in order to conserve nuclear norm
    
        norm_abk = 0.0
                
        for i in range(self.numstates):
            if i == istate:
                norm_abi = tmp_pop[i]
            elif i == jstate:
                norm_abj = tmp_pop[i]
            else:
                norm_abk += tmp_pop[i]
        
        print "total pop =", sum(tmp_pop)
        n_iter = 0
        S_act = 1.01
        S_trial = 0.5
        S_prev_trial= 0.81
        tol = 1e-10
        guess = (0.2, 0.9, 0.5, 0.7, 0.2, 0.9)  
        print "norm_abi =", norm_abi
        print "norm_abj =", norm_abj
        print "norm_abk =", norm_abk
        
        print "\nSolving a system of equations numerically to find QM and MCE amplitudes"
        
        while abs(S_act - S_prev_trial) > tol:
            
            n_iter += 1
            
            if n_iter > 50:
                print "Aborting cloning procedure, no solution that preserves norm"
                return False
            print "\nIteration ", n_iter, ":", "trying S_nuc =", S_trial
            c1, c2, b1k, b2k, b1j, b2i, success \
            = self.solve_nonlin_system(S_trial, norm_abk, norm_abi, norm_abj, guess)
            print "guess = ", ["%0.6f" % i for i in guess]
            guess_try = (c1, c2, b1k, b2k, b1j, b2i)
            s = sum(n < 0 for n in guess_try)
#             print "sum of negatives", s
            if s > 0 or not success: 
                np.random.seed(105)
                S_trial = np.random.rand()
                continue
            else:
                guess = guess_try
#             guess = ((c1+guess[0])/2, (c2+guess[1])/2, (b1k+guess[2])/2,\
#                      (b2k+guess[3])/2, (b1j+guess[4])/2, (b2i+guess[5])/2)
            print "new guess =", ["%0.6f" % i for i in guess]
            child_wf = np.zeros((self.numstates), dtype=np.complex128) 
            parent_wf = np.zeros((self.numstates), dtype=np.complex128) 
            
            for kstate in range(self.numstates):
                if kstate == istate:
                    # the population is removed from this state, so nothing to do here
                    scaling_factor = b2i
                    parent_wf += eigenvectors[:, kstate] * tmp_amp[kstate] * scaling_factor
                     
                elif kstate == jstate:
                    # the population from istate is transferred to jstate
                    scaling_factor = b1j                 
                    child_wf += eigenvectors[:, kstate] * tmp_amp[kstate] * scaling_factor
                 
                else:
                    # the rest of the states remain unchanged 
                    child_wf += eigenvectors[:, kstate] * tmp_amp[kstate] * b1k
                    parent_wf += eigenvectors[:, kstate] * tmp_amp[kstate] * b2k

            child_wf_T = np.conjugate(np.transpose(child_wf))
            parent_wf_T = np.conjugate(np.transpose(parent_wf))
                           
            child_pop = np.zeros(self.numstates)
            child_amp = np.zeros((self.numstates), dtype=np.complex128) 
            parent_pop = np.zeros(self.numstates)
            parent_amp = np.zeros((self.numstates), dtype=np.complex128)         
    
            for j in range(self.numstates):
                child_amp[j] = np.dot(eigenvectors[:, j], child_wf)
                child_pop[j] = np.real(np.dot(np.conjugate(child_amp[j]), child_amp[j]))
                parent_amp[j] = np.dot(eigenvectors[:, j], parent_wf)
                parent_pop[j] = np.real(np.dot(np.conjugate(parent_amp[j]), parent_amp[j]))
            
            parent_force = np.zeros((self.numdims))    
            child_force = np.zeros((self.numdims)) 
            print self.numdims
            for n in range(self.numdims):
                parent_force[n] = -np.real(np.dot(np.dot(parent_wf_T, Force[n]), parent_wf))
            for n in range(self.numdims):
                child_force[n] = -np.real(np.dot(np.dot(child_wf_T, Force[n]), child_wf))
            
            child_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(child_wf)), H_elec), child_wf))
            parent_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(parent_wf)), H_elec), parent_wf))
            
            # Also need to update momentum 
    #         self.momenta = mom_t
            
            print "Rescaling child's momentum:"
            child_rescale_ok, child_rescaled_momenta = self.rescale_momentum(tmp_energy, child_energy, mom_t)
    #         print "child_rescale =", child_rescale_ok
            
            if child_rescale_ok:
#                 print "\nmom_t=", mom_t
                parent_E_total = tmp_energy + parent.calc_kin_en(mom_t, parent.masses)
                child_E_total = child_energy +\
                self.calc_kin_en(child_rescaled_momenta, self.masses)
                print "child_E after rescale =", child_E_total
                print "parent E before rescale=", parent_E_total 
                print "Rescaling parent's momentum"
                parent_rescale_ok, parent_rescaled_momenta\
                = parent.rescale_momentum(tmp_energy, float(parent_energy), mom_t)
                print "parent E after rescale = ",\
                parent.calc_kin_en(parent_rescaled_momenta, parent.masses) + parent_energy
                
                Sij = self.overlap_nuc(pos_t, pos_t, parent_rescaled_momenta,\
                                       child_rescaled_momenta, parent.widths,\
                                       parent.widths)
                S_act = np.real(Sij)
                 
                print "S_actual =", S_act
                print "S_trial =", S_trial
                S_prev_trial = S_trial
                S_trial = S_act
#                 print "S_trial_next_step =", S_trial
                
        # need to update wave function at half step since the parent changed ES properties 
        # during cloning
        if parent_rescale_ok:
            # Setting mintime to current time to avoid backpropagation
            mintime = parent.time
            self.mintime = mintime
            self.firsttime = time
            self.positions = parent.positions

            # updating quantum parameters for child    
            self.td_wf_full_ts = child_wf
            self.td_wf = child_wf
            self.av_energy = float(child_energy)
            self.mce_amps = child_amp
            self.populations = child_pop
            self.av_force = child_force
            self.first_step = True
            self.momenta = child_rescaled_momenta
            self.approx_eigenvecs = eigenvectors
            self.energies = eigenvals
            # IS THIS OK?!
            self.h5_output()
            
            parent.momenta = parent_rescaled_momenta
            parent.td_wf = parent_wf
#             parent.td_wf_full_ts_qm = parent_wf
            parent.td_wf_full_ts = parent_wf
            parent.av_energy = float(parent_energy)
            parent.mce_amps = parent_amp
            parent.populations = parent_pop
            parent.av_force = parent_force
            parent.energies = eigenvals
            parent.approx_eigenvecs = eigenvectors
            
            # this makes sure the parent trajectory in VV propagated as first step
            # because the wave function is at the full TS, should be half step ahead
            parent.first_step = True
#                 self.first_step = True
#                 print "child av_energy", self.av_energy
#                 print "parent av_energy", parent.av_energy
            
            self.rescale_amp[0] = c1 * np.sqrt(nuc_norm)
            parent.rescale_amp[0] = c2 * np.sqrt(nuc_norm)
            print "Rescaling parent amplitude by a factor ", parent.rescale_amp
            print "Rescaling child amplitude by a factor ", self.rescale_amp
#                 sys.exit()
            return True
        else:
            return False
#     else:    
#         return False

    def rescale_momentum(self, v_ini, v_fin, p_ini):
        """This subroutine rescales the momentum of the child basis function
        The difference from spawning here is that the average Ehrenfest energy is rescaled,
        not of the pure electronic states"""
        
        m = self.masses
        t_ini = self.calc_kin_en(p_ini, m)
#         print "v_ini =", v_ini
#         print "v_fin =", v_fin
#         print "t_ini =", t_ini
        factor = ( ( v_ini + t_ini - v_fin ) / t_ini )
#         print "factor =", factor

        if factor < 0.0:
            print "Aborting cloning because because there is not enough energy for momentum adjustment"
            return False, factor
        factor = math.sqrt(factor)
        print "Rescaling momentum by factor ", factor
        p_fin = factor * p_ini
#         self.momenta = p_fin
                
        # Computing kinetic energy of child to make sure energy is conserved
        t_fin = 0.0
        for idim in range(self.numdims):
            t_fin += 0.5 * p_fin[idim] * p_fin[idim] / m[idim]
        if v_ini + t_ini - v_fin - t_fin > 1e-9: 
            print "ENERGY NOT CONSERVED!!!"
            sys.exit
        return True, factor*p_ini

    def propagate_step(self):
        """When cloning happens we start a parent wf
        as a new trajectory because electronic structure properties changed,
        first_step variable here ensures that we don't write to h5 file twice! (see vv.py)"""
        
        if float(self.time) - float(self.firsttime) < (self.timestep -1e-6) or self.first_step:
            self.prop_first_step()
        else:
            self.prop_not_first_step()

        # computing pairwise cloning probabilities
        self.clone_p = self.compute_cloning_probabilities()
        
    def compute_cloning_probabilities(self):
        """Computing pairwise cloning probabilities according to:
        tau_ij = (1 + t_dec / KE) / (E_i - E_j)"""
        
        print "Computing cloning probabilities"
        
        m = self.masses
        ke_tot = 0.0
        tau = np.zeros((self.numstates, self.numstates))
        for idim in range(self.numdims):
            ke_tot += 0.5 * self.momenta_tpdt[idim] * self.momenta_tpdt[idim] / m[idim]
        if ke_tot > 0.0:
            for istate in range(self.numstates):
                for jstate in range(self.numstates):
                    if istate == jstate:
                        tau[istate, jstate] = 0.0
                    else:
                        dE = np.abs(self.energies[jstate] - self.energies[istate])
                        tau[istate, jstate] = 1 / ((1 + self.t_decoherence_par / ke_tot) / dE)

#         print "tau =\n", tau
        return tau
            
    def h5_output(self, zdont_half_step=False):
        """This subroutine outputs all datasets into an h5 file at each timestep"""

        if len(self.h5_datasets) == 0:
            self.init_h5_datasets()
        filename = "working.hdf5"

        h5f = h5py.File(filename, "a")
        groupname = "traj_" + self.label
        if groupname not in h5f.keys():
            self.create_h5_traj(h5f,groupname)
        trajgrp = h5f.get(groupname)
        all_datasets = self.h5_datasets.copy()
        if not zdont_half_step:
            all_datasets.update(self.h5_datasets_half_step)
        for key in all_datasets:
            n = all_datasets[key]
#             print "key", key
            dset = trajgrp.get(key)
            l = dset.len()
            dset.resize(l+1, axis=0)
            ipos=l

#             if key == "forces_i":
#                 getcom = "self.get_" + key + "()"
#             else:
            getcom = "self." + key 

#             print getcom
            tmp = eval(getcom)
#             print "\nkey =", key
            if n!=1:
                dset[ipos,0:n] = tmp[0:n]
            else:
                dset[ipos,0] = tmp
        h5f.flush()
        h5f.close()

    def create_h5_traj(self, h5f, groupname):
        """create a new trajectory group in hdf5 output file"""
        
        trajgrp = h5f.create_group(groupname)
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            if key == "td_wf" or key == "mce_amps" or key == "td_wf_full_ts":
                dset = trajgrp.create_dataset(key, (0,n), maxshape=(None,n), dtype="complex128")
            else:
                dset = trajgrp.create_dataset(key, (0,n), maxshape=(None,n), dtype="float64")

        for key in self.h5_datasets_half_step:
            n = self.h5_datasets_half_step[key]
            if key == "td_wf" or key == "mce_amps" or key == "td_wf_full_ts":
                dset = trajgrp.create_dataset(key, (0,n), maxshape=(None,n), dtype="complex128")
            else:
                dset = trajgrp.create_dataset(key, (0,n), maxshape=(None,n), dtype="float64")
        # add some metadata
#         trajgrp.attrs["istate"] = self.istate
        trajgrp.attrs["masses"] = self.masses
        trajgrp.attrs["widths"] = self.widths
        if hasattr(self,"atoms"):
            trajgrp.attrs["atoms"] = self.atoms
        
    def get_data_at_time_from_h5(self, t, dset_name):
        """This subroutine gets trajectory data from "dset_name" array at a certain time"""
        
        h5f = h5py.File("working.hdf5", "r")
        groupname = "traj_" + self.label
        filename = "working.hdf5"
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time"][:]
        #print "size", dset_time.size
        ipoint = -1
        for i in range(len(dset_time)):
            if (dset_time[i] < t+1.0e-6) and (dset_time[i] > t-1.0e-6):
                ipoint = i
                #print "dset_time[i] ", dset_time[i]
                #print "i ", i
        dset = trajgrp[dset_name][:]            
        data = np.zeros(len(dset[ipoint, :]))
        data = dset[ipoint, :]
        #print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()
        return data

    def get_all_qm_data_at_time_from_h5(self, t, suffix=""):
        """This subroutine pulls all arrays from the trajectory at a certain time and assigns
        results to the same variable names with _qm suffix. The _qm variables essentially
        match the values at _t, added for clarity. 
        !!!!!WRONG? _tpdt?"""
        
        h5f = h5py.File("working.hdf5", "r")
        groupname = "traj_" + self.label
        filename = "working.hdf5"
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time"][:]
        #print "size", dset_time.size
        ipoint = -1
        for i in range(len(dset_time)):
            if (dset_time[i] < t+1.0e-6) and (dset_time[i] > t-1.0e-6):
                ipoint = i
                #print "dset_time[i] ", dset_time[i]
                #print "i ", i
        for dset_name in self.h5_datasets:
            dset = trajgrp[dset_name][:]            
            data = np.zeros(len(dset[ipoint, :]))
            data = dset[ipoint, :]
            comm = "self." + dset_name + "_qm" + suffix + " = data"
            exec(comm)
#             print "comm ", comm
#             print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()
            
    def get_all_qm_data_at_time_from_h5_half_step(self, t):
        "Same as get_all_qm_data_at_time_from_h5, but pulls data from a half timestep"
        
        h5f = h5py.File("working.hdf5", "r")
        groupname = "traj_" + self.label
        filename = "working.hdf5"
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time_half_step"][:]
        #print "size", dset_time.size
        ipoint = -1
        for i in range(len(dset_time)):
            if (dset_time[i] < t+1.0e-6) and (dset_time[i] > t-1.0e-6):
                ipoint = i
                #print "dset_time[i] ", dset_time[i]
                #print "i ", i
        for dset_name in self.h5_datasets_half_step:
            dset = trajgrp[dset_name][:]            
            data = np.zeros(len(dset[ipoint, :]))
            data = dset[ipoint, :]
            comm = "self." + dset_name + "_qm = data"
            exec(comm)
            #print "comm ", comm
            #print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()
            
    def initial_wigner(self, iseed):
        print "Randomly selecting Wigner initial conditions"
        ndims = self.numdims

        h5f = h5py.File('hessian.hdf5', 'r')
        
        pos = h5f['geometry'][:].flatten()

        h = h5f['hessian'][:]

        m = self.masses

        sqrtm = np.sqrt(m)

        #build mass weighted hessian
        h_mw = np.zeros_like(h)

        for idim in range(ndims):
            h_mw[idim, :] = h[idim, :] / sqrtm

        for idim in range(ndims):
            h_mw[:, idim] = h_mw[:, idim] / sqrtm

        # symmetrize mass weighted hessian
        h_mw = 0.5 * (h_mw + h_mw.T)

        # diagonalize mass weighted hessian
        evals, modes = np.linalg.eig(h_mw)

        # sort eigenvectors
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        modes = modes[:, idx]

        print 'Eigenvalues of the mass-weighted hessian are (a.u.)'
        print evals
        
        # seed random number generator
        np.random.seed(iseed)
        
        alphax = np.sqrt(evals[0:ndims-6]) / 2.0

        sigx = np.sqrt(1.0 / ( 4.0 * alphax))
        sigp = np.sqrt(alphax)

        dtheta = 2.0 * np.pi * np.random.rand(ndims - 6)
        dr = np.sqrt( np.random.rand(ndims - 6) )

        dx1 = dr * np.sin(dtheta)
        dx2 = dr * np.cos(dtheta)

        rsq = dx1 * dx1 + dx2 * dx2

        fac = np.sqrt( -2.0 * np.log(rsq) / rsq )

        x1 = dx1 * fac
        x2 = dx2 * fac

        posvec = np.append(sigx * x1, np.zeros(6))
        momvec = np.append(sigp * x2, np.zeros(6))

        deltaq = np.matmul(modes, posvec) / sqrtm
        pos += deltaq
        mom = np.matmul(modes, momvec) * sqrtm

        self.positions = pos
        self.momenta = mom

        zpe = np.sum(alphax[0 : (ndims - 6)])
        ke = 0.5 * np.sum(mom * mom / m)

        print "ZPE = ", zpe
        print "Kinetic energy = ", ke

    def overlap_nuc_1d(self, xi, xj, di, dj, xwi, xwj):
        """Compute 1-dimensional nuclear overlaps"""
        
        c1i = (complex(0.0, 1.0))
        deltax = xi - xj
        pdiff = di - dj
        osmwid = 1.0 / (xwi + xwj)
        
        xrarg = osmwid * (xwi*xwj*deltax*deltax + 0.25*pdiff*pdiff)
        if (xrarg < 10.0):
            gmwidth = math.sqrt(xwi*xwj)
            ctemp = (di*xi - dj*xj)
            ctemp = ctemp - osmwid * (xwi*xi + xwj*xj) * pdiff
            cgold = math.sqrt(2.0 * gmwidth * osmwid)
            cgold = cgold * math.exp(-1.0 * xrarg)
            cgold = cgold * cmath.exp(ctemp * c1i)
        else:
            cgold = 0.0
               
        return cgold
    
    def overlap_nuc(self, pos_i, pos_j, mom_i, mom_j, widths_i, widths_j):
        
        Sij = 1.0
        for idim in range(self.numdims):
            xi = pos_i[idim]
            xj = pos_j[idim]
            di = mom_i[idim]
            dj = mom_j[idim]
            xwi = widths_i[idim]
            xwj = widths_j[idim]
            Sij *= self.overlap_nuc_1d(xi, xj, di, dj, xwi, xwj)

        return Sij
        