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
from datashape.coretypes import int32

class traj(fmsobj):
        
    numdims = 1
    t_decoherence_par = 0.1
        
    def __init__(self):
        self.time = 0.0
        self.maxtime = -1.0
        self.mintime = 0.0
        self.firsttime = 0.0
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.widths = np.zeros(self.numdims)
        self.masses = np.zeros(self.numdims)
        self.label = "00"
        self.h5_datasets = dict()

        self.timestep = 0.0
        
        self.numstates = 5
        self.krylov_sub_n = self.numstates
        
        self.length_wf = self.numstates
        self.wf = np.zeros((self.numstates, self.length_wf))
        self.prev_wf = np.zeros((self.numstates, self.length_wf))
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates, self.numdims))
        self.S_elec_flat = np.zeros(self.numstates*self.numstates)

        self.numchildren = 0
        
        self.positions_qm = np.zeros(self.numdims)
        self.momenta_qm = np.zeros(self.numdims)
        self.energies_qm = np.zeros(self.numstates)
        self.forces_i_qm = np.zeros(self.numdims)

        #In the following block there are variables needed for ehrenfest
        self.H_elec = np.zeros((self.numstates, self.numstates), dtype = np.complex128)
        self.first_step = False
        self.full_H = bool()
        self.new_amp = np.zeros((1), dtype = np.complex128)
        self.rescale_amp = np.zeros((1), dtype = np.complex128)
        self.n_el_steps = np.zeros((1), dtype = np.int32)
#         self.n_el_steps = 1000
        self.td_wf_full_ts = np.zeros((self.numstates), dtype = np.complex128)
        self.td_wf = np.zeros((self.numstates), dtype = np.complex128)
        self.mce_amps = np.zeros((self.numstates), dtype = np.complex128)
        self.populations = np.zeros(self.numstates)
        self.av_energy = 0.0
        self.av_force = np.zeros(self.numdims)
        self.eigenvecs = np.zeros((self.numstates, self.numstates), dtype = np.complex128)
        self.approx_eigenvecs = np.zeros((self.krylov_sub_n, self.krylov_sub_n),\
                                         dtype = np.complex128)
        self.approx_energies = np.zeros(self.krylov_sub_n)
        self.approx_amp = np.zeros((self.krylov_sub_n), dtype = np.complex128)
        self.approx_pop = np.zeros(self.krylov_sub_n)
        self.approx_wf_full_ts = np.zeros((self.krylov_sub_n), dtype = np.complex128)
        self.wf_store_full_ts = np.zeros((self.numstates, self.krylov_sub_n),\
                                         dtype = np.complex128)
        self.wf_store = np.zeros((self.numstates, self.krylov_sub_n), dtype = np.complex128)
        self.clone_p = np.zeros((self.numstates, self.numstates))
        self.clone_E_diff = np.zeros(self.numstates)
        self.clone_E_diff_prev = np.zeros(self.numstates)
    
    def solve_nonlin_system(self, nuc_norm, S12nuc, norm_abk, norm_abi, norm_abj, guess):
    
        def Jacobian(X, *data):
            S12nuc, norm_abk, norm_abi, norm_abj, nuc_norm = data
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
            S12nuc, norm_abk, norm_abi, norm_abj, nuc_norm = data
            tot_el_norm = norm_abk + norm_abi + norm_abj
            
            return (b1j**2 * norm_abj + b1k**2 * norm_abk - 1,\
                b2i**2 * norm_abi + b2k**2 * norm_abk - 1,\
                c1 * b1j - np.sqrt(c1**2 * b1j**2 * norm_abj + c2**2 * b2i**2 * norm_abi +\
                                  (c1 * b1k + c2 * b2k)**2 * norm_abk),\
                c2 * b2i - np.sqrt(c1**2 * b1j**2 * norm_abj + c2**2 * b2i**2 * norm_abi +\
                                  (c1 * b1k + c2 * b2k)**2 * norm_abk),\
                (c1*b1k + c2 * b2k) - np.sqrt(c1**2 * b1j**2 * norm_abj + c2**2 * b2i**2 * norm_abi +\
                                  (c1 * b1k + c2 * b2k)**2 * norm_abk),\
                c1**2 + c2**2 + 2 * S12nuc * c1 * c2 * b1k * b2k * norm_abk - nuc_norm)
        
        tolerance = 1e-08
        data = (S12nuc, norm_abk, norm_abi, norm_abj, nuc_norm)
#         print "data =\n", data
        q =  root(equations, guess, method="hybr", tol=tolerance, jac=Jacobian,\
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
        check_tot_norm = c1**2 + c2**2 + 2 * S12nuc * c1 * c2 * b1k * b2k * norm_abk - nuc_norm
        
        check = (check_el_norm_j, check_el_norm_i, check_wf_j, check_wf_i, check_wf_k,\
                 check_tot_norm)
        
        err = sum(abs(el) > tolerance for el in check)
        if err > 0: 
            print "WARNING: accuracy not achieved"
            print check
        return c1, c2, b1k, b2k, b1j, b2i, success  
           
    def calc_kin_en(self, p, m):
        """Calculate kinetic energy of a trajectory"""

        ke = sum(0.5 * p[idim]**2 / m[idim] for idim in range(self.numdims))
        
        return ke
            
    def init_traj(self, t, ndims, pos, mom, wid, m, nstates, istat, lab):
        """Initialize trajectory"""
        
        self.time = t
        self.positions = pos
        self.momenta = mom
        self.widths = wid
        self.masses = m
        self.label = lab
        self.numstates = nstates
        self.firsttime = t
        
    def init_clone_traj(self, parent, istate, jstate, label, nuc_norm):

        self.numstates = parent.numstates
        self.timestep = parent.timestep
        self.maxtime = parent.maxtime
        self.full_H = parent.full_H
        self.widths = parent.widths
        self.masses = parent.masses
        self.n_el_steps = parent.n_el_steps
        
        
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
        
#         self.clonethresh = parent.clonethresh
        self.potential_specific_traj_copy(parent)
        
        time = parent.time
        self.time = time
        self.label = label
        pos_t = parent.positions
        mom_t = parent.momenta_full_ts
        tmp_pop = parent.populations
        tmp_amp = parent.mce_amps
        tmp_force = parent.av_force
        tmp_energy = parent.av_energy

#         tmp_wf = self.td_wf_full_ts
        H_elec, Force = self.construct_el_H(pos_t)
        eigenvals, eigenvectors = lin.eigh(H_elec)
        
#         print "\nparent pot_E before cloning =", parent.av_energy
#         print "parent momenta before cloning =", parent.momenta
#         print "parent E before cloning = ", parent.av_energy\
#         + parent.calc_kin_en(parent.momenta, parent.masses)
#         print "parent E before cloning  full_ts = ", parent.av_energy\
#         + parent.calc_kin_en(parent.momenta_full_ts, parent.masses)
#         print "wf before", parent.td_wf_full_ts
#         print "mce_amps before", parent.mce_amps
#       During the cloning procedure we look at pairwise decoherence times, all population
#       is going from istate to jstate. The rest of the amplitudes change too in order to
#       conserve nuclear norm
    
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
            
            if n_iter > 15:
                print "Aborting cloning procedure, no solution that preserves norm"
                return False
            print "\nIteration ", n_iter, ":", "trying S_nuc =", S_trial
            c1, c2, b1k, b2k, b1j, b2i, success \
            = self.solve_nonlin_system(nuc_norm, S_trial, norm_abk, norm_abi, norm_abj, guess)
            print "guess = ", ["%0.6f" % i for i in guess]
            guess_try = (c1, c2, b1k, b2k, b1j, b2i)
            s = sum(n < 0 for n in guess_try)
#             print "sum of negatives", s
            if s > 0 or not success: 
                np.random.seed(n_iter)
                S_trial = np.random.rand()
                continue
            else:
                guess = guess_try

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
            parent_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(parent_wf)),\
                                                  H_elec), parent_wf))
            
            # Also need to update momentum 
    #         self.momenta = mom_t
            
            print "Rescaling child's momentum:"
            child_rescale_ok, child_rescaled_momenta = self.rescale_momentum(tmp_energy,\
                                                                             child_energy,\
                                                                             mom_t)
            
            if child_rescale_ok:

                parent_E_total = tmp_energy + parent.calc_kin_en(mom_t, parent.masses)
                child_E_total = child_energy +\
                self.calc_kin_en(child_rescaled_momenta, self.masses)
                print "child_E after rescale =", child_E_total
                print "parent E before rescale=", parent_E_total 
                print "Rescaling parent's momentum"
                parent_rescale_ok, parent_rescaled_momenta\
                = parent.rescale_momentum(tmp_energy, float(parent_energy), mom_t)
                if not parent_rescale_ok:
                    continue
                parent_E_after_rescale\
                = parent.calc_kin_en(parent_rescaled_momenta, parent.masses) + parent_energy
                print "parent E after rescale = ", parent_E_after_rescale
                
                Sij = self.overlap_nuc(pos_t, pos_t, parent_rescaled_momenta,\
                                       child_rescaled_momenta, parent.widths,\
                                       parent.widths)
                S_act = np.real(Sij)
                 
                print "S_actual =", S_act
                print "S_trial =", S_trial
                S_prev_trial = S_trial
                S_trial = S_act                

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
                    self.momenta_full_ts = child_rescaled_momenta
                    self.eigenvecs = eigenvectors
                    self.energies = eigenvals
                    # IS THIS OK?!
                    self.h5_output()
                    
                    parent.momenta = parent_rescaled_momenta
                    parent.momenta_full_ts = parent_rescaled_momenta
                    parent.td_wf = parent_wf
                    parent.td_wf_full_ts = parent_wf
                    parent.av_energy = float(parent_energy)
                    parent.mce_amps = parent_amp
                    parent.populations = parent_pop
                    parent.av_force = parent_force
                    parent.energies = eigenvals
                    parent.eigenvecs = eigenvectors
                    
                    # this makes sure the parent trajectory in VV propagated as first step
                    # because the wave function is at the full TS, should be half step ahead
                    parent.first_step = True
#                     parent.h5_output()
                    
                    self.rescale_amp[0] = c1 * np.sqrt(nuc_norm)
                    parent.rescale_amp[0] = c2 * np.sqrt(nuc_norm)
                    print "Rescaling parent amplitude by a factor ", parent.rescale_amp
                    print "Rescaling child amplitude by a factor ", self.rescale_amp
        
                    
        #                 sys.exit()
#                     print "\nparent pot_E after cloning = ", parent.av_energy
#                     print "parent momenta after cloning =", parent.momenta
#                     print "total E after rescale =", parent_E_after_rescale
#                     print "wf after =", parent.td_wf_full_ts
#                     print "mce_amps after =", parent.mce_amps
                    return True
        
                else:
                    return False
           
    def init_clone_traj_approx(self, parent, istate, label, nuc_norm):
        
        self.numstates = parent.numstates
        self.timestep = parent.timestep
        self.maxtime = parent.maxtime
        self.full_H = parent.full_H
        self.n_el_steps = parent.n_el_steps
        
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
        
#         self.clonethresh = parent.clonethresh
        self.potential_specific_traj_copy(parent)
        
        time = parent.time
        self.time = time
        self.label = label
        pos_t = parent.positions
        mom_t = parent.momenta_full_ts
        tmp_pop = parent.approx_pop
        tmp_amp = parent.approx_amp
        tmp_force = parent.av_force
        tmp_energy = parent.av_energy
        eigenvals = parent.energies
#         eigenvectors = parent.approx_eigenvecs
        tmp_wf = parent.approx_wf_full_ts
        
        H_elec, Force = parent.construct_el_H(pos_t)
        eigenvals, eigenvectors = np.linalg.eigh(H_elec)
        
        q, r = np.linalg.qr(parent.wf_store_full_ts)

        Hk = np.dot(np.transpose(np.conjugate(q)), np.dot(H_elec, q))
        approx_force = np.zeros((self.numdims, self.krylov_sub_n, self.krylov_sub_n),\
                                dtype=np.complex128) 
        for n in range(self.numdims):
            approx_force[n] = np.dot(np.transpose(np.conjugate(q)), np.dot(Force[n], q))
        
        approx_e, approx_eigenvecs = np.linalg.eigh(Hk)
        
#         tmp_amp = np.dot(np.transpose(np.conjugate(q)), parent.mce_amps)
#         print "approx_eigenvecs =", eigenvectors
        print "approx_amp =", tmp_amp
        print "approx e =", approx_e
#       During the cloning procedure we look at pairwise decoherence times, all population
#       is going from istate to jstate. The rest of the amplitudes change too in order to
#       conserve nuclear norm
    
#         norm_abk = 0.0
#                 
#         for i in range(self.krylov_sub_n):
#             if i == istate:
#                 norm_abi = tmp_pop[i]
#             else:
#                 norm_abk += tmp_pop[i]
#         print "total pop =", sum(tmp_pop)
#         print "norm_abi =", norm_abi
#         print "norm_abk =", norm_abk        
        
        child_wf = np.zeros((self.krylov_sub_n), dtype=np.complex128) 
        parent_wf = np.zeros((self.krylov_sub_n), dtype=np.complex128) 
        
        for kstate in range(self.krylov_sub_n):
            if kstate == istate:
                # the population is removed from this state, so nothing to do here
                child_wf += approx_eigenvecs[:, kstate] * tmp_amp[kstate] / np.abs(tmp_amp[kstate])

            else:
                # the rest of the states remain unchanged 
                parent_wf += approx_eigenvecs[:, kstate] * tmp_amp[kstate]\
                           / np.sqrt(1 - np.abs(tmp_amp[istate])**2)

        child_wf_T = np.conjugate(np.transpose(child_wf))
        parent_wf_T = np.conjugate(np.transpose(parent_wf))
                       
        child_pop = np.zeros(self.krylov_sub_n)
        child_amp = np.zeros((self.krylov_sub_n), dtype=np.complex128) 
        parent_pop = np.zeros(self.krylov_sub_n)
        parent_amp = np.zeros((self.krylov_sub_n), dtype=np.complex128)         

        for j in range(self.krylov_sub_n):
            child_amp[j] = np.dot(np.conjugate(approx_eigenvecs[:, j]), child_wf)
            child_pop[j] = np.real(np.dot(np.conjugate(child_amp[j]), child_amp[j]))
            parent_amp[j] = np.dot(np.conjugate(approx_eigenvecs[:, j]), parent_wf)
            parent_pop[j] = np.real(np.dot(np.conjugate(parent_amp[j]), parent_amp[j]))
        
        parent_force = np.zeros((self.numdims))    
        child_force = np.zeros((self.numdims)) 
#         print "tmp_wf", tmp_wf

        for n in range(self.numdims):
            parent_force[n] = -np.real(np.dot(np.dot(parent_wf_T, approx_force[n]), parent_wf))
        for n in range(self.numdims):
            child_force[n] = -np.real(np.dot(np.dot(child_wf_T, approx_force[n]), child_wf))
        
        child_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(child_wf)), Hk),\
                                      child_wf))
        parent_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(parent_wf)), Hk),\
                                       parent_wf))
        
        approx_e = np.dot(np.transpose(np.conjugate(tmp_wf)), np.dot(Hk, tmp_wf))
        exact_e = np.dot(np.transpose(np.conjugate(parent.td_wf_full_ts)), np.dot(H_elec, parent.td_wf_full_ts))
        print "exact e =", exact_e
        print "approx_e =", approx_e
        print "Child E =", child_energy
        print "Parent E =", parent_energy
        print "child_pop =", child_pop
        print "parent_pop", parent_pop
#         sys.exit()
        
        print "Rescaling child's momentum:"
        child_rescale_ok, child_rescaled_momenta = self.rescale_momentum(tmp_energy,\
                                                                         child_energy,\
                                                                         mom_t) 
        if child_rescale_ok:

            parent_E_total = tmp_energy + parent.calc_kin_en(mom_t, parent.masses)
            child_E_total = child_energy +\
            self.calc_kin_en(child_rescaled_momenta, self.masses)
            print "child_E after rescale =", child_E_total
            print "parent E before rescale=", parent_E_total 
            print "Rescaling parent's momentum"
            parent_rescale_ok, parent_rescaled_momenta\
            = parent.rescale_momentum(tmp_energy, float(parent_energy), mom_t)
            if not parent_rescale_ok:
                return False
            print "parent E after rescale = ",\
            parent.calc_kin_en(parent_rescaled_momenta, parent.masses) + parent_energy         

            if parent_rescale_ok:
                # Setting mintime to current time to avoid backpropagation
                mintime = parent.time
                self.mintime = mintime
                self.firsttime = time
                self.positions = parent.positions
    
                # updating quantum parameters for child    
                child_wf_orig_basis = np.dot(q, child_wf)
                self.td_wf_full_ts = child_wf_orig_basis
                self.td_wf = child_wf_orig_basis
                approx_amp = np.zeros((self.numstates), dtype=np.complex128) 
                approx_pop = np.zeros(self.numstates) 
                for j in range(self.numstates):
                    approx_amp[j] = np.dot(np.conjugate(np.transpose(eigenvectors[:, j])), child_wf_orig_basis)
                    approx_pop[j] = np.real(np.dot(np.transpose(np.conjugate(approx_amp[j])), approx_amp[j]))
                print "child_full_pop", approx_pop
                self.av_energy = float(child_energy)
                self.approx_amp = child_amp
                self.approx_pop = child_pop
                self.av_force = child_force
                self.first_step = True
                self.momenta = child_rescaled_momenta
                self.momenta_full_ts = child_rescaled_momenta
                self.eigenvecs = eigenvectors
                self.energies = eigenvals
                self.approx_eigenvecs = approx_eigenvecs
                self.approx_energies = float(approx_e)
                # IS THIS OK?!
                self.h5_output()
                
                parent_wf_orig_basis = np.dot(q, parent_wf)
                parent.momenta = parent_rescaled_momenta
                parent.momenta_full_ts = parent_rescaled_momenta
                parent.td_wf = parent_wf_orig_basis
                parent.td_wf_full_ts = parent_wf_orig_basis
                parent.av_energy = float(parent_energy)
                parent.approx_amp = parent_amp
                parent.approx_pop = parent_pop
                parent.av_force = parent_force
                parent.energies = eigenvals
                parent.eigenvecs = eigenvectors
                parent.approx_eigenvecs = approx_eigenvecs
                parent.approx_energies = float(approx_e)
                # this makes sure the parent trajectory in VV propagated as first step
                # because the wave function is at the full TS, should be half step ahead
                parent.first_step = True
                print "parent wf =", parent_wf_orig_basis
                print "AMP_i coeff=", np.abs(tmp_amp[istate])
                print "child_pop =", child_pop
                print "parent_pop =", parent_pop
                self.rescale_amp[0] = np.abs(tmp_amp[istate])
                parent.rescale_amp[0] = np.sqrt(1 - np.abs(tmp_amp[istate])**2)
                print "Rescaling parent amplitude by a factor ", parent.rescale_amp
                print "Rescaling child amplitude by a factor ", self.rescale_amp
    #                 sys.exit()
                return True
            else:
                return False
            
    def init_clone_traj_to_a_state(self, parent, istate, label, nuc_norm):

        self.numstates = parent.numstates
        self.timestep = parent.timestep
        self.maxtime = parent.maxtime
        self.full_H = parent.full_H
        self.n_el_steps = parent.n_el_steps
        
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
        
#         self.clonethresh = parent.clonethresh
        self.potential_specific_traj_copy(parent)
        
        time = parent.time
        self.time = time
        self.label = label
        pos_t = parent.positions
        mom_t = parent.momenta_full_ts
        tmp_pop = parent.populations
        tmp_amp = parent.mce_amps
        tmp_force = parent.av_force
        tmp_energy = parent.av_energy

        tmp_wf = self.td_wf_full_ts
        H_elec, Force = self.construct_el_H(pos_t)
        eigenvals, eigenvectors = lin.eigh(H_elec)

#       During the cloning procedure we look at pairwise decoherence times, all population
#       is going from istate to jstate. The rest of the amplitudes change too in order to
#       conserve nuclear norm
    
        norm_abk = 0.0
                
        for i in range(self.numstates):
            if i == istate:
                norm_abi = tmp_pop[i]
            else:
                norm_abk += tmp_pop[i]
        
        print "total pop =", sum(tmp_pop)
        print "norm_abi =", norm_abi
        print "norm_abk =", norm_abk        
        
        child_wf = np.zeros((self.numstates), dtype=np.complex128) 
        parent_wf = np.zeros((self.numstates), dtype=np.complex128) 
        
        for kstate in range(self.numstates):
            if kstate == istate:
                # the population is removed from this state, so nothing to do here
                child_wf += eigenvectors[:, kstate] * tmp_amp[kstate] / np.abs(tmp_amp[kstate])
               
            else:
                # the rest of the states remain unchanged 
                parent_wf += eigenvectors[:, kstate] * tmp_amp[kstate]\
                           / np.sqrt(1 - np.abs(tmp_amp[istate])**2)
#             if kstate == istate:
#                 # the population is removed from this state, so nothing to do here
#                 scaling_factor = np.sqrt(1 + np.dot(np.transpose(np.conjugate(tmp_amp[jstate])), tmp_amp[jstate])\
#                                          / np.dot(np.transpose(np.conjugate(tmp_amp[kstate])), tmp_amp[kstate]))
#                 parent_wf += eigenvectors[:, kstate] * tmp_amp[kstate] * scaling_factor
#              
#             elif kstate == jstate:
#                 # the population from istate is transferred to jstate
#                 scaling_factor = np.sqrt(1 + np.dot(np.transpose(np.conjugate(tmp_amp[istate])), tmp_amp[istate])\
#                                          / np.dot(np.transpose(np.conjugate(tmp_amp[kstate])), tmp_amp[kstate]))
#                 child_wf += eigenvectors[:, kstate] * tmp_amp[kstate] * scaling_factor
#              
#             else:
#                 # the rest of the states remain unchanged 
#                 child_wf += eigenvectors[:, kstate] * tmp_amp[kstate]
#                 parent_wf += eigenvectors[:, kstate] * tmp_amp[kstate]
        
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
        parent_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(parent_wf)),\
                                              H_elec), parent_wf))
        print "Child E =", child_energy
        print "Parent E =", parent_energy        
#         sys.exit()
        # Also need to update momentum 
#         self.momenta = mom_t
        
        print "Rescaling child's momentum:"
        child_rescale_ok, child_rescaled_momenta = self.rescale_momentum(tmp_energy,\
                                                                         child_energy,\
                                                                         mom_t) 
        if child_rescale_ok:

            parent_E_total = tmp_energy + parent.calc_kin_en(mom_t, parent.masses)
            child_E_total = child_energy +\
            self.calc_kin_en(child_rescaled_momenta, self.masses)
            print "child_E after rescale =", child_E_total
            print "parent E before rescale=", parent_E_total 
            print "Rescaling parent's momentum"
            parent_rescale_ok, parent_rescaled_momenta\
            = parent.rescale_momentum(tmp_energy, float(parent_energy), mom_t)
            if not parent_rescale_ok:
                return False
            print "parent E after rescale = ",\
            parent.calc_kin_en(parent_rescaled_momenta, parent.masses) + parent_energy         

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
                self.momenta_full_ts = child_rescaled_momenta
                self.eigenvecs = eigenvectors
                self.energies = eigenvals
                # IS THIS OK?!
                self.h5_output()
                
                parent.momenta = parent_rescaled_momenta
                parent.momenta_full_ts = parent_rescaled_momenta
                parent.td_wf = parent_wf
                parent.td_wf_full_ts = parent_wf
                parent.av_energy = float(parent_energy)
                parent.mce_amps = parent_amp
                parent.populations = parent_pop
                parent.av_force = parent_force
                parent.energies = eigenvals
                parent.eigenvecs = eigenvectors
                
                # this makes sure the parent trajectory in VV propagated as first step
                # because the wave function is at the full TS, should be half step ahead
                parent.first_step = True
                print "AMP_i coeff=", np.abs(tmp_amp[istate])
                print "child_pop =", child_pop
                print "parent_pop =", parent_pop
                self.rescale_amp[0] = np.abs(tmp_amp[istate])
                parent.rescale_amp[0] = np.sqrt(1 - np.abs(tmp_amp[istate])**2)
                print "Rescaling parent amplitude by a factor ", parent.rescale_amp
                print "Rescaling child amplitude by a factor ", self.rescale_amp
    #                 sys.exit()
                return True
            else:
                return False

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
        print "Rescaling momentum by a factor ", factor
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
        self.clone_E_diff_prev = self.clone_E_diff
        self.clone_E_diff = self.compute_cloning_E_diff()
        
    def compute_cloning_E_diff(self):
        """Computing the energy differences between each state and the average"""
        
        print "Computing cloning parameters"
        
        clone_dE = np.zeros(self.numstates)
        if self.full_H:
            clone_dE = np.zeros(self.numstates)
            for istate in range(self.numstates):
                dE = np.abs(self.energies[istate] - self.av_energy)
                clone_dE[istate] = dE
        if not self.full_H:
            clone_dE = np.zeros(self.krylov_sub_n)
            for istate in range(self.krylov_sub_n):
                dE = np.abs(self.approx_energies[istate] - self.av_energy)
                clone_dE[istate] = dE
            print "clone_dE_approx =", clone_dE
#             sys.exit()
        print "clone_dE =\n", clone_dE
        return clone_dE
        
    def compute_cloning_probabilities(self):
        """Computing pairwise cloning probabilities according to:
        tau_ij = (1 + t_dec / KE) / (E_i - E_j)"""
        
        print "Computing cloning probabilities"
        
        m = self.masses
        ke_tot = 0.0
        tau = np.zeros((self.numstates, self.numstates))
        for idim in range(self.numdims):
            ke_tot += 0.5 * self.momenta_full_ts[idim] * self.momenta_full_ts[idim] / m[idim]
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
            
    def h5_output(self):
        """This subroutine outputs all datasets into an h5 file at each timestep"""

        if len(self.h5_datasets) == 0:
            self.init_h5_datasets()
        filename = "working.hdf5"

        h5f = h5py.File(filename, "a")
        groupname = "traj_" + self.label
        if groupname not in h5f.keys():
            self.create_h5_traj(h5f, groupname)
        trajgrp = h5f.get(groupname)
        all_datasets = self.h5_datasets.copy()
        dset_time = trajgrp["time"][:]
        
        for key in all_datasets:
            n = all_datasets[key]
            dset = trajgrp.get(key)
            l = dset.len()
            
            """This is not ideal, but couldn't find a better way to do this:
               During cloning the parent ES parameters change, but the hdf5 file already 
               has the data for the timestep, so this just overwrites the previous values
               if parameter first step is true"""
            if self.first_step and self.time > 1e-6 and l>0: 
                ipos = l - 1
            else:    
                dset.resize(l+1, axis=0)
                ipos = l

            getcom = "self." + key 
            tmp = eval(getcom)
#             print "key =", key
            if n!=1:
                dset[ipos, 0:n] = tmp[0:n]
            else:
                dset[ipos, 0] = tmp
        
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

        # add some metadata
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
        