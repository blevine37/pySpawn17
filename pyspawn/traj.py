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
        self.positions_tmdt = np.zeros(self.numdims)
        self.momenta_tpdt = np.zeros(self.numdims)
        self.momenta_t = np.zeros(self.numdims)
        self.momenta_tmdt = np.zeros(self.numdims)
        
        self.energies_tpdt = np.zeros(self.numstates)
        self.energies_t = np.zeros(self.numstates)
        self.energies_tmdt = np.zeros(self.numstates)
        
        self.av_energy_tpdt = 0.0
        self.av_energy_t = 0.0
        self.av_energy_tmdt = 0.0       
        
        self.av_force_tmdt = np.zeros(self.numdims)
        self.av_force_t = np.zeros(self.numdims)
        self.av_force_tpdt = np.zeros(self.numdims)
        
        self.td_wf_full_ts_tmdt = np.zeros((self.numstates), dtype = np.complex128)
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

    def init_clone_traj(self, parent, istate, jstate, label):

        self.numstates = parent.numstates
        self.timestep = parent.timestep
        self.maxtime = parent.maxtime
        
        self.widths = parent.widths
        self.masses = parent.masses
        
        time = parent.time
        self.time = time
        self.label = label
        pos_t = parent.positions
        mom_tmhdt = parent.momenta
        
        # In this block we construct Hamiltonian and obtain ES properties
        # Also propagating wave function and momentum              
        self.positions = pos_t
        H_elec, Force = self.construct_el_H(pos_t)
        tmp_wf_tmhdt = parent.td_wf
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

#        During the cloning procedure we look at pairwise decoherence times, all population is going from
#        istate to jstate. The rest of the amplitudes remain unchanged
        child_wf = np.zeros((self.numstates), dtype=np.complex128) 
        parent_wf = np.zeros((self.numstates), dtype=np.complex128) 
        child_wf_T = np.conjugate(np.transpose(child_wf))
        parent_wf_T = np.conjugate(np.transpose(parent_wf))

        for kstate in range(self.numstates):
            if kstate == istate:
                # the population is removed from this state, so nothing to do here
                scaling_factor = np.sqrt(1 + tmp_pop[jstate] / tmp_pop[kstate])
                parent_wf += eigenvectors[:, kstate] * tmp_amp[kstate] * scaling_factor
                 
            elif kstate == jstate:
                # the population from istate is transferred to jstate
                scaling_factor = np.sqrt(1 + tmp_pop[istate] / tmp_pop[kstate])                 
                child_wf += eigenvectors[:, kstate] * tmp_amp[kstate] * scaling_factor
             
            else:
                # the rest of the states remain unchanged 
                child_wf += eigenvectors[:, kstate] * tmp_amp[kstate]
                parent_wf += eigenvectors[:, kstate] * tmp_amp[kstate]
                       
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
        for n in range(self.numdims):
            parent_force[n] = -np.real(np.dot(np.dot(parent_wf_T, Force[n]), parent_wf))
        for n in range(self.numdims):
            child_force[n] = -np.real(np.dot(np.dot(child_wf_T, Force[n]), child_wf))
        
        a_t = parent_force / parent.masses
        child_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(child_wf)), H_elec), child_wf))
        parent_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(parent_wf)), H_elec), parent_wf))
        
        # updating quantum parameters for child    
        self.td_wf_full_ts = child_wf
        self.td_wf = child_wf
        self.av_energy = float(child_energy)
        self.mce_amps = child_amp
        self.populations = child_pop
        self.av_force = child_force
        self.first_step = True
        
        # Also need to update momentum 
        self.momenta = mom_t
                
        # Setting mintime to current time to avoid backpropagation
        mintime = parent.time
        self.mintime = mintime
        self.firsttime = time

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
        
        print "Rescaling child's momentum:"
        child_rescale_ok = self.rescale_momentum(tmp_energy, child_energy, mom_t)
        print "child_rescale =", child_rescale_ok

        if child_rescale_ok:
            parent_E_total = tmp_energy + parent.calc_kin_en(mom_t, parent.masses)
            child_E_total = child_energy + self.calc_kin_en(self.momenta, self.masses)
            print "child_E after rescale =", child_E_total
            print "parent E before rescale=", parent_E_total 
            print "Rescaling parent's momentum"
            parent_rescale_ok = parent.rescale_momentum(tmp_energy, float(parent_energy), mom_t)
            print "parent E after rescale = ", parent.calc_kin_en(parent.momenta, parent.masses) + parent_energy
        
#             sys.exit()
        # need to update wave function at half step since the parent changed ES properties 
        # during cloning
            if parent_rescale_ok:
                parent.td_wf = parent_wf
                parent.td_wf_full_ts_qm = parent_wf
                parent.av_energy = float(parent_energy)
                parent.mce_amps = parent_amp
                parent.populations = parent_pop
                # this makes sure the parent trajectory in VV propagated as first step
                parent.first_step = True
#                 self.first_step = True
                print "child av_energy", self.av_energy
                print "parent av_energy", parent.av_energy
#                 sys.exit()
                return True
            else:
                return False
        else:    
            return False

    def rescale_momentum(self, v_ini, v_fin, p_ini):
        """This subroutine rescales the momentum of the child basis function
        The difference from spawning here is that the average Ehrenfest energy is rescaled,
        not of the pure elecronic states"""
        
        m = self.masses
        t_ini = self.calc_kin_en(p_ini, m)
#         print "v_ini =", v_ini
#         print "v_fin =", v_fin
#         print "t_ini =", t_ini
        factor = ( ( v_ini + t_ini - v_fin ) / t_ini )
#         print "factor =", factor

        if factor < 0.0:
            print "Aborting cloning because because there is not enough energy for momentum adjustment"
            return False
        factor = math.sqrt(factor)
        print "Rescaling momentum by factor ", factor
        p_fin = factor * p_ini
        self.momenta = p_fin
                
        # Computing kinetic energy of child to make sure energy is conserved
        t_fin = 0.0
        for idim in range(self.numdims):
            t_fin += 0.5 * p_fin[idim] * p_fin[idim] / m[idim]
        if v_ini + t_ini - v_fin - t_fin > 1e-9: 
            print "ENERGY NOT CONSERVED!!!"
            sys.exit
        return True
    
#     def rescale_parent_momentum(self, v_ini, v_fin, accel):
#         """This subroutine rescales the momentum of the child basis function
#         The difference from spawning here is that the average Ehrenfest energy is rescaled,
#         not of the pure elecronic states"""
#         
#         # computing kinetic energy of parent.  Remember that, at this point,
#         # the child's momentum is still that of the parent, so we compute
#         # t_parent from the child's momentum
# 
#         p_ini = self.momenta_tpdt
#         m = self.masses
#         t_ini = self.calc_kin_en(p_ini, m)
# #         print "pot energies: initial and final", v_ini, v_fin
# #         print "kinetic energy: initial", t_ini
#                 
#         factor = ( ( v_ini + t_ini - v_fin ) / t_ini )
#         if factor < 0.0:
#             print "Aborting cloning because parent does not have enough energy for momentum adjustment"
#             return False
#         factor = math.sqrt(factor)
#         print "Rescaling  parent momentum by factor ", factor
#         p_fin = factor * p_ini
#         self.momenta_tpdt = p_fin
# 
# #         # need to update velocity at half step and position at step ahead
#         self.momenta = self.momenta_tpdt
#         self.positions = self.positions_tpdt
#         
#         # Computing kinetic energy of child to make sure energy is conserved
#         t_fin = 0.0
#         for idim in range(self.numdims):
#             t_fin += 0.5 * p_fin[idim] * p_fin[idim] / m[idim]
#         if v_ini + t_ini - v_fin - t_fin > 1e-9: 
#             print "ENERGY NOT CONSERVED!!!"
#             sys.exit
#         
#         return True

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
        tau_ij = (1 + t_dec / KE) / (E_i - E_j)
        p_ij = 1 - exp( -dt / tau_ij)"""
        
        print "Computing cloning probabilities"
        
        m = self.masses
        ke_tot = 0.0
        p = np.zeros((self.numstates, self.numstates))
        tau = np.zeros((self.numstates, self.numstates))
        for idim in range(self.numdims):
            ke_tot += 0.5 * self.momenta_tpdt[idim] * self.momenta_tpdt[idim] / m[idim]
        if ke_tot > 0.0:
            for istate in range(self.numstates):
                for jstate in range(self.numstates):
                    if istate == jstate:
                        tau[istate, jstate] = 0.0
                        p[istate, jstate] = 0.0
                    else:
                        dE = np.abs(self.energies[jstate] - self.energies[istate])
                        tau[istate, jstate] = (1 + self.t_decoherence_par / ke_tot) / dE
                        p[istate, jstate] = 1 - np.exp(-self.timestep / tau[istate, jstate])
        print "p =\n", p
        return p
            
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
        match the values at _t, added for clarity."""
        
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
            #print "dset[ipoint,:] ", dset[ipoint,:]        
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
        