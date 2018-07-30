# trajectory objects contain individual trajectory basis functions
import numpy as np
import sys
import math
from pyspawn.fmsobj import fmsobj
import os
import shutil
import h5py
from numpy import dtype
from pyspawn.potential.test_cone_td import propagate_symplectic

class traj(fmsobj):
    numstates = 2    
    numdims = 2
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
        self.istate = 0
        self.label = "00"
        self.h5_datasets = dict()
        self.h5_datasets_half_step = dict()

        self.timestep = 0.0

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
        
        self.spawnlastcoup = np.zeros(self.numstates)
        self.numchildren = 0
        
        self.positions_qm = np.zeros(self.numdims)
        self.momenta_qm = np.zeros(self.numdims)
        self.energies_qm = np.zeros(self.numstates)
        self.forces_i_qm = np.zeros(self.numdims)

        #In the following block there are variables needed for ehrenfest
        self.n_el_steps = 4000
        self.td_wf = np.zeros((self.numstates), dtype = np.complex128)
        self.td_wf_full_ts = np.zeros((self.numstates), dtype = np.complex128)
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
    
    def remove_state_pop(self, jstate):
        """After cloning of the wave function to the jstate the cloned wf have all population on jstate
        the rest of the population stays on the parent wf. This subroutine removes all population on jstate
        from the parent wf and updates the affected quantities"""
        for istate in range(self.numstates):
            new_wf = np.zeros((self.numstates), dtype = np.complex128)
            if istate != jstate:
                new_wf += self.approx_eigenvecs[istate, :] * self.mce_amps[istate]\
                                                    / np.sqrt(1 - self.populations[jstate])

        self.td_wf = new_wf
    # End of Ehrenfest block
    
    def set_forces_i_qm(self, f):
        if f.shape == self.forces_i_qm.shape:
            self.forces_i_qm = f.copy()
        else:
            print "Error in set_forces_i_qm"
            sys.exit

    def get_forces_i_qm(self):
        return self.forces_i_qm.copy()
            
    def init_traj(self, t, ndims, pos, mom, wid, m, nstates, istat, lab):

        self.time = t
        self.positions = pos
        self.momenta = mom
        self.widths = wid
        self.masses = m
        self.label = lab
        self.numstates = nstates
        self.istate = istat
        self.firsttime = t

    def init_clone_traj(self, parent, istate, jstate, label):

        self.numstates = parent.numstates

        self.istate = istate
        
        time = parent.time 
        self.time = time
        print "init_clone_traj: time =", time
        print "energies at tpdt", parent.energies_tpdt
        print "energies at tmdt", parent.energies_tmdt
        print "energies at t", parent.energies_t
        print "energies", parent.energies
        
        self.label = label
        
        pos = parent.positions_tpdt
        mom = parent.momenta_tpdt
        e = parent.energies_tpdt
        e_av = parent.av_energy_tpdt
                      
        self.positions = pos
        self.momenta = mom
        self.energies = e
        self.av_energy = e_av
#       cloning routines

#        During the cloning procedure we look at pairwise decoherence times, all population is going from
#        istate to jstate. The rest of the amplitudes remain unchanged
        parent_amp = parent.mce_amps
        wf = 0 + 0*1j
        parent_wf = 0 + 0*1j
        for kstate in range(self.numstates):
            if kstate == istate:
                # the population is removed from this state, so nothing to do here
                scaling_factor = np.sqrt(1 + np.dot(np.transpose(np.conjugate(parent_amp[jstate])), parent_amp[jstate])\
                                         / np.dot(np.transpose(np.conjugate(parent_amp[kstate])), parent_amp[kstate]))
                parent_wf += parent.approx_eigenvecs[:, kstate] * parent_amp[kstate] * scaling_factor
            
            elif kstate == jstate:
                # the population from istate is transferred to jstate
                scaling_factor = np.sqrt(1 + np.dot(np.transpose(np.conjugate(parent_amp[istate])), parent_amp[istate])\
                                         / np.dot(np.transpose(np.conjugate(parent_amp[kstate])), parent_amp[kstate]))
                wf += parent.approx_eigenvecs[:, kstate] * parent_amp[kstate] * scaling_factor
            
            else:
                # the rest of the states remain unchanged 
                wf += parent.approx_eigenvecs[:, kstate] * parent_amp[kstate]
                parent_wf += parent.approx_eigenvecs[:, kstate] * parent_amp[kstate]
                
        average_energy = 0.0
        pop = np.zeros(self.numstates)
        amp = np.zeros((self.numstates), dtype=np.complex128) 
        parent_pop = np.zeros(self.numstates)
        parent_amp = np.zeros((self.numstates), dtype=np.complex128)         
        eigenvectors_t = np.transpose(np.conjugate(parent.approx_eigenvecs))    
        parent_wf_T = np.transpose(np.conjugate(parent_wf))
        H_elec = self.construct_el_H(self.positions[0], self.positions[1])
        Hx, Hy = parent.construct_force(parent.positions_tpdt[0], parent.positions_tpdt[1])
        
        av_force = np.zeros((self.numdims))    
        av_force[0] = -np.real(np.dot(np.dot(parent_wf_T, Hx), parent_wf))
        av_force[1] = -np.real(np.dot(np.dot(parent_wf_T, Hy), parent_wf))
        print "init_clone_traj: parent force =", av_force
        a_tpdt = av_force / parent.masses
        print "init_clone_traj: parent accel =", a_tpdt
        for j in range(self.numstates):
            amp[j] = np.dot(eigenvectors_t[:, j], wf)
            pop[j] = np.real(np.dot(np.transpose(np.conjugate(amp[j])), amp[j]))
            parent_amp[j] = np.dot(eigenvectors_t[:, j], parent_wf)
            parent_pop[j] = np.real(np.dot(np.transpose(np.conjugate(parent_amp[j])), parent_amp[j]))
        
        average_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(wf)), H_elec), wf))
        parent_energy = np.real(np.dot(np.dot(np.transpose(np.conjugate(parent_wf)), H_elec), parent_wf))
        
        print "parent eigenvecs = ", parent.approx_eigenvecs
        print "parent wf = ", parent_wf
        print "child wf = ", wf
        print "child pop = ", pop
        print "parent pop = ", parent_pop
        print "clone init clone traj norm =", np.dot(np.transpose(np.conjugate(wf)), wf)
        
        # updating quantum parameters for child    
        self.td_wf = wf
        self.av_energy = float(average_energy)
        self.mce_amps = amp
        self.populations = pop
        
        print "momenta =", self.momenta
        print "positions =", pos
        print "energy of child =", self.av_energy
        print "energy of parent =", parent_energy
        
#         parent.rescale_momentum(e_av)
        
        # Setting mintime to current time to avoid backpropagation
        mintime = parent.time
        self.mintime = mintime

        self.firsttime = time

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
        
        self.timestep = parent.timestep

        z_dont = np.zeros(parent.numstates)
        z_dont[parent.istate] = 1.0
        self.clonethresh = parent.clonethresh
        
        self.potential_specific_traj_copy(parent)
        child_rescale_ok = self.rescale_momentum(e_av)
        if child_rescale_ok:
            parent_E_total = e_av + parent.calc_kin_en(mom, parent.masses)
            child_E_total = self.av_energy + self.calc_kin_en(self.momenta, self.masses)
            print "parent E before rescale=", parent_E_total 
#             print "child E =", child_E_total
#             print "parent momentum before rescale = ", parent.momenta
#             print "parent energy before rescale = ", parent.av_energy + parent.calc_kin_en(parent.momenta_tpdt, parent.masses)
            parent_rescale_ok = parent.rescale_parent_momentum(e_av, float(parent_energy), a_tpdt)
            print "parent E after rescale = ", parent.calc_kin_en(parent.momenta_tpdt, parent.masses) + parent_energy
#             print "parent momentum after rescale = ", parent.momenta
        # need to update wave function at half step since the parent changed ee properties 
        # during cloning
            if parent_rescale_ok:
                parent.td_wf = propagate_symplectic(parent, H_elec, parent_wf, parent.timestep/2, self.n_el_steps/2)
                parent.av_energy = float(parent_energy)
                parent.mce_amps = parent_amp
                parent.populations = parent_pop
                #parent.approx_eigenvecs = 
                return True
        else:
            return False
#  sys.exit()

    def rescale_momentum(self, v_ini):
        """This subroutine rescales the momentum of the child basis function
        The difference from spawning here is that the average Ehrenfest energy is rescaled,
        not of the pure elecronic states"""
        
        v_fin = self.av_energy

        # computing kinetic energy of parent.  Remember that, at this point,
        # the child's momentum is still that of the parent, so we compute
        # t_parent from the child's momentum
        p_ini = self.momenta
        m = self.masses
        t_ini = self.calc_kin_en(p_ini, m)
        
        factor = ( ( v_ini + t_ini - v_fin ) / t_ini )
        if factor < 0.0:
            print "# Aborting cloning because child does not have enough energy for momentum adjustment"
            return False
        factor = math.sqrt(factor)
        print "# rescaling momentum by factor ", factor
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
    
    def rescale_parent_momentum(self, v_ini, v_fin, accel):
        
        """This subroutine rescales the momentum of the child basis function
        The difference from spawning here is that the average Ehrenfest energy is rescaled,
        not of the pure elecronic states"""
        
        # computing kinetic energy of parent.  Remember that, at this point,
        # the child's momentum is still that of the parent, so we compute
        # t_parent from the child's momentum

        p_ini = self.momenta_tpdt
        m = self.masses
        t_ini = self.calc_kin_en(p_ini, m)
        print "pot energies: initial and final", v_ini, v_fin
        print "kinetic energy: initial", t_ini
                
        factor = ( ( v_ini + t_ini - v_fin ) / t_ini )
        if factor < 0.0:
            print "# Aborting cloning because child does not have enough energy for momentum adjustment"
            return False
        factor = math.sqrt(factor)
        print "# rescaling momentum by factor ", factor
        p_fin = factor * p_ini
        self.momenta_tpdt = p_fin
        # need to update velocity at half step and position at step ahead
        self.momenta = p_fin + 0.5 * self.timestep * accel
        print "rescale momentum: position 1", self.positions
        self.positions = self.positions + self.momenta / self.masses * self.timestep 
        print "rescale momentum: position 2", self.positions
        
        # Computing kinetic energy of child to make sure energy is conserved
        t_fin = 0.0
        for idim in range(self.numdims):
            print "m =", m[idim]
            t_fin += 0.5 * p_fin[idim] * p_fin[idim] / m[idim]
        if v_ini + t_ini - v_fin - t_fin > 1e-9: 
            print "ENERGY NOT CONSERVED!!!"
            sys.exit
        
        return True
                
    def set_forces(self,f):
        if f.shape == self.forces.shape:
            self.forces = f.copy()
        else:
            print "Error in set_forces"
            sys.exit

    def get_forces(self):
        return self.forces.copy()

    def get_forces_i(self):
        fi = self.get_forces()[self.istate, :]
        return fi

    def propagate_step(self):

        if float(self.time) - float(self.firsttime) < self.timestep:
            self.prop_first_step()
        else:
            self.prop_not_first_step()

        # consider whether to clone
        self.consider_cloning()
        
    def consider_cloning(self):
        """Marking trajectories for cloning using z_clone_now variable
        as a parameter we use (Ei - Eav) * POPi"""
        
        z = self.z_clone_now
        print "CONSIDERING CLONING:"
        clone_parameter = np.zeros(self.numstates)
        
        m = self.masses
        ke_tot = 0.0
        p = np.zeros((self.numstates, self.numstates))
        tau = np.zeros((self.numstates, self.numstates))
        for idim in range(self.numdims):
            ke_tot += 0.5 * self.momenta_tmdt[idim] * self.momenta_tmdt[idim] / m[idim]
        
        for istate in range(self.numstates):
            for jstate in range(self.numstates):
                if istate == jstate:
                    tau[istate, jstate] = 0.0
                    p[istate, jstate] = 0.0
                else:
                    dE = np.abs(self.energies[jstate] - self.energies[istate])
                    tau[istate, jstate] = (1 + self.t_decoherence_par / ke_tot) / dE
                    p[istate, jstate] = 1 - np.exp(-self.timestep / tau[istate, jstate])
        print "p = ", p                    
        self.clone_p = p

#             if clone_parameter[jstate] > self.clonethresh and jstate == np.argmax(clone_parameter):
#                 print "CLONING TO STATE ", jstate, " at time ", self.time
#                 # setting z_clone_now indicates that
#                 # this trajectory should clone to jstate
#                 z[jstate] = 1.0
#         print "max threshold", np.argmax(clone_parameter)        
#         self.z_clone_now = z 
            
    def h5_output(self, zdont_half_step=False):

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
            #print "key", key
            dset = trajgrp.get(key)
            l = dset.len()
            dset.resize(l+1, axis=0)
            ipos=l

            if key == "forces_i":
                getcom = "self.get_" + key + "()"
            else:
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
            if key == "td_wf" or key == "mce_amps":
                dset = trajgrp.create_dataset(key, (0,n), maxshape=(None,n), dtype="complex128")
            else:
                dset = trajgrp.create_dataset(key, (0,n), maxshape=(None,n), dtype="float64")

        for key in self.h5_datasets_half_step:
            n = self.h5_datasets_half_step[key]
            if key == "td_wf" or key == "mce_amps":
                dset = trajgrp.create_dataset(key, (0,n), maxshape=(None,n), dtype="complex128")
            else:
                dset = trajgrp.create_dataset(key, (0,n), maxshape=(None,n), dtype="float64")
        # add some metadata
        trajgrp.attrs["istate"] = self.istate
        trajgrp.attrs["masses"] = self.masses
        trajgrp.attrs["widths"] = self.widths
        if hasattr(self,"atoms"):
            trajgrp.attrs["atoms"] = self.atoms
        
    def get_data_at_time_from_h5(self, t, dset_name):
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
            #print "comm ", comm
            #print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()
            
    def get_all_qm_data_at_time_from_h5_half_step(self, t):
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
        print "## randomly selecting Wigner initial conditions"
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

        print '# eigenvalues of the mass-weighted hessian are (a.u.)'
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

        print "# ZPE = ", zpe
        print "# kinetic energy = ", ke
        