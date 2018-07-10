# trajectory objects contain individual trajectory basis functions
import numpy as np
import sys
import math
from pyspawn.fmsobj import fmsobj
import os
import shutil
import h5py
from numpy import dtype

class traj(fmsobj):
    def __init__(self):
        self.time = 0.0
        self.time_half_step = 0.0
        self.maxtime = -1.0
        self.mintime = 0.0
        self.firsttime = 0.0
        self.numdims = 2
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.widths = np.zeros(self.numdims)
        self.masses = np.zeros(self.numdims)
        self.istate = 0
        self.label = "00"
        self.h5_datasets = dict()
        self.h5_datasets_half_step = dict()

        self.timestep = 0.0
        
        self.numstates = 2
        self.length_wf = self.numstates
        self.wf = np.zeros((self.numstates,self.length_wf))
        self.prev_wf = np.zeros((self.numstates,self.length_wf))
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates,self.numdims))
        self.timederivcoups = np.zeros(self.numstates)
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
        
        self.spawnlastcoup = np.zeros(self.numstates)
        self.numchildren = 0
        
        self.positions_qm = np.zeros(self.numdims)
        self.momenta_qm = np.zeros(self.numdims)
        self.energies_qm = np.zeros(self.numstates)
        self.forces_i_qm = np.zeros(self.numdims)
        self.timederivcoups_qm = np.zeros(self.numstates)

        #In the following block there are variables needed for ehrenfest
        self.td_wf = np.zeros((self.numstates), dtype = np.complex128)
        self.mce_amps = np.zeros((self.numstates), dtype = np.complex128)
        self.populations = np.zeros(self.numstates)
        self.av_energy = 0.0
        self.av_force = np.zeros(self.numdims)
        self.approx_eigenvecs = np.zeros((self.numstates, self.numstates))
        self.z_clone_now = np.zeros(self.numstates)
        
        self.clonethresh = 0.0
        self.clonetimes = -1.0 * np.ones(self.numstates)
        self.z_clone_now = np.zeros(self.numstates)

    def get_mce_amps(self):
        return self.mce_amps.copy()
 
    def set_mce_amps(self, amps):
        self.mce_amps = amps

    def get_approx_eigenvecs(self):
        return self.approx_eigenvecs.copy()

    def set_approx_eigenvecs(self, eigenvecs):
        self.approx_eigenvecs = eigenvecs

    def get_clonethresh(self):
        return self.clonethresh

    def set_clonethresh(self, thresh):
        self.clonethresh = thresh

    def get_av_force(self):
        return self.av_force.copy()

    def set_av_force(self, av_force):
        self.av_force = av_force

    def get_av_energy(self):
        return self.av_energy

    def set_av_energy(self, e):
        self.av_energy = e

    def get_td_wf(self):
        return self.td_wf.copy()

    def set_td_wf(self, wf):
        self.td_wf = wf

    def get_populations(self):
        return self.populations
        
    def set_populations(self, pop):
        self.populations = pop

    def remove_state_pop(self, jstate):
        """After cloning of the wave function to the jstate the cloned wf have all population on jstate
        the rest of the population stays on the parent wf. This subroutine removes all population on jstate
        from the parent wf and updates the affected quantities"""
        for istate in range(self.numstates):
            new_wf = np.zeros((self.numstates), dtype = np.complex128)
            if istate != jstate:
                new_wf += self.approx_eigenvecs[istate, :] * self.mce_amps[istate]\
                                                    / np.sqrt(1-self.populations[jstate])

        self.td_wf = new_wf
    # End of Ehrenfest block
    
    def set_time(self,t):
        self.time = t
    
    def get_time(self):
        return self.time
    
    def set_time_half_step(self,t):
        self.time_half_step = t
    
    def get_time_half_step(self):
        return self.time_half_step
    
    def set_timestep(self,h):
        self.timestep = h
    
    def get_timestep(self):
        return self.timestep
    
    def set_maxtime(self,t):
        self.maxtime = t
    
    def get_maxtime(self):
        return self.maxtime
    
    def set_firsttime(self,t):
        self.firsttime = t
    
    def get_firsttime(self):
        return self.firsttime
    
    def get_mintime(self):
        return self.mintime
    
    def set_mintime(self,t):
        self.mintime = t
    
    def set_numdims(self,ndims):
        self.numdims = ndims
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.widths = np.zeros(self.numdims)
        self.masses = np.zeros(self.numdims)
        self.forces = np.zeros((self.numstates,self.numdims))
        
        self.positions_t = np.zeros(self.numdims)
        self.positions_tmdt = np.zeros(self.numdims)
        self.positions_tpdt = np.zeros(self.numdims)
        self.momenta_t = np.zeros(self.numdims)
        self.momenta_tmdt = np.zeros(self.numdims)
        self.momenta_tpdt = np.zeros(self.numdims)
        self.positions_qm = np.zeros(self.numdims)
        self.momenta_qm = np.zeros(self.numdims)
        self.forces_i_qm = np.zeros(self.numdims)
        
    def set_numstates(self,nstates):
        self.numstates = nstates
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates,self.numdims))

        self.spawnlastcoup = np.zeros(self.numstates)
        self.z_clone_now = np.zeros(self.numstates)

        self.energies_tpdt = np.zeros(self.numstates)
        self.energies_t = np.zeros(self.numstates)
        self.energies_tmdt = np.zeros(self.numstates)
        
    def set_istate(self,ist):
        self.istate = ist

    def get_istate(self):
        return self.istate

    def set_jstate(self,ist):
        self.jstate = ist

    def get_jstate(self):
        return self.jstate

    def get_numstates(self):
        return self.numstates

    def get_numdims(self):
        return self.numdims

    def set_numchildren(self,ist):
        self.numchildren = ist

    def get_numchildren(self):
        return self.numchildren

    def incr_numchildren(self):
        self.set_numchildren(self.get_numchildren() + 1)

    def set_positions(self,pos):
        if pos.shape == self.positions.shape:
            self.positions = pos.copy()
        else:
            print "Error in set_positions"
            sys.exit

    def get_positions(self):
        return self.positions.copy()
            
    def set_positions_qm(self,pos):
        if pos.shape == self.positions_qm.shape:
            self.positions_qm = pos.copy()
        else:
            print "Error in set_positions_qm"
            sys.exit

    def get_positions_qm(self):
        return self.positions_qm.copy()
            
    def set_positions_t(self,pos):
        if pos.shape == self.positions_t.shape:
            self.positions_t = pos.copy()
        else:
            print "Error in set_positions_t"
            sys.exit

    def get_positions_t(self):
        return self.positions_t.copy()
            
    def set_positions_tmdt(self,pos):
        if pos.shape == self.positions_tmdt.shape:
            self.positions_tmdt = pos.copy()
        else:
            print "Error in set_positions_tmdt"
            sys.exit

    def get_positions_tmdt(self):
        return self.positions_tmdt.copy()
            
    def set_positions_tpdt(self,pos):
        if pos.shape == self.positions_tpdt.shape:
            self.positions_tpdt = pos.copy()
        else:
            print "Error in set_positions_tpdt"
            sys.exit

    def get_positions_tpdt(self):
        return self.positions_tpdt.copy()
            
    def set_momenta_qm(self,mom):
        if mom.shape == self.momenta_qm.shape:
            self.momenta_qm = mom.copy()
        else:
            print "Error in set_momenta"
            sys.exit

    def get_momenta_qm(self):
        return self.momenta_qm.copy()
            
    def set_forces_i_qm(self,f):
        if f.shape == self.forces_i_qm.shape:
            self.forces_i_qm = f.copy()
        else:
            print "Error in set_forces_i_qm"
            sys.exit

    def get_forces_i_qm(self):
        return self.forces_i_qm.copy()
            
    def set_momenta(self,mom):
        if mom.shape == self.momenta.shape:
            self.momenta = mom.copy()
        else:
            print "Error in set_momenta"
            sys.exit

    def get_momenta(self):
        return self.momenta.copy()
            
    def set_momenta_t(self,mom):
        if mom.shape == self.momenta_t.shape:
            self.momenta_t = mom.copy()
        else:
            print "Error in set_momenta_t"
            sys.exit

    def get_momenta_t(self):
        return self.momenta_t.copy()
            
    def set_momenta_tmdt(self,mom):
        if mom.shape == self.momenta_tmdt.shape:
            self.momenta_tmdt = mom.copy()
        else:
            print "Error in set_momenta_tmdt"
            sys.exit

    def get_momenta_tmdt(self):
        return self.momenta_tmdt.copy()
            
    def set_momenta_tpdt(self,mom):
        if mom.shape == self.momenta_tpdt.shape:
            self.momenta_tpdt = mom.copy()
        else:
            print "Error in set_momenta_tpdt"
            sys.exit

    def get_momenta_tpdt(self):
        return self.momenta_tpdt.copy()
            
    def set_energies_qm(self,e):
        if e.shape == self.energies_qm.shape:
            self.energies_qm = e.copy()
        else:
            print "Error in set_energies_qm"
            sys.exit

    def get_energies_qm(self):
        return self.energies_qm.copy()

    def set_energies_t(self,e):
        if e.shape == self.energies_t.shape:
            self.energies_t = e.copy()
        else:
            print "Error in set_energies_t"
            sys.exit

    def get_energies_t(self):
        return self.energies_t.copy()
            
    def set_energies_tmdt(self,e):
        if e.shape == self.energies_tmdt.shape:
            self.energies_tmdt = e.copy()
        else:
            print "Error in set_energies_tmdt"
            sys.exit

    def get_energies_tmdt(self):
        return self.energies_tmdt.copy()
            
    def set_energies_tpdt(self,e):
        if e.shape == self.energies_tpdt.shape:
            self.energies_tpdt = e.copy()
        else:
            print "Error in set_energies_tpdt"
            sys.exit

    def get_energies_tpdt(self):
        return self.energies_tpdt.copy()
            
    def set_widths(self,wid):
        if wid.shape == self.widths.shape:
            self.widths = wid.copy()
        else:
            print "Error in set_widths"
            sys.exit

    def get_widths(self):
        return self.widths.copy()

    def set_masses(self,m):
        if m.shape == self.masses.shape:
            self.masses = m.copy()
        else:
            print "Error in set_masses"
            sys.exit

    def get_masses(self):
        return self.masses.copy()

    def set_atoms(self,a):
        self.atoms = a[:]

    def get_atoms(self):
        return self.atoms[:]

    def set_label(self,lab):
        self.label = lab

    def get_label(self):
        #if self.label.type == 'unicode':
        #    self.set_label(str(self.label))
        return self.label

    def init_traj(self,t,ndims,pos,mom,wid,m,nstates,istat,lab):
        self.set_time(t)
        self.set_numdims(ndims)
        self.set_positions(pos)
        self.set_momenta(mom)
        self.set_widths(wid)
        self.set_masses(m)
        self.set_label(lab)
        self.set_numstates(nstates)
        self.set_istate(istat)

        self.set_firsttime(t)

    def init_clone_traj(self, parent, istate, label):
        self.set_numstates(parent.get_numstates())
        self.set_numdims(parent.get_numdims())

        self.set_istate(istate)
        
        time = parent.get_time() - 2.0 * parent.get_timestep()
        self.set_time(time)

        self.set_label(label)
        
        pos = parent.get_positions_tmdt()
        mom = parent.get_momenta_tmdt()
        e = parent.get_energies_tmdt()
                      
        self.set_positions(pos)
        self.set_momenta(mom)
        self.set_energies(e)

#       cloning routines

#        Projecting out everything but the population on the desired state
        parent_amp = parent.mce_amps
        self.td_wf = parent.get_approx_eigenvecs()[self.istate, :] * parent_amp / np.abs(parent_amp)
        average_energy = 0.0
        pop = np.zeros(self.numstates)
        amp = np.zeros((self.numstates), dtype=np.complex128) 
        eigenvectors_t = np.transpose(np.conjugate(parent.get_approx_eigenvecs()))    
        wf = self.td_wf
         
        for k in range(self.numstates):
            amp[k] = np.dot(eigenvectors_t[k, :], wf)
            pop[k] = np.real(np.dot(np.transpose(np.conjugate(amp[k])), amp[k]))
            average_energy += pop[k] * parent.energies[k]
            
        self.av_energy = float(average_energy)
        
        print "energy of child =", self.av_energy
        print "energy of parent =", parent.av_energy
        # Setting mintime to current time to avoid backpropagation
        mintime = parent.get_time()
        self.set_mintime(mintime)

        self.set_firsttime(time)

        self.set_maxtime(parent.get_maxtime())
        self.set_widths(parent.get_widths())
        self.set_masses(parent.get_masses())
        if hasattr(parent,'atoms'):
            self.set_atoms(parent.get_atoms())
        if hasattr(parent,'civecs'):
            self.set_civecs(parent.get_civecs())
            self.set_backprop_civecs(parent.get_civecs())
            self.set_ncivecs(parent.get_ncivecs())
        if hasattr(parent,'orbs'):
            self.set_orbs(parent.get_orbs())
            self.set_backprop_orbs(parent.get_orbs())
            self.set_norbs(parent.get_norbs())
        if hasattr(parent,'prev_wf_positions'):
            self.set_prev_wf_positions(parent.get_prev_wf_positions())
            self.set_backprop_prev_wf_positions(parent.get_prev_wf_positions())
        if hasattr(parent,'electronic_phases'):
            self.set_electronic_phases(parent.get_electronic_phases())
            self.set_backprop_electronic_phases(parent.get_electronic_phases())
        
        self.set_timestep(parent.get_timestep())

        z_dont = np.zeros(parent.get_numstates())
        z_dont[parent.get_istate()] = 1.0
        self.set_clonethresh(parent.get_clonethresh())
        
        self.potential_specific_traj_copy(parent)

    def rescale_momentum(self, v_parent):
        v_child = self.av_energy
        print "rescale v_child ", v_child
        print "rescale v_parent ", v_parent

        # computing kinetic energy of parent.  Remember that, at this point,
        # the child's momentum is still that of the parent, so we compute
        # t_parent from the child's momentum
        p_parent = self.get_momenta()
        m = self.get_masses()
        t_parent = 0.0
        for idim in range(self.get_numdims()):
            t_parent += 0.5 * p_parent[idim] * p_parent[idim] / m[idim]
        print "rescale t_parent ", t_parent
        factor = ( ( v_parent + t_parent - v_child ) / t_parent )
        if factor < 0.0:
            print "# Aborting cloning because child does not have"
            print "# enough energy for momentum adjustment"
            return False
        factor = math.sqrt(factor)
        print "# rescaling momentum by factor ", factor
        p_child = factor * p_parent
        self.set_momenta(p_child)
        
        # Computing kinetic energy of child to make sure energy is conserved
        t_child = 0.0
        for idim in range(self.get_numdims()):
            t_child += 0.5 * p_child[idim] * p_child[idim] / m[idim]
        print "total energy of parent =", v_parent + t_parent
        print "total energy of child =", v_child + t_child
        if v_parent + t_parent - v_child - t_child > 1e-9: 
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
        fi = self.get_forces()[self.get_istate(),:]
        return fi

    def set_energies(self,e):
        if e.shape == self.energies.shape:
            self.energies = e.copy()
        else:
            print "Error in set_forces"
            sys.exit

    def get_energies(self):
        return self.energies.copy()

    def set_wf(self,wf):
        if wf.shape == self.wf.shape:
            self.wf = wf.copy()
        else:
            print "Error in set_wf"
            sys.exit

    def get_wf(self):
        return self.wf.copy()

    def set_prev_wf(self,wf):
        if wf.shape == self.prev_wf.shape:
            self.prev_wf = wf.copy()
        else:
            print "Error in set_prev_wf"
            sys.exit

    def get_prev_wf(self):
        return self.prev_wf.copy()
    
    def set_S_elec_flat(self,S):
        self.S_elec_flat = S.copy()
    
    def get_S_elec_flat(self):
        return self.S_elec_flat.copy()

    def set_z_clone_now(self,z):
        if z.shape == self.z_clone_now.shape:
            self.z_clone_now = z.copy()
        else:
            print "Error in set_z_clone_now"
            sys.exit
    
    def get_z_clone_now(self):
        return self.z_clone_now.copy()
    
    def set_z_compute_me(self,z):
        self.z_compute_me = z
    
    def get_z_compute_me(self):
        return self.z_compute_me

    def propagate_step(self, zbackprop=False):

        if abs(eval("self.get_time()") - self.get_firsttime()) < 1.0e-6:
            self.prop_first_step(zbackprop=zbackprop)
        else:
            self.prop_not_first_step(zbackprop=zbackprop)

        # consider whether to clone
        self.consider_cloning()
          
    def consider_cloning(self):

        thresh = self.clonethresh
        z = self.get_z_clone_now()
        print "CONSIDERING CLONING:"
        clone_parameter = np.zeros(self.numstates)
        
        for jstate in range(self.numstates):
            dE = self.energies[jstate] - self.av_energy
            clone_parameter[jstate] = np.abs(dE*self.populations[jstate])
            print "cloning parameter = ", clone_parameter[jstate]
            print "dE =", dE
            print "pop = ", self.populations[jstate]
            print "Threshold =", thresh
            print "z < 1", z.any() < 0.5

            if clone_parameter[jstate] > thresh and jstate == np.argmax(clone_parameter):
                print "CLONING TO STATE ", jstate, " at time ", self.get_time()
                # setting z_clone_now indicates that
                # this trajectory should clone to jstate
                z[jstate] = 1.0
        print "max threshold", np.argmax(clone_parameter)        
        self.set_z_clone_now(z) 
            
    def h5_output(self, zbackprop,zdont_half_step=False):

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
            dset.resize(l+1,axis=0)
            if not zbackprop:
                ipos=l
            else:
                ipos=0
                dset[1:(l+1),0:n] = dset[0:(l),0:n]
            getcom = "self.get_" + key + "()"
#             print getcom
            tmp = eval(getcom)
#             print "\nkey =", key
            if n!=1:
                dset[ipos,0:n] = tmp[0:n]
            else:
                dset[ipos,0] = tmp
        h5f.flush()
        h5f.close()

    # create a new trajectory group in hdf5 output file
    def create_h5_traj(self, h5f, groupname):
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
        
    def get_data_at_time_from_h5(self,t,dset_name):
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
        data = np.zeros(len(dset[ipoint,:]))
        data = dset[ipoint,:]
        #print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()
        return data

    def get_all_qm_data_at_time_from_h5(self,t,suffix=""):
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
            data = np.zeros(len(dset[ipoint,:]))
            data = dset[ipoint,:]
            comm = "self." + dset_name + "_qm" + suffix + " = data"
            exec(comm)
            #print "comm ", comm
            #print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()
            
    def get_all_qm_data_at_time_from_h5_half_step(self,t):
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
            data = np.zeros(len(dset[ipoint,:]))
            data = dset[ipoint,:]
            comm = "self." + dset_name + "_qm = data"
            exec(comm)
            #print "comm ", comm
            #print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()
            
    def initial_wigner(self,iseed):
        print "## randomly selecting Wigner initial conditions"
        ndims = self.get_numdims()

        h5f = h5py.File('hessian.hdf5', 'r')
        
        pos = h5f['geometry'][:].flatten()

        h = h5f['hessian'][:]

        m = self.get_masses()

        sqrtm = np.sqrt(m)

        #build mass weighted hessian
        h_mw = np.zeros_like(h)

        for idim in range(ndims):
            h_mw[idim,:] = h[idim,:] / sqrtm

        for idim in range(ndims):
            h_mw[:,idim] = h_mw[:,idim] / sqrtm

        # symmetrize mass weighted hessian
        h_mw = 0.5 * (h_mw + h_mw.T)

        # diagonalize mass weighted hessian
        evals, modes = np.linalg.eig(h_mw)

        # sort eigenvectors
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        modes = modes[:,idx]

        print '# eigenvalues of the mass-weighted hessian are (a.u.)'
        print evals
        
        # seed random number generator
        np.random.seed(iseed)
        
        alphax = np.sqrt(evals[0:ndims-6])/2.0

        sigx = np.sqrt(1.0 / ( 4.0 * alphax))
        sigp = np.sqrt(alphax)

        dtheta = 2.0 * np.pi * np.random.rand(ndims-6)
        dr = np.sqrt( np.random.rand(ndims-6) )

        dx1 = dr * np.sin(dtheta)
        dx2 = dr * np.cos(dtheta)

        rsq = dx1 * dx1 + dx2 * dx2

        fac = np.sqrt( -2.0 * np.log(rsq) / rsq )

        x1 = dx1 * fac
        x2 = dx2 * fac

        posvec = np.append(sigx * x1, np.zeros(6))
        momvec = np.append(sigp * x2, np.zeros(6))

        deltaq = np.matmul(modes,posvec)/sqrtm
        pos += deltaq
        mom = np.matmul(modes,momvec)*sqrtm

        self.set_positions(pos)
        self.set_momenta(mom)

        zpe = np.sum(alphax[0:ndims-6])
        ke = 0.5 * np.sum(mom * mom / m)

        print "# ZPE = ", zpe
        print "# kinetic energy = ", ke
        
