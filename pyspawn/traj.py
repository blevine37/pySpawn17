# trajectory objects contain individual trajectory basis functions
import numpy as np
import sys
import math
from pyspawn.fmsobj import fmsobj
import os
import shutil
import h5py

class traj(fmsobj):
    def __init__(self):
        self.time = 0.0
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

        self.timestep = 0.0
        self.propagator = "vv"
        
        self.numstates = 2
        self.software = "pyspawn"
        self.method = "cone"
        self.length_wf = self.numstates
        self.wf = np.zeros((self.numstates,self.length_wf))
        self.prev_wf = np.zeros((self.numstates,self.length_wf))
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates,self.numdims))
        self.timederivcoups = np.zeros(self.numstates)

        self.backprop_time = 0.0
        self.backprop_energies = np.zeros(self.numstates)
        self.backprop_forces = np.zeros((self.numstates,self.numdims))
        self.backprop_positions = np.zeros(self.numdims)
        self.backprop_momenta = np.zeros(self.numdims)
        self.backprop_wf = np.zeros((self.numstates,self.length_wf))
        self.backprop_prev_wf = np.zeros((self.numstates,self.length_wf))

        self.spawntimes = -1.0 * np.ones(self.numstates)
        self.spawnthresh = 0.0
        self.spawnlastcoup = np.zeros(self.numstates)
        self.positions_tpdt = np.zeros(self.numdims)
        self.positions_t = np.zeros(self.numdims)
        self.positions_tmdt = np.zeros(self.numdims)
        self.momenta_tpdt = np.zeros(self.numdims)
        self.momenta_t = np.zeros(self.numdims)
        self.momenta_tmdt = np.zeros(self.numdims)
        self.energies_tpdt = np.zeros(self.numstates)
        self.energies_t = np.zeros(self.numstates)
        self.energies_tmdt = np.zeros(self.numstates)
        self.z_spawn_now = np.zeros(self.numstates)
        self.z_dont_spawn = np.zeros(self.numstates)
        self.numchildren = 0
        
        self.positions_qm = np.zeros(self.numdims)
        self.momenta_qm = np.zeros(self.numdims)
        self.energies_qm = np.zeros(self.numstates)

    def set_time(self,t):
        self.time = t
    
    def get_time(self):
        return self.time
    
    def set_timestep(self,h):
        self.timestep = h
    
    def get_timestep(self):
        return self.timestep
    
    def set_backprop_time(self,t):
        self.backprop_time = t
    
    def get_backprop_time(self):
        return self.backprop_time
    
    def set_maxtime(self,t):
        self.maxtime = t
    
    def get_maxtime(self):
        return self.maxtime
    
    def set_firsttime(self,t):
        self.firsttime = t
    
    def get_firsttime(self):
        return self.firsttime
    
    def get_propagator(self):
        return self.propagator
    
    def set_propagator(self,prop):
        self.propagator = prop
    
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
        #self.last_es_positions = np.zeros(self.numdims)
        self.forces = np.zeros((self.numstates,self.numdims))

        self.backprop_positions = np.zeros(self.numdims)
        self.backprop_momenta = np.zeros(self.numdims)
        self.backprop_forces = np.zeros((self.numstates,self.numdims))
        
        self.positions_t = np.zeros(self.numdims)
        self.positions_tmdt = np.zeros(self.numdims)
        self.positions_tpdt = np.zeros(self.numdims)
        self.momenta_t = np.zeros(self.numdims)
        self.momenta_tmdt = np.zeros(self.numdims)
        self.momenta_tpdt = np.zeros(self.numdims)
        #self.prev_positions = np.zeros(self.numdims)
        #self.prev_forces = np.zeros((self.numstates,self.numdims))
        self.positions_qm = np.zeros(self.numdims)
        self.momenta_qm = np.zeros(self.numdims)
        
    def set_numstates(self,nstates):
        self.numstates = nstates
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates,self.numdims))

        self.backprop_energies = np.zeros(self.numstates)
        self.backprop_forces = np.zeros((self.numstates,self.numdims))

        self.spawntimes = -1.0 * np.ones(self.numstates)
        self.timederivcoups = np.zeros(self.numstates)
        self.spawnlastcoup = np.zeros(self.numstates)
        self.z_spawn_now = np.zeros(self.numstates)
        self.z_dont_spawn = np.zeros(self.numstates)

        self.energies_tpdt = np.zeros(self.numstates)
        self.energies_t = np.zeros(self.numstates)
        self.energies_tmdt = np.zeros(self.numstates)
        #self.prev_energies = np.zeros(self.numstates)
        #self.prev_forces = np.zeros((self.numstates,self.numdims))

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

    def set_software(self,sw):
        self.software = sw

    def set_method(self,meth):
        self.method = meth

    def get_software(self):
        return self.software

    def get_method(self):
        return self.method

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
            
    def set_backprop_positions(self,pos):
        if pos.shape == self.backprop_positions.shape:
            self.backprop_positions = pos.copy()
        else:
            print "Error in set_backprop_positions"
            sys.exit

    def get_backprop_positions(self):
        return self.backprop_positions.copy()
            
    def set_backprop_momenta(self,mom):
        if mom.shape == self.backprop_momenta.shape:
            self.backprop_momenta = mom.copy()
        else:
            print "Error in set_backprop_momenta"
            sys.exit

    def get_backprop_momenta(self):
        return self.backprop_momenta.copy()
            
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

        self.set_backprop_time(t)
        self.set_backprop_positions(pos)
        self.set_backprop_momenta(mom)

        self.set_firsttime(t)

    def init_spawn_traj(self, parent, istate, label):
        self.set_numstates(parent.get_numstates())
        self.set_numdims(parent.get_numdims())

        self.set_istate(istate)
        
        time = parent.get_time() - 2.0 * parent.get_timestep()
        self.set_time(time)

        self.set_label(label)
        
        pos = parent.get_positions_tmdt()
        mom = parent.get_momenta_tmdt()
        e = parent.get_energies_tmdt()

# adjust momentum
        
        self.set_positions(pos)
        self.set_momenta(mom)
        self.set_energies(e)

        mintime = parent.get_spawntimes()[istate]
        print "init_st mintime", mintime
        self.set_mintime(mintime)
        print "init_st time", time
        self.set_backprop_time(time)
        self.set_backprop_positions(pos)
        self.set_backprop_momenta(mom)
        print "init_st mintime2", self.get_mintime()

        self.set_firsttime(time)

        self.set_maxtime(parent.get_maxtime())
        self.set_widths(parent.get_widths())
        self.set_masses(parent.get_masses())
        
        self.set_timestep(parent.get_timestep())
        self.set_propagator(parent.get_propagator())

        z_dont = np.zeros(parent.get_numstates())
        z_dont[parent.get_istate()] = 1.0
        self.set_z_dont_spawn(z_dont)
        print "init_st mintime0", self.get_mintime()

    def init_centroid(self, existing, child, label):
        self.set_numstates(child.get_numstates())
        self.set_numdims(child.get_numdims())

        self.set_istate(child.get_istate())
        self.set_jstate(existing.get_istate())
        
        time = child.get_time()
        self.set_time(time)

        self.set_label(label)
        
        self.set_mintime(child.get_mintime())
        self.set_backprop_time(time)
        self.set_firsttime(time)
        self.set_maxtime(child.get_maxtime())
        
        self.set_widths(child.get_widths())
        self.set_masses(child.get_masses())
        
        self.set_timestep(child.get_timestep())

    def rescale_momentum(self, v_parent):
        v_child = self.get_energies()[self.get_istate()]
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
            print "Aborting spawn because child does not have"
            print "enough energy for momentum adjustment"
            return False
        print "rescale factor ", factor
        factor = math.sqrt(factor)
        print "rescale factor ", factor
        p_child = factor * p_parent
        print "rescale p_child ", p_child
        print "rescale p_parent ", p_parent
        self.set_momenta(p_child)
        return True

    def set_forces(self,f):
        if f.shape == self.forces.shape:
            self.forces = f.copy()
        else:
            print "Error in set_forces"
            sys.exit

    def get_forces(self):
        return self.forces.copy()

    def set_backprop_forces(self,f):
        if f.shape == self.backprop_forces.shape:
            self.backprop_forces = f.copy()
        else:
            print "Error in set_forces"
            sys.exit

    def get_backprop_forces(self):
        return self.backprop_forces.copy()

    def set_energies(self,e):
        if e.shape == self.energies.shape:
            self.energies = e.copy()
        else:
            print "Error in set_forces"
            sys.exit

    def get_energies(self):
        return self.energies.copy()

    def set_backprop_energies(self,e):
        if e.shape == self.backprop_energies.shape:
            self.backprop_energies = e.copy()
        else:
            print "Error in set_forces"
            sys.exit

    def get_backprop_energies(self):
        return self.backprop_energies.copy()

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

    def set_backprop_wf(self,wf):
        if wf.shape == self.backprop_wf.shape:
            self.backprop_wf = wf.copy()
        else:
            print "Error in set_backprop_wf"
            sys.exit

    def get_backprop_wf(self):
        return self.wf.copy()

    def set_backprop_prev_wf(self,wf):
        if wf.shape == self.backprop_prev_wf.shape:
            self.backprop_prev_wf = wf.copy()
        else:
            print "Error in set_backprop_prev_wf"
            sys.exit

    def get_backprop_prev_wf(self):
        return self.prev_wf.copy()

    def set_spawntimes(self,st):
        if st.shape == self.spawntimes.shape:
            self.spawntimes = st.copy()
        else:
            print "Error in set_spawntimes"
            sys.exit

    def get_spawntimes(self):
        return self.spawntimes.copy()

    def set_timederivcoups(self,t):
        if t.shape == self.timederivcoups.shape:
            self.timederivcoups = t.copy()
        else:
            print "Error in set_spawntimes"
            sys.exit
    
    def get_timederivcoups(self):
        return self.timederivcoups.copy()
    
    def set_timederivcoups_qm(self,t):
        if t.shape == self.timederivcoups_qm.shape:
            self.timederivcoups_qm = t.copy()
        else:
            print "Error in set_spawntimes"
            sys.exit
    
    def get_timederivcoups_qm(self):
        return self.timederivcoups_qm.copy()
    
    def set_spawnlastcoup(self,tdc):
        if tdc.shape == self.spawnlastcoup.shape:
            self.spawnlastcoup = tdc.copy()
        else:
            print "Error in set_spawnlastcoup"
            sys.exit
    
    def get_spawnlastcoup(self):
        return self.spawnlastcoup.copy()
    
    def set_z_spawn_now(self,z):
        if z.shape == self.z_spawn_now.shape:
            self.z_spawn_now = z.copy()
        else:
            print "Error in set_z_spawn_now"
            sys.exit
    
    def get_z_spawn_now(self):
        return self.z_spawn_now.copy()
    
    def set_z_dont_spawn(self,z):
        if z.shape == self.z_dont_spawn.shape:
            self.z_dont_spawn = z.copy()
        else:
            print "Error in set_z_dont_spawn"
            sys.exit
    
    def get_z_dont_spawn(self):
        return self.z_dont_spawn.copy()
    
    def set_spawnthresh(self,t):
        self.spawnthresh = t
    
    def get_spawnthresh(self):
        return self.spawnthresh
    
    def set_z_compute_me(self,z):
        self.z_compute_me = z
    
    def get_z_compute_me(self):
        return self.z_compute_me
    
    def set_z_compute_me_backprop(self,z):
        self.z_compute_me_backprop = z
    
    def get_z_compute_me_backprop(self):
        return self.z_compute_me_backprop
    
    def compute_elec_struct(self,zbackprop):
        tmp = "self.compute_elec_struct_" + self.get_software() + "_" + self.get_method() + "(zbackprop)"
        eval(tmp)

    def propagate_step(self, zbackprop=False):
        if not zbackprop:
            cbackprop = ""
        else:
            cbackprop = "backprop_"
        if eval("self.get_" + cbackprop + "time()") == self.get_firsttime():
            tmp = "self.prop_first_" + self.propagator + "(zbackprop=" + str(zbackprop) + ")"
            eval(tmp)
        else:
            tmp = "self.prop_" + self.propagator + "(zbackprop=" + str(zbackprop) + ")"
            eval(tmp)

        # consider whether to spawn
        if not zbackprop:
            self.consider_spawning()

    def compute_centroid(self, zbackprop=False):
        self.compute_elec_struct(zbackprop)
        if zbackprop:
            backprop_time = self.get_backprop_time()
            firsttime = self.get_firsttime()
            self.set_backprop_time(backprop_time - self.get_timestep())
            if (backprop_time + 1.0e-6) < firsttime:
                self.h5_output(zbackprop)
        else:
            self.set_time(self.get_time() + self.get_timestep())
            self.h5_output(zbackprop)
                       
    def consider_spawning(self):
        tdc = self.get_timederivcoups()
        lasttdc = self.get_spawnlastcoup()
        spawnt = self.get_spawntimes()
        thresh = self.get_spawnthresh()
        z_dont_spawn = self.get_z_dont_spawn()
        z = self.get_z_spawn_now()
        
        for jstate in range(self.numstates):
            print "consider1 ", jstate, self.get_istate()
            if (jstate != self.get_istate()):
                print "consider2 ",spawnt[jstate]
                if spawnt[jstate] > -1.0e-6:
        #check to see if a trajectory in a spawning region is ready to spawn
                    print "consider3 ",tdc[jstate], lasttdc[jstate]
                    if abs(tdc[jstate]) < abs(lasttdc[jstate]):
                        print "Spawning to state ", jstate, " at time ", self.get_time()
                        # setting z_spawn_now indicates that
                        # this trajectory should spawn to jstate
                        z[jstate] = 1.0
                else:
        #check to see if a trajectory is entering a spawning region
                    print "consider4 ",tdc[jstate], thresh
                    if (abs(tdc[jstate]) > thresh) and (z_dont_spawn[jstate] < 0.5):
                        print "Entered spawning region ", jstate, " at time ", self.get_time()
                        spawnt[jstate] = self.get_time()
                    else:
                        if (abs(tdc[jstate]) < (0.9*thresh)) and (z_dont_spawn[jstate] > 0.5):
                            z_dont_spawn[jstate] = 0.0
                        
                        
        self.set_z_spawn_now(z)
        self.set_z_dont_spawn(z_dont_spawn)
        self.set_spawnlastcoup(tdc)
        self.set_spawntimes(spawnt)
                        
            
            
    def h5_output(self, zbackprop):
        if not zbackprop:
            cbackprop = ""
        else:
            cbackprop = "backprop_"
        if "_&_" not in self.get_label():
            traj_or_cent = "traj_"
        else:
            traj_or_cent = "cent_"
        if len(self.h5_datasets) == 0:
            init = "self.init_h5_datasets_" + self.get_software() + "_" + self.get_method() + "()"
            eval(init)
        extensions = [3,2,1,0]
        for i in extensions :
            if i==0:
                ext = ""
            else:
                ext = str(i) + "."
            filename = "sim." + ext + "hdf5"
            if os.path.isfile(filename):
                if (i == extensions[0]):
                    os.remove(filename)
                else:
                    ext = str(i+1) + "."
                    filename2 = "sim." + ext + "hdf5"
                    if (i == extensions[-1]):
                        shutil.copy2(filename, filename2)
                    else:
                        shutil.move(filename, filename2)
        h5f = h5py.File(filename, "a")
        groupname = traj_or_cent + self.label
        if groupname not in h5f.keys():
            self.create_h5_traj(h5f,groupname)
        trajgrp = h5f.get(groupname)
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            print "key", key
            dset = trajgrp.get(key)
            l = dset.len()
            dset.resize(l+1,axis=0)
            if not zbackprop:
                ipos=l
            else:
                ipos=0
                dset[1:(l+1),0:n] = dset[0:(l),0:n]
            getcom = "self.get_" + cbackprop + key + "()"
            print getcom
            tmp = eval(getcom)
            if n!=1:
                dset[ipos,0:n] = tmp[0:n]
            else:
                dset[ipos,0] = tmp
        h5f.close()
        
    def create_h5_traj(self, h5f, groupname):
        trajgrp = h5f.create_group(groupname)
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            dset = trajgrp.create_dataset(key, (0,n), maxshape=(None,n))
            
    def get_data_at_time_from_h5(self,t,dset_name):
        h5f = h5py.File("sim.hdf5", "r")
        if "_&_" not in self.get_label():
            traj_or_cent = "traj_"
        else:
            traj_or_cent = "cent_"
        groupname = traj_or_cent + self.label
        filename = "sim.hdf5"
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time"][:]
        print "size", dset_time.size
        ipoint = -1
        for i in range(len(dset_time)):
            if (dset_time[i] < t+1.0e-6) and (dset_time[i] > t-1.0e-6):
                ipoint = i
                print "dset_time[i] ", dset_time[i]
                print "i ", i
        dset = trajgrp[dset_name][:]            
        data = np.zeros(len(dset[ipoint,:]))
        data = dset[ipoint,:]
        print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()
        return data

    def get_all_qm_data_at_time_from_h5(self,t):
        h5f = h5py.File("sim.hdf5", "r")
        if "_&_" not in self.get_label():
            traj_or_cent = "traj_"
        else:
            traj_or_cent = "cent_"
        groupname = traj_or_cent + self.label
        filename = "sim.hdf5"
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time"][:]
        print "size", dset_time.size
        ipoint = -1
        for i in range(len(dset_time)):
            if (dset_time[i] < t+1.0e-6) and (dset_time[i] > t-1.0e-6):
                ipoint = i
                print "dset_time[i] ", dset_time[i]
                print "i ", i
        for dset_name in self.h5_datasets:
            dset = trajgrp[dset_name][:]            
            data = np.zeros(len(dset[ipoint,:]))
            data = dset[ipoint,:]
            comm = "self." + dset_name + "_qm = data"
            exec(comm)
            print "comm ", comm
            print "dset[ipoint,:] ", dset[ipoint,:]        
        h5f.close()
            
    #def get_e_at_time_from_h5(self,t):
    #    h5f = h5py.File("sim.hdf5", "r")
    #    if "_&_" not in self.get_label():
    #        traj_or_cent = "traj_"
    #    else:
    #        traj_or_cent = "cent_"
    #    groupname = traj_or_cent + self.label
    #    filename = "sim.hdf5"
    #    trajgrp = h5f.get(groupname)

    #dset_time = trajgrp["time"][:]
    #    print "size", dset_time.size
    #    dset_e = trajgrp["energies"][:]
    #    ipoint = -1
    #    for i in range(len(dset_time)):
    #        if (dset_time[i] < t+1.0e-6) and (dset_time[i] > t-1.0e-6):
    #            ipoint = i
    #            print "dset_time[i] ", dset_time[i]
    #            print "i ", i
    #
    #   e = np.zeros(self.get_numstates())
    #    e = dset_e[ipoint,:]
    #    print "dset_e[ipoint,:] ", dset_e[ipoint,:]
        
    #    h5f.close()
    #    return e

    #def load_nac_data_from_h5(self, qm_time):
    #    routine = "load_nac_data_from_h5_" + self.get_software() + "_" + self.get_method() + "(qm_time)"
    #    exec(routine)
    
    def compute_tdc(self,W):
        Atmp = np.arccos(W[0,0]) - np.arcsin(W[0,1])
        Btmp = np.arccos(W[0,0]) + np.arcsin(W[0,1])
        Ctmp = np.arccos(W[1,1]) - np.arcsin(W[1,0])
        Dtmp = np.arccos(W[1,1]) + np.arcsin(W[1,0])
        if Atmp < 1.0e-6:
            A = -1.0
        else:
            A = -1.0 * np.sin(Atmp) / Atmp
        if Btmp < 1.0e-6:
            B = 1.0
        else:
            B = np.sin(Btmp) / Btmp
        if Ctmp < 1.0e-6:
            C = 1.0
        else:
            C = np.sin(Ctmp) / Ctmp
        if Dtmp < 1.0e-6:
            D = 1.0
        else:
            D = np.sin(Dtmp) / Dtmp
        h = self.get_timestep()
        tdc = 0.5 / h * (np.arccos(W[0,0])*(A+B) + np.arcsin(W[1,0])*(C+D))
        return tdc

#################################################
### electronic structure routines go here #######
#################################################

#each electronic structure method requires at least two routines:
#1) compute_elec_struct_, which computes energies, forces, and wfs
#2) init_h5_datasets_, which defines the datasets to be output to hdf5
#other ancillary routines may be included as well

### pyspawn_cone electronic structure ###
    def compute_elec_struct_pyspawn_cone(self,zbackprop):
        if not zbackprop:
            cbackprop = ""
        else:
            cbackprop = "backprop_"

        
        #self.prev_energies = self.energies.copy()
        #self.prev_forces = self.forces.copy()
        exec("self.set_" + cbackprop + "prev_wf(self.get_" + cbackprop + "wf())")
        #self.prev_wf = self.wf.copy()
        #self.prev_positions = self.last_es_positions.copy()
        #self.last_es_positions = self.positions.copy()

        exec("x = self.get_" + cbackprop + "positions()[0]")
        exec("y = self.get_" + cbackprop + "positions()[1]")
        r = math.sqrt( x * x + y * y )
        theta = (math.atan2(y,x)) / 2.0

        e = np.zeros(self.numstates)
        e[0] = ( r - 1.0 ) * ( r - 1.0 ) - 1.0
        e[1] = ( r + 1.0 ) * ( r + 1.0 ) - 1.0
        exec("self.set_" + cbackprop + "energies(e)")

        f = np.zeros((self.numstates,self.numdims))
        ftmp = -2.0 * ( r - 1.0 )
        f[0,0] = ( x / r ) * ftmp
        f[0,1] = ( y / r ) * ftmp
        ftmp = -2.0 * ( r + 1.0 )
        f[1,0] = ( x / r ) * ftmp
        f[1,1] = ( y / r ) * ftmp
        exec("self.set_" + cbackprop + "forces(f)")

        wf = np.zeros((self.numstates,self.length_wf))
        wf[0,0] = math.sin(theta)
        wf[0,1] = math.cos(theta)
        wf[1,0] = math.cos(theta)
        wf[1,1] = -math.sin(theta)
        exec("prev_wf = self.get_" + cbackprop + "prev_wf()")
        # phasing wave funciton to match previous time step
        W = np.matmul(prev_wf,wf.T)
        if W[0,0] < 0.0:
            wf[0,:] = -1.0*wf[0,:]
            W[:,0] = -1.0 * W[:,0]
        if W[1,1] < 0.0:
            wf[1,:] = -1.0*wf[1,:]
            W[:,1] = -1.0 * W[:,1]
        # computing NPI derivative coupling
        tmp=self.compute_tdc(W)
        tdc = np.zeros(self.numstates)
        if self.istate == 1:
            jstate = 0
        else:
            jstate = 1
        tdc[jstate] = tmp
        self.set_timederivcoups(tdc)
        
        exec("self.set_" + cbackprop + "wf(wf)")

    def init_h5_datasets_pyspawn_cone(self):
        self.h5_datasets["time"] = 1
        self.h5_datasets["energies"] = self.numstates
        self.h5_datasets["positions"] = self.numdims
        self.h5_datasets["momenta"] = self.numdims
        self.h5_datasets["wf0"] = self.numstates
        self.h5_datasets["wf1"] = self.numstates
        self.h5_datasets["timederivcoups"] = self.numstates

    def get_wf0(self):
        return self.wf[0,:].copy()

    def get_wf1(self):
        return self.wf[1,:].copy()

    def get_backprop_wf0(self):
        return self.backprop_wf[0,:].copy()

    def get_backprop_wf1(self):
        return self.backprop_wf[1,:].copy()

    def get_backprop_timederivcoups(self):
        return np.zeros(self.numstates)

###end pyspawn_cone electronic structure section###

#################################################
### integrators go here #########################
#################################################

#each integrator requires at least two routines:
#1) prop_first_, which propagates the first step
#2) prop_, which propagates all other steps
#other ancillary routines may be included as well

### velocity Verlet (vv) integrator section ###
    def prop_first_vv(self,zbackprop):
        if not zbackprop:
            cbackprop = ""
            dt = self.get_timestep()
        else:
            cbackprop = "backprop_"
            dt = -1.0 * self.get_timestep()
            
        exec("x_t = self.get_" + cbackprop + "positions()")
        self.compute_elec_struct(zbackprop)
        exec("f_t = self.get_" + cbackprop + "forces()[self.istate,:]")
        exec("p_t = self.get_" + cbackprop + "momenta()")
        exec("e_t = self.get_" + cbackprop + "energies()")
        m = self.get_masses()
        v_t = p_t / m
        a_t = f_t / m
        exec("t = self.get_" + cbackprop + "time()")
        
        print "t ", t
        print "x_t ", x_t
        print "f_t ", f_t
        print "v_t ", v_t
        print "a_t ", a_t

        if not zbackprop:
            self.h5_output(zbackprop)
        
        v_tphdt = v_t + 0.5 * a_t * dt
        x_tpdt = x_t + v_tphdt * dt
        
        exec("self.set_" + cbackprop + "positions(x_tpdt)")

        self.compute_elec_struct(zbackprop)
        exec("f_tpdt = self.get_" + cbackprop + "forces()[self.istate,:]")
        exec("e_tpdt = self.get_" + cbackprop + "energies()")
        
        a_tpdt = f_tpdt / m
        v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
        p_tpdt = v_tpdt * m
        
        exec("self.set_" + cbackprop + "momenta(p_tpdt)")

        t += dt

        exec("self.set_" + cbackprop + "time(t)")

        print "t ", t
        print "x_t ", x_tpdt
        print "f_t ", f_tpdt
        print "v_t ", v_tpdt
        print "a_t ", a_tpdt

        self.h5_output(zbackprop)
     
        v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt
        p_tp3hdt = v_tp3hdt * m

        exec("self.set_" + cbackprop + "momenta(p_tp3hdt)")

        x_tp2dt = x_tpdt + v_tp3hdt * dt

        if not zbackprop:
            self.set_positions_t(x_t)
            self.set_positions_tpdt(x_tpdt)
            self.set_momenta_t(p_t)
            self.set_momenta_tpdt(p_tpdt)
            self.set_energies_t(e_t)
            self.set_energies_tpdt(e_tpdt)
            
        exec("self.set_" + cbackprop + "positions(x_tp2dt)")
        
    def prop_vv(self,zbackprop):
        if not zbackprop:
            cbackprop = ""
            dt = self.get_timestep()
        else:
            cbackprop = "backprop_"
            dt = -1.0 * self.get_timestep()

        exec("x_tpdt = self.get_" + cbackprop + "positions()")
        self.compute_elec_struct(zbackprop)
        exec("f_tpdt = self.get_" + cbackprop + "forces()[self.istate,:]")
        exec("e_tpdt = self.get_" + cbackprop + "energies()")

        exec("p_tphdt = self.get_" + cbackprop + "momenta()")
        m = self.get_masses()
        v_tphdt = p_tphdt / m
        a_tpdt = f_tpdt / m
        exec("t = self.get_" + cbackprop + "time()")

        v_tpdt = v_tphdt + 0.5 * a_tpdt * dt

        p_tpdt = v_tpdt * m

        if not zbackprop:
            self.set_positions_tmdt(self.get_positions_t())
            self.set_positions_t(self.get_positions_tpdt())
            self.set_positions_tpdt(x_tpdt)
            self.set_energies_tmdt(self.get_energies_t())
            self.set_energies_t(self.get_energies_tpdt())
            self.set_energies_tpdt(e_tpdt)
            self.set_momenta_tmdt(self.get_momenta_t())
            self.set_momenta_t(self.get_momenta_tpdt())
            self.set_momenta_tpdt(p_tpdt)
        exec("self.set_" + cbackprop + "momenta(p_tpdt)")
        
        t += dt

        exec("self.set_" + cbackprop + "time(t)")

        print "t ", t
        print "x_t ", x_tpdt
        print "f_t ", f_tpdt
        print "v_t ", v_tpdt
        print "a_t ", a_tpdt

        self.h5_output(zbackprop)

        v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt

        p_tp3hdt = v_tp3hdt * m

        exec("self.set_" + cbackprop + "momenta(p_tp3hdt)")

        x_tp2dt = x_tpdt + v_tp3hdt * dt

        exec("self.set_" + cbackprop + "positions(x_tp2dt)")
### end velocity Verlet (vv) integrator section ###
