# trajectory objects contain individual trajectory basis functions
import numpy as np
import sys
import math
from fms.fmsobj import fmsobj
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
        self.ilabel = 0
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
        #self.last_es_positions = np.zeros(self.numdims)
        #self.prev_positions = np.zeros(self.numdims)
        self.energies = np.zeros(self.numstates)
        #self.prev_energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates,self.numdims))
        #self.prev_forces = np.zeros((self.numstates,self.numdims))

        self.backprop_time = 0.0
        self.backprop_energies = np.zeros(self.numstates)
        self.backprop_forces = np.zeros((self.numstates,self.numdims))
        self.backprop_positions = np.zeros(self.numdims)
        self.backprop_momenta = np.zeros(self.numdims)
        self.backprop_wf = np.zeros((self.numstates,self.length_wf))
        self.backprop_prev_wf = np.zeros((self.numstates,self.length_wf))
        
    def set_time(self,t):
        self.time = t
    
    def get_time(self):
        return self.time
    
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
    
    def set_propagator(self,prop):
        self.propagator = prop
    
    def get_mintime(self):
        return self.mintime
    
    def set_mintime(self,t):
        self.time = t
    
    def set_numdims(self,ndims):
        self.numdims = ndims
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.widths = np.zeros(self.numdims)
        self.masses = np.zeros(self.numdims)
        self.last_es_positions = np.zeros(self.numdims)
        self.forces = np.zeros((self.numstates,self.numdims))
        #self.prev_positions = np.zeros(self.numdims)
        #self.prev_forces = np.zeros((self.numstates,self.numdims))
        
    def set_numstates(self,nstates):
        self.numstates = nstates
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates,self.numdims))
        #self.prev_energies = np.zeros(self.numstates)
        #self.prev_forces = np.zeros((self.numstates,self.numdims))

    def set_istate(self,ist):
        self.istate = ist

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
            
    def set_momenta(self,mom):
        if mom.shape == self.momenta.shape:
            self.momenta = mom.copy()
        else:
            print "Error in set_momenta"
            sys.exit

    def get_momenta(self):
        return self.momenta.copy()
            
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

    def set_timestep(self,h):
        self.timestep = h

    def set_ilabel(self,i):
        self.ilabel = i

    def get_ilabel(self):
        return self.ilabel
    
    def set_label(self,lab):
        self.label = lab

    def get_label(self):
        return self.label

    def init_traj(self,t,ndims,pos,mom,wid,m,nstates,istat,ilabel,lab):
        self.set_time(t)
        self.set_numdims(ndims)
        self.set_positions(pos)
        self.set_momenta(mom)
        self.set_widths(wid)
        self.set_masses(m)
        self.set_ilabel(ilabel)
        self.set_label(lab)
        self.set_numstates(nstates)
        self.set_istate(istat)

        self.set_backprop_time(t)
        self.set_backprop_positions(pos)
        self.set_backprop_momenta(mom)

        self.set_firsttime(t)

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
        
    def h5_output(self, zbackprop):
        if not zbackprop:
            cbackprop = ""
        else:
            cbackprop = "backprop_"
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
        groupname = "traj_" + self.label
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
        dot0 = wf[0,0] * prev_wf[0,0] + wf[0,1] * prev_wf[0,1]
        dot1 = wf[1,0] * prev_wf[1,0] + wf[1,1] * prev_wf[1,1]
        if dot0 < 0.0:
            self.wf[0,:] = -1.0*self.wf[0,:]
            dot0 = -1.0 * dot0
        if dot1 < 0.0:
            self.wf[1,:] = -1.0*self.wf[1,:]
            dot1 = -1.0 * dot1
        exec("self.set_" + cbackprop + "wf(wf)")

    def init_h5_datasets_pyspawn_cone(self):
        self.h5_datasets["time"] = 1
        self.h5_datasets["energies"] = self.numstates
        self.h5_datasets["positions"] = self.numdims
        self.h5_datasets["momenta"] = self.numdims
        self.h5_datasets["wf0"] = self.numstates
        self.h5_datasets["wf1"] = self.numstates

    def get_wf0(self):
        return self.wf[0,:].copy()

    def get_wf1(self):
        return self.wf[1,:].copy()

    def get_backprop_wf0(self):
        return self.backprop_wf[0,:].copy()

    def get_backprop_wf1(self):
        return self.backprop_wf[1,:].copy()

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
            dt = self.timestep
        else:
            cbackprop = "backprop_"
            dt = -1.0 * self.timestep
            
        exec("x_t = self.get_" + cbackprop + "positions()")
        self.compute_elec_struct(zbackprop)
        exec("f_t = self.get_" + cbackprop + "forces()[self.istate,:]")
        exec("p_t = self.get_" + cbackprop + "momenta()")
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

        exec("self.set_" + cbackprop + "positions(x_tp2dt)")
        
    def prop_vv(self,zbackprop):
        if not zbackprop:
            cbackprop = ""
            dt = self.timestep
        else:
            cbackprop = "backprop_"
            dt = -1.0 * self.timestep
            
        exec("x_tpdt = self.get_" + cbackprop + "positions()")
        self.compute_elec_struct(zbackprop)
        exec("f_tpdt = self.get_" + cbackprop + "forces()[self.istate,:]")
        exec("p_tphdt = self.get_" + cbackprop + "momenta()")
        m = self.get_masses()
        v_tphdt = p_tphdt / m
        a_tpdt = f_tpdt / m
        exec("t = self.get_" + cbackprop + "time()")

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

        exec("self.set_" + cbackprop + "positions(x_tp2dt)")
### end velocity Verlet (vv) integrator section ###
        
            
                                
