# trajectory objects contain individual trajectory basis functions
import numpy as np
import sys
import math
from fms.fmsobj import fmsobj

class traj(fmsobj):
    def __init__(self):
        self.time = 0.0
        self.numdims = 2
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.widths = np.zeros(self.numdims)
        self.masses = np.zeros(self.numdims)
        self.istate = 0
        
        self.timestep = 0.0
        self.propagator = "vv"
        
        self.numstates = 2
        self.software = "pyspawn"
        self.method = "cone"
        self.length_wf = self.numstates
        self.wf = np.zeros((self.numstates,self.length_wf))
        self.prev_wf = np.zeros((self.numstates,self.length_wf))
        self.last_es_positions = np.zeros(self.numdims)
        self.prev_positions = np.zeros(self.numdims)
        self.output_positions = np.zeros(self.numdims)
        self.output_momenta = np.zeros(self.numdims)
        self.energies = np.zeros(self.numstates)
        self.prev_energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates,self.numdims))
        self.prev_forces = np.zeros((self.numstates,self.numdims))


    def set_time(self,t):
        self.time = t
    
    def set_numdims(self,ndims):
        self.numdims = ndims
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.widths = np.zeros(self.numdims)
        self.masses = np.zeros(self.numdims)
        self.last_es_positions = np.zeros(self.numdims)
        self.forces = np.zeros((self.numstates,self.numdims))
        self.prev_positions = np.zeros(self.numdims)
        self.output_positions = np.zeros(self.numdims)
        self.output_momenta = np.zeros(self.numdims)
        self.prev_forces = np.zeros((self.numstates,self.numdims))
        
    def set_numstates(self,nstates):
        self.numstates = numstates
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates,self.numdims))
        self.prev_energies = np.zeros(self.numstates)
        self.prev_forces = np.zeros((self.numstates,self.numdims))

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
            self.positions = pos
        else:
            print "Error in set_positions"
            sys.exit

    def set_momenta(self,mom):
        if mom.shape == self.momenta.shape:
            self.momenta = mom
        else:
            print "Error in set_momenta"
            sys.exit

    def set_widths(self,wid):
        if wid.shape == self.widths.shape:
            self.widths = wid
        else:
            print "Error in set_widths"
            sys.exit

    def set_masses(self,m):
        if m.shape == self.masses.shape:
            self.masses = m
        else:
            print "Error in set_masses"
            sys.exit

    def set_timestep(self,h):
        self.timestep = h

    def init_traj(self,t,ndims,pos,mom,wid,m):
        self.set_time(t)
        self.set_numdims(ndims)
        self.set_positions(pos)
        self.set_momenta(mom)
        self.set_widths(wid)
        self.set_masses(m)

    def get_forces(self):
        self.compute_elec_struct()
        return self.forces

    def get_energies(self):
        self.compute_elec_struct()
        return self.energies

    def get_wf(self):
        self.compute_elec_struct()
        return self.wf

    def get_prev_forces(self):
        self.compute_elec_struct()
        return self.prev_forces

    def get_prev_energies(self):
        self.compute_elec_struct()
        return self.prev_energies

    def get_prev_wf(self):
        self.compute_elec_struct()
        return self.prev_wf

    def compute_elec_struct(self):
        tmp = "self.compute_elec_struct_" + self.get_software() + "_" + self.get_method() + "()"
        eval(tmp)

    def compute_elec_struct_pyspawn_cone(self):
        self.prev_energies = self.energies.copy()
        self.prev_forces = self.forces.copy()
        self.prev_wf = self.wf.copy()
        self.prev_positions = self.last_es_positions.copy()
        self.last_es_positions = self.positions.copy()

        x = self.positions[0]
        y = self.positions[1]
        r = math.sqrt( x * x + y * y )
        theta = (math.atan2(y,x)) / 2.0
        
        self.energies[0] = ( r - 1.0 ) * ( r - 1.0 ) - 1.0
        self.energies[1] = ( r + 1.0 ) * ( r + 1.0 ) - 1.0
        
        ftmp = -2.0 * ( r - 1.0 )
        self.forces[0,0] = ( x / r ) * ftmp
        self.forces[0,1] = ( y / r ) * ftmp
        ftmp = -2.0 * ( r + 1.0 )
        self.forces[1,0] = ( x / r ) * ftmp
        self.forces[1,1] = ( y / r ) * ftmp
        
        self.wf[0,0] = math.sin(theta)
        self.wf[0,1] = math.cos(theta)
        self.wf[1,0] = math.cos(theta)
        self.wf[1,1] = -math.sin(theta)
        dot0 = self.wf[0,0] * self.prev_wf[0,0] + self.wf[0,1] * self.prev_wf[0,1]
        dot1 = self.wf[1,0] * self.prev_wf[1,0] + self.wf[1,1] * self.prev_wf[1,1]
        if dot0 < 0.0:
            self.wf[0,:] = -self.wf[0,:]
        if dot1 < 0.0:
            self.wf[1,:] = -self.wf[1,:]

    def propagate(self):
        print self.time
        if self.time == 0.0 :
            tmp = "self.prop_" + self.propagator + "_first()"
            eval(tmp)
        else:
            tmp = "self.prop_" + self.propagator + "()"
            eval(tmp)
        
    def prop_vv_first(self):
        x_t = self.positions
        f_t = self.get_forces()[self.istate,:]
        v_t = self.momenta / self.masses
        a_t = f_t / self.masses
        dt = self.timestep

        print "x_t ", x_t
        print "f_t ", f_t
        print "v_t ", v_t
        print "a_t ", a_t
     
        v_tphdt = v_t + 0.5 * a_t * dt
        x_tpdt = x_t + v_tphdt * dt

        self.positions = x_tpdt.copy()
        self.output_positions = x_tpdt.copy()

        f_tpdt = self.get_forces()[self.istate,:]
        
        a_tpdt = f_tpdt / self.masses
        v_tpdt = v_tphdt + 0.5 * a_tpdt * dt

        self.output_momenta = v_tpdt * self.masses

        v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt

        self.momenta = v_tp3hdt * self.masses

        x_tp2dt = x_tpdt + v_tp3hdt * dt

        self.positions = x_tp2dt.copy()

        self.timestep += dt
        
    def prop_vv(self):
        x_tpdt = self.positions.copy()
        v_tphdt = self.momenta / self.masses
        f_tpdt = self.get_forces()[self.istate,:]
        a_tpdt = f_tpdt / self.masses
        dt = self.timestep

        self.output_positions = x_tpdt.copy()

        v_tpdt = v_tphdt + 0.5 * a_tpdt * dt

        self.output_momenta = v_tpdt * self.masses

        v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt

        self.momenta = v_tp3hdt * self.masses

        x_tp2dt = x_tpdt + v_tp3hdt * dt

        self.positions = x_tp2dt.copy()
        
        self.timestep += dt
    
            
                                
