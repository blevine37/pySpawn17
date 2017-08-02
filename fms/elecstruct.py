import numpy as np
import math
from fms.fmsobj import fmsobj

class elecstruct(fmsobj):
    def __init__(self):
        self.numstates = 2
        self.numdims = 2
        self.software = "pyspawn"
        self.method = "cone"
        self.length_wf = self.numstates
        self.wf = np.zeros((self.numstates,self.length_wf))
        self.prev_wf = np.zeros((self.numstates,self.length_wf))
        self.last_positions = np.zeros(self.numdims)
        self.prev_positions = np.zeros(self.numdims)
        self.energies = np.zeros(self.numstates)
        self.prev_energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates,self.numdims))
        self.prev_forces = np.zeros((self.numstates,self.numdims))

    def set_numstates(self,nstates):
        self.numstates = numstates
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates,self.numdims))

    def set_numdims(self,ndims):
        self.numdims = ndims
        self.last_positions = np.zeros(self.numdims)
        self.forces = np.zeros((self.numstates,self.numdims))

    def set_software(self,sw):
        self.software = sw

    def get_software(self):
        return self.software

    def get_method(self):
        return self.method

    def get_forces(self):
        return self.forces

    def get_energies(self):
        return self.energies

    def get_wf(self):
        return self.wf

    def get_prev_forces(self):
        return self.forces

    def get_prev_energies(self):
        return self.energies

    def get_prev_wf(self):
        return self.prev_wf

    def compute_elec_struct_pyspawn_cone(self):
        self.prev_energies = self.energies.copy()
        self.prev_forces = self.forces.copy()
        self.prev_wf = self.wf.copy()

        x = self.positions[0]
        y = self.positions[1]
        r = math.sqrt( x * x + y * y )
        theta = (math.atan( y / x )) / 2.0
        
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
        self.wf[1,0] = -math.cos(theta)
        self.wf[1,1] = math.sin(theta)
        dot0 = self.wf[0,0] * self.prev_wf[0,0] + self.wf[0,1] * self.prev_wf[0,1]
        dot1 = self.wf[1,0] * self.prev_wf[1,0] + self.wf[1,1] * self.prev_wf[1,1]
        if dot0 < 0.0:
            self.wf[0,0:2] = -self.wf[0:2,0]
#            self.wf[0,1] = -self.wf[0,1]
        if dot1 < 0.0:
            self.wf[1,0] = -self.wf[1,0]
            self.wf[1,1] = -self.wf[1,1]
        
    
    

    
        
        
        
