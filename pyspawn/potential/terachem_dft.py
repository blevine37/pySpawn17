import math
import numpy as np
try:
    from tcpb.tcpb import TCProtobufClient
except ImportError:
    pass
import os
import errno

#################################################
### electronic structure routines go here #######
#################################################

#each electronic structure method requires at least two routines:
#1) compute_elec_struct_, which computes energies, forces, and wfs
#2) init_h5_datasets_, which defines the datasets to be output to hdf5
#3) potential_specific_traj_copy, which copies data that is potential specific 
#   from one traj data structure to another.  This is used when new trajectories
#   and centroids are spawned.
#other ancillary routines may be included as well

### terachem_cas electronic structure ###
def compute_elec_struct(self,zbackprop):
    if not zbackprop:
        cbackprop = ""
    else:
        cbackprop = "backprop_"    

    istate = self.get_istate()
    nstates = self.get_numstates()

    # initialize electronic_phases if not present
    if not hasattr(self,'electronic_phases'):
        self.electronic_phases = np.ones(nstates)
    if not hasattr(self,'backprop_electronic_phases'):
        self.backprop_electronic_phases = np.ones(nstates)

    exec("pos = self.get_" + cbackprop + "positions()")
    pos_list = pos.tolist()
        
    TC = TCProtobufClient(host='localhost', port=54321)

    options = self.get_tc_options()

#    options["castarget"] = istate

#    TC.update_options(**base_options)

    TC.connect()

    # Check if the server is available
    avail = TC.is_available()
    #print "TCPB Server available: {}".format(avail)

    # Write CI vectors and orbitals for initial guess and overlaps
    cwd = os.getcwd()
    # Gradient calculation

    # here we call TC once for energies and once for the gradient
    # will eventually be replaced by a more efficient interface
    #options = {}
    results = TC.compute_job_sync("energy", pos_list, "bohr", **options)
    #print results

    e = np.zeros(nstates)
    e[0] = results['energy']

    results = TC.compute_job_sync("gradient", pos_list, "bohr", **options)
    #print results

    f = np.zeros((nstates,self.numdims))
    #print "results['gradient'] ", results['gradient']
    #print "results['gradient'].flatten() ", results['gradient'].flatten()
    f[self.istate,:] = -1.0 * results['gradient'].flatten()

    exec("self.set_" + cbackprop + "energies(e)")

    exec("self.set_" + cbackprop + "forces(f)")

def init_h5_datasets(self):
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["forces_i"] = self.numdims
    self.h5_datasets_half_step["time_half_step"] = 1

def potential_specific_traj_copy(self,from_traj):
    self.set_tc_options(from_traj.get_tc_options())
    return

def set_tc_options(self, tco):
    self.tc_options = tco.copy()

def get_tc_options(self):
    return self.tc_options.copy()

def get_prev_wf_positions(self):
    return self.prev_wf_positions.copy()
            
def get_backprop_prev_wf_positions(self):
    return self.backprop_prev_wf_positions.copy()
            
def get_prev_wf_positions_in_angstrom(self):
    return 0.529177*self.prev_wf_positions
            
def get_backprop_prev_wf_positions_in_angstrom(self):
    return 0.529177*self.backprop_prev_wf_positions
            
def set_prev_wf_positions(self,pos):
    self.prev_wf_positions = pos.copy()

def set_backprop_prev_wf_positions(self,pos):
    self.backprop_prev_wf_positions = pos.copy()

def get_electronic_phases(self):
    return self.electronic_phases.copy()

def set_electronic_phases(self, v):
    self.electronic_phases = v.copy()

def get_backprop_electronic_phases(self):
    return self.backprop_electronic_phases.copy()

def set_backprop_electronic_phases(self, v):
    self.backprop_electronic_phases = v.copy()




###end terachem_cas electronic structure section###
