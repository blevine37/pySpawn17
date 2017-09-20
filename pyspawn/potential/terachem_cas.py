import math
import numpy as np
from tcpb.tcpb import TCProtobufClient

#################################################
### electronic structure routines go here #######
#################################################

#each electronic structure method requires at least two routines:
#1) compute_elec_struct_, which computes energies, forces, and wfs
#2) init_h5_datasets_, which defines the datasets to be output to hdf5
#other ancillary routines may be included as well

### terachem_cas electronic structure ###
def compute_elec_struct(self,zbackprop):
    if not zbackprop:
        cbackprop = ""
    else:
        cbackprop = "backprop_"

    istate = self.get_istate()

    exec("pos = self.get_" + cbackprop + "positions()")
    pos_list = pos.tolist()
        
    exec("self.set_" + cbackprop + "prev_wf(np.identity(2))")
    
    atoms = ['C', 'C', 'H', 'H', 'H', 'H']    

    TC = TCProtobufClient(host='localhost', port=54321)

    base_options = self.get_tc_options()

    TC.update_options(**base_options)

    TC.connect()

    # Check if the server is available
    avail = TC.is_available()
    print "TCPB Server available: {}".format(avail)

    # Gradient calculation
    options = {
        "directci":     "yes",
        "caswritevecs": "yes"
    }
    results = TC.compute_job_sync("gradient", pos_list, "bohr", **options)
    print results

    e = np.zeros(self.numstates)
    e[self.istate] = results['energy']
    f = np.zeros((self.numstates,self.numdims))
    print "results['gradient'] ", results['gradient']
    print "results['gradient'].flatten() ", results['gradient'].flatten()
    f[self.istate,:] = -1.0 * results['gradient'].flatten()

    exec("self.set_" + cbackprop + "energies(e)")

    exec("self.set_" + cbackprop + "forces(f)")
        
    wf = np.identity(2)
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
    exec("self.set_" + cbackprop + "timederivcoups(tdc)")
        
    exec("self.set_" + cbackprop + "wf(wf)")

def init_h5_datasets(self):
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["forces_i"] = self.numdims
    self.h5_datasets["wf0"] = 2
    self.h5_datasets["wf1"] = 2
    self.h5_datasets_half_step["time_half_step"] = 1
    self.h5_datasets_half_step["timederivcoups"] = self.numstates

def get_wf0(self):
    return self.wf[0,:].copy()

def get_wf1(self):
    return self.wf[1,:].copy()

def get_backprop_wf0(self):
    return self.backprop_wf[0,:].copy()

def get_backprop_wf1(self):
    return self.backprop_wf[1,:].copy()

def set_tc_options(self, tco):
    self.tc_options = tco.copy()

def get_tc_options(self):
    return self.tc_options.copy()

###end terachem_cas electronic structure section###
