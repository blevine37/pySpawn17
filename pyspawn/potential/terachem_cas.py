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

    base_options = self.get_tc_options()

    base_options["castarget"] = istate

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

    # here we call TC once for energies and once for the gradient
    # will eventually be replaced by a more efficient interface
    results = TC.compute_job_sync("energy", pos_list, "bohr", **options)
    print results

    e = np.zeros(nstates)
    e[:] = results['energy'][:]

    results = TC.compute_job_sync("gradient", pos_list, "bohr", **options)
    print results

    if hasattr(self,'civecs'):
        cwd = os.getcwd()
        civecout = os.path.join(cwd,"CIvecs.Singlet.dat")
        orbout = os.path.join(cwd,"c0")
        self.civecs.tofile(civecout)
        self.orbs.tofile(orbout)
        print "old civecs",  self.civecs
        print "old orbs", self.orbs
        zolaps = True
    else:
        zolaps = False


    civecfilename = os.path.join(results['job_scr_dir'], "CIvecs.Singlet.dat")
    self.civecs = np.fromfile(civecfilename)
    print "new civecs", self.civecs    

    orbfilename = os.path.join(results['job_scr_dir'], "c0")
    self.orbs = (np.fromfile(orbfilename)).flatten()
    #self.orbs = results['mo_coeffs'].flatten()
    print "new orbs", self.orbs

    self.ncivecs = self.civecs.size
    self.norbs = self.orbs.size

    f = np.zeros((nstates,self.numdims))
    print "results['gradient'] ", results['gradient']
    print "results['gradient'].flatten() ", results['gradient'].flatten()
    f[self.istate,:] = -1.0 * results['gradient'].flatten()

    exec("self.set_" + cbackprop + "energies(e)")

    exec("self.set_" + cbackprop + "forces(f)")

    #if False:        
    if zolaps:
        exec("pos2 = self.get_" + cbackprop + "prev_wf_positions_in_angstrom()")
        print 'pos2.tolist()', pos2.tolist()
        print 'civecfilename', civecfilename
        print 'civecout', civecout
        print 'orbfilename', orbfilename
        print 'orbout', orbout
        options = {
            "geom2":        pos2.tolist(),
            "cvec1file":    civecfilename,
            "cvec2file":    civecout,
            "orb1afile":    orbfilename,
            "orb2afile":    orbout
            }

        print 'pos_list', pos_list
        results2 = TC.compute_job_sync("ci_vec_overlap", pos_list, "bohr", **options)
        print "results2", results2
        S = results2['ci_overlap']
        print "S before phasing ", S

        # phasing electronic overlaps 
        for jstate in range(nstates):
            S[:,jstate] *= self.electronic_phases[jstate]
            S[jstate,:] *= self.electronic_phases[jstate]

        for jstate in range(nstates):
            if S[jstate,jstate] < 0.0:
                self.electronic_phases[jstate] *= -1.0
                # I'm not sure if this line is right, but it seems to be working
                S[jstate,:] *= -1.0
                
        print "S", S

        W = np.zeros((2,2))
        W[0,0] = S[istate,istate]

        tdc = np.zeros(nstates)

        for jstate in range(nstates):
            if istate == jstate:
                tdc[jstate] = 0.0
            else:
                W[1,0] = S[jstate,istate]
                W[0,1] = S[istate,jstate]
                W[1,1] = S[jstate,jstate]
                tdc[jstate] = self.compute_tdc(W)
                print "tdc", tdc[jstate]

        #tmp=self.compute_tdc(W)
        #tdc = np.zeros(self.numstates)
        #if self.istate == 1:
        #    jstate = 0
        #else:
        #    jstate = 1
        #    tdc[jstate] = tmp

        exec("self.set_" + cbackprop + "timederivcoups(tdc)")    
    
    exec("self.set_" + cbackprop + "prev_wf_positions(pos)")
    exec("self.set_" + cbackprop + "timederivcoups(np.zeros(self.numstates))")

def init_h5_datasets(self):
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["forces_i"] = self.numdims
    self.h5_datasets["civecs"] = self.ncivecs
    self.h5_datasets["orbs"] = self.norbs
    self.h5_datasets_half_step["time_half_step"] = 1
    self.h5_datasets_half_step["timederivcoups"] = self.numstates

def potential_specific_traj_copy(self,from_traj):
    self.set_tc_options(from_traj.get_tc_options())
    return

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

def get_prev_wf_positions(self):
    return self.pref_wf_positions.copy()
            
def get_prev_wf_positions_in_angstrom(self):
    return 0.529177*self.pref_wf_positions
            
def set_prev_wf_positions(self,pos):
    self.pref_wf_positions = pos.copy()

def get_civecs(self):
    return self.civecs.copy()

def get_orbs(self):
    return self.orbs.copy()


###end terachem_cas electronic structure section###
