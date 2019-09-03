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

def compute_elec_struct(self, zbackprop):
    """Subroutine that calls electronic structure calculation in Terachem
    through tcpb interface. This version is compatible with tcpb-0.5.0
    When running multiple job on the same server we need to make sure we use
    different ports for Terachem server. Every trajectory or centroid 
    has a port variable, which is passed along to children. Needs to be provided 
    at input in start file as a traj_param"""

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
        
    TC = TCProtobufClient(host='localhost', port=self.tc_port)

    base_options = self.get_tc_options()

    options = base_options

    options["castarget"] = istate

    #TC.update_options(**base_options)

    TC.connect()

    # Check if the server is available
    avail = TC.is_available()
    #print "TCPB Server available: {}".format(avail)

    # Write CI vectors and orbitals for initial guess and overlaps
    cwd = os.getcwd()
    if hasattr(self,'civecs'):
        civecout = os.path.join(cwd,"CIvecs.Singlet.old")
        orbout = os.path.join(cwd,"c0.old")
        orbout_t = os.path.join(cwd,"c0_t.old")
        eval("self.get_" + cbackprop + "civecs()").tofile(civecout)
        eval("self.get_" + cbackprop + "orbs()").tofile(orbout)
        n = int(math.floor(math.sqrt(self.get_norbs())))
        ((np.resize(eval("self.get_" + cbackprop + "orbs()"),(n,n)).T).flatten()).tofile(orbout_t)
        #print "old civecs", eval("self.get_" + cbackprop + "civecs()")
        #print "old orbs", eval("self.get_" + cbackprop + "orbs()")
        zolaps = True
        if ("casscf" in self.tc_options):
            if (self.tc_options["casscf"]=="yes"):
                options["caswritevecs"]="yes"
                options["casguess"]=orbout_t
            else:
                options["caswritevecs"]="yes"
                options["guess"]=orbout
        else:
            options["caswritevecs"]="yes"
            options["guess"]=orbout
    else:
        zolaps = False
        options["caswritevecs"]= "yes"

    # Gradient calculation

    # here we call TC once for energies and once for the gradient
    # will eventually be replaced by a more efficient interface
    results = TC.compute_job_sync("energy", pos_list, "bohr", **options)
    #print results

    e = np.zeros(nstates)
    e[:] = results['energy'][:]

    results = TC.compute_job_sync("gradient", pos_list, "bohr", **options)
    #print results


    civecfilename = os.path.join(results['job_scr_dir'], "CIvecs.Singlet.dat")
    exec("self.set_" + cbackprop + "civecs(np.fromfile(civecfilename))")
    #print "new civecs", self.civecs    

    #orbfilename = os.path.join(results['job_scr_dir'], "c0")
    orbfilename = results['orbfile']
    exec("self.set_" + cbackprop + "orbs((np.fromfile(orbfilename)).flatten())")

    self.set_norbs(self.get_orbs().size)

    # BGL transpose hack is temporary
    n = int(math.floor(math.sqrt(self.get_norbs())))
    clastchar = orbfilename.strip()[-1]
    #print "n", n
    #print "clastchar", clastchar
    if clastchar != '0':
        tmporbs = eval("self.get_" + cbackprop + "orbs()")
        exec("self.set_" + cbackprop + "orbs(((tmporbs.reshape((n,n))).T).flatten())")
    # end transpose hack

    #print "new orbs", eval("self.get_" + cbackprop + "orbs()")
    orbout2 = os.path.join(cwd,"c0.new")
    eval("self.get_" + cbackprop + "orbs()").tofile(orbout2)

    self.set_ncivecs(self.get_civecs().size)

    f = np.zeros((nstates,self.numdims))
    #print "results['gradient'] ", results['gradient']
    #print "results['gradient'].flatten() ", results['gradient'].flatten()
    f[self.istate,:] = -1.0 * results['gradient'].flatten()

    exec("self.set_" + cbackprop + "energies(e)")

    exec("self.set_" + cbackprop + "forces(f)")

    #if False:        
    if zolaps:
        exec("pos2 = self.get_" + cbackprop + "prev_wf_positions_in_angstrom()")
        #print 'pos2.tolist()', pos2.tolist()
        #print 'civecfilename', civecfilename
        #print 'civecout', civecout
        #print 'orbfilename', orbfilename
        #print 'orbout2', orbout2
        #print 'orbout', orbout
        options = base_options

        options["geom2"]=pos2.tolist()
        options["cvec1file"]=civecfilename
        options["cvec2file"]=civecout
        options["orb1afile"]=orbout2
        options["orb2afile"]=orbout

        #print 'pos_list', pos_list
        results2 = TC.compute_job_sync("ci_vec_overlap", pos_list, "bohr", **options)
        #print "results2", results2
        S = results2['ci_overlap']
        #print "S before phasing ", S

        # phasing electronic overlaps 
        for jstate in range(nstates):
            S[:,jstate] *= eval("self.get_" + cbackprop + "electronic_phases()[jstate]")
            S[jstate,:] *= eval("self.get_" + cbackprop + "electronic_phases()[jstate]")

        for jstate in range(nstates):
            if S[jstate,jstate] < 0.0:
                ep = eval("self.get_" + cbackprop + "electronic_phases()")
                ep[jstate] *= -1.0
                exec("self.set_" + cbackprop + "electronic_phases(ep)")
                # I'm not sure if this line is right, but it seems to be working
                S[jstate,:] *= -1.0
                
        #print "S", S
        exec("self.set_" + cbackprop + "S_elec_flat(S.flatten())")

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
                #print "tdc", tdc[jstate]

        #tmp=self.compute_tdc(W)
        #tdc = np.zeros(self.numstates)
        #if self.istate == 1:
        #    jstate = 0
        #else:
        #    jstate = 1
        #    tdc[jstate] = tmp

        #print "tdc2 ", tdc
        exec("self.set_" + cbackprop + "timederivcoups(tdc)")
    else:
        exec("self.set_" + cbackprop + "timederivcoups(np.zeros(self.numstates))")
    
    exec("self.set_" + cbackprop + "prev_wf_positions(pos)")

def compute_electronic_overlap(self,pos1,civec1,orbs1,pos2,civec2,orbs2):
    orbout1 = os.path.join(cwd,"c0.1")
    orbs1.tofile(orbout1)
    orbout2 = os.path.join(cwd,"c0.2")
    orbs2.tofile(orbout2)

    civecout1 = os.path.join(cwd,"civec.1")
    civec1.tofile(civecout1)
    civecout2 = os.path.join(cwd,"civec.2")
    civec2.tofile(civecout2)
    
    TC = TCProtobufClient(host='localhost', port=self.tc_port)
    options = self.get_tc_options()
    #TC.update_options(**base_options)
    TC.connect()
    # Check if the server is available
    avail = TC.is_available()

    options["geom2"]=(0.529177*pos2).tolist()
    options["cvec1file"]=civecfilename
    options["cvec2file"]=civecout
    options["orb1afile"]=orbout2
    options["orb2afile"]=orbout
     
    results2 = TC.compute_job_sync("ci_vec_overlap", pos1.tolist(), "bohr", **options)

    S = results2['ci_overlap']
    
    return S

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
    self.h5_datasets_half_step["S_elec_flat"] = self.numstates*self.numstates

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

def get_civecs(self):
    return self.civecs.copy()

def set_civecs(self, v):
    self.civecs = v.copy()

def get_ncivecs(self):
    return self.ncivecs

def set_ncivecs(self, n):
    self.ncivecs = n

def get_orbs(self):
    return self.orbs.copy()

def set_orbs(self, v):
    self.orbs = v.copy()

def get_norbs(self):
    return self.norbs

def set_norbs(self, n):
    self.norbs = n

def get_backprop_civecs(self):
    return self.backprop_civecs.copy()

def set_backprop_civecs(self, v):
    self.backprop_civecs = v.copy()

def get_backprop_orbs(self):
    return self.backprop_orbs.copy()

def set_backprop_orbs(self, v):
    self.backprop_orbs = v.copy()

def get_electronic_phases(self):
    return self.electronic_phases.copy()

def set_electronic_phases(self, v):
    self.electronic_phases = v.copy()

def get_backprop_electronic_phases(self):
    return self.backprop_electronic_phases.copy()

def set_backprop_electronic_phases(self, v):
    self.backprop_electronic_phases = v.copy()




###end terachem_cas electronic structure section###
