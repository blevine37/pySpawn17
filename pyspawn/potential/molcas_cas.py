import math
import numpy as np
import os
import shutil
import sys
from molcas_interface import Input, Environment, ReadOutput

#################################################
### electronic structure routines go here #######
#################################################

# each electronic structure method requires at least two routines:
# 1) compute_elec_struct_, which computes energies, forces, and wfs
# 2) init_h5_datasets_, which defines the datasets to be output to hdf5
# 3) potential_specific_traj_copy, which copies data that is potential specific
#    from one traj data structure to another.  This is used when new
#    trajectories and centroids are spawned.
#    other ancillary routines may be included as well

def compute_elec_struct(self, zbackprop):
    """Subroutine that calls electronic structure calculation in molcas
    through molcas_interface. This version is compatible with tcpb-0.5.0
    When running multiple job on the same server we need to make sure we use
    different ports for Terachem server. Every trajectory or centroid
    has a port variable, which is passed along to children.
    Needs to be provided at input in start file as a traj_param"""

    initDir = os.getcwd()

    if not zbackprop:
        cbackprop = ""
    else:
        cbackprop = "backprop_"
   
    dir_index=0
    dir_exists=False
    while dir_exists == False:
        QMDIR   = "QMDIR_" + str(dir_index)
        if os.path.exists(QMDIR):
            dir_index += 1
        else:
            dir_exists = True
            os.makedirs(QMDIR)
    

    istate = self.get_istate()
    nstates = self.get_numstates()

    # initialize electronic_phases if not present
    if not hasattr(self, 'electronic_phases'):
        self.electronic_phases = np.ones(nstates)
    if not hasattr(self, 'backprop_electronic_phases'):
        self.backprop_electronic_phases = np.ones(nstates)
    exec("pos = self.get_" + cbackprop + "positions()")
    pos_list = pos.tolist()

    base_options = self.get_molcas_options()

    options = base_options

    options["castarget"] = istate
    project=options["project"] + 'qm_' + str(dir_index) 
    
    if hasattr(self, 'wfn'):
        if options["method"] == 'caspt2':
            jobmix_old = QMDIR + "/JobMix.old"
            wfnout = os.path.join(initDir, jobmix_old)
        else:
            jobiph_old = QMDIR + "/JobIph.old"
            wfnout = os.path.join(initDir, jobiph_old)
        eval("self.get_" + cbackprop + "wfn()").tofile(wfnout)

        inporb= QMDIR + "/INPORB"
        orbout = os.path.join(initDir, inporb)
        eval("self.get_" + cbackprop + "inporbs()").tofile(orbout)
        zolaps = True
    else:
        zolaps = False
        # check INPORB exists
        if not os.path.exists("INPORB"):
            print "Error: INPORB file not found. Please provide an initial guess for propagation.\n"
            sys.exit(1)
        if not os.path.exists(QMDIR):
            os.makedirs(QMDIR)
        if self.time==0.0:
            shutil.copy("INPORB", QMDIR)

    MolcasEnv = Environment(project,QMDIR,'INPORB', '$MOLCAS',options["python3"],tmpdir='TMPDIR')
    MolcasEnv.setup_molcas()

    input_molcas = Input(project,options["atoms"],pos_list,options["basis"],options["charge"],
            options["spinmult"],options["nactel"],options["inactive"],options["actorb"],
            nstates, cbackprop,castarget=options["castarget"], method=options['method'],pt2=options["pt2"])
    
    input_molcas.write_gateway()
    input_molcas.write_seward()
    input_molcas.write_rasscf()
    if options["method"] == 'caspt2':
        input_molcas.write_caspt2()
    input_molcas.write_alaska()
    if zolaps:
        input_molcas.write_rassi()
    input_molcas.write_input()

    MolcasEnv.run_molcas()
    
    output = ReadOutput(project,options["atoms"],nstates,options["method"],options["pt2"])
    output.check_happy_landing()
    output.get_energy()
    output.get_gradients()
    #output.get_civectors()
    results = output.results()


    e = np.zeros(nstates)
    e = results['energy']
    g = results['gradient']
    workdir=os.getcwd()

    if options["method"] == 'caspt2':
        wfnfile = os.path.join(workdir, 'JobMix')
    else:
        wfnfile = os.path.join(workdir, "JobIph")
    
    exec("self.set_" + cbackprop + "wfn(np.fromfile(wfnfile))")

    orbfilename = "RasOrb"
    exec("self.set_" + cbackprop + "inporbs(np.fromfile(orbfilename))")


    f = np.zeros((nstates, self.numdims))
    f[self.istate, :] = -1.0 * results['gradient'].flatten()

    exec("self.set_" + cbackprop + "energies(e)")
    exec("self.set_" + cbackprop + "forces(f)")

    if zolaps:
        S = output.get_overlap()

        for jstate in range(nstates):
            S[:, jstate] *= eval("self.get_" + cbackprop +
                                 "electronic_phases()[jstate]")
            S[jstate, :] *= eval("self.get_" + cbackprop +
                                 "electronic_phases()[jstate]")

        for jstate in range(nstates):
            if S[jstate, jstate] < 0.0:
                ep = eval("self.get_" + cbackprop + "electronic_phases()")
                ep[jstate] *= -1.0
                exec("self.set_" + cbackprop + "electronic_phases(ep)")
                # I'm not sure if this line is right, but it seems to be working
                S[jstate, :] *= -1.0

        exec("self.set_" + cbackprop + "S_elec_flat(S.flatten())")

        W = np.zeros((2, 2))
        W[0, 0] = S[istate, istate]

        tdc = np.zeros(nstates)

        for jstate in range(nstates):
            if istate == jstate:
                tdc[jstate] = 0.0
            else:
                W[1,0] = S[jstate,istate]
                W[0,1] = S[istate,jstate]
                W[1,1] = S[jstate,jstate]
                 
                tdc[jstate] = self.compute_tdc(W)

        exec("self.set_" + cbackprop + "timederivcoups(tdc)")
    else:
        exec("self.set_" + cbackprop +
             "timederivcoups(np.zeros(self.numstates))")

    exec("self.set_" + cbackprop + "prev_wf_positions(pos)")
   
    os.chdir(initDir)
    shutil.rmtree(QMDIR)

#def compute_electronic_overlap():
#    #it doesn't seem to be called anywhere
#    print("function called")
#    wfn1 = os.path.joint(cwd, "JOB001")
#    wfn2 = os.path.joint(cwd, "JOB002")
#
#    Input.write_rassi(input_name="rassi.inp")
#    Environment.run_rassi()
#    S = Output.get_overlap()
#
#    #S = results2['ci_overlap']
#
#    return S


def init_h5_datasets(self):
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["forces_i"] = self.numdims
    #self.h5_datasets["jobiph"] = '' #self.jobiph
    #self.h5_datasets["civecs"] = ""#self.ncivecs
    #self.h5_datasets["orbs"] = "" #self.norbs
    self.h5_datasets['wfn'] = 2 
    self.h5_datasets_half_step["time_half_step"] = 1
    self.h5_datasets_half_step["timederivcoups"] = self.numstates
    self.h5_datasets_half_step["S_elec_flat"] = self.numstates*self.numstates

def potential_specific_traj_copy(self, from_traj):
    self.set_molcas_options(from_traj.get_molcas_options())
    return

def get_wf0(self):
    return self.wf[0, :].copy()

def get_wf1(self):
    return self.wf[1, :].copy()

def get_backprop_wf0(self):
    return self.backprop_wf[0, :].copy()

def get_backprop_wf1(self):
    return self.backprop_wf[1, :].copy()

def set_molcas_options(self, tco):
    self.molcas_options = tco.copy()

def get_molcas_options(self):
    return self.molcas_options.copy()

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

#################################################
def set_jobiph(self, v):
    self.jobiph = v

def get_jobiph(self):
    return self.jobiph

def set_backprop_jobiph(self, v):
    self.backprop_jobiph = v

def get_backprop_jobiph(self):
    return self.backprop_jobiph

def set_wfn(self, v):
    self.wfn = v.copy()

def get_wfn(self):
    return self.wfn.copy()

def set_backprop_wfn(self, v):
    self.backprop_wfn = v.copy()

def get_backprop_wfn(self):
    return self.backprop_wfn.copy()

def get_inporbs(self):
    return self.inporbs.copy()

def set_inporbs(self, v):
    self.inporbs = v.copy()

def get_backprop_inporbs(self):
    return self.inporbs.copy()

def set_backprop_inporbs(self, v):
    self.inporbs = v.copy()
#################################################
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
