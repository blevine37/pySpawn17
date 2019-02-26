# simulation object contains the current state of the simulation.
# It is analagous to the "bundle" object in the original FMS code.
import sys
import types
import math
import numpy as np
import h5py
from pyspawn.fmsobj import fmsobj
from pyspawn.traj import traj
import general as gen
import os
import shutil
import complexgaussian as cg
import datetime
import time
import cmath 
import copy
from astropy.coordinates.builtin_frames.utils import norm

class simulation(fmsobj):
    
    def __init__(self):
        # traj is a dictionary of trajectory basis functions (TBFs)
        self.traj = dict()

        # queue is a list of tasks to be run
        self.queue = ["END"]
        # tasktimes is a list of the simulation times associated with each task
        self.tasktimes = [1e10]

        # olapmax is the maximum overlap allowed for a spawn.  Above this,
        # the spawn is cancelled
        self.olapmax = 0.1
        
        # Number of electronic states (this needs to be fixed since this is already in traj object)
        self.num_el_states = 9
        
        # if pij > p_threshold  and pop > pop_threshold cloning is initiated 
        self.pop_threshold = 0.1
        self.p_threshold = 0.03
        
        # Type of cloning: either 'toastate' or 'pairwise'
        # 'pairwise' version seems to be not a good thing to do
        self.cloning_type = "toastate" 
        
        # quantum time is the current time of the quantum amplitudes
        self.quantum_time = 0.0
        
        # timestep for quantum propagation
        self.timestep = 0.0
        self.clone_again = False
        
        # quantum propagator
        #self.qm_propagator = "RK2"
        # quantum hamiltonian
        #self.qm_hamiltonian = "adiabatic"

        # maps trajectories to matrix element indices 
        # (the order of trajectories in the dictionary is not the same as amplitudes)
        self.traj_map = dict()

        # quantum amplitudes
        self.qm_amplitudes = np.zeros(0,dtype=np.complex128)
        
        # Total electronic population on each electronic states 
        # takes into account all nuclear basis functions
        self.el_pop = np.zeros(self.num_el_states)
        
        # energy shift for quantum propagation (better accuracy if energy is close to 0)
        self.qm_energy_shift = 0.0

        # variables to be output to hdf5 mapped to the size of each data point
        self.h5_datasets = dict()
        self.h5_types = dict()

        # maximium walltime in seconds
        self.max_quantum_time = -1.0

        # maximium walltime in seconds
        self.max_walltime = -1.0

    def set_maxtime_all(self,maxtime):

        self.max_quantum_time = maxtime
        h = self.timestep
        for key in self.traj:
            self.traj[key].maxtime = maxtime + h

    def from_dict(self, **tempdict):
        """Convert dict to simulation data structure"""
        
        for key in tempdict:
            if isinstance(tempdict[key], types.UnicodeType) :
                tempdict[key] = str(tempdict[key])
            if isinstance(tempdict[key], types.ListType) :
                if isinstance((tempdict[key])[0], types.FloatType) :
                    # convert 1d float lists to np arrays
                    tempdict[key] = np.asarray(tempdict[key])
                if isinstance((tempdict[key])[0], types.StringTypes) :
                    if (tempdict[key])[0][0] == "^":
                        for i in range(len(tempdict[key])):
                            tempdict[key][i] = eval(tempdict[key][i][1:])
                        tempdict[key] = np.asarray(tempdict[key], dtype=np.complex128)
                else:
                    if isinstance((tempdict[key])[0], types.ListType):
                        if isinstance((tempdict[key])[0][0], types.FloatType) :
                            # convert 2d float lists to np arrays
                           tempdict[key] = np.asarray(tempdict[key])
                        if isinstance((tempdict[key])[0][0], types.StringTypes) :
                            if (tempdict[key])[0][0][0] == "^":
                                for i in range(len(tempdict[key])):
                                    for j in range(len(tempdict[key][i])):
                                        tempdict[key][i][j] = eval(tempdict[key][i][j][1:])
                                tempdict[key] = np.asarray(tempdict[key], dtype=np.complex128)
            if isinstance(tempdict[key], types.DictType) :
                if 'fmsobjlabel' in (tempdict[key]).keys():
                    fmsobjlabel = (tempdict[key]).pop('fmsobjlabel')
                    obj = eval(fmsobjlabel[8:])()
                    obj.from_dict(**(tempdict[key]))
                    tempdict[key] = obj
                else:
                    for key2 in tempdict[key]:
                        if isinstance((tempdict[key])[key2], types.DictType) :
                            fmsobjlabel = ((tempdict[key])[key2]).pop('fmsobjlabel')
                            obj = eval(fmsobjlabel[8:])()
                            obj.from_dict(**((tempdict[key])[key2]))
                            (tempdict[key])[key2] = obj
        self.__dict__.update(tempdict)

    def add_traj(self, t1):
        """Add a trajectory to the simulation"""
        
        key = t1.label
        print "Trajectory added:", key
        mintime = t1.mintime
        index = -1
        for key2 in self.traj:
            if mintime < self.traj[key2].mintime:
                if index < 0:
                    index = self.traj_map[key2]
                self.traj_map[key2] += 1
        if index < 0:
            index = len(self.traj)
        self.traj[key] = t1
        self.traj_map[key] = index

    def propagate(self):
        """This is the main propagation loop for the simulation"""
        
        gen.print_splash()
        t0 = time.clock()
        while True:

            # update the queue (list of tasks to be computed)
            print "\nUpdating task queue"
            self.update_queue()

            # if the queue is empty, we're done!
            print "Time =", self.quantum_time
            print "Checking if we are at the end of the simulation"
            #if (self.queue[0] == "END"):
            if (self.quantum_time + 1.0e-6 > self.max_quantum_time):
                print "Propagate DONE, simulation ended gracefully!"
                return

            # end simulation if walltime has expired
            print "Checking if maximum wall time is reached"
            if (self.max_walltime < time.time() and self.max_walltime > 0):
                print "Wall time expired, simulation ended gracefully!"
                return
            
            # it is possible for the queue to run empty but for the job not to be done
            if (self.queue[0] != "END"):            
                # Right now we just run a single task per cycle,
                # but we could parallelize here and send multiple tasks
                # out for simultaneous processing.
                current = self.pop_task()
                print "\nStarting " + current            
                eval(current)
                print "Done with " + current
            else:
                print "Task queue is empty"

            
            # propagate quantum variables if possible
            print "\nPropagating quantum amplitudes if we have enough information to do so"
            
            self.propagate_quantum_as_necessary()
            
            cond_num = np.linalg.cond(self.S)
            if cond_num > 1000:
                print "BAD S matrix: condition number =", cond_num, "\nExiting"
                return
            
            # moved cloning routine into the propagation to update h5 file on time
#             print "\nNow we will clone new trajectories if necessary:"
#             self.clone_as_necessary()
                        
            # print restart output - this must be the last line in this loop!
            print "Updating restart output"
            self.restart_output()
            print "Elapsed wall time: %6.1f" % (time.clock() - t0)
                
#             if np.shape(self.S)[0] > 1:
#                 print "S12 =", np.abs(self.S[0,1])
#             
    def propagate_quantum_as_necessary(self):
        """here we will propagate the quantum amplitudes if we have
        the necessary information to do so.
        we have to determine what the maximum time is for which
        we have all the necessary information to propagate the amplitudes"""
        
        max_info_time = 1.0e10
        # first check trajectories
        for key in self.traj:

            timestep = self.traj[key].timestep
            time = self.traj[key].time
            if time  < max_info_time:
                max_info_time = time #- timestep

        print "We have enough information to propagate to time ", max_info_time

        # now, if we have the necessary info, we propagate
        while max_info_time > (self.quantum_time + 1.0e-6):

            if self.quantum_time > 1.0e-6:
                print "Propagating quantum amplitudes at time", self.quantum_time
                self.qm_propagate_step()
            else:
                print "Propagating quantum amplitudes at time", self.quantum_time,\
                " (first step)"
                self.qm_propagate_step(zoutput_first_step=True)

            print "\nOutputing quantum information to hdf5"
            self.h5_output()
            self.calc_approx_el_populations()
            print "\nNow we will clone new trajectories if necessary:"
            
            if self.cloning_type == "toastate":
                self.clone_to_a_state()
                if self.clone_again:
                    for key in self.traj:
                        self.traj[key].compute_cloning_E_diff()
                    self.clone_to_a_state()
            
            if self.cloning_type == "pairwise":
                self.clone_pairwise()
                if self.clone_again:
                    self.clone_pairwise()
#             print "\nOutputing quantum information to hdf5"
#             self.h5_output()
            
    def init_amplitudes_one(self):
        """Sets the first amplitude to 1.0 and all others to zero"""

        self.compute_num_traj_qm()
        self.qm_amplitudes = np.zeros_like(self.qm_amplitudes, dtype=np.complex128)
        self.qm_amplitudes[0] = 1.0
        
    def compute_num_traj_qm(self):
        """Get number of trajectories. Note that the order of trajectories in the dictionary 
        is not the same as in Hamiltonian!
        The new_amp variable is non-zero only for the child and parent cloned BFs
        we assign amplitudes during the timestep after cloning happened, 
        new_amp variable is zeroed out"""
        
        n = 0
        qm_time = self.quantum_time

        for key in self.traj:
            n += 1

            if self.traj_map[key] + 1 > len(self.qm_amplitudes):
                print "Adding trajectory ", key, "to the nuclear propagation" 
                # Adding the quantum amplitude for a new trajectory 
                self.qm_amplitudes = np.append(self.qm_amplitudes, self.traj[key].new_amp)
                self.traj[key].new_amp = 0.0 
                
                for key2 in self.traj:
                    if np.abs(self.traj[key2].new_amp) > 1e-6 and key2 != key:
                        # when new trajectory added we need to update the amplitude of parent
                        self.qm_amplitudes[self.traj_map[key2]] = self.traj[key2].new_amp
                        # zeroing out new_amp variable
                        self.traj[key2].new_amp = 0.0 
                  
        self.num_traj_qm = n
                                        
    def invert_S(self):
        """compute Sinv from S"""
        
        cond_num = np.linalg.cond(self.S)
        if cond_num > 500:
            print "BAD S matrix: condition number =", cond_num
            #sys.exit()
        else:
            #print "S condition number =", cond_num
            pass
        self.Sinv = np.linalg.inv(self.S)
        
    def build_Heff(self):
        """built Heff form H, Sinv, and Sdot"""
        
        c1i = (complex(0.0,1.0))
        self.Heff = np.matmul(self.Sinv, (self.H - c1i * self.Sdot))
        
    def build_S_elec(self):
        """Build matrix of electronic overlaps"""
        
        ntraj = self.num_traj_qm
        self.S_elec = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        if i == j:
                            self.S_elec[i,j] = 1.0
                        else:

                            wf_i_T = np.transpose(\
                                     np.conjugate(self.traj[keyi].td_wf_full_ts_qm))
                            wf_j = self.traj[keyj].td_wf_full_ts_qm
                            self.S_elec[i, j] = np.dot(wf_i_T, wf_j)
    
    def build_S(self):
        """Build the overlap matrix, S"""
        
        if self.quantum_time > 0.0:
            self.S_prev = self.S
            
        ntraj = self.num_traj_qm
        self.S = np.zeros((ntraj,ntraj), dtype=np.complex128)
        self.S_nuc = np.zeros((ntraj,ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.S_nuc[i,j] = cg.overlap_nuc(self.traj[keyi],\
                                                         self.traj[keyj],\
                                                         positions_i="positions_qm",\
                                                         positions_j="positions_qm",\
                                                         momenta_i="momenta_qm",\
                                                         momenta_j="momenta_qm") 
                        
                        self.S[i,j] = self.S_nuc[i,j] * self.S_elec[i,j]
        
                        
    def build_Sdot(self, first_half = None):
        """build the right-acting time derivative operator"""
        
        ntraj = self.num_traj_qm
        self.Sdot = np.zeros((ntraj, ntraj), dtype=np.complex128)
        self.S_dot_elec = np.zeros((ntraj, ntraj), dtype=np.complex128)
        self.S_dot_nuc = np.zeros((ntraj, ntraj), dtype=np.complex128)
        
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.S_dot_nuc[i,j] = cg.Sdot_nuc(self.traj[keyi],\
                                                     self.traj[keyj],\
                                                     positions_i="positions_qm",\
                                                     positions_j="positions_qm",\
                                                     momenta_i="momenta_qm",\
                                                     momenta_j="momenta_qm",\
                                                     forces_j="av_force_qm")
                                                
                        # Here we will call ES program to get Hamiltonian
                        H_elec, Force\
                        = self.traj[keyj].construct_el_H(self.traj[keyj].positions_qm)
                        wf_j_dot = -1j * np.dot(H_elec,\
                                              self.traj[keyj].td_wf_full_ts_qm)
                        wf_i_T = np.conjugate(np.transpose(self.traj[keyi].td_wf_full_ts_qm))
                        self.S_dot_elec[i, j] =  np.dot(wf_i_T, wf_j_dot)                                             

                        self.Sdot[i, j] = np.dot(self.S_dot_elec[i, j], self.S_nuc[i, j])\
                                       + np.dot(self.S_elec[i, j], self.S_dot_nuc[i, j])
            
    def build_H(self):
        """Building the Hamiltonian"""
        
        self.build_V()
        self.build_T()
        ntraj = self.num_traj_qm
        shift = self.qm_energy_shift * np.identity(ntraj)
        self.H = self.T + self.V + shift
        
    def build_V(self):
        """Build the potential energy matrix, V
        This routine assumes that S is already built"""
        
        c1i = (complex(0.0, 1.0))
        cm1i = (complex(0.0, -1.0))
        ntraj = self.num_traj_qm
        self.V = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        if i == j:
                            self.V[i, j] = self.traj[keyi].av_energy_qm
                        else:
                            nuc_overlap = self.S_nuc[i, j]

                            H_elec_i, Force_i\
                            =self.traj[keyi].construct_el_H(self.traj[keyi].positions_qm)
                            
                            H_elec_j, Force_j\
                            =self.traj[keyj].construct_el_H(self.traj[keyj].positions_qm)
                                                    
                            wf_i = self.traj[keyi].td_wf_full_ts_qm
                            wf_i_T = np.transpose(np.conjugate(wf_i))
                            wf_j = self.traj[keyj].td_wf_full_ts_qm
                            H_i = np.dot(wf_i_T, np.dot(H_elec_i, wf_j))
                            H_j = np.dot(wf_i_T, np.dot(H_elec_j, wf_j))
                            V_ij = 0.5 * (H_i + H_j)
                            self.V[i, j] = V_ij * nuc_overlap
                
    def build_T(self):
        "Building kinetic energy, needs electronic overlap S_elec"
        
        ntraj = self.num_traj_qm
        self.T = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.T[i, j] = cg.kinetic_nuc(self.traj[keyi], self.traj[keyj],\
                                                      positions_i="positions_qm",\
                                                      positions_j="positions_qm",\
                                                      momenta_i="momenta_qm",\
                                                      momenta_j="momenta_qm")\
                                                     *self.S_elec[i,j]

    def clone_pairwise(self):
        """Cloning routine. Trajectories that are cloning will be established from the 
        cloning probabilities variable clone_p.
        When a new basis function is added the labeling is done in a following way:
        a trajectory labeled 00b1b5 means that the initial trajectory "00" spawned
        a trajectory "1" (its second child) which then spawned another (it's 6th child)"""
        
        clonetraj = dict()
        for key in self.traj:
            for istate in range(self.traj[key].numstates):
#               # sorting jstate according to decreasing cloning probabilty
                jstates = (-self.traj[key].clone_p[istate, :]).argsort()
#                 print "jstates = ", jstates
                for jstate in jstates:
#                     print "jstate =", jstate
#                     sys.exit()
                    """Is this the state i to clone to state j?
                    If the following conditions are satisfied then we clone to that state
                    and we're done. If done we go to another state with lower probability"""
                    
                    if self.traj[key].clone_p[istate, jstate] > self.p_threshold and\
                    self.traj[key].populations[istate] > self.pop_threshold and\
                    self.traj[key].populations[jstate] > self.pop_threshold: #and\
#                     self.traj[key].populations[istate] > self.traj[key].populations[jstate]:
                        print "Trajectory " + key + " cloning from ", istate,\
                        "to", jstate, " state at time",\
                        self.traj[key].time, "with p =",\
                        self.traj[key].clone_p[istate, jstate]
    #                         print self.p_threshold
    
                        label = str(self.traj[key].label) + "b" +\
                        str(self.traj[key].numchildren)
                        # create and initiate new trajectory structure
                        newtraj = traj()
                        
                        # making a copy of a parent BF in order not to overwrite the original
                        # in case cloning fails due to large overlap 
                        parent_copy = copy.deepcopy(self.traj[key])
                        # total nuclear norm
#                         nuc_norm = np.abs(np.dot(np.conjugate(np.transpose(self.qm_amplitudes)),\
#                                           np.dot(self.S, self.qm_amplitudes)))
                        nuc_norm = 1.0
                        # okay, now we finally decide whether to clone or not
                        clone_ok = newtraj.init_clone_traj(parent_copy,\
                                                           istate, jstate, label, nuc_norm)
                        
                        if clone_ok:
                            
                            """This will be a separate subroutine eventually probably,
                            here we need to rescale the coefficients of two cloned nuclear
                            basis (1 and 2 in this notation) functions to conserve the norm. 
                            The problem is that to do this we need overlaps with all other
                            trajectories, which are not available yet.
                            So we need to calculate all overlap here, it is not a big deal
                            because all trajectories and quantum amplitudes are propagated to
                            the same time t"""
                            
                            # the dimensionality increases when we add new function but the
                            # matrices are not updated yet
                            new_dim = np.shape(self.S)[0]+1
                            ind_1 = self.traj_map[key]
                            ind_2 = np.shape(self.S)[0]
                            traj_1 = parent_copy
                            traj_2 = newtraj
                            
                            amps = np.zeros((np.shape(self.S)[0]+1), dtype=np.complex128)
                            S = np.zeros((np.shape(self.S)[0]+1, np.shape(self.S)[0]+1),\
                                         dtype=np.complex128)
                            
                            new_amp_1 = self.qm_amplitudes[ind_1] * traj_1.rescale_amp
                            new_amp_2 = self.qm_amplitudes[ind_1] * traj_2.rescale_amp
                            
                            # check for overlap S12

                            wf_1_T = np.transpose(np.conjugate(traj_1.td_wf_full_ts))
                            wf_2 = traj_2.td_wf_full_ts
                            S_elec_12 = np.real(np.dot(wf_1_T, wf_2))
                            S_nuc_12 = cg.overlap_nuc(traj_1,\
                                                             traj_2,\
                                                             positions_i="positions",\
                                                             positions_j="positions",\
                                                             momenta_i="momenta",\
                                                             momenta_j="momenta")
                            
                            S[ind_1, ind_2] = S_elec_12 * S_nuc_12
                            S[ind_2, ind_1] = np.conjugate(S[ind_1, ind_2])                            
                            
                            pop_12 = 0.0
                            pop_12n = 0.0
                            pop_n = 0.0
                            
                            # this is population of the cloning BFs
                            pop_12 = np.dot(np.conjugate(new_amp_1), new_amp_1)\
                                   + np.dot(np.conjugate(new_amp_2), new_amp_2)\
                                   + np.dot(np.conjugate(new_amp_1), new_amp_2)\
                                   * S[ind_1, ind_2]\
                                   + np.dot(np.conjugate(new_amp_2), new_amp_1)\
                                   * S[ind_2, ind_1]

                            x = 1.0

                            for ind in range(np.shape(self.S)[0] + 1): S[ind, ind] = 1.0
                            
                            if np.shape(self.S)[0] != 1:
                                for key_n in self.traj:
                                    
                                    ind_n = self.traj_map[key_n]
                                    if key_n != key and key_n != label:                            
                                        traj_n = self.traj[key_n]
                                        
                                        amp_n = self.qm_amplitudes[ind_n]
                                        amps[ind_n] = amp_n
                                        
                                        wf_1 = traj_1.td_wf_full_ts
                                        wf_1_T = np.transpose(np.conjugate(wf_1))
                                        
                                        wf_n = traj_n.td_wf_full_ts
                                        S_elec_1n = np.dot(wf_1_T, wf_n)
                                        S_nuc_1n = self.overlap_nuc(traj_1.positions,\
                                                                    traj_n.positions,\
                                                                    traj_1.momenta,\
                                                                    traj_n.momenta_qm,\
                                                                    traj_1.widths,\
                                                                    traj_n.widths) 
                                        
                                        S[ind_1, ind_n] = S_elec_1n * S_nuc_1n
                                        S[ind_n, ind_1] = np.conjugate(S[ind_1, ind_n])
                                        
                                        
                                        wf_2 = traj_2.td_wf_full_ts
                                        wf_2_T = np.transpose(np.conjugate(wf_2))
                                        
                                        S_elec_2n = np.dot(wf_2_T, wf_n)
                                        S_nuc_2n = self.overlap_nuc(traj_2.positions,\
                                                                    traj_n.positions,\
                                                                    traj_2.momenta,\
                                                                    traj_n.momenta_qm,\
                                                                    traj_2.widths,\
                                                                    traj_n.widths) 
                                        
                                        S[ind_2, ind_n] = S_elec_2n * S_nuc_2n
                                        S[ind_n, ind_2] = np.conjugate(S[ind_2, ind_n])
                                        
                                        # adding population from the overlap of cloning BFs
                                        # with noncloning
                                        pop_12n += np.dot(np.conjugate(new_amp_1),\
                                                          amps[ind_n])\
                                                 * S[ind_1, ind_n]\
                                                 + np.dot(np.conjugate(amps[ind_n]),\
                                                          new_amp_1)\
                                                 * S[ind_n, ind_1]\
                                                 + np.dot(np.conjugate(new_amp_2),\
                                                          amps[ind_n])\
                                                 * S[ind_2, ind_n]\
                                                 + np.dot(np.conjugate(amps[ind_n]),\
                                                          new_amp_2)\
                                                 * S[ind_n, ind_2]
                                        
                                        # adding population from noncloning BFs, 
                                        # only diagonal contribution
                                        pop_n += np.dot(np.conjugate(amp_n), amp_n)
                                        
                                        for key_n2 in self.traj:
                                            
                                            if key_n2 != key\
                                            and key_n2 != label\
                                            and key_n2 != key_n:
                                                
                                                traj_n2 = self.traj[key_n2]
                                                ind_n2 = self.traj_map[key_n2]
                                                amp_n2 = self.qm_amplitudes[ind_n2]
                                                amps[ind_n2] = amp_n2                                                

                                                S[ind_n, ind_n2] = self.S[ind_n, ind_n2]
                                                S[ind_n2, ind_n] = self.S[ind_n2, ind_n]
                                                
                                                # adding population of noncloning BFs, 
                                                # only off-diagonal contributions
                                                pop_n += 1/4 * S[ind_n, ind_n2]\
                                                        * np.dot(np.conjugate(amps[ind_n]),\
                                                          amps[ind_n2])\
                                                        + S[ind_n2, ind_n]\
                                                        * np.dot(np.conjugate(amps[ind_n2]),\
                                                                amps[ind_n])
  
                            # Solving quadratic equation for the factor 
                            # that ensures norm conservation
                            x = (-pop_12n + np.sqrt(pop_12n**2\
                                - 4 * (pop_n - nuc_norm) * pop_12)) / (2 * pop_12) 
                            alpha = np.dot(np.conjugate(x), x)
                            if np.abs(alpha) < 1e-6: alpha = 1.0
                            print "alpha =", alpha
                            print "total_pop =", pop_n + x * pop_12n + pop_12 * alpha
                            print "total pop before rescale =", pop_n + pop_12n + pop_12
                            traj_1.new_amp = new_amp_1 * x
                            traj_2.new_amp = new_amp_2 * x
                            
                            # For debugging purposes
                            amps[ind_1] = traj_1.new_amp
                            amps[ind_2] = traj_2.new_amp
                            norm = np.dot(np.conjugate(np.transpose(amps)), np.dot(S, amps))
                            
                            print "NORM =", norm
                            
                            # Number overlap elements that are larger than threshold
                            s = (S > self.olapmax).sum()
                            print "SUM =", s
                            if s > new_dim:
                                
                                print "Aborting cloning due to large overlap with\
                                existing trajectory"
                                self.clone_again = False
                                continue
                            
                            else:
                                # if overlap is less than threshold
                                # we can update the actual parent

                                self.traj[key] = traj_1
                            
                                print "Overlap OK, creating new trajectory ", label
                                clonetraj[label] = traj_2
                                self.traj[key].numchildren += 1
                                print "Cloning successful"
                                self.add_traj(clonetraj[label])
                                # update matrices here in case other trajectories clone
                                self.compute_num_traj_qm()
                                self.build_Heff_half_timestep()
                                self.clone_again = True
                                return
                     
                        else:
                            self.clone_again = False
                            continue

    def clone_to_a_state(self):
        """Cloning routine. Trajectories that are cloning will be established from the 
        cloning probabilities variable clone_p.
        When a new basis function is added the labeling is done in a following way:
        a trajectory labeled 00b1b5 means that the initial trajectory "00" spawned
        a trajectory "1" (its second child) which then spawned another (it's 6th child)"""
        
        clonetraj = dict()
        for key in self.traj:
#           # sorting states according to decreasing cloning probability
            istates = (-self.traj[key].clone_E_diff[:]).argsort()
            
            for istate in istates:
                """Do we clone to istate?
                If the following conditions are satisfied then we clone to that state
                and we're done. If done we go to another state with lower probability"""
                if self.traj[key].full_H:
                    # If we use eigenstates
                    pop_to_check = self.traj[key].populations
                else:
                    # If we use approximate eigenstates
                    pop_to_check = self.traj[key].approx_pop
#                 if self.traj[key].clone_E_diff[istate] > self.p_threshold and\
#                 self.traj[key].populations[istate] > self.pop_threshold:
                if self.traj[key].clone_E_diff[istate] > self.p_threshold and\
                pop_to_check[istate] > self.pop_threshold and\
                pop_to_check[istate] < 1.0 - self.pop_threshold and\
                self.traj[key].clone_E_diff[istate] >= self.traj[key].clone_E_diff_prev[istate]:
                    # the last condition ensures that we clone when gap increases, not decreases
                    print "Trajectory " + key + " cloning to ",\
                    istate, " state at time",\
                    self.traj[key].time, "with p =",\
                    self.traj[key].clone_E_diff[istate]

                    label = str(self.traj[key].label) + "b" +\
                    str(self.traj[key].numchildren)
                    # create and initiate new trajectory structure
                    newtraj = traj()
                    
                    # making a copy of a parent BF in order not to overwrite the original
                    # in case cloning fails due to large overlap 
                    parent_copy = copy.deepcopy(self.traj[key])
                    # total nuclear norm
#                         nuc_norm = np.abs(np.dot(np.conjugate(np.transpose(self.qm_amplitudes)),\
#                                           np.dot(self.S, self.qm_amplitudes)))
                    nuc_norm = 1.0
                    # okay, now we finally decide whether to clone or not
                    if parent_copy.full_H:
                        # cloning based on real eigenstates
                        clone_ok = newtraj.init_clone_traj_to_a_state(parent_copy,\
                                                       istate, label, nuc_norm)
                    else:
                        # cloning based on approximate eigenstates
                        clone_ok = newtraj.init_clone_traj_approx(parent_copy,\
                                                       istate, label, nuc_norm)                    
                    if clone_ok:
                        
                        """This will be a separate subroutine eventually probably,
                        here we need to rescale the coefficients of two cloned nuclear
                        basis (1 and 2 in this notation) functions to conserve the norm. 
                        The problem is that to do this we need overlaps with all other
                        trajectories, which are not available yet.
                        So we need to calculate all overlaps here, it is not a big deal
                        because all trajectories and quantum amplitudes are propagated to
                        the same time t"""
                        
                        # the dimensionality increases when we add new function but the
                        # matrices are not updated yet
                        new_dim = np.shape(self.S)[0]+1
                        ind_1 = self.traj_map[key]
                        ind_2 = np.shape(self.S)[0]
                        traj_1 = parent_copy 
                        traj_2 = newtraj
                        
                        amps = np.zeros((np.shape(self.S)[0]+1), dtype=np.complex128)
                        S = np.zeros((np.shape(self.S)[0]+1, np.shape(self.S)[0]+1),\
                                     dtype=np.complex128)
                        
                        new_amp_1 = self.qm_amplitudes[ind_1] * traj_1.rescale_amp
                        new_amp_2 = self.qm_amplitudes[ind_1] * traj_2.rescale_amp
                        
                        # check for overlap S12

                        wf_1_T = np.transpose(np.conjugate(traj_1.td_wf_full_ts))
                        wf_2 = traj_2.td_wf_full_ts
                        S_elec_12 = np.real(np.dot(wf_1_T, wf_2))
                        S_nuc_12 = cg.overlap_nuc(traj_1,\
                                                         traj_2,\
                                                         positions_i="positions",\
                                                         positions_j="positions",\
                                                         momenta_i="momenta",\
                                                         momenta_j="momenta")
                        
                        S[ind_1, ind_2] = S_elec_12 * S_nuc_12
                        S[ind_2, ind_1] = np.conjugate(S[ind_1, ind_2])                            
                        
                        pop_12 = 0.0
                        pop_12n = 0.0
                        pop_n = 0.0
                        
                        # this is population of the cloning BFs
                        pop_12 = np.dot(np.conjugate(new_amp_1), new_amp_1)\
                               + np.dot(np.conjugate(new_amp_2), new_amp_2)\
                               + np.dot(np.conjugate(new_amp_1), new_amp_2)\
                               * S[ind_1, ind_2]\
                               + np.dot(np.conjugate(new_amp_2), new_amp_1)\
                               * S[ind_2, ind_1]

                        x = 1.0

                        for ind in range(np.shape(self.S)[0] + 1): S[ind, ind] = 1.0
                        
                        if np.shape(self.S)[0] != 1:
                            for key_n in self.traj:
                                
                                ind_n = self.traj_map[key_n]
                                if key_n != key and key_n != label:                            
                                    traj_n = self.traj[key_n]
                                    
                                    amp_n = self.qm_amplitudes[ind_n]
                                    amps[ind_n] = amp_n

                                    wf_1 = traj_1.td_wf_full_ts
                                    wf_1_T = np.transpose(np.conjugate(wf_1))
                                    
                                    wf_n = traj_n.td_wf_full_ts
                                    S_elec_1n = np.dot(wf_1_T, wf_n)
                                    S_nuc_1n = self.overlap_nuc(traj_1.positions,\
                                                                traj_n.positions,\
                                                                traj_1.momenta,\
                                                                traj_n.momenta_qm,\
                                                                traj_1.widths,\
                                                                traj_n.widths) 
                                    
                                    S[ind_1, ind_n] = S_elec_1n * S_nuc_1n
                                    S[ind_n, ind_1] = np.conjugate(S[ind_1, ind_n])
                                    
                                    
                                    wf_2 = traj_2.td_wf_full_ts
                                    wf_2_T = np.transpose(np.conjugate(wf_2))
                                    
                                    S_elec_2n = np.dot(wf_2_T, wf_n)
                                    S_nuc_2n = self.overlap_nuc(traj_2.positions,\
                                                                traj_n.positions,\
                                                                traj_2.momenta,\
                                                                traj_n.momenta_qm,\
                                                                traj_2.widths,\
                                                                traj_n.widths) 
                                    
                                    S[ind_2, ind_n] = S_elec_2n * S_nuc_2n
                                    S[ind_n, ind_2] = np.conjugate(S[ind_2, ind_n])
                                    
                                    # adding population from the overlap of cloning BFs
                                    # with noncloning
                                    pop_12n += np.dot(np.conjugate(new_amp_1),\
                                                      amps[ind_n])\
                                             * S[ind_1, ind_n]\
                                             + np.dot(np.conjugate(amps[ind_n]),\
                                                      new_amp_1)\
                                             * S[ind_n, ind_1]\
                                             + np.dot(np.conjugate(new_amp_2),\
                                                      amps[ind_n])\
                                             * S[ind_2, ind_n]\
                                             + np.dot(np.conjugate(amps[ind_n]),\
                                                      new_amp_2)\
                                             * S[ind_n, ind_2]
                                    
                                    # adding population from noncloning BFs, 
                                    # only diagonal contribution
                                    pop_n += np.dot(np.conjugate(amp_n), amp_n)
                                    
                                    for key_n2 in self.traj:
                                        
                                        if key_n2 != key\
                                        and key_n2 != label\
                                        and key_n2 != key_n:
                                            
                                            traj_n2 = self.traj[key_n2]
                                            ind_n2 = self.traj_map[key_n2]
                                            amp_n2 = self.qm_amplitudes[ind_n2]
                                            amps[ind_n2] = amp_n2                                                

                                            S[ind_n, ind_n2] = self.S[ind_n, ind_n2]
                                            S[ind_n2, ind_n] = self.S[ind_n2, ind_n]
                                            
                                            # adding population of noncloning BFs, 
                                            # only off-diagonal contributions
                                            pop_n += 1/4 * S[ind_n, ind_n2]\
                                                    * np.dot(np.conjugate(amps[ind_n]),\
                                                      amps[ind_n2])\
                                                    + S[ind_n2, ind_n]\
                                                    * np.dot(np.conjugate(amps[ind_n2]),\
                                                            amps[ind_n])
                                            
                        # Solving quadratic equation for the factor 
                        # that ensures norm conservation
                        x = (-pop_12n + np.sqrt(pop_12n**2\
                            - 4 * (pop_n - nuc_norm) * pop_12)) / (2 * pop_12) 
                        alpha = np.dot(np.conjugate(x), x)
                        if np.abs(alpha) < 1e-6: alpha = 1.0
                        print "alpha =", alpha
                        print "total_pop =", pop_n + x * pop_12n + pop_12 * alpha
                        print "total pop before rescale =", pop_n + pop_12n + pop_12
                        traj_1.new_amp = new_amp_1 * x
                        traj_2.new_amp = new_amp_2 * x
                        
                        # For debugging purposes
                        amps[ind_1] = traj_1.new_amp
                        amps[ind_2] = traj_2.new_amp
                        norm = np.dot(np.conjugate(np.transpose(amps)), np.dot(S, amps))
                        
                        print "NORM =", norm
                        
                        # Number overlap elements that are larger than threshold
                        s = (abs(S) > self.olapmax).sum()
#                         print "SUM =", s
#                         print "S =", np.abs(S[0,:])
                        if s > new_dim:
                            
                            print "Aborting cloning due to large overlap with existing trajectory"
                            self.clone_again = False
                            continue
                        
                        else:
                            # if overlap is less than threshold
                            # we can update the actual parent

                            self.traj[key] = traj_1
                        
                            print "Overlap OK, creating new trajectory ", label
                            clonetraj[label] = traj_2
                            self.traj[key].numchildren += 1
                            print "Cloning successful"
                            self.add_traj(clonetraj[label])
                            # update matrices here in case other trajectories clone
                            self.compute_num_traj_qm()
                            self.build_Heff_half_timestep()
                            self.clone_again = True
                            return
                 
                    else:
                        self.clone_again = False
                        continue
    
    def restart_from_file(self, json_file, h5_file):
        """restarts from the current json file and copies
        the simulation data into working.hdf5"""
        
        self.read_from_file(json_file)
        shutil.copy2(h5_file, "working.hdf5")
        
    def restart_output(self):
        """output json restart file
        The json file is meant to represent the *current* state of the
        simulation.  There is a separate hdf5 file that stores the history of
        the simulation.  Both are needed for restart."""

#         print "Creating new sim.json" 
#         we keep copies of the last 3 json files just to be safe
        extensions = [3, 2, 1, 0]
        for i in extensions :
            if i==0:
                ext = ""
            else:
                ext = str(i) + "."
            filename = "sim." + ext + "json"
            if os.path.isfile(filename):
                if (i == extensions[0]):
                    os.remove(filename)
                else:
                    ext = str(i+1) + "."
                    filename2 = "sim." + ext + "json"
                    if (i == extensions[-1]):
                        shutil.copy2(filename, filename2)
                    else:
                        shutil.move(filename, filename2)
                        
        # now we write the current json file
        self.write_to_file("sim.json")
        extensions = [3, 2, 1, 0]
        for i in extensions :
            if i == 0:
                ext = ""
            else:
                ext = str(i) + "."
            filename = "sim." + ext + "hdf5"
            if os.path.isfile(filename):
                if (i == extensions[0]):
                    os.remove(filename)
                else:
                    ext = str(i + 1) + "."
                    filename2 = "sim." + ext + "hdf5"
                    if (i == extensions[-1]):
                        shutil.copy2(filename, filename2)
                    else:
                        shutil.move(filename, filename2)
        shutil.copy2("working.hdf5", "sim.hdf5")
        print "hdf5 and json output are synchronized"

    def calc_approx_el_populations(self):
        """This calculates population on each electronic state taking into account all nuclear BFs
        First we calculate nuclear population (Mulliken) out of two terms that cancel imaginary part.
        Then we multiply nuclear population by electronic population"""
        
        n_el_states = self.traj["00"].numstates
        norm = np.zeros(n_el_states)

        for i in range(n_el_states):
            nt = np.shape(self.S)[0]
            c_t = self.qm_amplitudes
            S_t = self.S
            sum_pop = 0.0
            for key1 in self.traj:
                pop_ist = 0.0
                ist = self.traj_map[key1]
                for key2 in self.traj:
                    ist2 = self.traj_map[key2]
                    
                    pop_ist += np.real(0.5 * (np.dot(np.conjugate(c_t[ist]),\
                                       np.dot(S_t[ist, ist2], c_t[ist2]))\
                                    + np.dot(np.conjugate(c_t[ist2]),\
                                       np.dot(S_t[ist2, ist], c_t[ist])))) 
                
                norm[i] += pop_ist * self.traj[key1].populations[i]
                self.el_pop[i] = norm[i]
    
    def h5_output(self):
        """"Writes output  to h5 file"""
        
        self.init_h5_datasets()
        filename = "working.hdf5"
        h5f = h5py.File(filename, "a")
        groupname = "sim"
        if groupname not in h5f.keys():
            # creating sim group in hdf5 output file
            self.create_h5_sim(h5f, groupname)
            grp = h5f.get(groupname)
            self.create_new_h5_map(grp)
        else:
            grp = h5f.get(groupname)
        znewmap = False
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            dset = grp.get(key)
            l = dset.len()
            if l > 0:
                lwidth = dset.size / l
                if n > lwidth:
                    dset.resize(n, axis=1)
                    if not znewmap:
                        self.create_new_h5_map(grp)
                        znewmap = True
            dset.resize(l+1, axis=0)
            ipos=l
#             getcom = "self.get_" + key + "()"
            getcom = "self." + key
#             print getcom

            tmp = eval(getcom)
            if type(tmp).__module__ == np.__name__:
                tmp = np.ndarray.flatten(tmp)
                dset[ipos, 0:n] = tmp[0:n]
            else:
                dset[ipos, 0] = tmp
        h5f.flush()
        h5f.close()

    def create_new_h5_map(self, grp):
        ntraj = self.num_traj_qm
        labels = np.empty(ntraj, dtype="S512")
        for key in self.traj_map:
            if self.traj_map[key] < ntraj:
                labels[self.traj_map[key]] = key
        grp.attrs["labels"] = labels
        grp.attrs["olapmax"] = self.olapmax
        grp.attrs["p_threshold"] = self.p_threshold
        grp.attrs["pop_threshold"] = self.pop_threshold
        
    def create_h5_sim(self, h5f, groupname):
        trajgrp = h5f.create_group(groupname)
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            dset = trajgrp.create_dataset(key, (0,n), maxshape=(None, None),\
                                          dtype=self.h5_types[key])

    def init_h5_datasets(self):
        """Initialization of the h5 datasets within the simulation object"""
        
        ntraj = self.num_traj_qm
        ntraj2 = ntraj * ntraj
        
        self.h5_datasets = dict()
        self.h5_datasets["quantum_time"] = 1
        self.h5_datasets["qm_amplitudes"] = ntraj
        self.h5_datasets["Heff"] = ntraj2
        self.h5_datasets["H"] = ntraj2
        self.h5_datasets["S"] = ntraj2
        self.h5_datasets["Sdot"] = ntraj2
        self.h5_datasets["Sinv"] = ntraj2
        self.h5_datasets["num_traj_qm"] = 1
        self.h5_datasets["el_pop"] = self.num_el_states
        
        self.h5_types = dict()
        self.h5_types["quantum_time"] = "float64"
        self.h5_types["qm_amplitudes"] = "complex128"
        self.h5_types["Heff"] = "complex128"
        self.h5_types["H"] = "complex128"
        self.h5_types["S"] = "complex128"
        self.h5_types["Sdot"] = "complex128"
        self.h5_types["Sinv"] = "complex128"
        self.h5_types["num_traj_qm"] = "int32"
        self.h5_types["el_pop"] = "float64"

    def get_qm_data_from_h5(self):
        """get the necessary geometries and energies from hdf5"""
        
        qm_time = self.quantum_time
        ntraj = self.num_traj_qm
        for key in self.traj:
            if self.traj_map[key] < ntraj:
                self.traj[key].get_all_qm_data_at_time_from_h5(qm_time)
    
    def get_numtasks(self):
        """get the number of tasks in the queue"""
        
        return (len(self.queue)-1)

    def pop_task(self):
        """pop the task from the top of the queue"""
        
        return self.queue.pop(0)

    def update_queue(self):
        """build a list of all tasks that need to be completed"""
        while self.queue[0] != "END":
            self.queue.pop(0)
        tasktimes=[1e10]
        
        # forward propagation tasks
        for key in self.traj:
            if (self.traj[key].maxtime + 1.0e-6) > self.traj[key].time:
                task_tmp = "self.traj[\"" + key  + "\"].propagate_step()"
                tasktime_tmp = self.traj[key].time
                self.insert_task(task_tmp, tasktime_tmp, tasktimes)
                
        print (len(self.queue)-1), "task(s) in queue:"
        for i in range(len(self.queue)-1):
            print self.queue[i] + ", time = " + str(tasktimes[i])
        print ""

    def insert_task(self, task, tt, tasktimes):
        """add a task to the queue"""
        
        for i in range(len(tasktimes)):
            if tt < tasktimes[i]:
                self.queue.insert(i,task)
                tasktimes.insert(i,tt)
                return
            
    def overlap_nuc_1d(self, xi, xj, di, dj, xwi, xwj):
        """Compute 1-dimensional nuclear overlaps"""
        
        c1i = (complex(0.0, 1.0))
        deltax = xi - xj
        pdiff = di - dj
        osmwid = 1.0 / (xwi + xwj)
        
        xrarg = osmwid * (xwi*xwj*deltax*deltax + 0.25*pdiff*pdiff)
        if (xrarg < 10.0):
            gmwidth = math.sqrt(xwi*xwj)
            ctemp = (di*xi - dj*xj)
            ctemp = ctemp - osmwid * (xwi*xi + xwj*xj) * pdiff
            cgold = math.sqrt(2.0 * gmwidth * osmwid)
            cgold = cgold * math.exp(-1.0 * xrarg)
            cgold = cgold * cmath.exp(ctemp * c1i)
        else:
            cgold = 0.0
               
        return cgold
    
    def overlap_nuc(self, pos_i, pos_j, mom_i, mom_j, widths_i, widths_j):
        
        Sij = 1.0
        for idim in range(self.traj["00"].numdims):
            xi = pos_i[idim]
            xj = pos_j[idim]
            di = mom_i[idim]
            dj = mom_j[idim]
            xwi = widths_i[idim]
            xwj = widths_j[idim]
            Sij *= self.overlap_nuc_1d(xi, xj, di, dj, xwi, xwj)

        return Sij
