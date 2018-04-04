# simulation object contains the current state of the simulation.
# It is analagous to the "bundle" object in the original FMS code.
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

class simulation(fmsobj):
    def __init__(self):
        # traj is a dictionary of trajectory basis functions (TBFs)
        self.traj = dict()

        # centroids is a dictionary of TBFs representing the centroids
        # between the basis functions
        self.centroids = dict()

        # queue is a list of tasks to be run
        self.queue = ["END"]
        # tasktimes is a list of the simulation times associated with each task
        self.tasktimes = [1e10]

        # olapmax is the maximum overlap allowed for a spawn.  Above this,
        # the spawn is cancelled
        self.olapmax = 0.8

        # quantum time is the current time of the quantum amplitudes
        self.quantum_time = 0.0
        # quantum time is the current time of the quantum amplitudes
        self.quantum_time_half_step = 0.0
        # timestep for qunatum propagation
        self.timestep = 0.0
        # quantum propagator
        #self.qm_propagator = "RK2"
        # quantum hamiltonian
        #self.qm_hamiltonian = "adiabatic"

        # maps trajectories to matrix element indices
        self.traj_map = dict()

        # quantum amplitudes
        self.qm_amplitudes = np.zeros(0,dtype=np.complex128)

        # energy shift for quantum propagation
        self.qm_energy_shift = 0.0

        # variables to be output to hdf5 mapped to the size of each data point
        self.h5_datasets = dict()
        self.h5_types = dict()

        # maximium walltime in seconds
        self.max_quantum_time = -1.0

        # maximium walltime in seconds
        self.max_walltime = -1.0

    # convert dict to simulation data structure
    def from_dict(self,**tempdict):
        for key in tempdict:
            if isinstance(tempdict[key],types.UnicodeType) :
                tempdict[key] = str(tempdict[key])
            if isinstance(tempdict[key],types.ListType) :
                if isinstance((tempdict[key])[0],types.FloatType) :
                    # convert 1d float lists to np arrays
                    tempdict[key] = np.asarray(tempdict[key])
                if isinstance((tempdict[key])[0],types.StringTypes) :
                    if (tempdict[key])[0][0] == "^":
                        for i in range(len(tempdict[key])):
                            tempdict[key][i] = eval(tempdict[key][i][1:])
                        tempdict[key] = np.asarray(tempdict[key],dtype=np.complex128)
                else:
                    if isinstance((tempdict[key])[0],types.ListType):
                        if isinstance((tempdict[key])[0][0],types.FloatType) :
                            # convert 2d float lists to np arrays
                           tempdict[key] = np.asarray(tempdict[key])
                        if isinstance((tempdict[key])[0][0],types.StringTypes) :
                            if (tempdict[key])[0][0][0] == "^":
                                for i in range(len(tempdict[key])):
                                    for j in range(len(tempdict[key][i])):
                                        tempdict[key][i][j] = eval(tempdict[key][i][j][1:])
                                tempdict[key] = np.asarray(tempdict[key],dtype=np.complex128)
            if isinstance(tempdict[key],types.DictType) :
                if 'fmsobjlabel' in (tempdict[key]).keys():
                    fmsobjlabel = (tempdict[key]).pop('fmsobjlabel')
                    obj = eval(fmsobjlabel[8:])()
                    obj.from_dict(**(tempdict[key]))
                    tempdict[key] = obj
                else:
                    for key2 in tempdict[key]:
                        if isinstance((tempdict[key])[key2],types.DictType) :
                            fmsobjlabel = ((tempdict[key])[key2]).pop('fmsobjlabel')
                            obj = eval(fmsobjlabel[8:])()
                            obj.from_dict(**((tempdict[key])[key2]))
                            (tempdict[key])[key2] = obj
        self.__dict__.update(tempdict)

    # add a trajectory to the simulation
    def add_traj(self,t1):
        key = t1.get_label()
        mintime = t1.get_mintime()
        index = -1
        for key2 in self.traj:
            if mintime < self.traj[key2].get_mintime():
                if index < 0:
                    index = self.traj_map[key2]
                self.traj_map[key2] += 1
        if index < 0:
            index = len(self.traj)
        self.traj[key] = t1
        self.traj_map[key] = index
        #sort traj_map by mintime
        

    # get number of trajectories
    def get_num_traj(self):
        return len(self.traj)
        
    # add a task to the queue
    def add_task(self,task):
        self.queue.append(task)

    # get the number of tasks in the queue
    def get_numtasks(self):
        return (len(self.queue)-1)

    def set_olapmax(self,s):
        self.olapmax = s

    # set the timestep on all trajectories and centroids
    def set_timestep_all(self,h):
        self.timestep = h
        for key in self.traj:
            self.traj[key].set_timestep(h)
        for key in self.centroids:
            self.centroids[key].set_timestep(h)

    # set the maximimum simulation time on all trajectories and centroids
    def set_maxtime_all(self,maxtime):
        self.set_max_quantum_time(maxtime)
        h = self.get_timestep()
        for key in self.traj:
            self.traj[key].set_maxtime(maxtime+h)
        for key in self.centroids:
            self.centroids[key].set_maxtime(maxtime+h)

    # set the minimimum simulation time on all trajectories and centroids
    def set_mintime_all(self,mintime):
        for key in self.traj:
            self.traj[key].set_mintime(mintime)
        for key in self.centroids:
            self.centroids[key].set_mintime(mintime)

    # set the propagator on all trajectories
    def set_propagator_all(self,prop):
        for key in self.traj:
            self.traj[key].set_propagator(prop)

    def get_num_traj_qm(self):
        return self.num_traj_qm
            
    def set_num_traj_qm(self,n):
        self.num_traj_qm = n

    def get_quantum_time(self):
        return self.quantum_time
            
    def set_quantum_time(self,t):
        self.quantum_time = t

    def get_quantum_time_half_step(self):
        return self.quantum_time_half_step
            
    def set_quantum_time_half_step(self,t):
        self.quantum_time_half_step = t

    def get_timestep(self):
        return self.timestep
            
    def set_timestep(self,h):
        self.timestep = h

    def get_max_quantum_time(self):
        return self.max_quantum_time
            
    def set_max_quantum_time(self,t):
        self.max_quantum_time = t

    def get_max_walltime(self):
        return self.max_walltime
            
    def set_max_walltime(self,t):
        current_t = time.time()
        self.max_walltime = current_t + t
        print "### simulation will end after ", t, " seconds wall time"

    def set_max_walltime_formatted(self,s):
        pt = datetime.datetime.strptime(s,'%H:%M:%S')
        self.set_max_walltime(pt.second+pt.minute*60+pt.hour*3600)

    def get_qm_energy_shift(self):
        return self.qm_energy_shift

    def set_qm_energy_shift(self, e):
        self.qm_energy_shift = e

    def get_qm_amplitudes(self):
        return self.qm_amplitudes.copy()
            
    def set_qm_amplitudes(self,amp):
        #if amp.shape == self.qm_amplitudes.shape:
        self.qm_amplitudes = amp.copy()
        #else:
        #    print "! error in set_qm_amplitudes"
        #    sys.exit

    def get_H(self):
        return self.H.copy()
    
    def get_Heff(self):
        return self.Heff.copy()
    
    def get_S(self):
        return self.S.copy()
    
    def get_Sdot(self):
        return self.Sdot.copy()
    
    def get_Sinv(self):
        return self.Sinv.copy()
    

    # this is the main propagation loop for the simulation
    def propagate(self):
        gen.print_splash()
        while True:
            # compute centroid positions and mark those centroids that
            # can presently be computed
            print "### updating centroids"
            self.update_centroids()

            # update the queue (list of tasks to be computed)
            print "### updating task queue"
            self.update_queue()

            # if the queue is empty, we're done!
            print "### checking if we are at the end of the simulation"
            #if (self.queue[0] == "END"):
            if (self.get_quantum_time() + 1.0e-6 > self.get_max_quantum_time()):
                print "### propagate DONE, simulation ended gracefully!"
                return

            # end simulation if walltime has expired
            print "### checking if maximum wall time is reached"
            if (self.get_max_walltime() < time.time() and self.get_max_walltime() > 0):
                print "### wall time expired, simulation ended gracefully!"
                return
            
            # it is possible for the queue to run empty but for the job not to be done
            if (self.queue[0] != "END"):            
                # Right now we just run a single task per cycle,
                # but we could parallelize here and send multiple tasks
                # out for simultaneous processing.
                current = self.pop_task()
                print "### starting " + current            
                eval(current)
                print "### done with " + current
            else:
                print "### task queue is empty"

            # spawn new trajectories if needed
            print "### now we will spawn new trajectories if necessary"
            self.spawn_as_necessary()
            
            # propagate quantum variables if possible
            print "### propagating quantum amplitudes if we have enough information to do so"
            self.propagate_quantum_as_necessary()
            
            # print restart output - this must be the last line in this loop!
            print "### updating restart output"
            self.restart_output()

    # here we will propagate the quantum amplitudes if we have
    # the necessary information to do so
    def propagate_quantum_as_necessary(self):
        # we have to determine what the maximum time is for which
        # we have all the necessary information to propogate the amplitudes
        max_info_time = 1.0e10
        # first check centroids
        for key in self.traj:
            print "traj key", key
            # if a trajectory is spawning, we can only propagate to the
            # spawntime
            timestep = self.traj[key].get_timestep()
            spawntimes = self.traj[key].get_spawntimes()
            for i in range(len(spawntimes)):
                if (spawntimes[i] - timestep) < max_info_time and spawntimes[i] > 0.0 :
                    max_info_time = spawntimes[i] - timestep
                    print "i spawntimes[i] max_info_time", i, spawntimes[i], max_info_time
            # if a trajectory is backpropagating, we can only propagate to
            # its mintime
            mintime = self.traj[key].get_mintime()
            print "mintime, backproptime", mintime, self.traj[key].get_backprop_time()
            if (mintime + 1.0e-6) < self.traj[key].get_backprop_time():
                if (mintime - timestep) < max_info_time:
                    max_info_time = mintime - timestep
                    print "mintime max_info_time", mintime, max_info_time
            # if a trajectory is neither spawning nor backpropagating, we can
            # only propagate to its current forward propagation time
            time = self.traj[key].get_time()
            if (time - timestep) < max_info_time:
                max_info_time = time - timestep
                print "time max_info_time", time, max_info_time
        # now centroids
        for key in self.centroids:
            print "centroid key", key
            # if a centroid is backpropagating, we can only propagate to
            # its mintime
            timestep = self.centroids[key].get_timestep()
            mintime = self.centroids[key].get_mintime()
            print "mintime, backprop_time", mintime, self.centroids[key].get_backprop_time()
            if (mintime + 1.0e-6) < self.centroids[key].get_backprop_time():
                if (mintime - timestep) < max_info_time:
                    max_info_time = mintime - timestep
                    print "mintime, max_info_time", mintime, max_info_time
            # if a centroid is not backpropagating, we can
            # only propagate to its current forward propagation time
            time = self.centroids[key].get_time()
            if (time - timestep) < max_info_time:
                # we subtract two timesteps because the spawning procedure
                # can take is back in time in a subsequent step
                max_info_time = time - timestep
                print "time max_info_time", time, max_info_time

        print "## we have enough information to propagate to time ", max_info_time

        # now, if we have the necessary info, we propagate
        while max_info_time > (self.get_quantum_time() + 1.0e-6):
            if self.get_quantum_time() > 1.0e-6:
                print "## propagating quantum amplitudes at time", self.get_quantum_time()
                self.qm_propagate_step()
            else:
                print "## propagating quantum amplitudes at time", self.get_quantum_time(), " (first step)"
                self.qm_propagate_step(zoutput_first_step=True)
                
            print "## outputing quantum information to hdf5"
            self.h5_output()

    # sets the first amplitude to 1.0 and all others to zero
    def init_amplitudes_one(self):
        self.compute_num_traj_qm()
        self.qm_amplitudes = np.zeros_like(self.qm_amplitudes,dtype=np.complex128)
        self.qm_amplitudes[0] = 1.0
        
    def compute_num_traj_qm(self):
        n = 0
        qm_time = self.get_quantum_time()
        for key in self.traj:
            if qm_time > (self.traj[key].get_mintime() - 1.0e-6):
                n += 1
        self.set_num_traj_qm(n)
        while n > len(self.get_qm_amplitudes()):
            self.qm_amplitudes = np.append(self.qm_amplitudes,0.0)
                
        
    # get the necessary geometries and energies from hdf5
    def get_qm_data_from_h5(self):
        qm_time = self.get_quantum_time()
        ntraj = self.get_num_traj_qm()
        for key in self.traj:
            if self.traj_map[key] < ntraj:
                self.traj[key].get_all_qm_data_at_time_from_h5(qm_time)
        for key in self.centroids:
            key1, key2 = str.split(key,"_a_")
            if self.traj_map[key1] < ntraj and self.traj_map[key2] < ntraj:
                self.centroids[key].get_all_qm_data_at_time_from_h5(qm_time)
            
    def get_qm_data_from_h5_half_step(self):
        qm_time = self.get_quantum_time_half_step()
        ntraj = self.get_num_traj_qm()
        for key in self.traj:
            if self.traj_map[key] < ntraj:
                self.traj[key].get_all_qm_data_at_time_from_h5_half_step(qm_time)
        for key in self.centroids:
            key1, key2 = str.split(key,"_a_")
            if self.traj_map[key1] < ntraj and self.traj_map[key2] < ntraj:
                self.centroids[key].get_all_qm_data_at_time_from_h5_half_step(qm_time)
                        
    # build the overlap matrix, S
    def build_S(self):
        ntraj = self.get_num_traj_qm()
        self.S = np.zeros((ntraj,ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.S[i,j] = cg.overlap_nuc_elec(self.traj[keyi], self.traj[keyj],positions_i="positions_qm",positions_j="positions_qm",momenta_i="momenta_qm",momenta_j="momenta_qm")

    # build the right-acting time derivative operator
    def build_Sdot(self):
        ntraj = self.get_num_traj_qm()
        self.Sdot = np.zeros((ntraj,ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.Sdot[i,j] = cg.Sdot_nuc_elec(self.traj[keyi], self.traj[keyj],positions_i="positions_qm",positions_j="positions_qm",momenta_i="momenta_qm",momenta_j="momenta_qm",forces_j="forces_i_qm")

    # compute Sinv from S
    def invert_S(self):
        self.Sinv = np.linalg.inv(self.S)
        
    # build the Hamiltonian matrix, H
    # This routine assumes that S is already built
    def build_H(self):
        print "# building potential energy matrix"
        self.build_V()
        print "# building NAC matrix"
        self.build_tau()
        print "# building kinetic energy matrix"
        self.build_T()
        ntraj = self.get_num_traj_qm()
        shift = self.get_qm_energy_shift() * np.identity(ntraj)
        print "# summing Hamiltonian"
        self.H = self.T + self.V + self.tau + shift

    # build the potential energy matrix, V
    # This routine assumes that S is already built
    def build_V(self):
        c1i = (complex(0.0,1.0))
        cm1i = (complex(0.0,-1.0))
        ntraj = self.get_num_traj_qm()
        self.V = np.zeros((ntraj,ntraj),dtype=np.complex128)
        for key in self.traj:
            i = self.traj_map[key]
            istate = self.traj[key].get_istate()
            if i < ntraj:
                self.V[i,i] = self.traj[key].get_energies_qm()[istate]
        for key in self.centroids:
            keyi, keyj = str.split(key,"_a_")
            i = self.traj_map[keyi]
            j = self.traj_map[keyj]
            if i < ntraj and j < ntraj:
                istate = self.centroids[key].get_istate()
                jstate = self.centroids[key].get_jstate()
                if istate == jstate:
                    E = self.centroids[key].get_energies_qm()[istate]
                    self.V[i,j] = self.S[i,j] * E
                    self.V[j,i] = self.S[j,i] * E

                
    # build the nonadiabatic coupling matrix, tau
    # This routine assumes that S is already built
    def build_tau(self):
        c1i = (complex(0.0,1.0))
        cm1i = (complex(0.0,-1.0))
        ntraj = self.get_num_traj_qm()
        self.tau = np.zeros((ntraj,ntraj),dtype=np.complex128)
        for key in self.centroids:
            keyi, keyj = str.split(key,"_a_")
            i = self.traj_map[keyi]
            j = self.traj_map[keyj]
            if i < ntraj and j < ntraj:
                istate = self.centroids[key].get_istate()
                jstate = self.centroids[key].get_jstate()
                if istate != jstate:
                    Sij = cg.overlap_nuc(self.traj[keyi], self.traj[keyj],positions_i="positions_qm",positions_j="positions_qm",momenta_i="momenta_qm",momenta_j="momenta_qm")
                    tdc = self.centroids[key].get_timederivcoups_qm()[jstate]
                    self.tau[i,j] = Sij * cm1i * tdc
                    self.tau[j,i] = Sij.conjugate() * c1i * tdc

                
    # build the kinetic energy matrix, T
    def build_T(self):
        ntraj = self.get_num_traj_qm()
        self.T = np.zeros((ntraj,ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.T[i,j] = cg.kinetic_nuc_elec(self.traj[keyi], self.traj[keyj],positions_i="positions_qm",positions_j="positions_qm",momenta_i="momenta_qm",momenta_j="momenta_qm")

    # built Heff form H, Sinv, and Sdot
    def build_Heff(self):
        print "# building effective Hamiltonian"
        c1i = (complex(0.0,1.0))
        self.Heff = np.matmul(self.Sinv, (self.H - c1i * self.Sdot))
        
    # pop the task from the top of the queue
    def pop_task(self):
        return self.queue.pop(0)

    # build a list of all tasks that need to be completed
    def update_queue(self):
        while self.queue[0] != "END":
            self.queue.pop(0)
        tasktimes=[1e10]
        
        # forward propagation tasks
        for key in self.traj:
            if (self.traj[key].get_maxtime()+1.0e-6) > self.traj[key].get_time():
                task_tmp = "self.traj[\"" + key  + "\"].propagate_step()"
                tasktime_tmp = self.traj[key].get_time()
                self.insert_task(task_tmp,tasktime_tmp, tasktimes)
                
        # backward propagation tasks
        for key in self.traj:
            if (self.traj[key].get_mintime()+1.0e-6) < self.traj[key].get_backprop_time():
                task_tmp = "self.traj[\"" + key  + "\"].propagate_step(zbackprop=True)"
                tasktime_tmp = self.traj[key].get_backprop_time()
                self.insert_task(task_tmp,tasktime_tmp, tasktimes)
                
        # centroid tasks (forward propagation)
        for key in self.centroids:
            if self.centroids[key].get_z_compute_me():
                task_tmp = "self.centroids[\"" + key  + "\"].compute_centroid()"
                tasktime_tmp = self.centroids[key].get_time()
                self.insert_task(task_tmp,tasktime_tmp, tasktimes)
                
        # centroid tasks (backward propagation)
        for key in self.centroids:
            if self.centroids[key].get_z_compute_me_backprop():
                task_tmp = "self.centroids[\"" + key  + "\"].compute_centroid(zbackprop=True)"
                tasktime_tmp = self.centroids[key].get_backprop_time()
                self.insert_task(task_tmp,tasktime_tmp, tasktimes)
                
        print "##", (len(self.queue)-1), "task(s) in queue:"
        for i in range(len(self.queue)-1):
            print self.queue[i] + ", time = " + str(tasktimes[i])
        print "END"

    # add a task to the queue
    def insert_task(self,task,tt,tasktimes):
        for i in range(len(tasktimes)):
            if tt < tasktimes[i]:
                self.queue.insert(i,task)
                tasktimes.insert(i,tt)
                return

    # compute the centroid positions and moment and check which centroids
    # can be computed
    def update_centroids(self):
        for key in self.centroids:
            key1, key2 = str.split(key,"_a_")
            timestep = self.centroids[key].get_timestep()

            #update backpropagating centroids
            self.centroids[key].set_z_compute_me_backprop(False)
            backprop_time = self.centroids[key].get_backprop_time() - timestep
            if (self.centroids[key].get_mintime()-1.0e-6) < backprop_time:
                backprop_time1 = self.traj[key1].get_backprop_time()
                if (backprop_time > backprop_time1 - 1.0e-6) and (backprop_time1  < (self.traj[key1].get_firsttime() - 1.0e-6) or backprop_time1  < (self.traj[key1].get_mintime() + 1.0e-6)):
                    backprop_time2 = self.traj[key2].get_backprop_time()
                    if (backprop_time > backprop_time2 - 1.0e-6) and (backprop_time2  < (self.traj[key2].get_firsttime() - 1.0e-6) or backprop_time2  < (self.traj[key2].get_mintime() + 1.0e-6)):
                        time1 = self.traj[key1].get_time()
                        time2 = self.traj[key2].get_time()
                        # this if takes care of the special case where we try to compute the backpropagating centroid at firsttime before
                        # forward propagation has begun
                        if (backprop_time + 1.0e-6 < time1) and (backprop_time + 1.0e-6 < time2):
                            pos1 = self.traj[key1].get_data_at_time_from_h5(backprop_time, "positions")
                            mom1 = self.traj[key1].get_data_at_time_from_h5(backprop_time, "momenta")
                            #if backprop_time2  < (self.traj[key2].get_mintime() + 1.0e-6):
                            #    pos2 = self.traj[key2].get_backprop_positions()
                            #    mom2 = self.traj[key2].get_backprop_momenta()
                            #else:
                            pos2 = self.traj[key2].get_data_at_time_from_h5(backprop_time, "positions")
                            mom2 = self.traj[key2].get_data_at_time_from_h5(backprop_time, "momenta")
                            #pos2, mom2 = self.traj[key2].get_q_and_p_at_time_from_h5(backprop_time)
                            absSij = abs(cg.overlap_nuc(self.traj[key1],self.traj[key2], positions_i=pos1, positions_j=pos2, momenta_i=mom1, momenta_j=mom2))
                            #print "absSij", absSij
                            # this definition of mom is only right if all basis functions have same
                            # width!!!!  I don't think the momentum is every used but still we 
                            # shoudl fix this soon.
                            width1 = self.traj[key1].get_widths()
                            width2 = self.traj[key2].get_widths()
                            pos_cent = ( width1 * pos1 + width2 * pos2 ) / (width1 + width2)
                            mom_cent = 0.5 * ( mom1 + mom2 )
                            self.centroids[key].set_backprop_positions(pos_cent)
                            self.centroids[key].set_backprop_momenta(mom_cent)
                            if absSij > 0.001:
                                self.centroids[key].set_z_compute_me_backprop(True)
                            else:
                                self.centroids[key].set_backprop_time(backprop_time)
                                dt = self.centroids[key].get_timestep()
                                self.centroids[key].set_backprop_time_half_step(backprop_time + 0.5 * dt)
                                self.centroids[key].set_backprop_energies(np.zeros(self.centroids[key].get_numstates()))
                                self.centroids[key].set_backprop_timederivcoups(np.zeros(self.centroids[key].get_numstates()))
                                firsttime = self.centroids[key].get_firsttime()
                                if abs(backprop_time - firsttime) > 1.0e-6:
                                    self.centroids[key].h5_output(True)

            # update forward propagating centroids
            self.centroids[key].set_z_compute_me(False)
            time = self.centroids[key].get_time() + timestep
            if (self.centroids[key].get_maxtime()+timestep+1.0e-6) > time:
                time1 = self.traj[key1].get_time()
                if (time < time1 + 1.0e-6) and time1 > self.traj[key1].get_firsttime() + 1.0e-6:
                    time2 = self.traj[key2].get_time()
                    if (time < time2 + 1.0e-6) and time2 > self.traj[key2].get_firsttime() + 1.0e-6:
                        pos1 = self.traj[key1].get_data_at_time_from_h5(time, "positions")
                        mom1 = self.traj[key1].get_data_at_time_from_h5(time, "momenta")
                        pos2 = self.traj[key2].get_data_at_time_from_h5(time, "positions")
                        mom2 = self.traj[key2].get_data_at_time_from_h5(time, "momenta")
                        #pos1, mom1 = self.traj[key1].get_q_and_p_at_time_from_h5(time)
                        #pos2, mom2 = self.traj[key2].get_q_and_p_at_time_from_h5(time)
                        absSij = abs(cg.overlap_nuc(self.traj[key1],self.traj[key2], positions_i=pos1, positions_j=pos2, momenta_i=mom1, momenta_j=mom2))
                        #print "absSij", absSij
                        # this definion of mom is only correct if all basis functions have same
                        # width!!!!  I don't think that the centroid momentum is ever used, but
                        # we should still fix this soon
                        width1 = self.traj[key1].get_widths()
                        width2 = self.traj[key2].get_widths()
                        pos_cent = ( width1 * pos1 + width2 * pos2 ) / ( width1 + width2)
                        mom_cent = 0.5 * ( mom1 + mom2 )
                        self.centroids[key].set_positions(pos_cent)
                        self.centroids[key].set_momenta(mom_cent)
                        if absSij > 0.001:
                            self.centroids[key].set_z_compute_me(True)
                        else:
                            self.centroids[key].set_time(time)
                            dt = self.centroids[key].get_timestep()
                            self.centroids[key].set_time_half_step(time - 0.5 * dt)
                            self.centroids[key].set_energies(np.zeros(self.centroids[key].get_numstates()))
                            self.centroids[key].set_timederivcoups(np.zeros(self.centroids[key].get_numstates()))
                            firsttime = self.centroids[key].get_firsttime()
                            if abs(time - firsttime) > 1.0e-6:
                                self.centroids[key].h5_output(False)
                            else:
                                self.centroids[key].h5_output(False,zdont_half_step=True)

    # this is the spawning routine
    def spawn_as_necessary(self):
        spawntraj = dict()
        for key in self.traj:
            # trajectories that are spawning or should start were marked
            # during propagation.  See "propagate_step" and "consider_spawning"
            # in traj.py
            z = self.traj[key].get_z_spawn_now()
            z_dont = self.traj[key].get_z_dont_spawn()
            spawnt = self.traj[key].get_spawntimes()
            for jstate in range(self.traj[key].get_numstates()):
                # is this trajectory marked to spawn to state j?
                if z[jstate] > 0.5:
                    # create label that indicates parentage
                    # for example:
                    # a trajectory labeled 00b1b5 means that the initial
                    # trajectory "00" spawned a trajecory "1" (its
                    # second child) which then spawned another (it's 6th child)
                    label = str(self.traj[key].get_label() + "b" + str(self.traj[key].get_numchildren()))

                    # create and initiate new trajectpory structure
                    newtraj = traj()
                    newtraj.init_spawn_traj(self.traj[key], jstate, label)

                    # checking to see if overlap with existing trajectories
                    # is too high.  If so, we abort spawn
                    z_add_traj_olap = self.check_overlap(newtraj)

                    # rescaling velocity.  We'll abort if there is not
                    # enough energy (aka a "frustrated spawn")
                    z_add_traj_rescale = newtraj.rescale_momentum(self.traj[key].get_energies_tmdt()[self.traj[key].get_istate()])

                    # okay, now we finally decide whether to spawn or not
                    if z_add_traj_olap and z_add_traj_rescale:
                        print "## creating new trajectory ", label
                        spawntraj[label] = newtraj
                        self.traj[key].incr_numchildren()

                    # whether we spawn or not, we reset the trajectory so
                    # that:
                    # it isn't slated to spawn
                    z[jstate] = 0.0
                    # it shouldn't spawn again until the coupling drops
                    # below a threshold
                    z_dont[jstate] = 1.0
                    # it isn't currently spawning
                    spawnt[jstate] = -1.0

            # once all states have been checked, we update the TBF structure
            self.traj[key].set_z_spawn_now(z)
            self.traj[key].set_z_dont_spawn(z_dont)
            self.traj[key].set_spawntimes(spawnt)

        # okay, now it's time to add the spawned trajectories
        for label in spawntraj:
            # create new centroid structures
            for key2 in self.traj:
                # "_a_" marks the centroid labels
                centkey = str(key2 + "_a_" + label)

                # create and initiate the trajectory structures!
                newcent = traj()
                newcent.init_centroid(self.traj[key2],spawntraj[label], centkey)

                # add the centroid
                self.centroids[centkey] = newcent
                print "# adding centroid ", centkey

            # finally, add the spawned trajectory
            self.add_traj(spawntraj[label])

    # check to make sure that a spawned trajectory doesn't overlap too much
    # with any existing trajectory
    def check_overlap(self,newtraj):
        z_add_traj=True
        for key2 in self.traj:
            # compute the overlap
            overlap = cg.overlap_nuc_elec(newtraj,self.traj[key2],positions_j="positions_tmdt",momenta_j="momenta_tmdt")

            # if the overlap is too high, don't spawn!
            if np.absolute(overlap) > self.olapmax:
                z_add_traj=False

            # let the user know what happened
            if not z_add_traj:
                print "# aborting spawn due to large overlap with existing trajectory"
        return z_add_traj

    # restarts from the current json file and copies the simulation data into working.hdf5
    def restart_from_file(self,json_file,h5_file):
        self.read_from_file(json_file)
        shutil.copy2(h5_file,"working.hdf5")
        
    # output json restart file
    # The json file is meant to represent the *current* state of the
    # simulation.  There is a separate hdf5 file that stores the history of
    # the simulation.  Both are needed for restart.
    def restart_output(self):
        print "## creating new sim.json" 
        # we keep copies of the last 3 json files just to be safe
        extensions = [3,2,1,0]
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
        print "## synchronizing sim.hdf5"
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
        shutil.copy2("working.hdf5", "sim.hdf5")
        print "## hdf5 and json output are synchronized"
        

    def h5_output(self):
        self.init_h5_datasets()
        filename = "working.hdf5"
        #extensions = [3,2,1,0]
        #for i in extensions :
        #    if i==0:
        #        ext = ""
        #    else:
        #        ext = str(i) + "."
        #    filename = "sim." + ext + "hdf5"
        #    if os.path.isfile(filename):
        #        if (i == extensions[0]):
        #            os.remove(filename)
        #        else:
        #            ext = str(i+1) + "."
        #            filename2 = "sim." + ext + "hdf5"
        #            if (i == extensions[-1]):
        #                shutil.copy2(filename, filename2)
        #            else:
        #                shutil.move(filename, filename2)
        h5f = h5py.File(filename, "a")
        groupname = "sim"
        if groupname not in h5f.keys():
            # creating sim group in hdf5 output file
            self.create_h5_sim(h5f,groupname)
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
                    dset.resize(n,axis=1)
                    if not znewmap:
                        self.create_new_h5_map(grp)
                        znewmap = True
            dset.resize(l+1,axis=0)
            ipos=l
            getcom = "self.get_" + key + "()"
            print getcom
            tmp = eval(getcom)
            if type(tmp).__module__ == np.__name__:
                tmp = np.ndarray.flatten(tmp)
                dset[ipos,0:n] = tmp[0:n]
            else:
                dset[ipos,0] = tmp
        h5f.flush()
        h5f.close()

    def create_new_h5_map(self,grp):
        ntraj = self.get_num_traj_qm()
        labels = np.empty(ntraj,dtype="S512")
        istates = np.zeros(ntraj,dtype=np.int32)
        for key in self.traj_map:
            if self.traj_map[key] < ntraj:
                labels[self.traj_map[key]] = key
                istates[self.traj_map[key]] = self.traj[key].get_istate()
        grp.attrs["labels"] = labels
        grp.attrs["istates"] = istates
        
    def create_h5_sim(self, h5f, groupname):
        trajgrp = h5f.create_group(groupname)
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            dset = trajgrp.create_dataset(key, (0,n), maxshape=(None,None), dtype=self.h5_types[key])

    def init_h5_datasets(self):
        ntraj = self.get_num_traj_qm()
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
        self.h5_types = dict()
        self.h5_types["quantum_time"] = "float64"
        self.h5_types["qm_amplitudes"] = "complex128"
        self.h5_types["Heff"] = "complex128"
        self.h5_types["H"] = "complex128"
        self.h5_types["S"] = "complex128"
        self.h5_types["Sdot"] = "complex128"
        self.h5_types["Sinv"] = "complex128"
        self.h5_types["num_traj_qm"] = "int32"
        
        
