import types
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
    """Simulation object contains the current state of the simulation.
    It is analagous to the "bundle" object in the original FMS code"""

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
        # timestep for quantum propagation
        self.timestep = 0.0
        # quantum propagator
#         self.qm_propagator = "RK2"
        # quantum hamiltonian
#         self.qm_hamiltonian = "adiabatic"

        # maps trajectories to matrix element indices
        self.traj_map = dict()

        # dictionary with istates for each trajectory, needed for analysis
        self.istates_dict = dict()

        # quantum amplitudes
        self.qm_amplitudes = np.zeros(0, dtype=np.complex128)

        # energy shift for quantum propagation
        self.qm_energy_shift = 0.0

        # variables to be output to hdf5 mapped to the size of each data point
        self.h5_datasets = dict()
        self.h5_types = dict()

        # maximium quantum walltime in seconds
        self.max_quantum_time = -1.0

        # maximium walltime in seconds
        self.max_walltime = -1.0

        # Stochastic-Selection AIMS controls (Added by A. Mehmood 10/02/2025)
        self.ssa_enabled = False             # master switch
        self.ssa_epsilon = 0.0               # |H_kl| threshold (a.u.) for connectivity
        self.ssa_seed = None                 # Random seed
        self.ssa_suspend_during_spawn = True # suspend selection while spawning
        self.ssa_spawn_delay_steps = 0       # wait this many quantum steps after each spawn
        self.ssa_min_tbf_to_start = 1        # minimum No. of TBFs before SSAIMS may act
        self.ssa_has_started = False         # once started, stays started        
        self.ssa_prev_ntraj = None           # internal trackers for the gates
        self.ssa_steps_since_spawn = 1e9     # large so first call isn't blocked unless a spawn occurs

    def from_dict(self, **tempdict):
        """Convert dict to simulation data structure"""

        for key in tempdict:
            if isinstance(tempdict[key], types.UnicodeType):
                tempdict[key] = str(tempdict[key])
            if isinstance(tempdict[key], types.ListType):
                if isinstance((tempdict[key])[0], types.FloatType):
                    # convert 1d float lists to np arrays
                    tempdict[key] = np.asarray(tempdict[key])
                if isinstance((tempdict[key])[0], types.StringTypes):
                    if (tempdict[key])[0][0] == "^":
                        for i in range(len(tempdict[key])):
                            tempdict[key][i] = eval(tempdict[key][i][1:])
                        tempdict[key] = np.asarray(tempdict[key],
                                                   dtype=np.complex128)
                else:
                    if isinstance((tempdict[key])[0], types.ListType):
                        if isinstance((tempdict[key])[0][0], types.FloatType):
                            # convert 2d float lists to np arrays
                            tempdict[key] = np.asarray(tempdict[key])
                        if isinstance((tempdict[key])[0][0],
                                      types.StringTypes):
                            if (tempdict[key])[0][0][0] == "^":
                                for i in range(len(tempdict[key])):
                                    for j in range(len(tempdict[key][i])):
                                        tempdict[key][i][j] = eval(tempdict[key][i][j][1:])
                                tempdict[key] = np.asarray(tempdict[key],
                                                           dtype=np.complex128)
            if isinstance(tempdict[key], types.DictType) :
                if 'fmsobjlabel' in (tempdict[key]).keys():
                    fmsobjlabel = (tempdict[key]).pop('fmsobjlabel')
                    obj = eval(fmsobjlabel[8:])()
                    obj.from_dict(**(tempdict[key]))
                    tempdict[key] = obj
                else:
                    for key2 in tempdict[key]:
                        if isinstance((tempdict[key])[key2],
                                      types.DictType):
                            if key == 'traj' or key == "centroids":
                                # This is a hack that fixes the previous hack lol
                                # initially trajectory's init didn't have numstates
                                # and numdims which caused certain issues
                                # so I'm adding the variables for traj initialization
                                # to make restart work
                                numdims = tempdict[key][key2]['numdims']
                                numstates = tempdict[key][key2]['numstates']
                                fmsobjlabel = ((tempdict[key])[key2]).pop('fmsobjlabel')
                                obj = eval(fmsobjlabel[8:])(numdims, numstates)
                                obj.from_dict(**((tempdict[key])[key2]))
                                (tempdict[key])[key2] = obj
                            else:
                                fmsobjlabel = ((tempdict[key])[key2]).pop('fmsobjlabel')
                                obj = eval(fmsobjlabel[8:])()
                                obj.from_dict(**((tempdict[key])[key2]))
                                (tempdict[key])[key2] = obj
        self.__dict__.update(tempdict)

    def add_traj(self, t1):
        """Add a trajectory to the simulation"""

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
        # sort traj_map by mintime

    def get_num_traj(self):
        """Get number of trajectories"""
        return len(self.traj)

    def add_task(self, task):
        """Add a task to the queue"""
        self.queue.append(task)

    def get_numtasks(self):
        """Get the number of tasks in the queue"""
        return (len(self.queue)-1)

    def set_olapmax(self, s):
        self.olapmax = s

    def set_timestep_all(self, h):
        """Set the timestep on all trajectories and centroids"""

        self.timestep = h
        for key in self.traj:
            self.traj[key].set_timestep(h)
        for key in self.centroids:
            self.centroids[key].set_timestep(h)

    def set_maxtime_all(self, maxtime):
        """Set the maximimum simulation time on all trajectories
        and centroids"""

        self.set_max_quantum_time(maxtime)
        h = self.get_timestep()
        for key in self.traj:
            self.traj[key].set_maxtime(maxtime+h)
        for key in self.centroids:
            self.centroids[key].set_maxtime(maxtime+h)

    def set_mintime_all(self, mintime):
        """Set the minimimum simulation time on all trajectories
        and centroids"""

        for key in self.traj:
            self.traj[key].set_mintime(mintime)
        for key in self.centroids:
            self.centroids[key].set_mintime(mintime)

    def set_propagator_all(self, prop):
        """Set the propagator on all trajectories"""

        for key in self.traj:
            self.traj[key].set_propagator(prop)

    def get_num_traj_qm(self):
        """Return number of trajectories"""
        return self.num_traj_qm

    def set_num_traj_qm(self, n):
        """Set number of trajectories"""
        self.num_traj_qm = n

    def get_quantum_time(self):
        """Return quantum time"""
        return self.quantum_time

    def set_quantum_time(self, t):
        """Set quantum time (for the whole simulation)"""
        self.quantum_time = t

    def get_quantum_time_half_step(self):
        """Return quantum time on half time step"""
        return self.quantum_time_half_step

    def set_quantum_time_half_step(self, t):
        """Get quantum time on half time step"""
        self.quantum_time_half_step = t

    def get_timestep(self):
        """Return timestep"""
        return self.timestep

    def set_timestep(self, h):
        """Set quantum timestep"""
        self.timestep = h

    def get_max_quantum_time(self):
        """Get max quantum time"""
        return self.max_quantum_time

    def set_max_quantum_time(self, t):
        """Set max quantum time"""
        self.max_quantum_time = t

    def get_max_walltime(self):
        """Return max walltime"""
        return self.max_walltime

    def set_max_walltime(self, t):
        """Set max walltime"""

        current_t = time.time()
        self.max_walltime = current_t + t
        print "### simulation will end after ", t, " seconds wall time"

    def set_max_walltime_formatted(self, s):
        """Formatted walltime"""

        pt = datetime.datetime.strptime(s, '%H:%M:%S')
        self.set_max_walltime(pt.second + pt.minute*60 + pt.hour*3600)

    def get_qm_energy_shift(self):
        """Return energy shift"""
        return self.qm_energy_shift

    def set_qm_energy_shift(self, e):
        """Set energy shift"""
        self.qm_energy_shift = e

    def get_qm_amplitudes(self):
        """Return QM amplitudes"""
        return self.qm_amplitudes.copy()

    def set_qm_amplitudes(self, amp):
        """Set QM amplitudes"""

#         if amp.shape == self.qm_amplitudes.shape:
        self.qm_amplitudes = amp.copy()
#         else:
#             print "! error in set_qm_amplitudes"
#             sys.exit

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

    def propagate(self):
        """This is the main propagation loop for the simulation"""

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
#             if (self.queue[0] == "END"):
            if (self.get_quantum_time() + 1.0e-6 > self.get_max_quantum_time()):
                print "### propagate DONE, simulation ended gracefully!"
                print "Removing working.hdf5, sim.1.hdf5 and sim.1.json files"
                os.remove('working.hdf5')
                os.remove('sim.1.hdf5')
                os.remove('sim.1.json')
                return

            # end simulation if walltime has expired
            print "### checking if maximum wall time is reached"
            if (self.get_max_walltime() < time.time() and self.get_max_walltime() > 0):
                print "### wall time expired, simulation ended gracefully!"
                return

            # it is possible for the queue to run empty but for the job not
            # to be done
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

    def propagate_quantum_as_necessary(self):
        """Here we will propagate the quantum amplitudes if we have
        the necessary information to do so.
        we have to determine what the maximum time is for which
        we have all the necessary information to propagate the amplitudes"""

        max_info_time = 1.0e10
        # first check trajectories
        for key in self.traj:
            # if a trajectory is spawning, we can only propagate to the
            # spawntime
            timestep = self.traj[key].get_timestep()
            spawntimes = self.traj[key].get_spawntimes()
            for i in range(len(spawntimes)):
                if (spawntimes[i] - timestep) < max_info_time\
                        and spawntimes[i] > 0.0:
                    max_info_time = spawntimes[i] - timestep
#                     print "i spawntimes[i] max_info_time", i, spawntimes[i], max_info_time
            # if a trajectory is backpropagating, we can only propagate to
            # its mintime
            mintime = self.traj[key].get_mintime()
#             print "mintime, backproptime", mintime, self.traj[key].get_backprop_time()
            if (mintime + 1.0e-6) < self.traj[key].get_backprop_time():
                if (mintime - timestep) < max_info_time:
                    max_info_time = mintime - timestep
#                     print "mintime max_info_time", mintime, max_info_time
            # if a trajectory is neither spawning nor backpropagating, we can
            # only propagate to its current forward propagation time
            time = self.traj[key].get_time()
            if (time - timestep) < max_info_time:
                max_info_time = time - timestep
#                 print "time max_info_time", time, max_info_time
        # now centroids
        for key in self.centroids:
            # if a centroid is backpropagating, we can only propagate to
            # its mintime
            timestep = self.centroids[key].get_timestep()
            mintime = self.centroids[key].get_mintime()
#             print "mintime, backprop_time", mintime, self.centroids[key].\
#                 get_backprop_time()
            if (mintime + 1.0e-6) < self.centroids[key].get_backprop_time():
                if (mintime - timestep) < max_info_time:
                    max_info_time = mintime - timestep
#                     print "mintime, max_info_time", mintime, max_info_time
            # if a centroid is not backpropagating, we can
            # only propagate to its current forward propagation time
            time = self.centroids[key].get_time()
            if (time - timestep) < max_info_time:
                # we subtract two timesteps because the spawning procedure
                # can take is back in time in a subsequent step
                max_info_time = time - timestep
#                 print "time max_info_time", time, max_info_time

        print "## we have enough information to propagate to time ",\
            max_info_time

        # now, if we have the necessary info, we propagate
        while max_info_time > (self.get_quantum_time() + 1.0e-6):
            if self.get_quantum_time() > 1.0e-6:
                print "## propagating quantum amplitudes at time",\
                    self.get_quantum_time()
                self.qm_propagate_step()
            else:
                print "## propagating quantum amplitudes at time",\
                    self.get_quantum_time(), " (first step)"
                self.qm_propagate_step(zoutput_first_step=True)

                        # SSAIMS stochastic selection (if enabled)
            self.ssaims_step()

            print "## outputing quantum information to hdf5"
            self.h5_output()

    def init_amplitudes_one(self):
        """Sets the first amplitude to 1.0 and all others to zero"""

        self.compute_num_traj_qm()
        self.qm_amplitudes = np.zeros_like(self.qm_amplitudes,
                                           dtype=np.complex128)
        self.qm_amplitudes[0] = 1.0

    def compute_num_traj_qm(self):
        """Compute number of trajectories, also increases the number of
        amplitudes after spawning"""

        n = 0
        qm_time = self.get_quantum_time()
        for key in self.traj:
            if qm_time > (self.traj[key].get_mintime() - 1.0e-6):
                n += 1
        self.set_num_traj_qm(n)
        while n > len(self.get_qm_amplitudes()):
            self.qm_amplitudes = np.append(self.qm_amplitudes, 0.0)

    def get_qm_data_from_h5(self):
        """Get the necessary geometries and energies from hdf5 at full ts"""

        qm_time = self.get_quantum_time()
        ntraj = self.get_num_traj_qm()
        for key in self.traj:
            if self.traj_map[key] < ntraj:
                self.traj[key].get_all_qm_data_at_time_from_h5(qm_time)
        for key in self.centroids:
            key1, key2 = str.split(key, "_a_")
            if self.traj_map[key1] < ntraj and self.traj_map[key2] < ntraj:
                self.centroids[key].get_all_qm_data_at_time_from_h5(qm_time)

    def get_qm_data_from_h5_half_step(self):
        """Get the necessary geometries and energies from hdf5 at half ts"""

        qm_time = self.get_quantum_time_half_step()
        ntraj = self.get_num_traj_qm()
        for key in self.traj:
            if self.traj_map[key] < ntraj:
                self.traj[key].\
                    get_all_qm_data_at_time_from_h5_half_step(qm_time)
        for key in self.centroids:
            key1, key2 = str.split(key, "_a_")
            if self.traj_map[key1] < ntraj and self.traj_map[key2] < ntraj:
                self.centroids[key].\
                    get_all_qm_data_at_time_from_h5_half_step(qm_time)

    def build_S(self):
        """Build the overlap matrix, S"""

        ntraj = self.get_num_traj_qm()
        self.S = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.S[i, j] = cg.overlap_nuc_elec(
                            self.traj[keyi], self.traj[keyj],
                            positions_i="positions_qm",
                            positions_j="positions_qm",
                            momenta_i="momenta_qm",
                            momenta_j="momenta_qm")
        #print("S is: ", self.S)

    def build_Sdot(self):
        """Build the right-acting time derivative operator"""

        ntraj = self.get_num_traj_qm()
        self.Sdot = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.Sdot[i, j] = cg.Sdot_nuc_elec(
                            self.traj[keyi],
                            self.traj[keyj],
                            positions_i="positions_qm",
                            positions_j="positions_qm",
                            momenta_i="momenta_qm",
                            momenta_j="momenta_qm",
                            forces_j="forces_i_qm")

    def invert_S(self):
        """Compute Sinv from S"""
        self.Sinv = np.linalg.inv(self.S)

    def build_H(self):
        """Build the Hamiltonian matrix, H
        This routine assumes that S is already built"""

        print "# building potential energy matrix"
        self.build_V()
        print "# building NAC matrix"
        self.build_tau()
        print "# building kinetic energy matrix"
        self.build_T()
        ntraj = self.get_num_traj_qm()
       # #shift = self.get_qm_energy_shift() * np.identity(ntraj)
        print "# summing Hamiltonian"
        self.H = self.T + self.V + self.tau #+ shift
        #print('T is: ', self.T)
        #print('V is:', self.V)
        #print('tau is: ', self.tau)
        #print('shift is: ', shift)
        #print("H is: ", self.H)

    def build_V(self):
        """Build the potential energy matrix, V
        This routine assumes that S is already built"""

        shift = self.get_qm_energy_shift()
        ntraj = self.get_num_traj_qm()
        self.V = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for key in self.traj:
            i = self.traj_map[key]
            istate = self.traj[key].get_istate()
            if i < ntraj:
                self.V[i, i] = self.traj[key].get_energies_qm()[istate] + shift
        for key in self.centroids:
            keyi, keyj = str.split(key, "_a_")
            i = self.traj_map[keyi]
            j = self.traj_map[keyj]
            if i < ntraj and j < ntraj:
                istate = self.centroids[key].get_istate()
                jstate = self.centroids[key].get_jstate()
                if istate == jstate:
                    E = self.centroids[key].get_energies_qm()[istate] + shift
                    self.V[i, j] = self.S[i, j] * E
                    self.V[j, i] = self.S[j, i] * E

    def build_tau(self):
        """Build the nonadiabatic coupling matrix, tau
        This routine assumes that S is already built"""

        c1i = (complex(0.0, 1.0))
        cm1i = (complex(0.0, -1.0))
        ntraj = self.get_num_traj_qm()
        self.tau = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for key in self.centroids:
            keyi, keyj = str.split(key, "_a_")
            i = self.traj_map[keyi]
            j = self.traj_map[keyj]
            if i < ntraj and j < ntraj:
                istate = self.centroids[key].get_istate()
                jstate = self.centroids[key].get_jstate()
                if istate != jstate:
                    Sij = cg.overlap_nuc(self.traj[keyi],
                                         self.traj[keyj],
                                         positions_i="positions_qm",
                                         positions_j="positions_qm",
                                         momenta_i="momenta_qm",
                                         momenta_j="momenta_qm")
                    tdc = self.centroids[key].get_timederivcoups_qm()[jstate]
                    self.tau[i, j] = Sij * cm1i * tdc
                    self.tau[j, i] = Sij.conjugate() * c1i * tdc

    def build_T(self):
        """build the kinetic energy matrix, T"""

        ntraj = self.get_num_traj_qm()
        self.T = np.zeros((ntraj, ntraj), dtype=np.complex128)
        for keyi in self.traj:
            i = self.traj_map[keyi]
            if i < ntraj:
                for keyj in self.traj:
                    j = self.traj_map[keyj]
                    if j < ntraj:
                        self.T[i, j] = cg.kinetic_nuc_elec(
                            self.traj[keyi],
                            self.traj[keyj],
                            positions_i="positions_qm",
                            positions_j="positions_qm",
                            momenta_i="momenta_qm",
                            momenta_j="momenta_qm")

    def build_Heff(self):
        """built Heff form H, Sinv, and Sdot"""

        print "# building effective Hamiltonian"
        c1i = (complex(0.0, 1.0))
        self.Heff = np.matmul(self.Sinv, (self.H - c1i * self.Sdot))

    def pop_task(self):
        """pop the task from the top of the queue"""

        return self.queue.pop(0)

    def update_queue(self):
        """build a list of all tasks that need to be completed"""

        while self.queue[0] != "END":
            self.queue.pop(0)
        tasktimes = [1e10]

        # forward propagation tasks
        for key in self.traj:
            if (self.traj[key].get_maxtime() + 1.0e-6) > self.traj[key].get_time():
                task_tmp = "self.traj[\""\
                    + key + "\"].propagate_step()"
                tasktime_tmp = self.traj[key].get_time()
                self.insert_task(task_tmp, tasktime_tmp, tasktimes)

        # backward propagation tasks
        for key in self.traj:
            if (self.traj[key].get_mintime()+1.0e-6) < self.traj[key].get_backprop_time():
                task_tmp = "self.traj[\"" + key\
                    + "\"].propagate_step(zbackprop=True)"
                tasktime_tmp = self.traj[key].get_backprop_time()
                self.insert_task(task_tmp, tasktime_tmp, tasktimes)

        # centroid tasks (forward propagation)
        for key in self.centroids:
            if self.centroids[key].get_z_compute_me():
                task_tmp = "self.centroids[\"" + key\
                    + "\"].compute_centroid()"
                tasktime_tmp = self.centroids[key].get_time()
                self.insert_task(task_tmp,tasktime_tmp, tasktimes)

        # centroid tasks (backward propagation)
        for key in self.centroids:
            if self.centroids[key].get_z_compute_me_backprop():
                task_tmp = "self.centroids[\"" + key +\
                    "\"].compute_centroid(zbackprop=True)"
                tasktime_tmp = self.centroids[key].get_backprop_time()
                self.insert_task(task_tmp, tasktime_tmp, tasktimes)

        print "##", (len(self.queue)-1), "task(s) in queue:"
        for i in range(len(self.queue)-1):
            print self.queue[i] + ", time = " + str(tasktimes[i])
        print "END"

    def insert_task(self, task, tt, tasktimes):
        """Add a task to the queue"""

        for i in range(len(tasktimes)):
            if tt < tasktimes[i]:
                self.queue.insert(i, task)
                tasktimes.insert(i, tt)
                return

    def update_centroids(self):
        """Compute the centroid positions and moment and check which centroids
        can be computed"""

        for key in self.centroids:
            key1, key2 = str.split(key, "_a_")
            timestep = self.centroids[key].get_timestep()

            # update backpropagating centroids
            self.centroids[key].set_z_compute_me_backprop(False)
            backprop_time = self.centroids[key].get_backprop_time() - timestep
            if (self.centroids[key].get_mintime()-1.0e-6) < backprop_time:
                backprop_time1 = self.traj[key1].get_backprop_time()
                if (backprop_time > backprop_time1 - 1.0e-6)\
                        and (backprop_time1  < (self.traj[key1].get_firsttime() - 1.0e-6)\
                        or backprop_time1  < (self.traj[key1].get_mintime() + 1.0e-6)):
                    backprop_time2 = self.traj[key2].get_backprop_time()
                    if (backprop_time > backprop_time2 - 1.0e-6)\
                            and (backprop_time2  < (self.traj[key2].get_firsttime() - 1.0e-6)\
                            or backprop_time2  < (self.traj[key2].get_mintime() + 1.0e-6)):
                        time1 = self.traj[key1].get_time()
                        time2 = self.traj[key2].get_time()
                        # this if takes care of the special case where we try 
                        # to compute the backpropagating centroid at firsttime before
                        # forward propagation has begun
                        if (backprop_time + 1.0e-6 < time1) and (backprop_time + 1.0e-6 < time2):
                            pos1 = self.traj[key1].get_data_at_time_from_h5(backprop_time, "positions")
                            mom1 = self.traj[key1].get_data_at_time_from_h5(backprop_time, "momenta")
#                             if backprop_time2  < (self.traj[key2].get_mintime() + 1.0e-6):
#                                 pos2 = self.traj[key2].get_backprop_positions()
#                                 mom2 = self.traj[key2].get_backprop_momenta()
#                             else:
                            pos2 = self.traj[key2].get_data_at_time_from_h5(backprop_time, "positions")
                            mom2 = self.traj[key2].get_data_at_time_from_h5(backprop_time, "momenta")
#                             pos2, mom2 = self.traj[key2].get_q_and_p_at_time_from_h5(backprop_time)
                            absSij = abs(cg.overlap_nuc(self.traj[key1],
                                                        self.traj[key2],
                                                        positions_i=pos1,
                                                        positions_j=pos2,
                                                        momenta_i=mom1,
                                                        momenta_j=mom2))
#                             print "absSij", absSij
                            # this definition of mom is only right if all basis functions have same
                            # width!!!!  I don't think the momentum is every used but still we 
                            # should fix this soon.
                            width1 = self.traj[key1].get_widths()
                            width2 = self.traj[key2].get_widths()
                            pos_cent = (width1 * pos1 + width2 * pos2) / (width1 + width2)
                            mom_cent = 0.5 * (mom1 + mom2)
                            self.centroids[key].set_backprop_positions(pos_cent)
                            self.centroids[key].set_backprop_momenta(mom_cent)
                            if absSij > 0.001:
                                self.centroids[key].set_z_compute_me_backprop(True)
                            else:
                                self.centroids[key].set_backprop_time(backprop_time)
                                dt = self.centroids[key].get_timestep()
                                self.centroids[key].set_backprop_time_half_step(
                                    backprop_time + 0.5 * dt)
                                self.centroids[key].set_backprop_energies(
                                    np.zeros(self.centroids[key].get_numstates()))
                                self.centroids[key].set_backprop_timederivcoups(
                                    np.zeros(self.centroids[key].get_numstates()))
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
                        absSij = abs(cg.overlap_nuc(self.traj[key1],
                                                    self.traj[key2],
                                                    positions_i=pos1,
                                                    positions_j=pos2,
                                                    momenta_i=mom1,
                                                    momenta_j=mom2))
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

    def spawn_as_necessary(self):
        """this is the spawning routine"""

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
                    # trajectory "00" spawned a trajectory "1" (its
                    # second child) which then spawned another (it's 6th child)
                    label = str(self.traj[key].get_label() + "b"
                                + str(self.traj[key].get_numchildren()))

                    # create and initiate new trajectory structure
                    newtraj = traj(self.traj[key].numdims, self.traj[key].numstates)
                    newtraj.init_spawn_traj(self.traj[key], jstate, label)

                    # checking if overlap between parent and child is not too small
                    # sometimes NAC coupling jumps at points of electronic wf discontinuity
                    # even though it is a warning sign, it is inevitable in many cases
                    # so here we calculate nuclear overlap to make sure there is going to
                    # be population transfer as a result of adding newtraj
                    parent_child_nuc_olap = cg.overlap_nuc(self.traj[key], newtraj)
                    if np.abs(parent_child_nuc_olap) < 0.02:
                        z_add_traj_olap = False
                    else:
                        z_add_traj_olap = True

                    # checking to see if overlap with existing trajectories
                    # is too high.  If so, we abort spawn
                    if z_add_traj_olap:
                        z_add_traj_olap = self.check_overlap(newtraj)

                    # rescaling velocity.  We'll abort if there is not
                    # enough energy (aka a "frustrated spawn")
                    z_add_traj_rescale = newtraj.rescale_momentum(
                        self.traj[key].get_energies_tmdt()[self.traj[key].get_istate()])
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
                newcent = traj(self.traj[key].numdims, self.traj[key].numstates)
                newcent.init_centroid(self.traj[key2], spawntraj[label], centkey)

                # add the centroid
                self.centroids[centkey] = newcent
                print "# adding centroid ", centkey

            # finally, add the spawned trajectory
            self.add_traj(spawntraj[label])

    def check_overlap(self, newtraj):
        """check to make sure that a spawned trajectory doesn't overlap too much
        with any existing trajectory"""

        z_add_traj = True
        for key2 in self.traj:
            # compute the overlap
            overlap = cg.overlap_nuc_elec(newtraj,
                                          self.traj[key2],
                                          positions_j="positions_tmdt",
                                          momenta_j="momenta_tmdt")

            # if the overlap is too high, don't spawn!
            if np.absolute(overlap) > self.olapmax:
                z_add_traj = False
                print "# aborting spawn due to large overlap with existing trajectory"

        return z_add_traj

    def restart_from_file(self, json_file, h5_file):
        """restarts from the current json file and copies the simulation data
        into working.hdf5"""

        self.read_from_file(json_file)
        shutil.copy2(h5_file, "working.hdf5")

    def restart_output(self):
        """output json restart file
        The json file is meant to represent the *current* state of the
        simulation.  There is a separate hdf5 file that stores the history of
        the simulation.  Both are needed for restart."""

        print "## creating new sim.json"
        # we keep copies of the last 3 json files just to be safe
        extensions = [2, 1, 0]
        for i in extensions:
            if i == 0:
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
        extensions = [2, 1, 0]
        for i in extensions:
            if i == 0:
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
        """Outputs info into h5 file"""

        self.init_h5_datasets()
        filename = "working.hdf5"
#         extensions = [3,2,1,0]
#         for i in extensions :
#             if i==0:
#                 ext = ""
#             else:
#                 ext = str(i) + "."
#             filename = "sim." + ext + "hdf5"
#             if os.path.isfile(filename):
#                 if (i == extensions[0]):
#                     os.remove(filename)
#                 else:
#                     ext = str(i+1) + "."
#                     filename2 = "sim." + ext + "hdf5"
#                     if (i == extensions[-1]):
#                         shutil.copy2(filename, filename2)
#                     else:
#                         shutil.move(filename, filename2)
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
            ipos = l
            getcom = "self.get_" + key + "()"
#             print getcom
            tmp = eval(getcom)
            if type(tmp).__module__ == np.__name__:
                tmp = np.ndarray.flatten(tmp)
                if l > 0:   # Start: Added by A. Mehmood to avoid basis shrinks after SSAIMS
                    row_width = dset.shape[1]
                else:
                    row_width = n
                if tmp.size < row_width:
                    row = np.zeros(row_width, dtype=dset.dtype)
                    row[:tmp.size] = tmp
                    dset[ipos, 0:row_width] = row
                else:
                    dset[ipos, 0:row_width] = tmp[0:row_width]   # End: Added by A. Mehmood                
            else:
                dset[ipos, 0] = tmp
        # ensure sim attrs up-to-date each write
        
        # Added by A.Mehmood: We need the procedure below to keep record of pruned trajs in sim dataset.
        # Save history of labels and corrsponding state per-step mapping so analysis can reconstruct information
        try:
            grp = h5f.get(groupname)

            # The row we just wrote is the last one (time is an append-only dataset)
            # If quantum_time is missing for some reason, fall back to 0.
            try:
                row_idx = int(grp["quantum_time"].shape[0]) - 1
                if row_idx < 0:
                    row_idx = 0
            except Exception:
                row_idx = 0

            # Build current label order (index -> label) for current("THIS") step
            n_now = int(self.get_num_traj_qm()) if hasattr(self, "get_num_traj_qm") else 0
            inv_map = [None]*n_now      # index -> label
            # self.traj_map: label -> current index
            for lab, idx in self.traj_map.items():
                if idx is not None and 0 <= idx < n_now:
                    inv_map[idx] = lab
            labels_list = [ (l if l is not None else "") for l in inv_map ]

            # Build current istates aligned with inv_map
            ist_list = []
            for l in labels_list:
                if l == "":
                    ist_list.append("-1")
                else:
                    try:
                        ist_list.append(str(int(self.traj[l].get_istate())))
                    except Exception:
                        try:
                            ist_list.append(str(int(self.centroids[l].get_istate())))
                        except Exception:
                            ist_list.append("-1")

            # Store as comma-separated strings
            labels_csv = ",".join(labels_list)
            istates_csv = ",".join(ist_list)

            # Create variable-length string datasets once and store label history to per time even if SSAIMS kill it
            if "labels_this_step" not in grp:
                vlen_str = h5py.special_dtype(vlen=bytes)
                grp.create_dataset("labels_this_step", shape=(0,), maxshape=(None,), dtype=vlen_str)
            # Create variable-length string datasets once and store TBS state history to per time even if SSAIMS kill it
            if "istates_this_step" not in grp:
                vlen_str = h5py.special_dtype(vlen=bytes)
                grp.create_dataset("istates_this_step", shape=(0,), maxshape=(None,), dtype=vlen_str)

            # Append at row_idx
            ds_lbl = grp["labels_this_step"]
            if ds_lbl.shape[0] <= row_idx:
                ds_lbl.resize((row_idx+1,))
            ds_lbl[row_idx] = labels_csv

            ds_ist = grp["istates_this_step"]
            if ds_ist.shape[0] <= row_idx:
                ds_ist.resize((row_idx+1,))
            ds_ist[row_idx] = istates_csv
        except Exception as history:
            try:
                print "### WARNING: failed to write per-step mapping/history", history
            except Exception:
                pass

        self.create_new_h5_map(grp)
        h5f.flush()
        h5f.close()

    def create_new_h5_map(self, grp):
        """Creates mapping of trajectory number to their labels
        This is important because traj dictionaries are not ordered
        with quantum aplitudes"""

        ntraj = self.get_num_traj_qm()
        labels = np.empty(ntraj, dtype="S512")
        istates = np.zeros(ntraj, dtype=np.int32)
        for key in self.traj_map:
            if self.traj_map[key] < ntraj:
                labels[self.traj_map[key]] = key
                istates[self.traj_map[key]] = self.traj[key].get_istate()
        grp.attrs["labels"] = labels
        grp.attrs["istates"] = istates

    def create_h5_sim(self, h5f, groupname):
        """Create h5 simulation datasets"""

        trajgrp = h5f.create_group(groupname)
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            dset = trajgrp.create_dataset(key, (0, n),
                                          maxshape=(None, None),
                                          dtype=self.h5_types[key])

    def init_h5_datasets(self):
        """Initialize simulation h5 file"""

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

    ###-----Start SSAIMS Implementation (A. Mehmood 10/01/2025)-----### 
    def enable_ssaims(self, epsilon=1.0e-4, ss_seed=None, suspend_during_spawn=True,
                    spawn_delay_steps=10, min_tbf_to_start=2, verbose=False):
        """Read user's provided parameters and/or set defaults."""
        self.ssa_enabled = True
        self.ssa_epsilon = float(epsilon)
        self.ssa_seed = ss_seed
        self.ssa_suspend_during_spawn = bool(suspend_during_spawn)    
        # start gates
        self.ssa_spawn_delay_steps = int(spawn_delay_steps)
        self.ssa_min_tbf_to_start  = int(min_tbf_to_start)
        self.ssa_verbose = bool(verbose)
        # JSON-safe RNG state
        self.ssa_rand_calls = 0
        # reset latches/trackers on enable
        self.ssa_has_started = False
        self.ssa_prev_ntraj = None
        self.ssa_steps_since_spawn = 1e9  # large so first check won't be blocked unless a spawn is detected
        self.ssa_age_in_basis = {}
        try:
            print "SSAIMS RNG seed =", self.ssa_seed
        except Exception:
            pass

    def disable_ssaims(self):
        """Disable SSAIMS (stochastic selection)."""
        self.ssa_enabled = False

    def ssa_spawn_in_progress(self):
        """Return True if any trajectory is currently preparing/performing a spawn."""      
        for key in self.traj:
            z = self.traj[key].get_z_spawn_now()
            spawnt = self.traj[key].get_spawntimes()
            # If any state is flagged to spawn or has a positive spawntime, suspend
            try:
                if (z is not None and (z > 0.5).any()) or (spawnt is not None and (spawnt > 0.0).any()):
                    return True
            except Exception:
                pass
        return False

    def ssaims_step(self):
        """
        Perform one SSAIMS selection step at the current quantum time.
        """
        
        if not getattr(self, "ssa_enabled", False):
            return

        if getattr(self, "ssa_suspend_during_spawn", True) and self.ssa_spawn_in_progress():
            # Avoid selection during spawning windows (as recommended by SSAIMS paper)
            try:
                print "### SSAIMS suspended: spawning in progress (no selection applied)"
            except Exception:
                pass
            return

        # Ensure we have at least 2 active TBFs
        self.compute_num_traj_qm()
        ntraj = self.get_num_traj_qm()
        if ntraj <= 1:
            try:
                print "### SSAIMS inactive: single TBF (no selection applied)"
            except Exception:
                pass
            return

        # Detect basis spawn to reset delay counter
        if self.ssa_prev_ntraj is None:
            self.ssa_prev_ntraj = ntraj
        else:
            if ntraj > self.ssa_prev_ntraj:
                self.ssa_steps_since_spawn = 0
                try:
                    print "### SSAIMS: spawn detected, resetting delay counter"
                except Exception:
                    pass
            self.ssa_prev_ntraj = ntraj            
        # Count the cooldown steps since last spawn
        self.ssa_steps_since_spawn += 1

        # Condition minimum TBFs
        if ntraj < self.ssa_min_tbf_to_start:
            try:
                print "### SSAIMS inactive: min-TBF criteria not met (ntraj=%d < min_tbf_to_start=%d)" % (
                ntraj, self.ssa_min_tbf_to_start)
            except Exception:
                pass
            return

        # Latch on first activation. once SSAIMS starts, keep it started
        if not self.ssa_has_started:
            self.ssa_has_started = True

        # Make sure QM matrices are available at the current quantum time
        self.get_qm_data_from_h5()
        self.build_S()
        self.build_H()

        H = self.get_H()
        S = self.get_S()
        eps = float(self.ssa_epsilon)

        # split TBFs into mature and immature and set age ciunter for immature to avoid killing 
        try:
            t_qm = self.get_quantum_time()
        except Exception:
            t_qm = None
            
        # Build reverse mapping index -> label for all active TBFs
        idx_to_label = [None]*ntraj
        for lab, idx in self.traj_map.items():
            if idx < ntraj:
                idx_to_label[idx] = lab
    
        active_indices = []
        active_labels  = []
        if t_qm is not None:
            for i in range(ntraj):
                lab = idx_to_label[i]
                if lab is None:
                    continue
                try:
                    # newborns not yet in the QM basis
                    if (self.traj[lab].get_mintime() - 1.0e-6) <= t_qm:
                        active_indices.append(i)
                        active_labels.append(lab)
                except Exception:
                    # if we cannot read mintime, consider it active to be safe
                    active_indices.append(i)
                    active_labels.append(lab)
    
        # Initialize/increment ages for currently active labels
        if not hasattr(self, "ssa_age_in_basis") or (self.ssa_age_in_basis is None):
            self.ssa_age_in_basis = {}
        for lab in active_labels:
            if lab not in self.ssa_age_in_basis:
                self.ssa_age_in_basis[lab] = 0
            else:
                # incease age by one time step
                self.ssa_age_in_basis[lab] += 1

        for lab, age in self.ssa_age_in_basis.items():
            if age < self.ssa_spawn_delay_steps:
                try:
                    print "### SSAIMS inactive: cooldown time not reached for %s (age=%d < delay_steps=%d)" % (
                        lab, age, self.ssa_spawn_delay_steps)
                except Exception:
                    pass
   
        # Partition active block into immature (< spawn_delay_steps) and mature (>= spawn_delay_steps)
        immature_labels  = []
        immature_indices = []
        mature_indices   = []
        delay_steps = int(getattr(self, "ssa_spawn_delay_steps", 0))
        if delay_steps > 0:
            for i, lab in zip(active_indices, active_labels):
                age = int(self.ssa_age_in_basis.get(lab, 0))
                if age < delay_steps:
                    immature_labels.append(lab)
                    immature_indices.append(i)
                else:
                    mature_indices.append(i)
        else:
            # If no per-TBF delay requested; everything is mature
            mature_indices = list(active_indices)

        # If there are fewer than 2 mature TBFs, selection is useless.
        if len(mature_indices) <= 1:
            return    
        
        # Build adjacency only over the mature subset. Let immature to age >= spawn_delay_steps
        m = len(mature_indices)        
        adj = [[False]*m for _ in range(m)]
        for a in range(m):
            adj[a][a] = True
        for a in range(m):
            i = mature_indices[a]
            for b in range(a+1, m):
                j = mature_indices[b]
                if abs(H[i, j]) >= eps:
                    adj[a][b] = True
                    adj[b][a] = True

        # Find connected components via BFS (the badass procedure, love it)
        visited = [False]*m
        components_local = []
        for i in range(m):
            if not visited[i]:
                queue = [i]
                visited[i] = True
                comp = []
                while queue:
                    u = queue.pop(0)
                    comp.append(u)
                    for v in range(m):
                        if adj[u][v] and not visited[v]:
                            visited[v] = True
                            queue.append(v)
                components_local.append(sorted(comp))

        if len(components_local) <= 1:
            # fully coupled or single TBF -> nothing to select  
            try:
                print "### SSAIMS inactive: fully coupled (no selection applied)"
            except Exception:
                pass
            return

        # Coherent populations P_alpha on mature components only
        c = self.get_qm_amplitudes()
        Ps = []
        comps_global = []
        for comp_loc in components_local:
            comp_glob = [mature_indices[a] for a in comp_loc]
            comps_global.append(comp_glob)
            idx = np.ix_(comp_glob, comp_glob)
            S_sub = S[idx]
            c_sub = c[comp_glob]
            P = np.dot(c_sub.conjugate(), np.dot(S_sub, c_sub))
            try:
                P_val = float(np.real(P))
            except Exception:
                P_val = 0.0
            if P_val < 0.0 and abs(P_val) < 1.0e-14:
                P_val = 0.0
            Ps.append(P_val)
    
        totalP = sum(Ps)
    
        # Fallback for coherence ~0: use sum |c|^2 on mature comps only
        if totalP <= 1.0e-16:
            Ps = []
            for comp_glob in comps_global:
                P_val = float(np.sum(np.abs(c[comp_glob])**2))
                Ps.append(P_val)
            totalP = sum(Ps)
    
        # If still zero, keep largest mature component by size
        if totalP <= 1.0e-16:
            sizes = [len(comp) for comp in comps_global]
            keep_idx = int(np.argmax(sizes))
            keep_comp_mature = comps_global[keep_idx]
        else:
            # Stochastic selection proportional to P_alpha (on mature comps)
            rnd = self.ssa_next_uniform()
            acc = 0.0
            keep_idx = 0
            for k, Pk in enumerate(Ps):
                acc += Pk/totalP
                if rnd <= acc:
                    keep_idx = k
                    break
            keep_comp_mature = comps_global[keep_idx]
    
        # Final keep set = selected mature component  U  all immature indices
        keep_all = sorted(set(keep_comp_mature).union(set(immature_indices)))
    
        # Safety: avoid empty keep set (shouldn't happen, but just in case)
        if len(keep_all) == 0:
            keep_all = list(active_indices)
                
        ### Start Verbose part. Developed for debugging, but kept if user wants.
        if getattr(self, 'ssa_verbose', False):
            try:    
                # |H| stats (off-diagonal only) over the *active* QM block
                Habs = np.abs(H.copy())
                for i in range(ntraj):
                    Habs[i, i] = 0.0
                above = (Habs >= float(self.ssa_epsilon))
                num_edges = int(np.sum(above) // 2)
                maxH = float(np.max(Habs)) if ntraj > 0 else 0.0
                minHnz = float(np.min(Habs[Habs > 0])) if (Habs > 0).any() else 0.0
        
                # component sizes (report the *mature* component sizes only that are just built)
                comp_sizes = [len(cmp) for cmp in comps_global]
                totalP_dbg = float(sum(Ps)) if len(Ps) > 0 else 0.0
                probs = [ (p/totalP_dbg if totalP_dbg > 0.0 else 0.0) for p in Ps ]
        
                # labels by active index
                labels_by_idx = [None]*ntraj
                for lab, idx in self.traj_map.items():
                    if idx < ntraj:
                        labels_by_idx[idx] = lab
        
                # What we're keeping (report the mature-chosen block AND the union keep_all)
                keep_set_mature = set(keep_comp_mature) if 'keep_comp_mature' in locals() else set()
                keep_set_all    = set(keep_all) if 'keep_all' in locals() else keep_set_mature
        
                # Which labels are being removed (indices in active block but not in keep_all)
                to_remove = [labels_by_idx[i] for i in range(ntraj)
                            if (i not in keep_set_all) and (labels_by_idx[i] is not None)]
        
                # Try to get current quantum time
                try:
                    tqm = self.get_quantum_time()
                    print "### SSAIMS VERBOSE @ t_qm =", tqm
                except Exception:
                    print "### SSAIMS VERBOSE"
        
                print "    epsilon (|H_kl| threshold): %.3e" % self.ssa_epsilon
                print "    ntraj(before): %d" % ntraj
                print "    edges(|H|>=eps): %d   max|H|: %.3e   min_nonzero|H|: %.3e" % (num_edges, maxH, minHnz)
                print "    components(mature): %d   sizes: %s" % (len(comps_global), comp_sizes)
                print "    Ps (mature): [%s]   totalP(mature): %.6f" % (', '.join(['%.6f' % p for p in Ps]),totalP_dbg)
        
                # If drew rnd earlier, show it and the cumulative distribution
                if 'rnd' in locals():
                    try:
                        print "    rnd: %.6f   cumulative probs: [%s]" % (rnd,', '.join(['%.6f' % x for x in np.cumsum(probs)]))
                    except Exception:
                        pass
        
                # If computed P_keep for the mature choice, show it; otherwise show the final kept norm
                if 'keep_idx' in locals() and 0 <= keep_idx < len(Ps):
                    P_keep_mature = Ps[keep_idx]
                else:
                    P_keep_mature = None
        
                print "    keep_idx(mature):", (keep_idx if 'keep_idx' in locals() else None), \
                    "   keep_size(mature):", (len(keep_comp_mature) if 'keep_comp_mature' in locals() else None), \
                    "   P_keep(mature):", (P_keep_mature if P_keep_mature is not None else 'n/a')
                print "    keep_size(final):", (len(keep_all) if 'keep_all' in locals() else len(keep_set_mature))
                print "    removing_labels:", to_remove        
                # Spawn status
                try:
                    sp = self.ssa_spawn_in_progress() if hasattr(self, 'ssa_spawn_in_progress') else self.ssa_spawn_in_progress()
                except Exception:
                    sp = None
                print "    spawn_in_progress:", sp
        
            except Exception as e:
                try:
                    print "### SSAIMS VERBOSE print failed:", e
                except Exception:
                    pass               
        ### End Verbose part
            
        ### Start prune and rebuild coefficients in new TBFs order ###
        
        # Convert the keep set to LABELS (After so many attempts, the working solution)
        # Procedure: Prune by indices and Re-map by labels, then renomalize in current basis
        
        # Build the keep labels list from indices
        keep_labels = []
        for i in keep_all:
            if 0 <= i < ntraj and idx_to_label[i] is not None:
                keep_labels.append(idx_to_label[i])
        
        # Remember old mapping and old amplitudes
        old_traj_map = dict(self.traj_map)
        c_old = c.copy()
        
        # Prune by indices (please) (not by P.I.T.A labels)
        self.ssa_prune_to_indices(set(keep_all))
        
        # Guard: if we somehow ended up with zero TBFs, restore by indices
        if self.get_num_traj_qm() == 0 or len(self.traj_map) == 0:
            restore_indices = [i for i in active_indices if 0 <= i < ntraj]
            if restore_indices:
                self.ssa_prune_to_indices(set(restore_indices))
        
        # Rebuild c in the new basis order using labels
        new_n = int(self.get_num_traj_qm())
        c_now = np.zeros(new_n, dtype=np.complex128)
        
        # Only map the labels we intended to keep
        for lab in keep_labels:
            new_i = self.traj_map.get(lab, None)        # index after prune
            old_i = old_traj_map.get(lab, None)         # index before prune
            if new_i is not None and old_i is not None and 0 <= old_i < len(c_old) and 0 <= new_i < new_n:
                c_now[new_i] = c_old[old_i]
        
        # Safety: if everything is still ~0 (e.g., label mismatch), fall back to intersection
        if float(np.sum(np.abs(c_now)**2)) < 1e-30:
            for lab, new_i in self.traj_map.items():        # labels that kept after SSAIMS
                old_i = old_traj_map.get(lab, None)
                if old_i is not None and 0 <= old_i < len(c_old) and 0 <= new_i < new_n:
                    c_now[new_i] = c_old[old_i]
                
        # Rebuild S for the pruned basis and renormalize so c* S c = 1
        self.build_S()
        S_now = self.get_S()
        P_now = np.dot(c_now.conjugate(), np.dot(S_now, c_now))
        try:
            P_now_val = float(np.real(P_now))
        except Exception:
            P_now_val = 0.0
        
        if P_now_val > 1.0e-16:
            c_now = c_now / np.sqrt(P_now_val)
        else:
            # last-resort |c|^2 norm
            norm2 = float(np.sum(np.abs(c_now)**2))
            if norm2 > 1.0e-16:
                c_now = c_now / np.sqrt(norm2)     
       
        self.set_qm_amplitudes(c_now)
        self.set_num_traj_qm(new_n)
        
        # Refresh index->state map to match current basis
        self.istates_dict = {}
        for lab, idx in self.traj_map.items():      # label -> current index
            try:
                st = int(self.traj[lab].get_istate())
            except Exception:
                try:
                    st = int(self.centroids[lab].get_istate())
                except Exception:
                    st = -1
            self.istates_dict[idx] = st
        
        # Rebuild all QM matrices so the write matches the pruned basis
        self.build_S()
        try: self.build_Sdot()
        except Exception: pass
        try: self.invert_S()
        except Exception: pass
        try:
            self.build_H()
            self.build_Heff()
        except Exception:
            pass
    ### End prune and rebuild coefficients in new TBFs order ###            
       
    def ssa_next_uniform(self):
        """
        JSON cannot save RandonState i.e. RNG array generated using random seed.
        This is JSON-safe RNG procedure which return next U(0,1) using (ssa_seed, ssa_rand_calls).
        No RandomState is stored on self, so restart JSON stays clean and do not show error.
        """

        if getattr(self, 'ssa_seed', None) is None:           
            return np.random.rand()            # No users seed provided then use global RNG

        k = int(getattr(self, 'ssa_rand_calls', 0))
        ssa_rng = np.random.RandomState(self.ssa_seed)        
        val = ssa_rng.random_sample(k + 1)[-1] # Jump ahead to (k+1)-th draw and take it
        self.ssa_rand_calls = k + 1
        return val

    def ssa_prune_to_indices(self, keep_indices):
        """
        Remove trajectories/centroids not in 'keep_indices' after SSAIM procedure and rebuild traj_map.  
        """
        
        ntraj = self.get_num_traj_qm()
        labels = [None]*ntraj
        for key in self.traj_map:
            idx = self.traj_map[key]
            if idx < ntraj:
                labels[idx] = key

        keep_labels = set([labels[i] for i in sorted(list(keep_indices)) if i < ntraj and labels[i] is not None])

        # Prune trajectories
        new_traj = {}
        for key in self.traj:
            if key in keep_labels:
                new_traj[key] = self.traj[key]
        self.traj = new_traj

        # Prune centroids where one endpoint is missing
        to_delete = []
        for cent in self.centroids:
            key1, key2 = str.split(cent, "_a_")
            if (key1 not in self.traj) or (key2 not in self.traj):
                to_delete.append(cent)
        for cent in to_delete:
            del self.centroids[cent]

        # Rebuild traj_map with contiguous indices ordered by mintime
        sorted_keys = sorted(self.traj.keys(), key=lambda k: self.traj[k].get_mintime())
        self.traj_map = {}
        for i, key in enumerate(sorted_keys):
            self.traj_map[key] = i

        self.set_num_traj_qm(len(sorted_keys))

    ###-----End SSAIMS Implementation-----###
