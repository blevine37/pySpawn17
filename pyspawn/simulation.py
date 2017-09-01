# simulation object contains the current state of the simulation.
# It is analagous to the "bundle" object in the original FMS code.
import types
import numpy as np
from pyspawn.fmsobj import fmsobj
from pyspawn.traj import traj
import os
import shutil
import complexgaussian as cg

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

    # convert dict to simulation data structure
    def from_dict(self,**tempdict):
        for key in tempdict:
            if isinstance(tempdict[key],types.ListType) :
                if isinstance((tempdict[key])[0],types.FloatType) :
                    # convert 1d float lists to np arrays
                    tempdict[key] = np.asarray(tempdict[key])
                else:
                    if isinstance((tempdict[key])[0],types.ListType):
                        if isinstance((tempdict[key])[0][0],types.FloatType) :
                            # convert 2d float lists to np arrays
                           tempdict[key] = np.asarray(tempdict[key])
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
        self.traj[key] = t1

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
        for key in self.traj:
            self.traj[key].set_timestep(h)
        for key in self.centroids:
            self.centroids[key].set_timestep(h)

    # set the maximimum simulation time on all trajectories and centroids
    def set_maxtime_all(self,maxtime):
        for key in self.traj:
            self.traj[key].set_maxtime(maxtime)
        for key in self.centroids:
            self.centroids[key].set_maxtime(maxtime)

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

    # this is the main propagation loop for the simulation
    def propagate(self):
        while True:
            # compute centroid positions and mark those centroids that
            # can presently be computed
            self.update_centroids()

            # update the queue (list of tasks to be computed)
            self.update_queue()

            # if the queue is empty, we're done!
            if (self.queue[0] == "END"):
                print "propagate DONE"
                return

            # Right now we just run a single task per cycle,
            # but we could parallelize here and send multiple tasks
            # out for simultaneous processing.
            current = self.pop_task()
            print "Starting " + current            
            eval(current)
            print "Done with " + current

            # spawn new trajectories if needed
            self.spawn_as_necessary()

            # print restart output
            self.json_output()

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
            print "key " + key
            if (self.traj[key].get_maxtime()-1.0e-6) > self.traj[key].get_time():
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
                
        print (len(self.queue)-1), " jobs in queue:"
        for task in self.queue:
            print task

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
            key1, key2 = str.split(key,"_&_")
            timestep = self.centroids[key].get_timestep()

            #update backpropagating centroids
            self.centroids[key].set_z_compute_me_backprop(False)
            backprop_time = self.centroids[key].get_backprop_time()
            if (self.centroids[key].get_mintime()+1.0e-6) < backprop_time:
                if (backprop_time > self.traj[key1].get_backprop_time() + timestep - 1.0e-6):
                    if (backprop_time > self.traj[key2].get_backprop_time() + timestep - 1.0e-6):
                        pos1, mom1 = self.traj[key1].get_q_and_p_at_time_from_h5(backprop_time)
                        pos2, mom2 = self.traj[key2].get_q_and_p_at_time_from_h5(backprop_time)
                        # this is only write if all basis functions have same
                        # width!!!!  Fix this soon
                        pos_cent = 0.5 * ( pos1 + pos2 )
                        mom_cent = 0.5 * ( mom1 + mom2 )
                        self.centroids[key].set_positions_backprop(pos_cent)
                        self.centroids[key].set_momenta_backprop(mom_cent)
                        self.centroids[key].set_z_compute_me_backprop(True)

            # update forward propagating centroids
            self.centroids[key].set_z_compute_me(False)
            time = self.centroids[key].get_time()
            if (self.centroids[key].get_maxtime()-1.0e-6) > time:
                if (time < self.traj[key1].get_time() - timestep + 1.0e-6):
                    if (time < self.traj[key2].get_time() - timestep + 1.0e-6):
                        pos1, mom1 = self.traj[key1].get_q_and_p_at_time_from_h5(time)
                        pos2, mom2 = self.traj[key2].get_q_and_p_at_time_from_h5(time)
                        # this is only write if all basis functions have same
                        # width!!!!  Fix this soon
                        pos_cent = 0.5 * ( pos1 + pos2 )
                        mom_cent = 0.5 * ( mom1 + mom2 )
                        self.centroids[key].set_positions(pos_cent)
                        self.centroids[key].set_momenta(mom_cent)
                        self.centroids[key].set_z_compute_me(True)
                        

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
                    # a trajectory labeled 00->1->5 means that the initial
                    # trajectory "00" spawned a trajecory "1" (its
                    # second child) which then spawned another (it's 6th child)
                    label = self.traj[key].get_label() + "->" + str(self.traj[key].get_numchildren())
                    print "Creating new traj, ", label

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
                # "_&_" marks the centroid labels
                centkey = str(key2 + "_&_" + label)

                # create and initiate the trajectory structures!
                newcent = traj()
                newcent.init_centroid(self.traj[key2],spawntraj[label], centkey)

                # add the centroid
                self.centroids[centkey] = newcent
                print "adding centroid ", centkey

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
                print "Aborting spawn due to large overlap with existing trajectory"
        return z_add_traj
        
    # output json restart file
    # The json file is meant to represent the *current* state of the
    # simulation.  There is a separate hdf5 file that stores the history of
    # the simulation.  Both are needed for restart.
    def json_output(self):
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
