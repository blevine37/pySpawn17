# simulation object contains the current state of the simulation
import types
import numpy as np
from pyspawn.fmsobj import fmsobj
from pyspawn.traj import traj
import os
import shutil
import complexgaussian as cg

class simulation(fmsobj):
    def __init__(self):
        self.traj = dict()

        self.centroids = dict()
        
        self.queue = ["END"]
        self.tasktimes = [1e10]
        
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

    # 
#    def set_spawntraj(self,t1):
#        self.spawntraj = t1

    def add_task(self,task):
        self.queue.append(task)

    def get_numtasks(self):
        return (len(self.queue)-1)

    def set_olapmax(self,s):
        self.olapmax = s

    def set_timestep_all(self,h):
        for key in self.traj:
            self.traj[key].set_timestep(h)

    def set_maxtime_all(self,maxtime):
        for key in self.traj:
            self.traj[key].set_maxtime(maxtime)
        for key in self.centroids:
            self.centroids[key].set_maxtime(maxtime)

    def set_mintime_all(self,mintime):
        for key in self.traj:
            self.traj[key].set_mintime(mintime)

    def set_propagator_all(self,prop):
        for key in self.traj:
            self.traj[key].set_propagator(prop)

    def propagate(self):
        while True:
            self.update_centroids()
            self.update_queue()
            if (self.queue[0] == "END"):
                print "propagate DONE"
                return
            print self.queue[0]
            print self.queue[1]
            print self.traj
            current = self.pop_task() 
            print "Starting " + current
            # Right now we just run a single task per cycle,
            # but we could parallelize here and send multiple tasks
            # out for simultaneous processing.
            eval(current)
            print "Done with " + current
            self.spawn_as_necessary()
            self.json_output()

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
                
        print (len(self.queue)-1), " jobs in queue:"
        for task in self.queue:
            print task

    def insert_task(self,task,tt,tasktimes):
        for i in range(len(tasktimes)):
            if tt < tasktimes[i]:
                self.queue.insert(i,task)
                tasktimes.insert(i,tt)
                return
        
    def update_centroids(self):
        #update backprop
        for key in self.centroids:
            print "key ", key
            print "type ", type(key)
            print "typ2 ", type(str(key))
            key1, key2 = str.split(key,"_&_")
            print "key1 ", key1
            print "key2 ", key2
            timestep = self.centroids[key].get_timestep()

            self.centroids[key].set_z_compute_me_backprop(False)
            backprop_time = self.centroids[key].get_backprop_time()
            print "u_cent backprop_time ", backprop_time
            if (self.centroids[key].get_mintime()+1.0e-6) < backprop_time:
                if (backprop_time > self.traj[key1].get_backprop_time() + timestep - 1.0e-6):
                    if (backprop_time > self.traj[key2].get_backprop_time() + timestep - 1.0e-6):
                        pos1, mom1 = self.traj[key1].get_q_and_p_at_time_from_h5(backprop_time)
                        pos2, mom2 = self.traj[key2].get_q_and_p_at_time_from_h5(backprop_time)
                        pos_cent = 0.5 * ( pos1 + pos2 )
                        mom_cent = 0.5 * ( mom1 + mom2 )
                        print "pos1 ", pos1
                        print "pos2 ", pos2
                        print "pos_cent ", pos_cent
                        print "mom1 ", mom1
                        print "mom2 ", mom2
                        print "mom_cent ", mom_cent
                        self.centroids[key].set_positions_backprop(pos_cent)
                        self.centroids[key].set_momenta_backprop(mom_cent)
                        self.centroids[key].set_z_compute_me_backprop(True)
                        
            self.centroids[key].set_z_compute_me(False)
            time = self.centroids[key].get_time()
            print "u_cent time ", time
            if (self.centroids[key].get_maxtime()-1.0e-6) > time:
                if (time < self.traj[key1].get_time() - timestep + 1.0e-6):
                    if (time < self.traj[key2].get_time() - timestep + 1.0e-6):
                        pos1, mom1 = self.traj[key1].get_q_and_p_at_time_from_h5(time)
                        pos2, mom2 = self.traj[key2].get_q_and_p_at_time_from_h5(time)
                        pos_cent = 0.5 * ( pos1 + pos2 )
                        mom_cent = 0.5 * ( mom1 + mom2 )
                        print "pos1 ", pos1
                        print "pos2 ", pos2
                        print "pos_cent ", pos_cent
                        self.centroids[key].set_positions(pos_cent)
                        self.centroids[key].set_momenta(mom_cent)
                        self.centroids[key].set_z_compute_me(True)
                        


    def spawn_as_necessary(self):
        spawntraj = dict()
        for key in self.traj:
            z = self.traj[key].get_z_spawn_now()
            z_dont = self.traj[key].get_z_dont_spawn()
            spawnt = self.traj[key].get_spawntimes()
            for jstate in range(self.traj[key].get_numstates()):
                if z[jstate] > 0.5:
                    label = self.traj[key].get_label() + "->" + str(self.traj[key].get_numchildren())
                    print "Creating new traj, ", label

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
                        
                    z[jstate] = 0.0
                    z_dont[jstate] = 1.0
                    spawnt[jstate] = -1.0
                    
            self.traj[key].set_z_spawn_now(z)
            self.traj[key].set_z_dont_spawn(z_dont)
            self.traj[key].set_spawntimes(spawnt)
                    
        for label in spawntraj:
            # create new centroids
            for key2 in self.traj:
                centkey = str(key2 + "_&_" + label)
                newcent = traj()
                newcent.init_centroid(self.traj[key2],spawntraj[label], centkey)
                self.centroids[centkey] = newcent
                print "adding centroid ", centkey
                            
            self.add_traj(spawntraj[label])

    def check_overlap(self,newtraj):
        z_add_traj=True
        for key2 in self.traj:
            overlap = cg.overlap_nuc_elec(newtraj,self.traj[key2],positions_j="positions_tmdt",momenta_j="momenta_tmdt")
            print "overlap ", np.absolute(overlap), self.olapmax
            if np.absolute(overlap) > self.olapmax:
                z_add_traj=False
            if not z_add_traj:
                print "Aborting spawn due to large overlap with existing trajectory"
        return z_add_traj
        
            
    def json_output(self):
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
        self.write_to_file("sim.json")
