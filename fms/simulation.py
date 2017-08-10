# simulation object contains the current state of the simulation
import types
import numpy as np
from fms.fmsobj import fmsobj
from fms.traj import traj
import os
import shutil

class simulation(fmsobj):
    def __init__(self):
        self.maxtime = -1.0
        self.numtraj = 0
        self.traj = dict()
        self.defaulttimestep = 0.0
        self.queue = ["END"]

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
                    obj = eval(fmsobjlabel[4:])()
                    obj.from_dict(**(tempdict[key]))
                    tempdict[key] = obj
                else:
                    for key2 in tempdict[key]:
                        if isinstance((tempdict[key])[key2],types.DictType) :
                            fmsobjlabel = ((tempdict[key])[key2]).pop('fmsobjlabel')
                            obj = eval(fmsobjlabel[4:])()
                            obj.from_dict(**((tempdict[key])[key2]))
                            (tempdict[key])[key2] = obj
        self.__dict__.update(tempdict)

    # add a trajectory to the simulation
    def add_traj(self,t1):
        key = t1.get_label()
        self.traj[key] = t1
        self.numtraj += 1

    # 
#    def set_spawntraj(self,t1):
#        self.spawntraj = t1

    def add_task(self,task):
        self.queue.append(task)

    def get_numtasks(self):
        return (len(self.queue)-1)

    def set_timestep_all(self,h):
        for key in self.traj:
            self.traj[key].set_timestep(h)

    def set_maxtime_all(self,maxtime):
        for key in self.traj:
            self.traj[key].set_maxtime(maxtime)

    def set_mintime_all(self,mintime):
        for key in self.traj:
            self.traj[key].set_mintime(mintime)

    def set_propagator_all(self,prop):
        for key in self.traj:
            self.traj[key].set_propagator(prop)

    def propagate(self):
        while True:
            self.update_queue()
            if (self.queue[0] == "END"):
                print "propagate DONE"
                return
            print self.queue[0]
            print self.queue[1]
            print self.traj
            current = self.queue.pop(0)
            print "Starting " + current
            # Right now we just run a single task per cycle,
            # but we could parallelize here and send multiple tasks
            # out for simultaneous processing.
            eval(current)
            print "Done with " + current
            self.spawn_as_necessary()
            self.json_output()

    def update_queue(self):
        while self.queue[0] != "END":
            self.queue.pop(0)
        for key in self.traj:
            print "key " + key
            if (self.traj[key].get_maxtime()-1.0e-6) > self.traj[key].get_time():
                task_tmp = "self.traj[\"" + key  + "\"].propagate_step()"
                self.queue.insert(0,task_tmp)
        for key in self.traj:
            print "key " + key

            if (self.traj[key].get_mintime()+1.0e-6) < self.traj[key].get_backprop_time():
                print "uq backprop ", self.traj[key].get_mintime(), self.traj[key].get_backprop_time()
                task_tmp = "self.traj[\"" + key  + "\"].propagate_step(zbackprop=True)"
                self.queue.insert(0,task_tmp)
        print (len(self.queue)-1), " jobs in queue"
        for task in self.queue:
            print task
        
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

                    spawntraj[label] = traj()
                    spawntraj[label].init_spawn_traj(self.traj[key], jstate, label)
                    
                    self.traj[key].incr_numchildren()
                    z[jstate] = 0.0
                    z_dont[jstate] = 1.0
                    spawnt[jstate] = -1.0
                    
            self.traj[key].set_z_spawn_now(z)
            self.traj[key].set_z_dont_spawn(z_dont)
            self.traj[key].set_spawntimes(spawnt)
                    
        for label in spawntraj:
            ### need to test overlaps before adding trajectories!!!!
            self.add_traj(spawntraj[label])

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
        

    

        
        
