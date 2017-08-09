# simulation object contains the current state of the simulation
import types
import numpy as np
from fms.fmsobj import fmsobj
from fms.traj import traj

class simulation(fmsobj):
    def __init__(self):
        self.maxtime = -1.0
        self.numtraj = 0
        self.traj = dict()
#        self.spawntraj = traj()
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
    def add_traj(self,t1, key):
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
            eval(current)
            print "Done with " + current

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
                task_tmp = "self.traj[\"" + key  + "\"].propagate_step(zbackprop=True)"
                self.queue.insert(0,task_tmp)
        print (len(self.queue)-1), " jobs in queue"
        for task in self.queue:
            print task
        
    
    

        
        
