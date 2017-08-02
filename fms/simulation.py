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
        self.spawntraj = traj()
        self.defaulttimestep = 0.0

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

    def add_traj(self,t1, key):
        self.traj[key] = t1
        self.numtraj += 1

    def set_spawntraj(self,t1):
        self.spawntraj = t1
    


        
        
