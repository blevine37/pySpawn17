# A class from which all fms classes should be derived.
# Includes methods for output of classes to json format.
# The ability to read/dump data from/to json is essential to the
# restartability that we intend.
# nested python dictionaries serve as an intermediate between json
# and the native python class
import simulation
import traj
import numpy as np
import h5py

class fmsanalysis(object):
    def __init__(self):
        self.h5file = h5py.File("sim.hdf5", "r")
        self.num_traj = len(self.retrieve_amplitudes()[0][:])

    def __del__(self):
        self.h5file.close()

    def get_num_traj(self):
        return self.num_traj

    def compute_expec(self,Op,c,zreal=True):
        expec = np.matmul(c.conjugate(), np.matmul(Op, c))
        if zreal:
            expec = expec.real
        return expec

    def retrieve_amplitudes(self):
        c = self.h5file["sim/qm_amplitudes"][()]
        return c
        
    def retrieve_Ss(self):
        S = self.h5file["sim/S"][()]
        return S

    def retrieve_times(self):
        times = np.ndarray.flatten(self.h5file["sim/quantum_time"][()])
        return times
        
    def compute_norms(self,outfilename):
        times = self.retrieve_times()
        c = self.retrieve_amplitudes()
        S = self.retrieve_Ss()
        ntraj = self.get_num_traj()
        for i in range(len(times)):
            c_t = c[i][:]
            S_t = S[i][:].reshape((ntraj,ntraj))
            N = self.compute_expec(S_t,c_t)
            print times[i], N
