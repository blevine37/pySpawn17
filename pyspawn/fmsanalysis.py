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
        self.labels = self.h5file["sim"].attrs["labels"]
        self.istates = self.h5file["sim"].attrs["istates"]

    def __del__(self):
        self.h5file.close()

    def get_num_traj(self):
        return self.num_traj

    def get_max_state(self):
        return (np.amax(self.istates)+1)

    def compute_expec(self,Op,c,zreal=True):
        expec = np.matmul(c.conjugate(), np.matmul(Op, c))
        if zreal:
            expec = expec.real
        return expec

    def compute_expec_istate_not_normalized(self,Op,c,istate,zreal=True):
        ctmp = np.zeros(len(c),dtype=np.complex128)
        for i in range(len(c)):
            if self.istates[i] == istate:
                ctmp[i] = c[i]
        expec = np.matmul(ctmp.conjugate(), np.matmul(Op, ctmp))
        if zreal:
            expec = expec.real
        return expec
    
    def retrieve_amplitudes(self):
        c = self.h5file["sim/qm_amplitudes"][()]
        return c
        
    def retrieve_Ss(self):
        S = self.h5file["sim/S"][()]
        return S

    def retrieve_num_traj_qm(self):
        ntraj = np.ndarray.flatten(self.h5file["sim/num_traj_qm"][()])
        return ntraj
        
    def retrieve_times(self):
        times = np.ndarray.flatten(self.h5file["sim/quantum_time"][()])
        return times
        
    def compute_norms(self,outfilename):
        of = open(outfilename, 'w')
        
        times = self.retrieve_times()
        c = self.retrieve_amplitudes()
        S = self.retrieve_Ss()
        ntraj = self.retrieve_num_traj_qm()
        maxstates = self.get_max_state()
        for i in range(len(times)):
            nt = ntraj[i]
            c_t = c[i][0:nt]
            nt2 = nt*nt
            S_t = S[i][0:nt2].reshape((nt,nt))
            Nstate = np.zeros(maxstates)
            for ist in range(maxstates):
                Nstate[ist] =  self.compute_expec_istate_not_normalized(S_t,c_t,ist)
            N = self.compute_expec(S_t,c_t)
            of.write(str(times[i])+ " "+" ".join(map(str,Nstate))+" "+str(N)+"\n")
        of.close()
