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
import math

class fafile(object):
    def __init__(self,h5filename):
        self.datasets = {}
        self.h5file = h5py.File(h5filename, "r")
        self.labels = self.h5file["sim"].attrs["labels"]
        self.istates = self.h5file["sim"].attrs["istates"]
        self.retrieve_num_traj_qm()
        self.retrieve_quantum_times()
        self.retrieve_qm_amplitudes()
        self.num_traj = len(self.datasets["qm_amplitudes"][0][:])
        self.retrieve_S()
        self.retrieve_traj_time()

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
    
    def retrieve_qm_amplitudes(self):
        c = self.h5file["sim/qm_amplitudes"][()]
        self.datasets["qm_amplitudes"] = c
        
    def retrieve_S(self):
        S = self.h5file["sim/S"][()]
        self.datasets["S"] = S

    def retrieve_num_traj_qm(self):
        self.ntraj = np.ndarray.flatten(self.h5file["sim/num_traj_qm"][()])
        
    def retrieve_quantum_times(self):
        times = np.ndarray.flatten(self.h5file["sim/quantum_time"][()])
        self.datasets["quantum_times"] = times

    def retrieve_traj_time(self):
        for key in self.labels:
            trajgrp = "traj_" + key
            time = self.h5file[trajgrp]['time'][()].flatten()
            key2 = key + "_time"
            self.datasets[key2] = time
        
    def get_amplitude_vector(self,i):
        nt = self.ntraj[i]
        c_t = self.datasets["qm_amplitudes"][i][0:nt]
        return c_t

    def get_overlap_matrix(self,i):
        nt = self.ntraj[i]
        nt2 = nt*nt
        S_t = self.datasets["S"][i][0:nt2].reshape((nt,nt))
        return S_t

    def get_traj_data_from_h5(self,label,key):
        trajgrp = "traj_" + label
        return self.h5file[trajgrp][key][()]

    def get_traj_attr_from_h5(self,label,key):
        trajgrp = "traj_" + label
        return self.h5file[trajgrp].attrs[key]

    def get_traj_dataset(self,label,key):
        return self.datasets[label + "_" + key]

    def get_traj_num_times(self,label):
        key = label + "_time"
        return len(self.datasets[key])


    def compute_electronic_state_populations(self,column_filename=None):
        if column_filename != None:
            of = open(column_filename, 'w')
        
        times = self.datasets["quantum_times"]
        ntimes = len(times)
        maxstates = self.get_max_state()
        Nstate = np.zeros((maxstates+1,ntimes))
        for i in range(ntimes):
            nt = self.ntraj[i]
            c_t = self.get_amplitude_vector(i)
            S_t = self.get_overlap_matrix(i)
            for ist in range(maxstates):
                Nstate[ist,i] =  self.compute_expec_istate_not_normalized(S_t,c_t,ist)
            Nstate[maxstates,i] = self.compute_expec(S_t,c_t)
            if column_filename != None:
                of.write(str(times[i])+ " "+" ".join(map(str,Nstate[:,i]))+"\n")
        if column_filename != None:
            of.close() 
        self.datasets["electronic_state_populations"] = Nstate
            
        return 

    def write_xyzs(self):
        for key in self.labels:
            #trajgrp = "traj_" + key
            #times = self.h5file[trajgrp]['time'][()].flatten()
            times = self.get_traj_dataset(key,"time")
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key,"positions")
            # convert to angstrom
            pos /= 1.8897161646321
            npos = pos.size / ntimes
            natoms = npos/3
            atoms = self.get_traj_attr_from_h5(key,"atoms")

            filename = "traj_" + key + ".xyz"
            of = open(filename,"w")

            for itime in range(ntimes):
                of.write(str(natoms)+"\n")
                of.write("T = "+str(times[itime])+"\n")
                for iatom in range(natoms):
                    of.write(atoms[iatom]+"  "+str(pos[itime,3*iatom])+"  "+str(pos[itime,3*iatom+1])+"  "+str(pos[itime,3*iatom+2])+"\n")

            of.close()

    def write_trajectory_energy_files(self):
        for key in self.labels:
            trajgrp = "traj_" + key
            times = self.h5file[trajgrp]['time'][()].flatten()
            ntimes = len(times)
            mom = self.h5file[trajgrp]['momenta'][()]
            nmom = mom.size / ntimes
            poten = self.h5file[trajgrp]['energies'][()]
            nstates = poten.size/ntimes

            istate = self.h5file[trajgrp].attrs['istate']

            m = self.h5file[trajgrp].attrs['masses']

            filename = trajgrp + ".energy"
            of = open(filename,"w")

            for itime in range(ntimes):
                p = mom[itime,:]
                kinen = 0.5 * np.sum(p * p / m)
                toten = kinen + poten[itime,istate]
                of.write(str(times[itime])+"  ")
                for jstate in range(nstates):
                    of.write(str(poten[itime,jstate])+"  ")
                of.write(str(kinen)+"  "+str(toten)+"\n")

            of.close()

    def write_trajectory_bond_files(self,bonds):
        for key in self.labels:
            trajgrp = "traj_" + key
            times = self.h5file[trajgrp]['time'][()].flatten()
            ntimes = len(times)
            pos = self.h5file[trajgrp]['positions'][()]
            npos = pos.size / ntimes

            nbonds = bonds.size / 2

            filename = trajgrp + ".bonds"
            of = open(filename,"w")

            for itime in range(ntimes):
                of.write(str(times[itime])+"  ")
                for ibond in range(nbonds):
                    ipos = 3*bonds[ibond,0]
                    jpos = 3*bonds[ibond,1]
                    ri = pos[itime,ipos:(ipos+3)]
                    rj = pos[itime,jpos:(jpos+3)]
                    r = ri-rj
                    d = math.sqrt(np.sum(r*r))
                    of.write(str(d)+"  ")
                of.write("\n")
            of.close()

    def write_trajectory_angle_files(self,angles):
        for key in self.labels:
            trajgrp = "traj_" + key
            times = self.h5file[trajgrp]['time'][()].flatten()
            ntimes = len(times)
            pos = self.h5file[trajgrp]['positions'][()]
            npos = pos.size / ntimes

            nangles = angles.size / 3

            filename = trajgrp + ".angles"
            of = open(filename,"w")

            for itime in range(ntimes):
                of.write(str(times[itime])+"  ")
                for iang in range(nangles):
                    ipos = 3*angles[iang,0]
                    jpos = 3*angles[iang,1]
                    kpos = 3*angles[iang,2]
                    ri = pos[itime,ipos:(ipos+3)]
                    rj = pos[itime,jpos:(jpos+3)]
                    rk = pos[itime,kpos:(kpos+3)]

                    rji = ri - rj
                    rjk = rk - rj

                    dji = math.sqrt(np.sum(rji*rji))
                    djk = math.sqrt(np.sum(rjk*rjk))

                    rji /= dji
                    rjk /= djk

                    dot = np.sum(rji * rjk)

                    ang = math.acos(dot) / math.pi * 180.0

                    of.write(str(ang)+"  ")
                of.write("\n")
            of.close()

    def write_trajectory_tdc_files(self):
        for key in self.labels:
            trajgrp = "traj_" + key
            times = self.h5file[trajgrp]['time_half_step'][()].flatten()
            ntimes = len(times)
            tdc = self.h5file[trajgrp]['timederivcoups'][()]
            nstates = tdc.size / ntimes

            filename = trajgrp + ".tdc"
            of = open(filename,"w")

            for itime in range(ntimes):
                of.write(str(times[itime])+"  ")
                for jstate in range(nstates):
                    of.write(str(tdc[itime,jstate])+"  ")
                of.write("\n")

            of.close()
