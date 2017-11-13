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
        self.fill_quantum_times()
        self.fill_qm_amplitudes()
        self.num_traj = len(self.datasets["qm_amplitudes"][0][:])
        self.fill_S()
        self.fill_traj_time()

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
    
    def fill_qm_amplitudes(self):
        c = self.h5file["sim/qm_amplitudes"][()]
        self.datasets["qm_amplitudes"] = c
        
    def fill_S(self):
        S = self.h5file["sim/S"][()]
        self.datasets["S"] = S

    def retrieve_num_traj_qm(self):
        self.ntraj = np.ndarray.flatten(self.h5file["sim/num_traj_qm"][()])
        
    def fill_quantum_times(self):
        times = np.ndarray.flatten(self.h5file["sim/quantum_time"][()])
        self.datasets["quantum_times"] = times

    def fill_traj_time(self):
        for key in self.labels:
            trajgrp = "traj_" + key
            time = self.h5file[trajgrp]['time'][()].flatten()
            key2 = key + "_time"
            self.datasets[key2] = time
            time = self.h5file[trajgrp]['time_half_step'][()].flatten()
            key3 = key + "_time_half_step"
            self.datasets[key3] = time
        

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

    def get_traj_num_times_half_step(self,label):
        key = label + "_time_half_step"
        return len(self.datasets[key])

    def list_datasets(self):
        for key in self.datasets:
            print key

    def write_columnar_data_file(self,times,dsets,filename):
        of = open(filename,"w")
        t = self.datasets[times][:]
        for i in range(len(t)):
            of.write(str(t[i]) + " ")
            for iset in range(len(dsets)):
                dat = self.datasets[dsets[iset]][i,:]
                for j in range(len(dat)):
                    of.write(str(dat[j]) + " ")
            of.write("\n")
        of.close()
        
    def fill_electronic_state_populations(self,column_filename=None):
        times = self.datasets["quantum_times"]
        ntimes = len(times)
        maxstates = self.get_max_state()
        Nstate = np.zeros((ntimes,maxstates+1))
        for i in range(ntimes):
            nt = self.ntraj[i]
            c_t = self.get_amplitude_vector(i)
            S_t = self.get_overlap_matrix(i)
            for ist in range(maxstates):
                Nstate[i,ist] =  self.compute_expec_istate_not_normalized(S_t,c_t,ist)
            Nstate[i,maxstates] = self.compute_expec(S_t,c_t)
        self.datasets["electronic_state_populations"] = Nstate
            
        if column_filename != None:
            self.write_columnar_data_file("quantum_times",["electronic_state_populations"],column_filename)

        return 

    def write_xyzs(self):
        for key in self.labels:
            times = self.get_traj_dataset(key,"time")
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key,"positions")
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

    def fill_trajectory_energies(self,column_file_prefix=None):
        for key in self.labels:
            times =  self.get_traj_dataset(key,"time")
            ntimes = self.get_traj_num_times(key)
            mom = self.get_traj_data_from_h5(key,"momenta")
            nmom = mom.size / ntimes
            poten = self.get_traj_data_from_h5(key,"energies")
            nstates = poten.size/ntimes

            istate = self.get_traj_attr_from_h5(key,'istate')

            m = self.get_traj_attr_from_h5(key, 'masses')

            kinen = np.zeros((ntimes,1))
            toten = np.zeros((ntimes,1))

            for itime in range(ntimes):
                p = mom[itime,:]
                kinen[itime,0] = 0.5 * np.sum(p * p / m)
                toten[itime,0] = kinen[itime,0] + poten[itime,istate]

            dset_poten = key + "_poten"
            dset_toten = key + "_toten"
            dset_kinen = key + "_kinen"

            self.datasets[dset_poten] = poten
            self.datasets[dset_toten] = toten
            self.datasets[dset_kinen] = kinen

            if column_file_prefix != None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key+"_time",[dset_poten,dset_kinen,dset_toten],column_filename)

    def fill_trajectory_bonds(self,bonds,column_file_prefix):
        for key in self.labels:
            times = self.get_traj_dataset(key,"time")
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key,"positions")
            npos = pos.size / ntimes
            nbonds = bonds.size / 2

            d = np.zeros((ntimes,nbonds))

            for itime in range(ntimes):
                for ibond in range(nbonds):
                    ipos = 3*bonds[ibond,0]
                    jpos = 3*bonds[ibond,1]
                    ri = pos[itime,ipos:(ipos+3)]
                    rj = pos[itime,jpos:(jpos+3)]
                    r = ri-rj
                    d[itime,ibond] = math.sqrt(np.sum(r*r))

            dset_bonds = key + "_bonds"

            self.datasets[dset_bonds] = d

            if column_file_prefix != None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key+"_time",[dset_bonds],column_filename)

    def fill_trajectory_angles(self,angles,column_file_prefix):
        for key in self.labels:
            times = self.get_traj_dataset(key,"time")
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key,"positions")
            npos = pos.size / ntimes
            nangles = angles.size / 3

            ang = np.zeros((ntimes,nangles))

            for itime in range(ntimes):
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

                    ang[itime,iang] = math.acos(dot) / math.pi * 180.0

            dset_angles = key + "_angles"

            self.datasets[dset_angles] = ang

            if column_file_prefix != None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key+"_time",[dset_angles],column_filename)


    def fill_trajectory_diheds(self,diheds,column_file_prefix):
        for key in self.labels:
            times = self.get_traj_dataset(key,"time")
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key,"positions")
            npos = pos.size / ntimes
            ndiheds = diheds.size / 4

            dih = np.zeros((ntimes,ndiheds))

            for itime in range(ntimes):
                for idih in range(ndiheds):
                    ipos = 3*diheds[idih,0]
                    jpos = 3*diheds[idih,1]
                    kpos = 3*diheds[idih,2]
                    lpos = 3*diheds[idih,3]
                    ri = pos[itime,ipos:(ipos+3)]
                    rj = pos[itime,jpos:(jpos+3)]
                    rk = pos[itime,kpos:(kpos+3)]
                    rl = pos[itime,lpos:(lpos+3)]

                    rji = ri - rj
                    rkj = rj - rk
                    rjk = -1.0 * rkj
                    rkl = rl - rk

                    rjicrossrjk = np.cross(rji,rjk)
                    rkjcrossrkl = np.cross(rkj,rkl)

                    normjijk = math.sqrt(np.sum(rjicrossrjk*rjicrossrjk))
                    normkjkl = math.sqrt(np.sum(rkjcrossrkl*rkjcrossrkl))

                    rjijk = rjicrossrjk / normjijk
                    rkjkl = rkjcrossrkl / normkjkl

                    dot = np.sum(rjijk * rkjkl)

                    dih[itime,idih] = math.acos(dot) / math.pi * 180.0

            dset_diheds = key + "_diheds"

            self.datasets[dset_diheds] = dih

            if column_file_prefix != None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key+"_time",[dset_diheds],column_filename)


    def fill_trajectory_tdcs(self,column_file_prefix=None):
        for key in self.labels:
            times =  self.get_traj_dataset(key,"time_half_step")
            ntimes = self.get_traj_num_times_half_step(key)
            mom = self.get_traj_data_from_h5(key,"momenta")
            nmom = mom.size / ntimes
            tdc = self.get_traj_data_from_h5(key,"timederivcoups")
            nstates = tdc.size/ntimes

            istate = self.get_traj_attr_from_h5(key,'istate')

            dset_tdc = key + "_tdc"

            self.datasets[dset_tdc] = tdc

            if column_file_prefix != None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key+"_time_half_step",[dset_tdc],column_filename)
