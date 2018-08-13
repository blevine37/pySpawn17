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

    def compute_expec(self, Op, c, zreal=True):
        expec = np.matmul(c.conjugate(), np.matmul(Op, c))
        if zreal:
            expec = expec.real
        return expec

    def compute_expec_istate_not_normalized(self, Op, c, istate, zreal=True):
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
        self.ntraj = self.h5file["sim/num_traj_qm"][()].flatten()
        
    def fill_quantum_times(self):
        times = self.h5file["sim/quantum_time"][()]
        self.datasets["quantum_times"] = times

    def fill_traj_time(self):
        for key in self.labels:
            trajgrp = "traj_" + key
            time = self.h5file[trajgrp]['time'][()]
            key2 = key + "_time"
            self.datasets[key2] = time
            time = self.h5file[trajgrp]['time_half_step'][()]
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
        t = self.datasets[times][:,0]
        for i in range(len(t)):
            of.write(str(t[i]) + " ")
            for iset in range(len(dsets)):
                dat = self.datasets[dsets[iset]][i,:]
                for j in range(len(dat)):
                    of.write(str(dat[j]) + " ")
            of.write("\n")
        of.close()
        
    def fill_mulliken_populations(self,column_filename=None):
        times = self.datasets["quantum_times"][:,0]
        ntimes = len(times)
        ntraj = self.get_num_traj()

        mull = np.zeros((ntimes,ntraj))

        for itime in range(ntimes):
            nt = self.ntraj[itime]
            c_t = self.get_amplitude_vector(itime)
            S_t = self.get_overlap_matrix(itime)
            for i in range(nt):
                for j in range(nt):
                    tmp = 0.5 * np.real(c_t[i] * np.conj(c_t[j]) * S_t[i,j])
                    mull[itime,i] += tmp
                    mull[itime,j] += tmp
        self.datasets["mulliken_populations"] = mull
        if column_filename != None:
            self.write_columnar_data_file("quantum_times",["mulliken_populations"],column_filename)

        return 

    def fill_expec_mulliken(self,dset_name,column_filename=None):
        times = self.datasets["quantum_times"][:,0]
        ntimes = len(times)
        ntraj = self.get_num_traj()
        
        ncol = self.datasets[self.labels[0] + "_" + dset_name].shape[1]

        if "mulliken_populations" not in self.datasets:
            self.fill_mulliken_populations()
        mull = self.datasets["mulliken_populations"]

        x = np.zeros((ntimes,ncol))

        denom = mull.sum(axis=1)

        for itraj in range(ntraj):
            key = self.labels[itraj]
            dset_x = key + "_" + dset_name
            xk = self.datasets[dset_x]

            dset_t = key + "_time"
            trajtimes = self.datasets[dset_t][:,0]
            firsttime = trajtimes[0]
            lasttime = times[-1]
            ntrajtimes = self.datasets[dset_t].size

            for itime in range(ntimes):
                #print times[itime], firsttime
                if ((times[itime]-1e-6) < firsttime) and ((times[itime]+1e-6) > firsttime):
                    ifirsttime = itime

            for itime in range(ntrajtimes):
                #print times[itime], firsttime
                if ((trajtimes[itime]-1e-6) < lasttime) and ((trajtimes[itime]+1e-6) > lasttime):
                    ilasttime = itime+1


            for icol in range(ncol):
                x[ifirsttime:ifirsttime+ilasttime,icol] += xk[0:ilasttime,icol] * mull[ifirsttime:ifirsttime+ilasttime,itraj]
                
        for icol in range(ncol):
            x[:,icol] = x[:,icol] / denom

        dset_expec = "expec_mull_" + dset_name

        self.datasets[dset_expec] = x

        if column_filename != None:
            self.write_columnar_data_file("quantum_times",[dset_expec],column_filename)

    def fill_electronic_state_populations(self, column_filename=None):
        times = self.datasets["quantum_times"][:, 0]
        ntimes = len(times)
        maxstates = self.get_max_state()
        Nstate = np.zeros((ntimes, maxstates + 1))
        for i in range(ntimes):
            nt = self.ntraj[i]
            c_t = self.get_amplitude_vector(i)
            S_t = self.get_overlap_matrix(i)
            for ist in range(maxstates):
                Nstate[i,ist] =  self.compute_expec_istate_not_normalized(S_t, c_t, ist)
            Nstate[i,maxstates] = self.compute_expec(S_t, c_t)
        self.datasets["electronic_state_populations"] = Nstate
            
        if column_filename != None:
            self.write_columnar_data_file("quantum_times", ["electronic_state_populations"], column_filename)

        return 
    
    def fill_trajectory_populations(self, column_file_prefix=None):
        for key in self.labels:
            times =  self.get_traj_dataset(key, "time")[:, 0]
            ntimes = self.get_traj_num_times(key)
            pop = self.get_traj_data_from_h5(key, "populations")
            nstates = pop.size/ntimes

            population = np.zeros((ntimes, nstates))

            for itime in range(ntimes):
                population = pop[itime, :]

            dset_pop = key + "_pop"
            
            self.datasets[dset_pop] = pop
            
            if column_file_prefix != None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key + "_time",\
                                              [dset_pop],\
                                              column_filename)        
    
    def write_xyzs(self):
        for key in self.labels:
            times = self.get_traj_dataset(key,"time")[:,0]
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

    def fill_trajectory_energies(self, column_file_prefix=None):
        for key in self.labels:
            times =  self.get_traj_dataset(key,"time")[:, 0]
            ntimes = self.get_traj_num_times(key)
            mom = self.get_traj_data_from_h5(key,"momenta")
            nmom = mom.size / ntimes
            poten = self.get_traj_data_from_h5(key,"energies")
            av_energy = self.get_traj_data_from_h5(key, "av_energy")
            nstates = poten.size/ntimes

            istate = self.get_traj_attr_from_h5(key,'istate')

            m = self.get_traj_attr_from_h5(key, 'masses')

            kinen = np.zeros((ntimes,1))
            toten = np.zeros((ntimes,1))

            for itime in range(ntimes):
                p = mom[itime, :]
                kinen[itime, 0] = 0.5 * np.sum(p * p / m)
                toten[itime, 0] = kinen[itime, 0] + av_energy[itime]

            dset_poten = key + "_poten"
            dset_toten = key + "_toten"
            dset_kinen = key + "_kinen"
            dset_aven = key + "_aven"
            
            self.datasets[dset_poten] = poten
            self.datasets[dset_toten] = toten
            self.datasets[dset_kinen] = kinen
            self.datasets[dset_aven] = av_energy
            
            if column_file_prefix != None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key + "_time",\
                                              [dset_poten, dset_kinen, dset_toten, dset_aven],\
                                              column_filename)

    def fill_trajectory_bonds(self,bonds,column_file_prefix):
        for key in self.labels:
            times = self.get_traj_dataset(key,"time")[:,0]
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
            times = self.get_traj_dataset(key,"time")[:,0]
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
            times = self.get_traj_dataset(key,"time")[:,0]
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

    def fill_trajectory_twists(self,twists,column_file_prefix):
        for key in self.labels:
            times = self.get_traj_dataset(key,"time")[:,0]
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key,"positions")
            npos = pos.size / ntimes
            ntwists = twists.size / 6

            twi = np.zeros((ntimes,ntwists))

            for itime in range(ntimes):
                for itwi in range(ntwists):
                    ipos = 3*twists[itwi,0]
                    jpos = 3*twists[itwi,1]
                    kpos = 3*twists[itwi,2]
                    lpos = 3*twists[itwi,3]
                    mpos = 3*twists[itwi,4]
                    npos = 3*twists[itwi,5]
                    ri = pos[itime,ipos:(ipos+3)]
                    rj = pos[itime,jpos:(jpos+3)]
                    rk = pos[itime,kpos:(kpos+3)]
                    rl = pos[itime,lpos:(lpos+3)]
                    rm = pos[itime,mpos:(mpos+3)]
                    rn = pos[itime,npos:(npos+3)]

                    rji = ri - rj
                    rkl = rl - rk
                    rmn = rn - rm

                    rjicrossrkl = np.cross(rji,rkl)
                    rjicrossrmn = np.cross(rji,rmn)

                    normjikl = math.sqrt(np.sum(rjicrossrkl*rjicrossrkl))
                    normjimn = math.sqrt(np.sum(rjicrossrmn*rjicrossrmn))

                    rjikl = rjicrossrkl / normjikl
                    rjimn = rjicrossrmn / normjimn

                    dot = np.sum(rjikl * rjimn)

                    twi[itime,itwi] = math.acos(dot) / math.pi * 180.0

            dset_twists = key + "_twists"

            self.datasets[dset_twists] = twi

            if column_file_prefix != None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key+"_time",[dset_twists],column_filename)

    def fill_trajectory_pyramidalizations(self,pyrs,column_file_prefix):
        for key in self.labels:
            times = self.get_traj_dataset(key,"time")[:,0]
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key,"positions")
            npos = pos.size / ntimes
            npyrs = pyrs.size / 4

            pyr = np.zeros((ntimes,npyrs))

            for itime in range(ntimes):
                for ipyr in range(npyrs):
                    ipos = 3*pyrs[ipyr,0]
                    jpos = 3*pyrs[ipyr,1]
                    kpos = 3*pyrs[ipyr,2]
                    lpos = 3*pyrs[ipyr,3]
                    ri = pos[itime,ipos:(ipos+3)]
                    rj = pos[itime,jpos:(jpos+3)]
                    rk = pos[itime,kpos:(kpos+3)]
                    rl = pos[itime,lpos:(lpos+3)]

                    rij = rj - ri
                    rik = rk - ri
                    ril = rl - ri

                    rikcrossril = np.cross(rik,ril)

                    normikil = math.sqrt(np.sum(rikcrossril*rikcrossril))
                    normij = math.sqrt(np.sum(rij*rij))

                    rikil = rikcrossril / normikil
                    rij /= normij

                    dot = np.sum(rikil * rij)

                    pyr[itime,ipyr] = math.asin(math.fabs(dot)) / math.pi * 180.0

            dset_pyrs = key + "_pyrs"

            self.datasets[dset_pyrs] = pyr

            if column_file_prefix != None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key+"_time",[dset_pyrs],column_filename)

    def fill_trajectory_tdcs(self,column_file_prefix=None):
        for key in self.labels:
            times =  self.get_traj_dataset(key,"time_half_step")[:,0]
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
