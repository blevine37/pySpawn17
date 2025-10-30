import math

import h5py
import numpy as np
from typing import Dict, Any


class fafile(object):
    """A class from which all fms classes should be derived.
    Includes methods for output of classes to json format.
    The ability to read/dump data from/to json is essential to the
    restartability that we intend.
    nested python dictionaries serve as an intermediate between json
    and the native python class"""

    def __init__(self, h5filename):
        self.datasets = {}
        self.h5file = h5py.File(h5filename, "r")
        self.read_step_mapping()
        self.labels = self.h5file["sim"].attrs["labels"]
        self.istates = self.h5file["sim"].attrs["istates"]
        self.numstates = self.h5file['traj_00'].attrs["numstates"]
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
        if hasattr(self, "states_sorted") and self.states_sorted:
            return int(max(self.states_sorted)) + 1
        return int(np.amax(self.istates)) + 1

    def compute_expec(self, Op, c, zreal=True):
        expec = np.matmul(c.conjugate(), np.matmul(Op, c))
        if zreal:
            expec = expec.real
        return expec

    def compute_expec_istate_not_normalized(self, Op, c, istate, zreal=True):
        ctmp = np.zeros(len(c), dtype=np.complex128)
        for i in range(len(c)):
            if self.istates[i] == istate:
                ctmp[i] = c[i]
        expec = np.matmul(ctmp.conjugate(), np.matmul(Op, ctmp))
        if zreal:
            expec = expec.real
        return expec

    def create_istate_dict(self):

        istates_dict = dict()
        for key in self.labels:
            istates_dict[key] = self.h5file['traj_' + key].attrs["istate"]
        self.datasets["istates_dict"] = istates_dict

    def fill_labels(self):
        """
        Fill a static list of all trajectory labels present in the file.
        Robust to SSAIMS (pruning) because we read labels from traj_* groups,
        not from sim.attrs['labels'] (which may only reflect the final basis).
        """
        labs = self.all_traj_labels()
        self.datasets["labels"] = labs

    def fill_istates(self):
        """
        Fill a list of per-trajectory initial electronic states aligned to self.datasets['labels'].
        Reads from each traj_* group's 'istate' attribute (SSAIMS-safe).
        """
        if "labels" not in self.datasets:
            self.fill_labels()
        labels = self.datasets.get("labels", [])
        
        istates = []
        for lab in labels:
            ist = -1
            try:
                g = self.h5file["traj_" + lab]
                ist = int(g.attrs.get("istate", -1))
            except Exception:
                pass
            istates.append(ist)        
        self.datasets["istates"] = istates


    def fill_qm_amplitudes(self):
        c = self.h5file["sim/qm_amplitudes"][()]
        self.datasets["qm_amplitudes"] = c

    def fill_S(self):
        S = self.h5file["sim/S"][()]
        self.datasets["S"] = S

    def retrieve_num_traj_qm(self):
        self.ntraj = self.h5file["sim/num_traj_qm"][()].flatten()

    def get_numstates(self):
        self.datasets['numstates'] = self.numstates

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

    def get_amplitude_vector(self, i):
        nt = self.ntraj[i]
        c_t = self.datasets["qm_amplitudes"][i][0:nt]
        return c_t

    def get_overlap_matrix(self, i):
        nt = self.ntraj[i]
        nt2 = nt * nt
        S_t = self.datasets["S"][i][0:nt2].reshape((nt, nt))
        return S_t

    def get_traj_data_from_h5(self, label, key):
        trajgrp = "traj_" + label
        return self.h5file[trajgrp][key][()]

    def get_traj_attr_from_h5(self, label, key):
        trajgrp = "traj_" + label
        return self.h5file[trajgrp].attrs[key]

    def get_traj_dataset(self, label, key):
        return self.datasets[label + "_" + key]

    def get_traj_num_times(self, label):
        key = label + "_time"
        return len(self.datasets[key])

    def get_traj_num_times_half_step(self, label):
        key = label + "_time_half_step"
        return len(self.datasets[key])

    def list_datasets(self):
        for key in self.datasets:
            print key

    def write_columnar_data_file(self, times, dsets, filename):
        """Subroutine to write to text files. Currently outputs in scientific notation
        in single precision for readability"""

        of = open(filename, "w")
        t = self.datasets[times][:, 0]
        for i in range(len(t)):
            number = t[i]
            of.write(str(number) + " ")
            for iset in range(len(dsets)):
                dat = self.datasets[dsets[iset]][i, :]
                for j in range(len(dat)):
                    number = dat[j]
                    of.write(str(number) + " ")
            of.write("\n")
        of.close()

    def decode_bytes_array(self, arr):
        """Reads and decode labels history"""
        out = []
        for x in arr:
            try:
                out.append(x.decode("utf-8"))
            except Exception:
                out.append(x)
        return out
    
    def row_mats_from_flat(self, S_flat_row, n):
        """Reshape the row-flattened S into an n*n matrix."""
        n2 = n * n
        return S_flat_row[:n2].reshape((n, n))

    def all_traj_labels(self):
        """Return trajectory labels by scanning traj_* groups"""
        labs = []
        for k in self.h5file.keys():
            if k.startswith("traj_"):
                labs.append(k[len("traj_"):])  # drop "traj_" prefix
        labs.sort()
        return labs

    def fill_mulliken_populations(self, column_filename=None):
        """
        SSAIMS-aware Mulliken populations, aligned by global label order.
        Output shape: (ntimes, L) where L = len(all_labels), each column j is Mulliken
        population of label all_labels[j] at each time. Inactive labels get 0 at that time.
        """
        all_labels = self.all_traj_labels()
        L = len(all_labels)
        self.datasets["labels"] = list(all_labels)
    
        grp_sim = self.h5file["sim"]
        times = grp_sim["quantum_time"][()][:, 0]
        ntimes = len(times)
        qampls = grp_sim["qm_amplitudes"][()]
        Sflat  = grp_sim["S"][()]
        ntraj_each = grp_sim["num_traj_qm"][()].flatten().astype(int)
    
        rows_labels = self.labels_this_step_rows()
        mull = np.zeros((ntimes, L), dtype=float)
        col_of = {}
        for j, lab in enumerate(all_labels):
            col_of[lab] = j
    
        for i in range(ntimes):
            nt = int(ntraj_each[i])
            if nt <= 0:
                continue
    
            c_t = qampls[i][:nt]
            S_t = self.row_mats_from_flat(Sflat[i], nt)    
            if rows_labels is not None and i < len(rows_labels):
                labs_i = rows_labels[i]
                if len(labs_i) != nt:
                    labs_i = list(labs_i[:nt]) + ([""] * max(0, nt - len(labs_i)))
            else:
                labs_i = all_labels[:nt]
    
            for a in range(nt):
                acc = 0.0 + 0.0j
                ca_conj = np.conjugate(c_t[a])
                for b in range(nt):
                    acc += 0.5 * (ca_conj * (S_t[a, b] * c_t[b]) +
                                np.conjugate(c_t[b]) * (S_t[b, a] * c_t[a]))
                mval = float(np.real(acc))
                if mval < 0.0 and abs(mval) < 1.0e-12:
                    mval = 0.0
    
                lab = labs_i[a]
                j = col_of.get(lab, None)
                if j is not None:
                    mull[i, j] = mval

        self.datasets["quantum_times"] = times.reshape(-1, 1) if times.ndim == 1 else times
        self.datasets["mulliken_populations"] = mull
    
        if column_filename is not None:
            with open(column_filename, "w") as fout:
                fout.write("# time")
                for lab in all_labels:
                    fout.write("   Mull(%s)" % lab)
                fout.write("\n")
                for i in range(ntimes):
                    fout.write("%.10f" % times[i])
                    for j in range(L):
                        fout.write("   %.10f" % mull[i, j])
                    fout.write("\n")
        
    def fill_trajectory_populations(self, column_file_prefix=None):
        """Prints out state populations for every trajectory"""

        for key in self.labels:

            pop = self.get_traj_data_from_h5(key, "populations")
            dset_pop = key + "_pop"
            self.datasets[dset_pop] = pop

            if column_file_prefix is not None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key + "_time", [dset_pop],
                                              column_filename)

    def labels_this_step_rows(self):
        """
        Return a list-of-lists of labels active at each time step, or None if dataset absent.
        Uses sim/labels_this_step (CSV strings) if present.
        """
        if "sim" not in self.h5file:
            return None
        grp = self.h5file["sim"]
        if "labels_this_step" not in grp:
            return None
        rows = []
        ds = grp["labels_this_step"][()]
        for entry in ds:
            s = self.decode_bytes(entry)
            s = str(s) if not isinstance(s, (str, unicode)) else s
            if s.strip() == "":
                rows.append([])
            else:
                rows.append([x.strip() for x in s.split(",") if x.strip() != ""])
        return rows

    def decode_bytes(self, b): 
        try:
            return b.decode("utf-8")
        except Exception:
            return b

    def fill_nuclear_bf_populations(self, column_filename=None): 
        """ 
        Build nuclear basis-function populations aligned by *global label order*.
        Returns N with shape (ntimes, 1 + L), where L = len(all_labels).
        - Column 0: total norm  c^* S c  (real)
        - Column j>0: population of label all_labels[j-1] at each time. If that label
            is inactive at a time, the column has 0 for that time.
        """   
        all_labels = self.all_traj_labels()
        L = len(all_labels)
        self.datasets["labels"] = list(all_labels)
    
        grp_sim = self.h5file["sim"]
        qtimes = grp_sim["quantum_time"][()][:, 0]
        ntimes = len(qtimes)   
        qampls = grp_sim["qm_amplitudes"][()]
        Sflat  = grp_sim["S"][()]
        ntraj_each = grp_sim["num_traj_qm"][()].flatten().astype(int)    
        rows_labels = self.labels_this_step_rows()       
        Nlab = np.zeros((ntimes, 1 + L), dtype=float)
    
        col_of_label = {}
        for j, lab in enumerate(all_labels):
            col_of_label[lab] = 1 + j
    
        for i in range(ntimes):
            nt = int(ntraj_each[i])
            if nt <= 0:
                continue
    
            c_t = qampls[i][:nt]
            S_t = self.row_mats_from_flat(Sflat[i], nt)
            nvec = np.zeros(nt, dtype=float)
            for a in range(nt):
                acc = 0.0 + 0.0j
                ca_conj = np.conjugate(c_t[a])
                for b in range(nt):
                    acc += 0.5 * (ca_conj * (S_t[a, b] * c_t[b]) + np.conjugate(c_t[b]) * (S_t[b, a] * c_t[a]))
                nval = float(np.real(acc))
                if nval < 0.0 and abs(nval) < 1.0e-12:
                    nval = 0.0
                nvec[a] = nval
    
            Ptot = float(np.real(np.dot(c_t.conjugate(), np.dot(S_t, c_t))))
            if Ptot < 0.0 and abs(Ptot) < 1.0e-12:
                Ptot = 0.0
            Nlab[i, 0] = Ptot
    
            if rows_labels is not None and i < len(rows_labels):
                labs_i = rows_labels[i]
                if len(labs_i) != nt:
                    labs_i = list(labs_i[:nt]) + ([""] * max(0, nt - len(labs_i)))
            else:
                labs_i = all_labels[:nt]
    
            for a in range(nt):
                lab = labs_i[a]
                j = col_of_label.get(lab, None)
                if j is not None:
                    Nlab[i, j] = nvec[a]

        self.datasets["quantum_times"] = qtimes
        self.datasets["nuclear_bf_populations"] = Nlab
    
    def fill_expec_mulliken(self, dset_name, column_filename=None):
        """
        Mulliken-weighted expectation of a per-trajectory dataset (e.g., 'positions', 'momenta', etc).
        Output: dset_expec = 'expec_mull_' + dset_name  with shape (ntimes, ncol)
        where ncol = dataset's per-time column count for a trajectory (e.g., 3N for positions).
        """
        times = self.datasets["quantum_times"][:, 0]
        ntimes = len(times)
    
        if "labels" not in self.datasets:
            self.fill_labels()
        if "mulliken_populations" not in self.datasets:
            self.fill_mulliken_populations()
        labels = self.datasets["labels"]
        mull = self.datasets["mulliken_populations"]
    
        ncol = None
        first_lab_with = None
        for lab in labels:
            gname = "traj_" + lab
            if gname in self.h5file and dset_name in self.h5file[gname]:
                shape = self.h5file[gname][dset_name].shape
                if len(shape) == 2:
                    ncol = int(shape[1])
                else:
                    ncol = 1
                first_lab_with = lab
                break
        if ncol is None:
            return
    
        X = np.zeros((ntimes, ncol), dtype=float)
        denom = np.sum(mull, axis=1)
        for itraj, lab in enumerate(labels):
            try:
                g = self.h5file["traj_" + lab]
                if dset_name not in g:
                    continue
                xk = g[dset_name][()]
                tk = g["time"][()][:, 0]
   
                if tk.size == 0:
                    continue
                firsttime = tk[0]
                lasttime = tk[-1]
    
                ifirsttime = None
                tol = 1.0e-6
                for it in range(ntimes):
                    dt = times[it] - firsttime
                    if (-tol < dt) and (dt < tol):
                        ifirsttime = it
                        break
                if ifirsttime is None:
                    diffs = np.abs(times - firsttime)
                    ifirsttime = int(np.argmin(diffs))
                ilasttime = int(tk.size)

                if ifirsttime + ilasttime > ntimes:
                    ilasttime = ntimes - ifirsttime
                if ilasttime <= 0:
                    continue
                    
                w = mull[ifirsttime:ifirsttime + ilasttime, itraj]

                if xk.ndim == 1:
                    X[ifirsttime:ifirsttime + ilasttime, 0] += xk[:ilasttime] * w
                else:
                    X[ifirsttime:ifirsttime + ilasttime, :] += (xk[:ilasttime, :] * w.reshape(-1, 1))
    
            except Exception:
                pass
                
        for i in range(ntimes):
            d = denom[i]
            if d > 1.0e-16:
                X[i, :] = X[i, :] / d
    
        dset_expec = "expec_mull_" + dset_name
        self.datasets[dset_expec] = X
    
        if column_filename is not None:
            self.write_columnar_data_file("quantum_times", [dset_expec], column_filename)
           
    def fill_electronic_state_populations(self, column_filename=None):
        """
        Calculates population on each electronic state.
        Uses per-time labels/istates if present, else falls back to legacy attrs.
        Writes dataset: "electronic_state_populations" with shape (ntimes, maxstates+1)
        where last column is total norm c^*Sc.
        """
        times = self.datasets["quantum_times"][:, 0]
        ntimes = len(times)
        ntraj_row = self.h5file["sim/num_traj_qm"][()].flatten()
        qampls = self.h5file["sim/qm_amplitudes"][()]
        S_flat = self.h5file["sim/S"][()]
    
        maxstates = self.get_max_state()
        Nstate = np.zeros((ntimes, maxstates+1), dtype=float)
    
        labels_rows = getattr(self, "labels_rows", None)
        istates_rows = getattr(self, "istates_rows", None)
    
        for i in range(ntimes):
            nt = int(ntraj_row[i])
            if nt <= 0:
                continue
                
            c_t = qampls[i][:nt]
            S_t = self.row_mats_from_flat(S_flat[i], nt)
    
            row_states = None
            if istates_rows and i < len(istates_rows):
                row_states = istates_rows[i]
            if (row_states is None) or (len(row_states) != nt):
                row_states = [int(x) for x in self.istates[:nt]]
    
            for I in range(maxstates):
                idxI = [k for k, s in enumerate(row_states) if int(s) == I]
                if not idxI:
                    Nstate[i, I] = 0.0
                    continue
                S_II = S_t[np.ix_(idxI, idxI)]
                c_I  = c_t[idxI]
                P_I  = np.dot(c_I.conjugate(), np.dot(S_II, c_I))
                P_I  = float(np.real(P_I))
                if P_I < 0.0 and abs(P_I) < 1e-12:
                    P_I = 0.0
                Nstate[i, I] = P_I
    
            Ptot = np.dot(c_t.conjugate(), np.dot(S_t, c_t))
            Ptot = float(np.real(Ptot))
            if Ptot < 0.0 and abs(Ptot) < 1e-12:
                Ptot = 0.0
            Nstate[i, maxstates] = Ptot
    
        self.datasets["electronic_state_populations"] = Nstate
    
        if column_filename is not None:
            self.write_columnar_data_file("quantum_times",
                                        ["electronic_state_populations"],
                                        column_filename)        

    def write_xyzs(self):
        """Prints out geometries into .xyz file"""
        
        if "labels" not in self.datasets:
            self.fill_labels()
        labels = self.datasets.get("labels", [])
        for key in labels:
            times = self.get_traj_dataset(key, "time")[:, 0]
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key, "positions")
            pos /= 1.8897161646321
            npos = pos.size / ntimes
            natoms = npos/3
            atoms = self.get_traj_attr_from_h5(key, "atoms")

            filename = "traj_" + key + ".xyz"
            of = open(filename, "w")

            for itime in range(ntimes):
                of.write(str(natoms)+"\n")
                of.write("T = "+str(times[itime])+"\n")
                for iatom in range(natoms):
                    of.write(atoms[iatom] + "  " + str(pos[itime, 3*iatom])
                             + "  " + str(pos[itime, 3*iatom + 1])
                             + "  " + str(pos[itime, 3*iatom + 2]) + "\n")

            of.close()

    def fill_trajectory_energies(self, column_file_prefix=None):
        """
        Creates datasets with energies for each trajectory and optionally
        prints into text file. SSAIMS-safe: includes all traj_* groups even if pruned.
        """
        traj_keys = self.all_traj_labels()
        for key in traj_keys:
            try:
                grp = self.h5file["traj_" + key]
                time = grp["time"][()]
                mom  = grp["momenta"][()]
                ener = grp["energies"][()]
                try:
                    masses = grp.attrs["masses"]
                except Exception:
                    masses = None
    
                try:
                    istate = int(grp.attrs["istate"])
                except Exception:
                    istate = 0
    
                T = len(time)
                if mom.shape[0] < T:
                    T = mom.shape[0]
                if ener.shape[0] < T:
                    T = ener.shape[0]
    
                time = time[:T]
                mom  = mom[:T, :]
                ener = ener[:T, :]
    
                kinen = np.zeros((T, 1), dtype=float)
                toten = np.zeros((T, 1), dtype=float)
    
                if masses is None:
                    has_masses = False
                else:
                    masses = np.asarray(masses, dtype=float)
                    has_masses = (masses.size == mom.shape[1])
    
                for it in range(T):
                    if has_masses:
                        p = mom[it, :]
                        kinen[it, 0] = 0.5 * np.sum((p * p) / masses)
                    else:
                        kinen[it, 0] = 0.0
                    if istate < ener.shape[1]:
                        toten[it, 0] = kinen[it, 0] + ener[it, istate]
                    else:
                        toten[it, 0] = kinen[it, 0] + ener[it, 0]

                dset_poten = key + "_poten"
                dset_kinen = key + "_kinen"
                dset_toten = key + "_toten"
                dset_time  = key + "_time"
    
                self.datasets[dset_poten] = ener
                self.datasets[dset_kinen] = kinen
                self.datasets[dset_toten] = toten
                self.datasets[dset_time]  = time
    
                if column_file_prefix is not None:
                    column_filename = column_file_prefix + "_" + key + ".dat"
                    cols = np.zeros((T, 4), dtype=float)
                    cols[:, 0] = time[:, 0]
                    cols[:, 1] = ener[:, istate] if istate < ener.shape[1] else ener[:, 0]
                    cols[:, 2] = kinen[:, 0]
                    cols[:, 3] = toten[:, 0]
                    with open(column_filename, "w") as fout:
                        fout.write("# time  poten(istate=%d)  kinen  toten\n" % istate)
                        for it in range(T):
                            fout.write("%.10f  %.10f  %.10f  %.10f\n" % (cols[it,0], cols[it,1], cols[it,2], cols[it,3]))
    
            except Exception as e:
                try:
                    print "Skipping trajectory %s due to error: %s" % (key, str(e))
                except Exception:
                    pass
                                                                                                   
    def fill_trajectory_bonds(self, bonds, column_file_prefix):
        """
        Calculates bond distances and writes it into datasets key_bonds
        and column_file_prefix file
        """
        
        if "labels" not in self.datasets:
            self.fill_labels()
        labels = self.datasets.get("labels", [])

        for key in labels:
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key, "positions")
            nbonds = bonds.size / 2

            d = np.zeros((ntimes, nbonds))

            for itime in range(ntimes):
                for ibond in range(nbonds):
                    ipos = 3*bonds[ibond, 0]
                    jpos = 3*bonds[ibond, 1]
                    ri = pos[itime, ipos:(ipos+3)]
                    rj = pos[itime, jpos:(jpos+3)]
                    r = ri-rj
                    d[itime, ibond] = math.sqrt(np.sum(r*r))

            dset_bonds = key + "_bonds"

            self.datasets[dset_bonds] = d

            if column_file_prefix is not None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key + "_time",
                                              [dset_bonds], column_filename)

    def fill_trajectory_angles(self, angles, column_file_prefix):
        """Calculates regular angles from 3 atom positions"""

        if "labels" not in self.datasets:
            self.fill_labels()
        labels = self.datasets.get("labels", [])

        for key in labels:
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key, "positions")
            nangles = angles.size / 3

            ang = np.zeros((ntimes, nangles))

            for itime in range(ntimes):
                for iang in range(nangles):
                    ipos = 3*angles[iang, 0]
                    jpos = 3*angles[iang, 1]
                    kpos = 3*angles[iang, 2]
                    ri = pos[itime, ipos:(ipos+3)]
                    rj = pos[itime, jpos:(jpos+3)]
                    rk = pos[itime, kpos:(kpos+3)]

                    rji = ri - rj
                    rjk = rk - rj

                    dji = math.sqrt(np.sum(rji*rji))
                    djk = math.sqrt(np.sum(rjk*rjk))

                    rji /= dji
                    rjk /= djk

                    dot = np.sum(rji * rjk)

                    ang[itime, iang] = math.acos(dot) / math.pi * 180.0

            dset_angles = key + "_angles"

            self.datasets[dset_angles] = ang

            if column_file_prefix is not None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key + "_time",
                                              [dset_angles], column_filename)

    def fill_trajectory_diheds(self, diheds, column_file_prefix):
        """Calculates dihedral angles (4 atoms input)"""
        
        if "labels" not in self.datasets:
            self.fill_labels()
        labels = self.datasets.get("labels", [])

        for key in labels:
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key, "positions")
            ndiheds = diheds.size / 4

            dih = np.zeros((ntimes, ndiheds))

            for itime in range(ntimes):
                for idih in range(ndiheds):
                    ipos = 3*diheds[idih, 0]
                    jpos = 3*diheds[idih, 1]
                    kpos = 3*diheds[idih, 2]
                    lpos = 3*diheds[idih, 3]
                    ri = pos[itime, ipos:(ipos+3)]
                    rj = pos[itime, jpos:(jpos+3)]
                    rk = pos[itime, kpos:(kpos+3)]
                    rl = pos[itime, lpos:(lpos+3)]

                    rji = ri - rj
                    rkj = rj - rk
                    rjk = -1.0 * rkj
                    rkl = rl - rk

                    rjicrossrjk = np.cross(rji, rjk)
                    rkjcrossrkl = np.cross(rkj, rkl)

                    normjijk = math.sqrt(np.sum(rjicrossrjk*rjicrossrjk))
                    normkjkl = math.sqrt(np.sum(rkjcrossrkl*rkjcrossrkl))

                    rjijk = rjicrossrjk / normjijk
                    rkjkl = rkjcrossrkl / normkjkl

                    dot = np.sum(rjijk * rkjkl)

                    dih[itime, idih] = math.acos(dot) / math.pi * 180.0

            dset_diheds = key + "_diheds"

            self.datasets[dset_diheds] = dih

            if column_file_prefix is not None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key + "_time",
                                              [dset_diheds], column_filename)

    def fill_trajectory_twists(self, twists, column_file_prefix):
        """Calculates twisting angles (6 atoms input)"""

        if "labels" not in self.datasets:
            self.fill_labels()
        labels = self.datasets.get("labels", [])

        for key in labels:
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key, "positions")
            ntwists = twists.size / 6

            twi = np.zeros((ntimes, ntwists))

            for itime in range(ntimes):
                for itwi in range(ntwists):
                    ipos = 3*twists[itwi, 0]
                    jpos = 3*twists[itwi, 1]
                    kpos = 3*twists[itwi, 2]
                    lpos = 3*twists[itwi, 3]
                    mpos = 3*twists[itwi, 4]
                    npos = 3*twists[itwi, 5]
                    ri = pos[itime, ipos:(ipos+3)]
                    rj = pos[itime, jpos:(jpos+3)]
                    rk = pos[itime, kpos:(kpos+3)]
                    rl = pos[itime, lpos:(lpos+3)]
                    rm = pos[itime, mpos:(mpos+3)]
                    rn = pos[itime, npos:(npos+3)]

                    rji = ri - rj
                    rkl = rl - rk
                    rmn = rn - rm

                    rjicrossrkl = np.cross(rji, rkl)
                    rjicrossrmn = np.cross(rji, rmn)

                    normjikl = math.sqrt(np.sum(rjicrossrkl*rjicrossrkl))
                    normjimn = math.sqrt(np.sum(rjicrossrmn*rjicrossrmn))

                    rjikl = rjicrossrkl / normjikl
                    rjimn = rjicrossrmn / normjimn

                    dot = np.sum(rjikl * rjimn)

                    twi[itime, itwi] = math.acos(dot) / math.pi * 180.0

            dset_twists = key + "_twists"

            self.datasets[dset_twists] = twi

            if column_file_prefix is not None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key + "_time",
                                              [dset_twists], column_filename)

    def fill_trajectory_pyramidalizations(self, pyrs, column_file_prefix):
        """Calculates pyramidalization angles"""

        if "labels" not in self.datasets:
            self.fill_labels()
        labels = self.datasets.get("labels", [])

        for key in labels:
            ntimes = self.get_traj_num_times(key)
            pos = self.get_traj_data_from_h5(key, "positions")
            npyrs = pyrs.size / 4

            pyr = np.zeros((ntimes, npyrs))

            for itime in range(ntimes):
                for ipyr in range(npyrs):
                    ipos = 3*pyrs[ipyr, 0]
                    jpos = 3*pyrs[ipyr, 1]
                    kpos = 3*pyrs[ipyr, 2]
                    lpos = 3*pyrs[ipyr, 3]
                    ri = pos[itime, ipos:(ipos+3)]
                    rj = pos[itime, jpos:(jpos+3)]
                    rk = pos[itime, kpos:(kpos+3)]
                    rl = pos[itime, lpos:(lpos+3)]

                    rij = rj - ri
                    rik = rk - ri
                    ril = rl - ri

                    rikcrossril = np.cross(rik, ril)

                    normikil = math.sqrt(np.sum(rikcrossril*rikcrossril))
                    normij = math.sqrt(np.sum(rij*rij))

                    rikil = rikcrossril / normikil
                    rij /= normij

                    dot = np.sum(rikil * rij)

                    pyr[itime, ipyr] =\
                        math.asin(math.fabs(dot)) / math.pi * 180.0

            dset_pyrs = key + "_pyrs"

            self.datasets[dset_pyrs] = pyr

            if column_file_prefix is not None:
                column_filename = column_file_prefix + "_" + key + ".dat"
                self.write_columnar_data_file(key + "_time",
                                              [dset_pyrs], column_filename)

    def fill_trajectory_tdcs(self, column_file_prefix=None):
        """
        Collect time-derivative couplings for ALL traj_* groups and
        write per-trajectory text files if requested.
        """
        if "labels" not in self.datasets:
            labs = []
            for k in self.h5file.keys():
                if k.startswith("traj_"):
                    labs.append(k[len("traj_"):])
            labs.sort()
            self.datasets["labels"] = labs
    
        labels = self.datasets.get("labels", [])
    
        for key in labels:
            try:
                tdc = self.get_traj_data_from_h5(key, "timederivcoups")
                dset_tdc = key + "_tdc"
    
                try:
                    ths = self.get_traj_data_from_h5(key, "time_half_step")
                except Exception:
                    ths = self.get_traj_data_from_h5(key, "time")
                    
                T_tdc = tdc.shape[0]
                T_ths = ths.shape[0]
                T_use = T_tdc if T_tdc < T_ths else T_ths
    
                if T_use <= 0:
                    continue

                if ths.ndim == 1:
                    ths = ths.reshape((-1, 1))
                ths_use = ths[:T_use, :]
                tdc_use = tdc[:T_use, :]
    
                dset_time_hs = key + "_time_half_step"
                self.datasets[dset_time_hs] = ths_use
                self.datasets[dset_tdc]     = tdc_use
    
                if column_file_prefix is not None:
                    column_filename = column_file_prefix + "_" + key + ".dat"
                    self.write_columnar_data_file(dset_time_hs, [dset_tdc], column_filename)
    
            except Exception as e:
                try:
                    print "Skipping TDC for %s due to error: %s" % (key, str(e))
                except Exception:
                    pass
    
    def read_step_mapping(self):
        """
        Read per-step labels/istates if present (SSAIMS), else fall back to static sim attrs.
        self.labels_rows   : list of lists of labels per time step
        self.istates_rows  : list of lists of istates per time step
        self.states_sorted : sorted unique set of states seen anywhere (int)
        """
        grp_sim = self.h5file["sim"]
        qtimes = grp_sim["quantum_time"][()][:,0]
        n_times = len(qtimes)
    
        self.labels_rows = []
        self.istates_rows = []
  
        has_perstep = ("labels_this_step" in grp_sim) and ("istates_this_step" in grp_sim)
        if has_perstep:
            lbl_ds = grp_sim["labels_this_step"]
            ist_ds = grp_sim["istates_this_step"]
            T = min(n_times, lbl_ds.shape[0], ist_ds.shape[0])
            for i in range(T):
                try:
                    lbl_csv = lbl_ds[i]
                    if isinstance(lbl_csv, bytes):
                        lbl_csv = lbl_csv.decode("utf-8")
                    labels_i = lbl_csv.split(",") if lbl_csv else []
                except Exception:
                    labels_i = []
                try:
                    ist_csv = ist_ds[i]
                    if isinstance(ist_csv, bytes):
                        ist_csv = ist_csv.decode("utf-8")
                    istates_i = [int(x) for x in ist_csv.split(",")] if ist_csv else []
                except Exception:
                    istates_i = []
                if len(istates_i) != len(labels_i):
                    if len(istates_i) < len(labels_i):
                        istates_i += [-1]*(len(labels_i)-len(istates_i))
                    else:
                        istates_i = istates_i[:len(labels_i)]
                self.labels_rows.append(labels_i)
                self.istates_rows.append(istates_i)
            for _ in range(T, n_times):
                self.labels_rows.append([])
                self.istates_rows.append([])
        else:
            labels0  = self.decode_bytes_array(grp_sim.attrs.get("labels", []))
            istates0 = grp_sim.attrs.get("istates", np.array([], dtype=np.int32))
            istates0 = [int(x) for x in istates0.tolist()] if hasattr(istates0, "tolist") else list(istates0)
        
            traj_labels = []
            traj_states = []
            for k in self.h5file.keys():
                if k.startswith("traj_"):
                    try:
                        label = k.split("traj_")[1]
                        istate = int(self.h5file[k].attrs.get("istate", -1))
                        traj_labels.append(label)
                        traj_states.append(istate)
                    except Exception:
                        continue
        
            merged_labels = list(dict.fromkeys(labels0 + traj_labels))
            merged_istates = istates0 + [traj_states[traj_labels.index(l)] if l in traj_labels else -1 for l in merged_labels[len(istates0):]]
        
            for _ in range(n_times):
                self.labels_rows.append(list(merged_labels))
                self.istates_rows.append(list(merged_istates))

        st_set = set()
        for row in self.istates_rows:
            for s in row:
                if s is not None and int(s) >= 0:
                    st_set.add(int(s))
        self.states_sorted = sorted(st_set)
    