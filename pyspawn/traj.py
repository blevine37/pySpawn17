import numpy as np
import sys
import math
from pyspawn.fmsobj import fmsobj
import h5py


class traj(fmsobj):
    """Trajectory objects contain individual trajectory basis functions"""

    def __init__(self, numdims, numstates):
        self.time = 0.0
        self.time_half_step = 0.0
        self.maxtime = -1.0
        self.mintime = 0.0
        self.firsttime = 0.0
        self.numdims = numdims
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.widths = np.zeros(self.numdims)
        self.masses = np.zeros(self.numdims)
        self.istate = 0
        self.label = "00"
        self.h5_datasets = dict()
        self.h5_datasets_half_step = dict()

        self.timestep = 0.0
        #         self.propagator = "vv"
        self.numstates = numstates
        #         self.software = "pyspawn"
        #         self.method = "cone"
        self.length_wf = self.numstates
        self.wf = np.zeros((self.numstates, self.length_wf))
        self.prev_wf = np.zeros((self.numstates, self.length_wf))
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates, self.numdims))
        self.timederivcoups = np.zeros(self.numstates)
        self.S_elec_flat = np.zeros(self.numstates * self.numstates)

        self.backprop_time = 0.0
        self.backprop_time_half_step = 0.0
        self.backprop_energies = np.zeros(self.numstates)
        self.backprop_forces = np.zeros((self.numstates, self.numdims))
        self.backprop_positions = np.zeros(self.numdims)
        self.backprop_momenta = np.zeros(self.numdims)
        self.backprop_wf = np.zeros((self.numstates, self.length_wf))
        self.backprop_prev_wf = np.zeros((self.numstates, self.length_wf))
        self.backprop_timederivcoups = np.zeros(self.numstates)
        self.backprop_S_elec_flat = np.zeros(self.numstates * self.numstates)

        self.spawntimes = -1.0 * np.ones(self.numstates)
        self.spawnthresh = 0.0
        self.spawnlastcoup = np.zeros(self.numstates)
        self.positions_tpdt = np.zeros(self.numdims)
        self.positions_t = np.zeros(self.numdims)
        self.positions_tmdt = np.zeros(self.numdims)
        self.momenta_tpdt = np.zeros(self.numdims)
        self.momenta_t = np.zeros(self.numdims)
        self.momenta_tmdt = np.zeros(self.numdims)
        self.energies_tpdt = np.zeros(self.numstates)
        self.energies_t = np.zeros(self.numstates)
        self.energies_tmdt = np.zeros(self.numstates)
        self.z_spawn_now = np.zeros(self.numstates)
        self.z_dont_spawn = np.zeros(self.numstates)
        self.numchildren = 0

        self.positions_qm = np.zeros(self.numdims)
        self.momenta_qm = np.zeros(self.numdims)
        self.energies_qm = np.zeros(self.numstates)
        self.forces_i_qm = np.zeros(self.numdims)
        self.timederivcoups_qm = np.zeros(self.numstates)
        # when running terachem jobs we need to have a different port for every
        # terachem server instance
        self.tc_port = 0

    def set_time(self, t):
        self.time = t

    def get_time(self):
        return self.time

    def set_time_half_step(self, t):
        self.time_half_step = t

    def get_time_half_step(self):
        return self.time_half_step

    def set_timestep(self, h):
        self.timestep = h

    def get_timestep(self):
        return self.timestep

    def set_backprop_time(self, t):
        self.backprop_time = t

    def get_backprop_time(self):
        return self.backprop_time

    def set_backprop_time_half_step(self, t):
        self.backprop_time_half_step = t

    def get_backprop_time_half_step(self):
        return self.backprop_time_half_step

    def set_maxtime(self, t):
        self.maxtime = t

    def get_maxtime(self):
        return self.maxtime

    def set_firsttime(self, t):
        self.firsttime = t

    def get_firsttime(self):
        return self.firsttime

    #     def get_propagator(self):
    #         return self.propagator

    #     def set_propagator(self, prop):
    #         self.propagator = prop

    def get_mintime(self):
        return self.mintime

    def set_mintime(self, t):
        self.mintime = t

    def set_numdims(self, ndims):
        self.numdims = ndims
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.widths = np.zeros(self.numdims)
        self.masses = np.zeros(self.numdims)
        #         self.last_es_positions = np.zeros(self.numdims)
        self.forces = np.zeros((self.numstates, self.numdims))

        self.backprop_positions = np.zeros(self.numdims)
        self.backprop_momenta = np.zeros(self.numdims)
        self.backprop_forces = np.zeros((self.numstates, self.numdims))

        self.positions_t = np.zeros(self.numdims)
        self.positions_tmdt = np.zeros(self.numdims)
        self.positions_tpdt = np.zeros(self.numdims)
        self.momenta_t = np.zeros(self.numdims)
        self.momenta_tmdt = np.zeros(self.numdims)
        self.momenta_tpdt = np.zeros(self.numdims)
        #         self.prev_positions = np.zeros(self.numdims)
        #         self.prev_forces = np.zeros((self.numstates,self.numdims))
        self.positions_qm = np.zeros(self.numdims)
        self.momenta_qm = np.zeros(self.numdims)
        self.forces_i_qm = np.zeros(self.numdims)

    def set_numstates(self, nstates):
        self.numstates = nstates
        self.energies = np.zeros(self.numstates)
        self.forces = np.zeros((self.numstates, self.numdims))

        self.backprop_energies = np.zeros(self.numstates)
        self.backprop_forces = np.zeros((self.numstates, self.numdims))

        self.spawntimes = -1.0 * np.ones(self.numstates)
        self.timederivcoups = np.zeros(self.numstates)
        self.backprop_timederivcoups = np.zeros(self.numstates)
        self.timederivcoups_qm = np.zeros(self.numstates)
        self.spawnlastcoup = np.zeros(self.numstates)
        self.z_spawn_now = np.zeros(self.numstates)
        self.z_dont_spawn = np.zeros(self.numstates)

        self.energies_tpdt = np.zeros(self.numstates)
        self.energies_t = np.zeros(self.numstates)
        self.energies_tmdt = np.zeros(self.numstates)

    #         self.prev_energies = np.zeros(self.numstates)
    #         self.prev_forces = np.zeros((self.numstates,self.numdims))

    def set_istate(self, ist):
        self.istate = ist

    def get_istate(self):
        return self.istate

    def set_jstate(self, ist):
        self.jstate = ist

    def get_jstate(self):
        return self.jstate

    def get_numstates(self):
        return self.numstates

    def get_numdims(self):
        return self.numdims

    def set_numchildren(self, ist):
        self.numchildren = ist

    def get_numchildren(self):
        return self.numchildren

    def incr_numchildren(self):
        self.set_numchildren(self.get_numchildren() + 1)

    #     def set_software(self,sw):
    #         self.software = sw
    #
    #     def set_method(self,meth):
    #         self.method = meth
    #
    #     def get_software(self):
    #         return self.software
    #
    #     def get_method(self):
    #         return self.method

    def set_positions(self, pos):
        if pos.shape == self.positions.shape:
            self.positions = pos.copy()
        else:
            print "Error in set_positions"
            sys.exit()

    def get_positions(self):
        return self.positions.copy()

    def set_positions_qm(self, pos):
        if pos.shape == self.positions_qm.shape:
            self.positions_qm = pos.copy()
        else:
            print "Error in set_positions_qm"
            sys.exit()

    def get_positions_qm(self):
        return self.positions_qm.copy()

    def set_positions_t(self, pos):
        if pos.shape == self.positions_t.shape:
            self.positions_t = pos.copy()
        else:
            print "Error in set_positions_t"
            sys.exit()

    def get_positions_t(self):
        return self.positions_t.copy()

    def set_positions_tmdt(self, pos):
        if pos.shape == self.positions_tmdt.shape:
            self.positions_tmdt = pos.copy()
        else:
            print "Error in set_positions_tmdt"
            sys.exit()

    def get_positions_tmdt(self):
        return self.positions_tmdt.copy()

    def set_positions_tpdt(self, pos):
        if pos.shape == self.positions_tpdt.shape:
            self.positions_tpdt = pos.copy()
        else:
            print "Error in set_positions_tpdt"
            sys.exit()

    def get_positions_tpdt(self):
        return self.positions_tpdt.copy()

    def set_momenta_qm(self, mom):
        if mom.shape == self.momenta_qm.shape:
            self.momenta_qm = mom.copy()
        else:
            print "Error in set_momenta"
            sys.exit()

    def get_momenta_qm(self):
        return self.momenta_qm.copy()

    def set_forces_i_qm(self, f):
        if f.shape == self.forces_i_qm.shape:
            self.forces_i_qm = f.copy()
        else:
            print "Error in set_forces_i_qm"
            sys.exit()

    def get_forces_i_qm(self):
        return self.forces_i_qm.copy()

    def set_momenta(self, mom):
        if mom.shape == self.momenta.shape:
            self.momenta = mom.copy()
        else:
            print "Error in set_momenta"
            sys.exit()

    def get_momenta(self):
        return self.momenta.copy()

    def set_momenta_t(self, mom):
        if mom.shape == self.momenta_t.shape:
            self.momenta_t = mom.copy()
        else:
            print "Error in set_momenta_t"
            sys.exit()

    def get_momenta_t(self):
        return self.momenta_t.copy()

    def set_momenta_tmdt(self, mom):
        if mom.shape == self.momenta_tmdt.shape:
            self.momenta_tmdt = mom.copy()
        else:
            print "Error in set_momenta_tmdt"
            sys.exit()

    def get_momenta_tmdt(self):
        return self.momenta_tmdt.copy()

    def set_momenta_tpdt(self, mom):
        if mom.shape == self.momenta_tpdt.shape:
            self.momenta_tpdt = mom.copy()
        else:
            print "Error in set_momenta_tpdt"
            sys.exit()

    def get_momenta_tpdt(self):
        return self.momenta_tpdt.copy()

    def set_energies_qm(self, e):
        if e.shape == self.energies_qm.shape:
            self.energies_qm = e.copy()
        else:
            print "Error in set_energies_qm"
            sys.exit()

    def get_energies_qm(self):
        return self.energies_qm.copy()

    def set_energies_t(self, e):
        if e.shape == self.energies_t.shape:
            self.energies_t = e.copy()
        else:
            print "Error in set_energies_t"
            sys.exit()

    def get_energies_t(self):
        return self.energies_t.copy()

    def set_energies_tmdt(self, e):
        if e.shape == self.energies_tmdt.shape:
            self.energies_tmdt = e.copy()
        else:
            print "Error in set_energies_tmdt"
            sys.exit()

    def get_energies_tmdt(self):
        return self.energies_tmdt.copy()

    def set_energies_tpdt(self, e):
        if e.shape == self.energies_tpdt.shape:
            self.energies_tpdt = e.copy()
        else:
            print "Error in set_energies_tpdt"
            sys.exit()

    def get_energies_tpdt(self):
        return self.energies_tpdt.copy()

    def set_backprop_positions(self, pos):
        if pos.shape == self.backprop_positions.shape:
            self.backprop_positions = pos.copy()
        else:
            print "Error in set_backprop_positions"
            sys.exit()

    def get_backprop_positions(self):
        return self.backprop_positions.copy()

    def set_backprop_momenta(self, mom):
        if mom.shape == self.backprop_momenta.shape:
            self.backprop_momenta = mom.copy()
        else:
            print "Error in set_backprop_momenta"
            sys.exit()

    def get_backprop_momenta(self):
        return self.backprop_momenta.copy()

    def set_widths(self, wid):
        if wid.shape == self.widths.shape:
            self.widths = wid.copy()
        else:
            print "Error in set_widths"
            sys.exit()

    def get_widths(self):
        return self.widths.copy()

    def set_masses(self, m):
        if m.shape == self.masses.shape:
            self.masses = m.copy()
        else:
            print "Error in set_masses"
            sys.exit()

    def get_masses(self):
        return self.masses.copy()

    def set_atoms(self, a):
        self.atoms = a[:]

    def get_atoms(self):
        return self.atoms[:]

    def set_label(self, lab):
        self.label = lab

    def get_label(self):
        #         if self.label.type == 'unicode':
        #             self.set_label(str(self.label))
        return self.label

    def get_tc_port(self):
        return self.tc_port

    def set_tc_port(self, port):
        self.tc_port = port

    def init_traj(self, t, ndims, pos, mom, wid, m, nstates, istat, lab):
        """Initializes trajectory, mainly used for tests"""

        self.set_time(t)
        self.set_numdims(ndims)
        self.set_positions(pos)
        self.set_momenta(mom)
        self.set_widths(wid)
        self.set_masses(m)
        self.set_label(lab)
        self.set_numstates(nstates)
        self.set_istate(istat)

        self.set_backprop_time(t)
        self.set_backprop_positions(pos)
        self.set_backprop_momenta(mom)

        self.set_firsttime(t)

    def init_spawn_traj(self, parent, istate, label):
        """Initializing a child and making on its new istate
        with a new label and making sure we copy appropriate
        parameters from parent"""

        self.set_numstates(parent.get_numstates())
        self.set_numdims(parent.get_numdims())

        self.set_istate(istate)

        time = parent.get_time() - 2.0 * parent.get_timestep()
        self.set_time(time)

        self.set_label(label)

        pos = parent.get_positions_tmdt()
        mom = parent.get_momenta_tmdt()
        e = parent.get_energies_tmdt()
        self.set_positions(pos)
        self.set_momenta(mom)
        self.set_energies(e)

        mintime = float(parent.get_spawntimes()[istate])
        self.set_mintime(mintime)
        self.set_backprop_time(time)
        self.set_backprop_positions(pos)

        self.set_firsttime(time)

        self.set_maxtime(parent.get_maxtime())
        self.set_widths(parent.get_widths())
        self.set_masses(parent.get_masses())
        if hasattr(parent, 'atoms'):
            self.set_atoms(parent.get_atoms())
        if hasattr(parent, 'civecs'):
            self.set_civecs(parent.get_civecs())
            self.set_backprop_civecs(parent.get_civecs())
            self.set_ncivecs(parent.get_ncivecs())
        if hasattr(parent, 'orbs'):
            self.set_orbs(parent.get_orbs())
            self.set_backprop_orbs(parent.get_orbs())
            self.set_norbs(parent.get_norbs())
        if hasattr(parent, 'prev_wf_positions'):
            self.set_prev_wf_positions(parent.get_prev_wf_positions())
            self.set_backprop_prev_wf_positions(parent.get_prev_wf_positions())
        if hasattr(parent, 'electronic_phases'):
            self.set_electronic_phases(parent.get_electronic_phases())
            self.set_backprop_electronic_phases(parent.get_electronic_phases())
        if hasattr(parent, 'wfn'):
            self.set_wfn(parent.get_wfn())
            self.set_backprop_wfn(parent.get_wfn())
        if hasattr(parent, 'inporbs'):
            self.set_inporbs(parent.get_inporbs())
            self.set_backprop_inporbs(parent.get_inporbs())
        

        self.set_timestep(parent.get_timestep())
        #         self.set_propagator(parent.get_propagator())

        z_dont = np.zeros(parent.get_numstates())
        z_dont[parent.get_istate()] = 1.0
        self.set_z_dont_spawn(z_dont)
        self.set_spawnthresh(parent.get_spawnthresh())

        self.potential_specific_traj_copy(parent)

        # copying port for terachem jobs
        self.set_tc_port(parent.tc_port)

    def init_centroid(self, existing, child, label):
        ts = child.get_timestep()

        self.set_numstates(child.get_numstates())
        self.set_numdims(child.get_numdims())

        self.set_istate(child.get_istate())
        self.set_jstate(existing.get_istate())

        time = child.get_time()
        self.set_time(time - ts)

        self.set_label(label)

        mintime = max(child.get_mintime(), existing.get_mintime())
        self.set_mintime(mintime)
        self.set_backprop_time(time + ts)
        self.set_firsttime(time)
        self.set_maxtime(child.get_maxtime())

        self.set_widths(child.get_widths())
        self.set_masses(child.get_masses())
        if hasattr(child, 'atoms'):
            self.set_atoms(child.get_atoms())
        if hasattr(child, 'civecs'):
            self.set_civecs(child.get_civecs())
            self.set_backprop_civecs(child.get_civecs())
            self.set_ncivecs(child.get_ncivecs())
        if hasattr(child, 'orbs'):
            self.set_orbs(child.get_orbs())
            self.set_backprop_orbs(child.get_orbs())
            self.set_norbs(child.get_norbs())
        if hasattr(child, 'prev_wf_positions'):
            self.set_prev_wf_positions(child.get_prev_wf_positions())
            self.set_backprop_prev_wf_positions(child.get_prev_wf_positions())
        if hasattr(child, 'electronic_phases'):
            self.set_electronic_phases(child.get_electronic_phases())
            self.set_backprop_electronic_phases(child.get_electronic_phases())
        if hasattr(child, 'wfn'):
            self.set_wfn(child.get_wfn())
            self.set_backprop_wfn(child.get_wfn())
        if hasattr(child, 'inporbs'):
            self.set_inporbs(child.get_inporbs())
            self.set_backprop_inporbs(child.get_inporbs())
        self.set_timestep(ts)
        self.potential_specific_traj_copy(existing)

        # copying port for tc job
        self.set_tc_port(existing.tc_port)

    def rescale_momentum(self, v_parent):
        """ Computing kinetic energy of parent.  Remember that, at this point,
        the child's momentum is still that of the parent, so we compute
        t_parent from the child's momentum"""

        v_child = self.get_energies()[self.get_istate()]
        #         print "rescale v_child ", v_child
        #         print "rescale v_parent ", v_parent
        p_parent = self.get_momenta()
        m = self.get_masses()
        t_parent = 0.0
        for idim in range(self.get_numdims()):
            t_parent += 0.5 * p_parent[idim] * p_parent[idim] / m[idim]
        #         print "rescale t_parent ", t_parent
        factor = ((v_parent + t_parent - v_child) / t_parent)
        if factor < 0.0:
            print "# Aborting spawn because child does not have"
            print "# enough energy for momentum adjustment"
            return False
        factor = math.sqrt(factor)
        print "# rescaling momentum by factor ", factor
        p_child = factor * p_parent
        self.set_momenta(p_child)
        self.set_backprop_momenta(p_child)

        return True

    def set_forces(self, f):
        if f.shape == self.forces.shape:
            self.forces = f.copy()
        else:
            print "Error in set_forces"
            sys.exit()

    def get_forces(self):
        return self.forces.copy()

    def get_forces_i(self):
        fi = self.get_forces()[self.get_istate(), :]
        return fi

    def set_backprop_forces(self, f):
        if f.shape == self.backprop_forces.shape:
            self.backprop_forces = f.copy()
        else:
            print "Error in set_forces"
            sys.exit()

    def get_backprop_forces(self):
        return self.backprop_forces.copy()

    def get_backprop_forces_i(self):
        fi = self.get_backprop_forces()[self.get_istate(), :]
        return fi

    def set_energies(self, e):
        if e.shape == self.energies.shape:
            self.energies = e.copy()
        else:
            print "Error in set_forces"
            sys.exit()

    def get_energies(self):
        return self.energies.copy()

    def set_backprop_energies(self, e):
        if e.shape == self.backprop_energies.shape:
            self.backprop_energies = e.copy()
        else:
            print "Error in set_forces"
            sys.exit()

    def get_backprop_energies(self):
        return self.backprop_energies.copy()

    def set_wf(self, wf):
        if wf.shape == self.wf.shape:
            self.wf = wf.copy()
        else:
            print "Error in set_wf"
            sys.exit()

    def get_wf(self):
        return self.wf.copy()

    def set_prev_wf(self, wf):
        if wf.shape == self.prev_wf.shape:
            self.prev_wf = wf.copy()
        else:
            print "Error in set_prev_wf"
            sys.exit()

    def get_prev_wf(self):
        return self.prev_wf.copy()

    def set_backprop_wf(self, wf):
        if wf.shape == self.backprop_wf.shape:
            self.backprop_wf = wf.copy()
        else:
            print "Error in set_backprop_wf"
            sys.exit()

    def get_backprop_wf(self):
        return self.backprop_wf.copy()

    def set_backprop_prev_wf(self, wf):
        if wf.shape == self.backprop_prev_wf.shape:
            self.backprop_prev_wf = wf.copy()
        else:
            print "Error in set_backprop_prev_wf"
            sys.exit()

    def get_backprop_prev_wf(self):
        return self.backprop_prev_wf.copy()

    def set_spawntimes(self, st):
        if st.shape == self.spawntimes.shape:
            self.spawntimes = st.copy()
        else:
            print "Error in set_spawntimes"
            sys.exit()

    def get_spawntimes(self):
        return self.spawntimes.copy()

    def set_timederivcoups(self, t):
        if t.shape == self.timederivcoups.shape:
            self.timederivcoups = t.copy()
        else:
            print "Error in set_spawntimes"
            sys.exit()

    def get_timederivcoups(self):
        return self.timederivcoups.copy()

    def set_backprop_timederivcoups(self, t):
        if t.shape == self.backprop_timederivcoups.shape:
            # we multiply by -1.0 because the tdc is computed as the
            # derivative with respect to -t during back propagation
            self.backprop_timederivcoups = -1.0 * t.copy()
        else:
            print "Error in set_spawntimes"
            sys.exit()

    def get_backprop_timederivcoups(self):
        return self.backprop_timederivcoups.copy()

    def set_S_elec_flat(self, S):
        self.S_elec_flat = S.copy()

    def get_S_elec_flat(self):
        return self.S_elec_flat.copy()

    def set_backprop_S_elec_flat(self, S):
        self.backprop_S_elec_flat = S.copy()

    def get_backprop_S_elec_flat(self):
        return self.backprop_S_elec_flat.copy()

    def set_timederivcoups_qm(self, t):
        if t.shape == self.timederivcoups_qm.shape:
            self.timederivcoups_qm = t.copy()
        else:
            print "Error in set_spawntimes"
            sys.exit()

    def get_timederivcoups_qm(self):
        return self.timederivcoups_qm.copy()

    def set_spawnlastcoup(self, tdc):
        if tdc.shape == self.spawnlastcoup.shape:
            self.spawnlastcoup = tdc.copy()
        else:
            print "Error in set_spawnlastcoup"
            sys.exit()

    def get_spawnlastcoup(self):
        return self.spawnlastcoup.copy()

    def set_z_spawn_now(self, z):
        if z.shape == self.z_spawn_now.shape:
            self.z_spawn_now = z.copy()
        else:
            print "Error in set_z_spawn_now"
            sys.exit()

    def get_z_spawn_now(self):
        return self.z_spawn_now.copy()

    def set_z_dont_spawn(self, z):
        if z.shape == self.z_dont_spawn.shape:
            self.z_dont_spawn = z.copy()
        else:
            print "Error in set_z_dont_spawn"
            sys.exit()

    def get_z_dont_spawn(self):
        return self.z_dont_spawn.copy()

    def set_spawnthresh(self, t):
        self.spawnthresh = t

    def get_spawnthresh(self):
        return self.spawnthresh

    def set_z_compute_me(self, z):
        self.z_compute_me = z

    def get_z_compute_me(self):
        return self.z_compute_me

    def set_z_compute_me_backprop(self, z):
        self.z_compute_me_backprop = z

    def get_z_compute_me_backprop(self):
        return self.z_compute_me_backprop

    #    def compute_elec_struct(self, zbackprop):
    #        tmp = "self.compute_elec_struct_" + self.get_software() + "_"\
    #            + self.get_method() + "(zbackprop)"
    #        eval(tmp)

    def propagate_step(self, zbackprop=False):
        """Performs classical propagation for one step"""

        if not zbackprop:
            cbackprop = ""
        else:
            cbackprop = "backprop_"
        if abs(eval("self.get_" + cbackprop + "time()")
               - self.get_firsttime()) < 1.0e-6:
            self.prop_first_step(zbackprop=zbackprop)
        else:
            self.prop_not_first_step(zbackprop=zbackprop)

        # consider whether to spawn
        if not zbackprop:
            self.consider_spawning()

    def compute_centroid(self, zbackprop=False):
        firsttime = self.get_firsttime()
        dt = self.get_timestep()
        if zbackprop:
            cbackprop = "backprop_"
            sign = -1.0
        else:
            cbackprop = ""
            sign = 1.0
        t = eval("self.get_" + cbackprop + "time()")
        t += sign * dt
        exec ("self.set_" + cbackprop + "time(t)")
        exec ("self.set_" + cbackprop + "time_half_step(t + sign * -0.5 * dt)")
        # if it is this trajectories first timestep (forward or backward)
        self.compute_elec_struct(zbackprop)
        # only output on forward propagation
        if abs(t - firsttime) < 1.0e-6:
            if not zbackprop:
                self.h5_output(zbackprop, zdont_half_step=True)
        else:
            self.h5_output(zbackprop)

    def consider_spawning(self):
        """Here we decide if trajectory is spawning at the current time step"""

        tdc = self.get_timederivcoups()
        lasttdc = self.get_spawnlastcoup()
        spawnt = self.get_spawntimes()
        thresh = self.get_spawnthresh()
        z_dont_spawn = self.get_z_dont_spawn()
        z = self.get_z_spawn_now()

        for jstate in range(self.numstates):
            if jstate != self.get_istate():
                if spawnt[jstate] > -1.0e-6:
                    # check to see if a trajectory in a spawning region 
                    # is ready to spawn (reached maximum coupling)
                    if abs(tdc[jstate]) < abs(lasttdc[jstate]):
                        # print "Spawning to state ", jstate, " at time ", self.get_time()
                        # setting z_spawn_now indicates that
                        # this trajectory should spawn to jstate
                        z[jstate] = 1.0
                else:
                    # check to see if trajectory is entering a spawning region
                    if (abs(tdc[jstate]) > thresh) and (z_dont_spawn[jstate] < 0.5):
                        spawnt[jstate] = self.get_time() - self.get_timestep()
                        print "## trajectory " + self.get_label() + \
                              " entered spawning region for state ", jstate, \
                            " at time ", spawnt[jstate]
                    else:
                        if (abs(tdc[jstate]) < (0.9 * thresh)) \
                                and (z_dont_spawn[jstate] > 0.5):
                            z_dont_spawn[jstate] = 0.0

        self.set_z_spawn_now(z)
        self.set_z_dont_spawn(z_dont_spawn)
        self.set_spawnlastcoup(tdc)
        self.set_spawntimes(spawnt)

    def h5_output(self, zbackprop, zdont_half_step=False):
        """Outputs data into h5 file"""

        if not zbackprop:
            cbackprop = ""
        else:
            cbackprop = "backprop_"
        if "_a_" not in self.get_label():
            traj_or_cent = "traj_"
        else:
            traj_or_cent = "cent_"
        if len(self.h5_datasets) == 0:
            self.init_h5_datasets()
        filename = "working.hdf5"
        #         extensions = [3,2,1,0]
        #         for i in extensions :
        #             if i==0:
        #                 ext = ""
        #             else:
        #                 ext = str(i) + "."
        #             filename = "sim." + ext + "hdf5"
        #             if os.path.isfile(filename):
        #                 if (i == extensions[0]):
        #                     os.remove(filename)
        #                 else:
        #                     ext = str(i+1) + "."
        #                     filename2 = "sim." + ext + "hdf5"
        #                     if (i == extensions[-1]):
        #                         shutil.copy2(filename, filename2)
        #                     else:
        #                         shutil.move(filename, filename2)
        h5f = h5py.File(filename, "a")
        groupname = traj_or_cent + self.label
        if groupname not in h5f.keys():
            self.create_h5_traj(h5f, groupname)
        trajgrp = h5f.get(groupname)
        all_datasets = self.h5_datasets.copy()
        if not zdont_half_step:
            all_datasets.update(self.h5_datasets_half_step)
        #for key, values in all_datasets.items():
            #print(key, values) 
        
        for key in all_datasets:
            n = all_datasets[key]
            #             print "key =", key
            dset = trajgrp.get(key)
            l = dset.len()
            dset.resize(l + 1, axis=0)
            if not zbackprop:
                ipos = l
            else:
                ipos = 0
                dset[1:(l + 1), 0:n] = dset[0:l, 0:n]
            getcom = "self.get_" + cbackprop + key + "()"
            #             print "getcom =", getcom
            tmp = eval(getcom)
            #             print "ipos =", ipos
            if n != 1:
                dset[ipos, 0:n] = tmp[0:n]
            else:
                dset[ipos, 0] = tmp
        h5f.flush()
        h5f.close()

    def create_h5_traj(self, h5f, groupname):
        """create a new trajectory group in hdf5 output file"""

        trajgrp = h5f.create_group(groupname)
        for key in self.h5_datasets:
            n = self.h5_datasets[key]
            #             print "key, n ", key, n
            trajgrp.create_dataset(key, (0, n), maxshape=(None, n),
                                   dtype="float64")
        for key in self.h5_datasets_half_step:
            n = self.h5_datasets_half_step[key]
            trajgrp.create_dataset(key, (0, n), maxshape=(None, n),
                                   dtype="float64")
        # add some metadata
        trajgrp.attrs["istate"] = self.istate
        trajgrp.attrs["masses"] = self.masses
        trajgrp.attrs["widths"] = self.widths
        trajgrp.attrs["tc_port"] = self.tc_port
        trajgrp.attrs["numstates"] = self.numstates
        trajgrp.attrs["spawnthresh"] = self.spawnthresh
        if hasattr(self, "atoms"):
            trajgrp.attrs["atoms"] = self.atoms

    def get_data_at_time_from_h5(self, t, dset_name):
        """Pulls data at full time step from h5 file"""

        h5f = h5py.File("working.hdf5", "r")
        if "_a_" not in self.get_label():
            traj_or_cent = "traj_"
        else:
            traj_or_cent = "cent_"
        groupname = traj_or_cent + self.label
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time"][:]
        #         print "size", dset_time.size
        ipoint = -1
        for i in range(len(dset_time)):
            if (dset_time[i] < t + 1.0e-6) and (dset_time[i] > t - 1.0e-6):
                ipoint = i
        #                 print "dset_time[i] ", dset_time[i]
        #                 print "i ", i
        dset = trajgrp[dset_name][:]
        data = np.zeros(len(dset[ipoint, :]))
        data = dset[ipoint, :]
        #         print "dset[ipoint,:] ", dset[ipoint,:]
        h5f.close()
        return data

    def get_all_qm_data_at_time_from_h5(self, t, suffix=""):
        """Pulls qm data from h5 file at full time step"""

        h5f = h5py.File("working.hdf5", "r")
        if "_a_" not in self.get_label():
            traj_or_cent = "traj_"
        else:
            traj_or_cent = "cent_"
        groupname = traj_or_cent + self.label
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time"][:]
        #         print "size", dset_time.size
        ipoint = -1
        for i in range(len(dset_time)):
            if (dset_time[i] < t + 1.0e-6) and (dset_time[i] > t - 1.0e-6):
                ipoint = i
        #                 print "dset_time[i] ", dset_time[i]
        #                 print "i ", i
        for dset_name in self.h5_datasets:
            dset = trajgrp[dset_name][:]
            data = np.zeros(len(dset[ipoint, :]))
            data = dset[ipoint, :]
            comm = "self." + dset_name + "_qm" + suffix + " = data"
            exec (comm)
        #             print "comm ", comm
        #             print "dset[ipoint,:] ", dset[ipoint,:]
        h5f.close()

    def get_all_qm_data_at_time_from_h5_half_step(self, t):
        """Pulls data from h5 file at half time step"""

        h5f = h5py.File("working.hdf5", "r")
        if "_a_" not in self.get_label():
            traj_or_cent = "traj_"
        else:
            traj_or_cent = "cent_"
        groupname = traj_or_cent + self.label
        trajgrp = h5f.get(groupname)
        dset_time = trajgrp["time_half_step"][:]
        #         print "size", dset_time.size
        ipoint = -1
        for i in range(len(dset_time)):
            if (dset_time[i] < t + 1.0e-6) and (dset_time[i] > t - 1.0e-6):
                ipoint = i
        #                 print "dset_time[i] ", dset_time[i]
        #                 print "i ", i
        for dset_name in self.h5_datasets_half_step:
            dset = trajgrp[dset_name][:]
            data = np.zeros(len(dset[ipoint, :]))
            data = dset[ipoint, :]
            comm = "self." + dset_name + "_qm = data"
            exec (comm)
            # print "comm ", comm
            # print "dset[ipoint,:] ", dset[ipoint,:]
        h5f.close()

    def compute_tdc(self, Win):
        """Computes derivative coupling matrix elements
        using NPI"""

        W = Win.copy()
        if 1.0 < W[0, 0]: # < 1.01:
            W[0, 0] = 1.0
        if -1.0 > W[0, 0]: # > -1.01:
            W[0, 0] = -1.0
        if 1.0 < W[1, 1]: # < 1.01:
            W[1, 1] = 1.0
        if -1.0 > W[1, 1]: # > -1.01:
            W[1, 1] = -1.0
        if 1.0 < W[0, 1]: # < 1.01:
            W[0, 0] = 1.0
        if -1.0 > W[0, 1]: # > -1.01:
            W[0, 0] = -1.0
        if 1.0 < W[1, 0]: # < 1.01:
            W[1, 1] = 1.0
        if -1.0 > W[1, 0]: # > -1.01:
            W[1, 1] = -1.0
        Atmp = np.arccos(W[0, 0]) - np.arcsin(W[0, 1])
        Btmp = np.arccos(W[0, 0]) + np.arcsin(W[0, 1])
        Ctmp = np.arccos(W[1, 1]) - np.arcsin(W[1, 0])
        Dtmp = np.arccos(W[1, 1]) + np.arcsin(W[1, 0])
        Wlj = np.sqrt(1 - W[0, 0] * W[0, 0] - W[1, 0] * W[1, 0])
        if Wlj != Wlj:
            Wlj = 0.0
        if np.absolute(Atmp) < 1.0e-6:
            A = -1.0
        else:
            A = -1.0 * np.sin(Atmp) / Atmp
        if np.absolute(Btmp) < 1.0e-6:
            B = 1.0
        else:
            B = np.sin(Btmp) / Btmp
        if np.absolute(Ctmp) < 1.0e-6:
            C = 1.0
        else:
            C = np.sin(Ctmp) / Ctmp
        if np.absolute(Dtmp) < 1.0e-6:
            D = 1.0
        else:
            D = np.sin(Dtmp) / Dtmp
        if Wlj < 1.0e-6:
            E = 0.0
        else:
            Wlk = -1.0 * (W[0, 1] * W[0, 0] + W[1, 1] * W[1, 0]) / Wlj
            sWlj = np.sin(Wlj)
            sWlk = np.sin(Wlk)
            Etmp = np.sqrt((1 - Wlj * Wlj) * (1 - Wlk * Wlk))
            denom = sWlj * sWlj - sWlk * sWlk
            E = 2.0 * Wlj * (Wlj * Wlk * sWlj + (Etmp - 1.0) * sWlk) / denom
        h = self.get_timestep()
        tdc = 0.5 / h * (np.arccos(W[0, 0]) * (A + B)
                         + np.arcsin(W[1, 0]) * (C + D) + E)
        return tdc

    def initial_wigner(self, iseed, temp=0.0):
        """Wigner distribution of positions and momenta
        Works at finite temperature if a temp parameter is passed
        If temp is not provided temp = 0 is assumed"""

        print "## randomly selecting Wigner initial conditions at T=", temp
        ndims = self.get_numdims()

        h5f = h5py.File('hessian.hdf5', 'r')

        pos = h5f['geometry'][:].flatten()

        h = h5f['hessian'][:]

        m = self.get_masses()

        sqrtm = np.sqrt(m)

        # build mass weighted hessian
        h_mw = np.zeros_like(h)

        for idim in range(ndims):
            h_mw[idim, :] = h[idim, :] / sqrtm

        for idim in range(ndims):
            h_mw[:, idim] = h_mw[:, idim] / sqrtm

        # symmetrize mass weighted hessian
        h_mw = 0.5 * (h_mw + h_mw.T)

        # diagonalize mass weighted hessian
        evals, modes = np.linalg.eig(h_mw)

        # sort eigenvectors
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        modes = modes[:, idx]

        print '# eigenvalues of the mass-weighted hessian are (a.u.)'
        print evals

        # Checking if frequencies make sense
        freq_cm = np.sqrt(evals[0:ndims - 6])*219474.63
        n_high_freq = 0
        print 'Frequencies in cm-1:'
        for freq in freq_cm:
            if freq > 5000: n_high_freq += 1
            print freq
            assert not np.isnan(freq), "NaN encountered in frequencies! Exiting"

        if n_high_freq > 0: print("Number of frequencies > 5000cm-1:", n_high_freq)

        # seed random number generator
        np.random.seed(iseed)
        alphax = np.sqrt(evals[0:ndims - 6]) / 2.0

        # finite temperature distribution
        if temp > 1e-05:
            beta = 1 / (temp * 0.000003166790852)
            print "beta = ", beta
            alphax = alphax * np.tanh(np.sqrt(evals[0:ndims - 6]) * beta / 2)
        sigx = np.sqrt(1.0 / (4.0 * alphax))
        sigp = np.sqrt(alphax)

        dtheta = 2.0 * np.pi * np.random.rand(ndims - 6)
        dr = np.sqrt(np.random.rand(ndims - 6))

        dx1 = dr * np.sin(dtheta)
        dx2 = dr * np.cos(dtheta)

        rsq = dx1 * dx1 + dx2 * dx2

        fac = np.sqrt(-2.0 * np.log(rsq) / rsq)

        x1 = dx1 * fac
        x2 = dx2 * fac

        posvec = np.append(sigx * x1, np.zeros(6))
        momvec = np.append(sigp * x2, np.zeros(6))

        deltaq = np.matmul(modes, posvec) / sqrtm
        pos += deltaq
        mom = np.matmul(modes, momvec) * sqrtm

        self.set_positions(pos)
        self.set_momenta(mom)

        zpe = np.sum(alphax[0:ndims - 6])
        ke = 0.5 * np.sum(mom * mom / m)
        #         print np.sqrt(np.tanh(evals[0:ndims-6]/(2*0.0031668)))
        print "# ZPE = ", zpe
        print "# kinetic energy = ", ke
    
    def read_initial_conds(self):
        ###NOT FINISHED
        """Get already sampled position and momenta from initial condition filesd"""

        print "## reading initial conditions from file:"
        ndims = self.get_numdims()
        m = self.get_masses()
        sqrtm = np.sqrt(m)

        #h5f = h5py.File('hessian.hdf5', 'r')
        with open('geometry.xyz', 'r') as file:
            geom= file.readlines()

        geom = geom[2:]
        pos = []
        for i in range(len(geom)):
            atom = geom[i].split()
            pos.append(float(atom[1]))
            pos.append(float(atom[2]))
            pos.append(float(atom[3]))
        pos=np.array(pos)

        with open('velocities.xyz', 'r') as file:
            lines = file.readlines()

        vel = []

        for atom in lines:
            vel.extend([float(x) for x in atom.split()])
        
        #mom = []
        #for i in range(len(vel)):
        #    atom = vel[i].split()
        #    mom.append(float(atom[0])*m[i])
        #    mom.append(float(atom[1])*m[i])
        #    mom.append(float(atom[2])*m[i])
        #mom=np.array(mom)
        #pos = h5f['geometry'][:].flatten()

        #h = h5f['hessian'][:]


        # build mass weighted hessian
        #h_mw = np.zeros_like(h)

        #for idim in range(ndims):
        #    h_mw[idim, :] = h[idim, :] / sqrtm

        #for idim in range(ndims):
        #    h_mw[:, idim] = h_mw[:, idim] / sqrtm

        # symmetrize mass weighted hessian
        #h_mw = 0.5 * (h_mw + h_mw.T)

        # diagonalize mass weighted hessian
        #evals, modes = np.linalg.eig(h_mw)

        # sort eigenvectors
        #idx = evals.argsort()[::-1]
        #evals = evals[idx]
        #modes = modes[:, idx]

        #print '# eigenvalues of the mass-weighted hessian are (a.u.)'
        #print evals

        ## Checking if frequencies make sense
        #freq_cm = np.sqrt(evals[0:ndims - 6])*219474.63
        #n_high_freq = 0
        #print 'Frequencies in cm-1:'
        #for freq in freq_cm:
        #    if freq > 5000: n_high_freq += 1
        #    print freq
        #    assert not np.isnan(freq), "NaN encountered in frequencies! Exiting"

        #if n_high_freq > 0: print("Number of frequencies > 5000cm-1:", n_high_freq)

        # seed random number generator
        #np.random.seed(iseed)
        #alphax = np.sqrt(evals[0:ndims - 6]) / 2.0
#
#        # finite temperature distribution
#        if temp > 1e-05:
#            beta = 1 / (temp * 0.000003166790852)
#            print "beta = ", beta
#            alphax = alphax * np.tanh(np.sqrt(evals[0:ndims - 6]) * beta / 2)
#        sigx = np.sqrt(1.0 / (4.0 * alphax))
#        sigp = np.sqrt(alphax)
#
#        dtheta = 2.0 * np.pi * np.random.rand(ndims - 6)
#        dr = np.sqrt(np.random.rand(ndims - 6))
#
#        dx1 = dr * np.sin(dtheta)
#        dx2 = dr * np.cos(dtheta)
#
#        rsq = dx1 * dx1 + dx2 * dx2
#
#        fac = np.sqrt(-2.0 * np.log(rsq) / rsq)
#
#        x1 = dx1 * fac
#        x2 = dx2 * fac
#
#        posvec = np.append(sigx * x1, np.zeros(6))
#        momvec = np.append(sigp * x2, np.zeros(6))
#
#        deltaq = np.matmul(modes, posvec) / sqrtm
#        pos += deltaq
#        mom = np.matmul(modes, momvec) * sqrtm

        
        self.set_positions(pos)
        m = self.get_masses()
        print("M ", m)
        print("V ", vel)
        veloc =np.array(vel)
        mom = veloc * m
        self.set_momenta(mom)
        print('geom: ', pos)
        print('mom: ', mom)

#        zpe = np.sum(alphax[0:ndims - 6])
        ke = 0.5 * np.sum(mom * mom / m)
        #         print np.sqrt(np.tanh(evals[0:ndims-6]/(2*0.0031668)))
#        print "# ZPE = ", zpe
        print "# kinetic energy = ", ke
