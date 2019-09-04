######################################################
# adiabatic Hamiltonian
######################################################


def build_Heff_first_half(self):
    """Build Heff for the first half of the time step in
    the adibatic rep (with NPI)"""

    self.get_qm_data_from_h5()

    qm_time = self.get_quantum_time()
    dt = self.get_timestep()
    t_half = qm_time + 0.5 * dt
    self.set_quantum_time_half_step(t_half)
    self.get_qm_data_from_h5_half_step()

    self.build_S()
    self.invert_S()
    self.build_Sdot()
    self.build_H()

    self.build_Heff()


def build_Heff_second_half(self):
    """Build Heff for the second half of the time step in
    the adibatic rep (with NPI)"""

    self.get_qm_data_from_h5()

    qm_time = self.get_quantum_time()
    dt = self.get_timestep()
    t_half = qm_time - 0.5 * dt
    self.set_quantum_time_half_step(t_half)
    self.get_qm_data_from_h5_half_step()

    self.build_S()
    self.invert_S()
    self.build_Sdot()
    self.build_H()

    self.build_Heff()
