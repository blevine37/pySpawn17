######################################################
# Ehrenfest Hamiltonian (diabatic basis)
######################################################
import sys
import numpy as np
import pyspawn.complexgaussian as cg

def build_Heff_half_timestep(self):
    """build Heff for the either half of the time step in the diabatic rep
    Since we don't need any information at half time step there is no difference between 
    first and second half"""
    
    self.get_qm_data_from_h5()
         
    self.build_S_elec()
    self.build_S()
    print "qm time =", self.quantum_time
    print "positions:"
    for key in self.traj: print self.traj[key].positions_qm
    print "S=\n", self.S
    print "amps =\n", self.qm_amplitudes
#     print "S_elec =\n", self.S_elec
#     print "S_nuc =\n", self.S_nuc
    self.invert_S()
    self.build_Sdot()
    self.build_H()
    self.build_Heff()
