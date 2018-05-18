import numpy as np
import numpy.linalg as la
import math

def qm_propagate_step(self,zoutput_first_step=False):
    
    
    
    """Split symplectic integrator, NOT DONE YET"""
    c1i = (complex(0.0,1.0))
    self.compute_num_traj_qm()
    qm_t = self.get_quantum_time()
    dt = self.get_timestep()
    qm_tpdt = qm_t + dt 
    ntraj = self.get_num_traj_qm()
    
    amps_t = self.get_qm_amplitudes()
    amp_r = np.real(amps_t)
    amp_i = np.imag(amps_t)
    
    self.build_Heff_first_half()
    
    amp_r_dot = np.dot(self.H, amp_i)
    amp_r = amp_r + 0.5 * dt * amp_r_dot
    amp_i_dot = -1.0 * np.dot(self.H, amp_r)
    amp_i = amp_i + dt * amp_i_dot
    
    self.set_quantum_time(qm_tpdt)
    self.build_Heff_second_half()
    
    amp_r_dot = np.dot(self.H, amp_i)
    amp_r = amp_r + 0.5 * dt * amp_r_dot
    
    amps = amp_r + 1j * amp_i
    print "amps_t", amps
    print "amp_i", amp_i
    print "amp_r", amp_r
    self.set_qm_amplitudes(amps)
    