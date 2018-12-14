import numpy as np
import numpy.linalg as la
import math
from astropy.coordinates.builtin_frames.utils import norm

######################################################
# exponential integrator
######################################################

def qm_propagate_step(self, zoutput_first_step=False):
    
    c1i = (complex(0.0, 1.0))
    self.compute_num_traj_qm()
    qm_t = self.quantum_time
    dt = self.timestep
    qm_tpdt = qm_t + dt 
    ntraj = self.num_traj_qm
    
    amps_t = self.qm_amplitudes
#     print "positions"
#     for trajectory in self.traj: print self.traj[trajectory].positions
    print "Building effective Hamiltonian for the first half step"
    
    self.build_Heff_half_timestep()
    norm = np.dot(np.conjugate(np.transpose(amps_t)), np.dot(self.S, amps_t))
#     print "amps =", amps_t
    print "Norm first half =", norm    
    # output the first step before propagating
    if zoutput_first_step:
        self.h5_output()

    #print "fulldiag Heff", self.Heff
 
    iHdt = (-0.5 * dt * c1i) * self.Heff

    #print "fulldiag iHdt", iHdt
 
    W,R = la.eig(iHdt)

    #LH = L.conj().T
    
    #print "fulldiag W", W
    #print "fulldiag LH", LH
    #print "fulldiag R", R
    
    X = np.exp( W )

    #print "fulldiag X", X
    
    amps = amps_t
    
    tmp1 = la.solve(R, amps)
    tmp2 = X * tmp1 # element-wise multiplication
    #amps = la.solve(LH,tmp2)
    amps = np.matmul(R, tmp2)

    self.quantum_time = qm_tpdt
    
    print "Building effective Hamiltonian for the second half step"
    self.build_Heff_half_timestep()
    print "Effective Hamiltonian built"    
 
    iHdt = (-0.5 * dt * c1i) * self.Heff

    #print "fulldiag iHdt2", iHdt
 
    W,R = la.eig(iHdt)

    #LH = L.conj().T
    
    #print "fulldiag W2", W
    #print "fulldiag LH2", LH
    #print "fulldiag R2", R
    
    X = np.exp( W )
    
    tmp1 = la.solve(R, amps)
    tmp2 = X * tmp1 # element-wise multiplication
    #amps = la.solve(LH,tmp2)
    amps = np.matmul(R, tmp2)

    self.qm_amplitudes = amps
     
    norm = np.dot(np.conjugate(np.transpose(amps)), np.dot(self.S, amps))

    print "Norm second half =", norm
    
    print "Done with quantum propagation"            

######################################################
