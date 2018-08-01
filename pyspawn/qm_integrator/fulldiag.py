import numpy as np
import numpy.linalg as la
import math

######################################################
# adaptive RK2 quantum integrator
######################################################

def qm_propagate_step(self, zoutput_first_step=False):
    c1i = (complex(0.0, 1.0))
    self.compute_num_traj_qm()
    qm_t = self.quantum_time
    dt = self.timestep
    qm_tpdt = qm_t + dt 
    ntraj = self.num_traj_qm
    
    amps_t = self.qm_amplitudes
    #print "amps_t", amps_t
    
    self.build_Heff_first_half()
    
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

    #print "fulldiag amps", amps
    
    tmp1 = la.solve(R, amps)
    tmp2 = X * tmp1 # elementwise multiplication
    #amps = la.solve(LH,tmp2)
    amps = np.matmul(R, tmp2)

    #print "fulldiag amps2", amps
    
    self.quantum_time = qm_tpdt

    self.build_Heff_second_half()
    print "Effective Hamiltonian built"    
    #print "fulldiag Heff2", self.Heff
 
    iHdt = (-0.5 * dt * c1i) * self.Heff

    #print "fulldiag iHdt2", iHdt
 
    W,R = la.eig(iHdt)

    #LH = L.conj().T
    
    #print "fulldiag W2", W
    #print "fulldiag LH2", LH
    #print "fulldiag R2", R
    
    X = np.exp( W )

    #print "fulldiag X2", X

    #print "fulldiag amps3", amps
    
    tmp1 = la.solve(R, amps)
    tmp2 = X * tmp1 # elementwise multiplication
    #amps = la.solve(LH,tmp2)
    amps = np.matmul(R, tmp2)

    #print "fulldiag amps4", amps
    
    self.qm_amplitudes = amps
                
######################################################
