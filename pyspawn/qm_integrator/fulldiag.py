import numpy as np
import scipy.linalg as la
import math

######################################################
# adaptive RK2 quantum integrator
######################################################

def qm_propagate_step(self,zoutput_first_step=False):
    c1i = (complex(0.0,1.0))
    self.compute_num_traj_qm()
    qm_t = self.get_quantum_time()
    dt = self.get_timestep()
    qm_tpdt = qm_t + dt 
    ntraj = self.get_num_traj_qm()
    
    amps_t = self.get_qm_amplitudes()
    #print "amps_t", amps_t
    
    self.build_Heff_first_half()
    
    # output the first step before propagating
    if zoutput_first_step:
        self.h5_output()

    print "fulldiag Heff", self.Heff
 
    iHdt = (-0.5 * dt * c1i) * self.Heff

    print "fulldiag iHdt", iHdt
 
    W,L,R = la.eig(iHdt,left=True)

    print "fulldiag W", W
    print "fulldiag L", L
    print "fulldiag R", R
    
    X = np.exp( W )

    print "fulldiag X", X
    
    amps = amps_t
    tmp1 = la.solve(R,amps)
    tmp2 = X*tmp1 # elementwise multiplication
    amps = la.solve(L,tmp2)
    
    self.set_quantum_time(qm_tpdt)

    self.build_Heff_second_half()
        
    iHdt = (-0.5 * dt * c1i) * self.Heff

    W,L,R = la.eig(iHdt,left=True)
    
    X = np.exp( W )

    amps = amps_t
    tmp1 = la.solve(R,amps)
    tmp2 = X*tmp1 # elementwise multiplication
    amps = la.solve(L,tmp2)
    
    self.set_qm_amplitudes(amps)
                
######################################################
