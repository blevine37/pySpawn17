import numpy as np
import math

######################################################
# adaptive RK2 quantum integrator
######################################################

def qm_propagate_step(self,zoutput_first_step=False):
    maxcut = 16
    c1i = (complex(0.0,1.0))
    self.compute_num_traj_qm()
    qm_t = self.get_quantum_time()
    dt = self.get_timestep()
    qm_tpdt = qm_t + dt 
    ntraj = self.get_num_traj_qm()
    
    amps_t = self.get_qm_amplitudes()
    #print "amps_t", amps_t
    
    self.build_Heff_first_half()
    
    #print "rk2 Heff ", self.Heff

    # output the first step before propagating
    if zoutput_first_step:
        self.h5_output()

    ncut = 0
    # adaptive integration
    while ncut <= maxcut and ncut >= 0:
        amps = amps_t
        # how many quantum time steps will we take
        nstep = 1
        for i in range(ncut):
            nstep *= 2
        print "# in adaptive RK integrator, nstep = ", nstep
        #dt_small = dt / float(nstep)
        dt_small = 0.5 * dt / float(nstep)
        #print "rk2 nstep2", nstep
        #print "rk2 dt_small", dt_small
        
        for istep in range(nstep):
            #print "istep nstep dt_small ", istep, nstep, dt_small
            k1 = (-1.0 * dt_small * c1i) * np.matmul(self.Heff,amps)
                #print "k1 ", k1
            tmp = amps + 0.5 * k1
                #print "temp ", tmp
            k2 = (-1.0 * dt_small * c1i) * np.matmul(self.Heff,tmp)
                #print "k2 ", k2
            amps = amps + k2
                #print "amps ", amps
            
        if ncut > 0:
            diff = amps - amps_save
            error = math.sqrt((np.sum(np.absolute(diff * np.conjugate(diff)))/ntraj)) 
            if error < 0.0001:
                ncut = -2
                print "# adaptive integration converged, error = ", error
                            
        ncut += 1
        amps_save = amps

    if ncut != -1:
        print "# problem in adaptive integration: error = ", error, "after maximum adaptation!"

    amps_tphdt = amps
        
    self.set_quantum_time(qm_tpdt)

    self.build_Heff_second_half()
        
    #print "rk2 Heff2 ", self.Heff

    ncut = 0
    # adaptive integration
    while ncut <= maxcut and ncut >= 0:
        amps = amps_tphdt
        # how many quantum time steps will we take
        nstep = 1
        for i in range(ncut):
            nstep *= 2
        print "# in adaptive RK integrator, nstep = ", nstep
        #dt_small = dt / float(nstep)
        dt_small = 0.5 * dt / float(nstep)
        #print "rk2 nstep2", nstep
        #print "rk2 dt_small2", dt_small
        
        for istep in range(nstep):
            #print "istep nstep dt_small ", istep, nstep, dt_small
            k1 = (-1.0 * dt_small * c1i) * np.matmul(self.Heff,amps)
            tmp = amps + 0.5 * k1
            k2 = (-1.0 * dt_small * c1i) * np.matmul(self.Heff,tmp)
            amps = amps + k2
            
        if ncut > 0:
            diff = amps - amps_save
            error = math.sqrt((np.sum(np.absolute(diff * np.conjugate(diff)))/ntraj)) 
            if error < 0.0001:
                ncut = -2
                print "# adaptive integration converged, error = ", error
                            
        ncut += 1
        amps_save = amps
        
    if ncut != -1:
        print "Problem in quantum integration: error = ", error, "after maximum adaptation!"

    #print "amps_tpdt ", amps

    self.set_qm_amplitudes(amps)
        
    #print "amps saved ", self.get_qm_amplitudes()
            
    #self.clean_up_matrices()
        
######################################################
