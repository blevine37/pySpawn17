#################################################
### integrators go here #########################
#################################################
from xlrd.xlsx import F_TAG
import sys
#each integrator requires at least two routines:
#1) prop_first_, which propagates the first step
#2) prop_, which propagates all other steps
#other ancillary routines may be included as well

### velocity Verlet (vv) integrator section ###
def prop_first_step(self):
    
    print "Performing the VV propagation for the first timestep"
#     print "self.first_step =", self.first_step
    dt = self.timestep
    x_t = self.positions
    p_t = self.momenta
    m = self.masses
    v_t = p_t / m
    t = self.time
    
    # first computing forces for the momentum propagation for t0 + dt/2
#     print "test 1\nwf =", self.td_wf_full_ts
#     print "eigenvecs =", self.approx_eigenvecs
#     print "mce_amps", self.mce_amps
#     print "Average energy =", self.av_energy
#     print "Energies =", self.energies
#     print "Force =", self.av_force
#     print "Population = ", self.populations
#     print "momenta", self.momenta 
#     if not self.first_step:
    print "wf:\n", self.td_wf
    self.compute_elec_struct()
    self.first_step = False 

    print "wf:\n", self.td_wf
#     print "test 2\nwf =", self.td_wf_full_ts
#     print "eigenvecs =", self.approx_eigenvecs
#     print "mce_amps", self.mce_amps
#     print "Average energy =", self.av_energy
#     print "Energies =", self.energies
#     print "Force =", self.av_force
#     print "Population = ", self.populations
#     print "momenta", self.momenta 
    f_t = self.av_force
    a_t = f_t / m
    e_t = self.energies
    e_av_t = self.av_energy    
    wf_t = self.td_wf_full_ts
    
    # propagating velocity half a timestep
    v_tphdt = v_t + 0.5 * a_t * dt

    # Output of parameters at t0
#     if not self.first_step:
       # after cloning we treat it as first step, h5 already has this timestep
    self.h5_output(zdont_half_step=True)
    
    # now we can propagate position full timestep
    x_tpdt = x_t + v_tphdt * dt
    self.positions = x_tpdt
    self.compute_elec_struct()
    f_tpdt = self.av_force
    e_tpdt = self.energies
    e_av_tpdt = self.av_energy
    wf_tpdt = self.td_wf_full_ts
    a_tpdt = f_tpdt / m
        
    # we can compute momentum value at full timestep (t0 + dt)
    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
    p_tpdt = v_tpdt * m
       
    # Positions and momenta values at t0 + dt
    
    self.momenta = p_tpdt
    
    t_half = t + 0.5 * dt
    t += dt

    self.time = t
    self.time_half_step = t_half
    
    # Output of parameters at t0 + dt
    self.h5_output()
    
    # Saving momentum at t0 + 3/2 * dt for the next iteration 
    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt
    p_tp3hdt = v_tp3hdt * m
    self.momenta = p_tp3hdt
    
    # Saving position at t0 + 2dt for the next iteration
#     x_tp2dt = x_tpdt + v_tp3hdt * dt
#     self.positions = x_tp2dt
    
    # Also saving positions, momenta and energies at t0 and t0 + dt
    self.positions_t = x_t
    self.positions_tpdt = x_tpdt
    
    self.momenta_t = p_t
    self.momenta_tpdt = p_tpdt
    
    self.energies_t = e_t
    self.energies_tpdt = e_tpdt
    
    self.av_energy_t = e_av_t
    self.av_energy_tpdt = e_av_tpdt
    
    self.td_wf_full_ts_t = wf_t
    self.td_wf_full_ts_tpdt = wf_tpdt

    self.av_force_t = f_t
    self.av_force_tpdt = f_tpdt
#     if self.first_step and self.time > self.timestep:
#         print "EXIT IN VV"
#         sys.exit()    
    
def prop_not_first_step(self):
    """Velocity Verlet propagator for not first timestep. Here we call electronic structure
    property calculation only once """

    dt = self.timestep

    x_t = self.positions
    m = self.masses
    
    f_t = self.av_force
    e_t = self.energies
    e_av_t = self.av_energy
    wf_t = self.td_wf_full_ts
    p_tphdt = self.momenta
    v_tphdt = p_tphdt / m
    a_t = f_t / m
    t = self.time
    
    # Propagating positions to t + dt and computing electronic structure
    x_tpdt = x_t + v_tphdt * dt
    self.positions = x_tpdt
    self.compute_elec_struct()
    
    wf_tpdt = self.td_wf_full_ts
    e_tpdt = self.energies
    e_av_tpdt = self.av_energy
    f_tpdt = self.av_force
    a_tpdt =  f_tpdt / m
    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
    p_tpdt = v_tpdt * m
    
    # Saving everything at t - dt, t, t + dt
    self.positions_t = self.positions_tpdt
    self.positions_tpdt = x_tpdt
    
    self.energies_t = self.energies_tpdt
    self.energies_tpdt = e_tpdt
        
    self.av_energy_t = self.av_energy_tpdt
    self.av_energy_tpdt = e_av_tpdt

    self.momenta_t = self.momenta_tpdt
    self.momenta_tpdt = p_tpdt

    self.td_wf_full_ts_t = self.td_wf_full_ts_tpdt
    self.td_wf_full_ts_tpdt = wf_tpdt
    
    self.av_force_t = self.av_force_tpdt
    self.av_force_tpdt = f_tpdt
    
    self.momenta = p_tpdt    
    
    t_half = t + 0.5 * dt
    t += dt

    self.time = t
    self.time_half_step = t_half

    # Output of parameters at t
    self.h5_output()
  
    # Computing and saving momentum at t + 1/2 dt
    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt
    p_tp3hdt = v_tp3hdt * m
    self.momenta = p_tp3hdt
    
    # Propagating positions to t + dt
#     x_tp2dt = x_tpdt + v_tp3hdt * dt
#     self.positions = x_tp2dt
### end velocity Verlet (vv) integrator section ###
