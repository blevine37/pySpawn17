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
    dt = self.timestep
    x_t = self.positions
    p_t = self.momenta
    m = self.masses
    v_t = p_t / m
    t = self.time
    
    # first computing forces for the momentum propagation for t0 + dt/2
    self.compute_elec_struct()
    f_t = self.av_force
    a_t = f_t / m
    e_t = self.energies
    e_av_t = self.av_energy    
    
    # numerically differentiating to test the forces
#     dx = 1e-6
#     self.positions[0] = self.positions[0] + dx
#     self.compute_elec_struct()
#     num_deriv = (self.av_energy - e_av_t) / dx 
#     print "num_deriv =", num_deriv
#     print "force =", f_t
#     print "new force =", self.av_force
#     sys.exit()
    
    # propagating velocity half a timestep
    v_tphdt = v_t + 0.5 * a_t * dt
#     print "x_t =", x_t
#     print "v_t =", v_t
#     print "a_t =", a_t
#     print "f_t =", f_t
#     print "e =", e_av_t
#     print "ke_t =", self.calc_kin_en(p_t, m)
#     print "\ntotal energy at t:", e_av_t + self.calc_kin_en(p_t, m)
#     print "v_tphdt =", v_tphdt
    
    # Output of parameters at t0
    self.h5_output(zdont_half_step=True)
    
    # now we can propagate position full timestep
    x_tpdt = x_t + v_tphdt * dt
    self.positions = x_tpdt
    self.compute_elec_struct()
    f_tpdt = self.av_force
    e_tpdt = self.energies
    e_av_tpdt = self.av_energy
    a_tpdt = f_tpdt / m
        
    # we can compute momentum value at full timestep (t0 + dt)
    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
    p_tpdt = v_tpdt * m
    
#     print "x_tpdt", x_tpdt
#     print "v_tpdt =", v_tpdt
#     print "a_tpdt =", a_tpdt
#     print "f_tpdt =", f_tpdt
#     print "e =", e_av_tpdt
#     print "ke_t =", self.calc_kin_en(p_tpdt, m)
#     print "\ntotal energy at t + dt:", e_av_tpdt + self.calc_kin_en(p_tpdt, m)
#     print "\nDE =", e_av_t + self.calc_kin_en(p_t, m) - (e_av_tpdt + self.calc_kin_en(p_tpdt, m))
    
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
    x_tp2dt = x_tpdt + v_tp3hdt * dt
    self.positions = x_tp2dt
    
    # Also saving positions, momenta and energies at t0 and t0 + dt
    self.positions_t = x_t
    self.positions_tpdt = x_tpdt
    
    self.momenta_t = p_t
    self.momenta_tpdt = p_tpdt
    
    self.energies_t = e_t
    self.energies_tpdt = e_tpdt
    
    self.av_energy_t = e_av_t
    self.av_energy_tpdt = e_av_tpdt
    
def prop_not_first_step(self):
    
    print "\nPerforming the VV propagation for NOT first timestep"
    dt = self.timestep

    x_tpdt = self.positions
    m = self.masses
    
    self.compute_elec_struct()
    f_tpdt = self.av_force
    e_tpdt = self.energies
    e_av_tpdt = self.av_energy
    p_tphdt = self.momenta
    v_tphdt = p_tphdt / m
    a_tpdt = f_tpdt / m
    t = self.time

    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
    p_tpdt = v_tpdt * m
    
    # Saving everything at t - dt, t, t + dt
    self.positions_tmdt = self.positions_t
    self.positions_t = self.positions_tpdt
    self.positions_tpdt = x_tpdt
    
    self.energies_tmdt = self.energies_t
    self.energies_t = self.energies_tpdt
    self.energies_tpdt = e_tpdt
        
    self.av_energy_tmdt = self.av_energy_t
    self.av_energy_t = self.av_energy_tpdt
    self.av_energy_tpdt = e_av_tpdt

    self.momenta_tmdt = self.momenta_t
    self.momenta_t = self.momenta_tpdt
    self.momenta_tpdt = p_tpdt
    
#     dE_PE = self.av_energy_t - self.av_energy_tpdt
#     dE_KE = self.calc_kin_en(self.momenta_t, m) - self.calc_kin_en(self.momenta_tpdt, m)
#     print "\ndE_PE =", dE_PE
#     print "dE_KE =", dE_KE
#     print "dE_total =", dE_PE + dE_KE
    
    # Computing momenta at t + dt and outputting paramaters
    self.momenta = p_tpdt

    t_half = t + 0.5 * dt
    t += dt

    self.time = t
    self.time_half_step = t_half

    # Output of parameters at t
    self.h5_output()

#     print "\nt =", t
#     print "x_t =", x_tpdt
#     print "f_t =", f_tpdt
#     print "v_t =", v_tpdt
#     print "a_t =", a_tpdt
    
    # Computing and saving momentum at t + 1/2 dt
    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt
    p_tp3hdt = v_tp3hdt * m
    self.momenta = p_tp3hdt
    
    # Propagating positions to t + dt
    x_tp2dt = x_tpdt + v_tp3hdt * dt
    self.positions = x_tp2dt
    
### end velocity Verlet (vv) integrator section ###
