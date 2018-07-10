#################################################
### integrators go here #########################
#################################################

#each integrator requires at least two routines:
#1) prop_first_, which propagates the first step
#2) prop_, which propagates all other steps
#other ancillary routines may be included as well

### velocity Verlet (vv) integrator section ###
def prop_first_step(self,zbackprop):

    dt = self.get_timestep()
            
    x_t = self.get_positions()
    self.compute_elec_struct(zbackprop)
    f_t = self.get_av_force()
    p_t = self.get_momenta()
    e_t = self.get_energies()
    m = self.get_masses()
    v_t = p_t / m
    a_t = f_t / m
    t = self.get_time()

    self.h5_output(zbackprop, zdont_half_step=True)
        
    v_tphdt = v_t + 0.5 * a_t * dt
    x_tpdt = x_t + v_tphdt * dt
    
    self.set_positions(x_tpdt)
    self.compute_elec_struct(zbackprop)
    f_tpdt = self.get_av_force()
    e_tpdt = self.get_energies()
    
    a_tpdt = f_tpdt / m
    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
    p_tpdt = v_tpdt * m
    
    self.set_momenta(p_tpdt)

    t_half = t + 0.5 * dt
    t += dt

    self.set_time(t)
    self.set_time_half_step(t_half)

    self.h5_output(zbackprop)
     
    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt
    p_tp3hdt = v_tp3hdt * m

    self.set_momenta(p_tp3hdt)

    x_tp2dt = x_tpdt + v_tp3hdt * dt

    self.set_positions_t(x_t)
    self.set_positions_tpdt(x_tpdt)
    self.set_momenta_t(p_t)
    self.set_momenta_tpdt(p_tpdt)
    self.set_energies_t(e_t)
    self.set_energies_tpdt(e_tpdt)
        
    self.set_positions(x_tp2dt)
        
def prop_not_first_step(self,zbackprop):

    dt = self.get_timestep()

    x_tpdt = self.get_positions()
    self.compute_elec_struct(zbackprop)
    f_tpdt = self.get_forces_i()
    e_tpdt = self.get_energies()

    p_tphdt = self.get_momenta()
    m = self.get_masses()
    v_tphdt = p_tphdt / m
    a_tpdt = f_tpdt / m
    t = self.get_time()

    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt

    p_tpdt = v_tpdt * m

    self.set_positions_tmdt(self.get_positions_t())
    self.set_positions_t(self.get_positions_tpdt())
    self.set_positions_tpdt(x_tpdt)
    self.set_energies_tmdt(self.get_energies_t())
    self.set_energies_t(self.get_energies_tpdt())
    self.set_energies_tpdt(e_tpdt)
    self.set_momenta_tmdt(self.get_momenta_t())
    self.set_momenta_t(self.get_momenta_tpdt())
    self.set_momenta_tpdt(p_tpdt)
    self.set_momenta(p_tpdt)

    t_half = t + 0.5 * dt
    t += dt

    self.set_time(t)
    self.set_time_half_step(t_half)
    
    self.h5_output(zbackprop)

    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt

    p_tp3hdt = v_tp3hdt * m

    self.set_momenta(p_tp3hdt)

    x_tp2dt = x_tpdt + v_tp3hdt * dt

    self.set_positions(x_tp2dt)
### end velocity Verlet (vv) integrator section ###
