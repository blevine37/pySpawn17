#################################################
### integrators go here #########################
#################################################

#each integrator requires at least two routines:
#1) prop_first_, which propagates the first step
#2) prop_, which propagates all other steps
#other ancillary routines may be included as well

### velocity Verlet (vv) integrator section ###
def prop_first_step(self,zbackprop):

    dt = self.timestep
            
    x_t = self.positions
    self.compute_elec_struct(zbackprop)
    f_t = self.av_force
    p_t = self.momenta
    e_t = self.energies
    m = self.masses
    v_t = p_t / m
    a_t = f_t / m
    t = self.time

    self.h5_output(zbackprop, zdont_half_step=True)
        
    v_tphdt = v_t + 0.5 * a_t * dt
    x_tpdt = x_t + v_tphdt * dt
    
    self.positions = x_tpdt
    self.compute_elec_struct(zbackprop)
    f_tpdt = self.av_force
    e_tpdt = self.energies
    
    a_tpdt = f_tpdt / m
    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
    p_tpdt = v_tpdt * m
    
    self.momenta = p_tpdt

    t_half = t + 0.5 * dt
    t += dt

    self.time = t
    self.time_half_step = t_half

    self.h5_output(zbackprop)
     
    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt
    p_tp3hdt = v_tp3hdt * m

    self.momenta = p_tp3hdt

    x_tp2dt = x_tpdt + v_tp3hdt * dt

    self.positions_t = x_t
    self.positions_tpdt = x_tpdt
    self.momenta_t = p_t
    self.momenta_tpdt = p_tpdt
    self.energies_t = e_t
    self.energies_tpdt = e_tpdt
        
    self.positions = x_tp2dt
        
def prop_not_first_step(self,zbackprop):

    dt = self.timestep

    x_tpdt = self.positions
    self.compute_elec_struct(zbackprop)
    f_tpdt = self.get_forces_i()
    e_tpdt = self.energies

    p_tphdt = self.momenta
    m = self.masses
    v_tphdt = p_tphdt / m
    a_tpdt = f_tpdt / m
    t = self.time

    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt

    p_tpdt = v_tpdt * m

    self.positions_tmdt = self.positions_t
    self.positions_t = self.positions_tpdt
    self.positions_tpdt = x_tpdt
    self.energies_tmdt = self.energies_t
    self.energies_t = self.energies_tpdt
    self.energies_tpdt = e_tpdt
    self.momenta_tmdt = self.momenta_t
    self.momenta_t = self.momenta_tpdt
    self.momenta_tpdt = p_tpdt
    self.momenta = p_tpdt

    t_half = t + 0.5 * dt
    t += dt

    self.time = t
    self.time_half_step = t_half
    
    self.h5_output(zbackprop)

    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt

    p_tp3hdt = v_tp3hdt * m

    self.momenta = p_tp3hdt

    x_tp2dt = x_tpdt + v_tp3hdt * dt

    self.positions = x_tp2dt

### end velocity Verlet (vv) integrator section ###
