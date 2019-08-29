#################################################
### integrators go here #########################
#################################################

#each integrator requires at least two routines:
#1) prop_first_, which propagates the first step
#2) prop_, which propagates all other steps
#other ancillary routines may be included as well

### velocity Verlet (vv) integrator section ###
def prop_first_step(self,zbackprop):
    if not zbackprop:
        cbackprop = ""
        dt = self.get_timestep()
    else:
        cbackprop = "backprop_"
        dt = -1.0 * self.get_timestep()
    x_t = getattr(self, "get_" + cbackprop + "positions")()        
    #exec("x_t = self.get_" + cbackprop + "positions()")
    self.compute_elec_struct(zbackprop)
    exec("f_t = self.get_" + cbackprop + "forces_i()")
    exec("p_t = self.get_" + cbackprop + "momenta()")
    exec("e_t = self.get_" + cbackprop + "energies()")
    m = self.get_masses()
    v_t = p_t / m
    a_t = f_t / m
    exec("t = self.get_" + cbackprop + "time()")

    if not zbackprop:
        self.h5_output(zbackprop, zdont_half_step=True)
        
    v_tphdt = v_t + 0.5 * a_t * dt
    x_tpdt = x_t + v_tphdt * dt
    
    exec("self.set_" + cbackprop + "positions(x_tpdt)")
    
    self.compute_elec_struct(zbackprop)
    exec("f_tpdt = self.get_" + cbackprop + "forces_i()")
    exec("e_tpdt = self.get_" + cbackprop + "energies()")
    
    a_tpdt = f_tpdt / m
    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt
    p_tpdt = v_tpdt * m
    
    exec("self.set_" + cbackprop + "momenta(p_tpdt)")

    t_half = t + 0.5 * dt
    t += dt

    exec("self.set_" + cbackprop + "time(t)")
    exec("self.set_" + cbackprop + "time_half_step(t_half)")

    self.h5_output(zbackprop)
     
    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt
    p_tp3hdt = v_tp3hdt * m

    exec("self.set_" + cbackprop + "momenta(p_tp3hdt)")

    x_tp2dt = x_tpdt + v_tp3hdt * dt

    if not zbackprop:
        self.set_positions_t(x_t)
        self.set_positions_tpdt(x_tpdt)
        self.set_momenta_t(p_t)
        self.set_momenta_tpdt(p_tpdt)
        self.set_energies_t(e_t)
        self.set_energies_tpdt(e_tpdt)
        
    exec("self.set_" + cbackprop + "positions(x_tp2dt)")
        
def prop_not_first_step(self,zbackprop):
    if not zbackprop:
        cbackprop = ""
        dt = self.get_timestep()
    else:
        cbackprop = "backprop_"
        dt = -1.0 * self.get_timestep()

    exec("x_tpdt = self.get_" + cbackprop + "positions()")
    self.compute_elec_struct(zbackprop)
    exec("f_tpdt = self.get_" + cbackprop + "forces_i()")
    exec("e_tpdt = self.get_" + cbackprop + "energies()")

    exec("p_tphdt = self.get_" + cbackprop + "momenta()")
    m = self.get_masses()
    v_tphdt = p_tphdt / m
    a_tpdt = f_tpdt / m
    exec("t = self.get_" + cbackprop + "time()")

    v_tpdt = v_tphdt + 0.5 * a_tpdt * dt

    p_tpdt = v_tpdt * m

    if not zbackprop:
        self.set_positions_tmdt(self.get_positions_t())
        self.set_positions_t(self.get_positions_tpdt())
        self.set_positions_tpdt(x_tpdt)
        self.set_energies_tmdt(self.get_energies_t())
        self.set_energies_t(self.get_energies_tpdt())
        self.set_energies_tpdt(e_tpdt)
        self.set_momenta_tmdt(self.get_momenta_t())
        self.set_momenta_t(self.get_momenta_tpdt())
        self.set_momenta_tpdt(p_tpdt)
    exec("self.set_" + cbackprop + "momenta(p_tpdt)")

    t_half = t + 0.5 * dt
    t += dt

    exec("self.set_" + cbackprop + "time(t)")
    exec("self.set_" + cbackprop + "time_half_step(t_half)")
    
    self.h5_output(zbackprop)

    v_tp3hdt = v_tpdt + 0.5 * a_tpdt * dt

    p_tp3hdt = v_tp3hdt * m

    exec("self.set_" + cbackprop + "momenta(p_tp3hdt)")

    x_tp2dt = x_tpdt + v_tp3hdt * dt

    exec("self.set_" + cbackprop + "positions(x_tp2dt)")
### end velocity Verlet (vv) integrator section ###
