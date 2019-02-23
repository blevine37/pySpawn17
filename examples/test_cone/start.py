# this script starts a new FMS calculation on a model cone potential
import numpy as np
import pyspawn        
import pyspawn.general
import os
pop_thresh_list = [0.1, 0.08, 0.05, 0.03, 0.01]
p_thresh_list = [0.1, 0.08, 0.05, 0.04, 0.03, 0.025, 0.02]
olapmax_list = [0.5, 0.1]
for olapmax in olapmax_list:
    for pop_thresh in pop_thresh_list:
        for p_thresh in p_thresh_list:
    
            # Velocity Verlet classical propagator
            clas_prop = "vv"
            
            # fulldiag exponential quantum propagator
            qm_prop = "fulldiag"
            
            # diabatic ehrenfest Hamiltonian
            qm_ham = "ehrenfest"
            
            # use TeraChem CASSCF or CASCI to compute potentials
            potential = "linear_slope"
            
            # initial time
            t0 = 0.0
            
            # time step
            ts = 0.1
            
            # final simulation time
            tfinal = 120.0
            
            # number of dimensions                                                                                           
            numdims = 1
            
            # number of electronic states                                                                                                                    
            numstates = 5
            
            # trajectory parameters
            traj_params = {
                # initial time
                "time": t0,
                # time step
                "timestep": ts,
                # final simulation time
                "maxtime": tfinal,
            #     # initial electronic state (indexed such that 0 is the ground state)
            #     "istate": 1,
                # Gaussian widths
                "widths": np.asarray([6.0]),
                # nuclear masses (in a.u)    
                "masses": np.asarray([1822.0]),
                # initial positions
                "positions": np.asarray([-0.2]),
                # inition momenta
                "momenta": np.asarray([10.0]),
            #     "momenta": np.asarray([0.3]),
            #     "numdims": numdims,
                # Use approximate eigenstates or full Hamiltonian diagonalization
                "full_H": False,
                # Size of Krylov subspace for full_H = False 
            #     "krylov_sub_n": 4,
                "numstates": numstates,
                # How many electronic timesteps in one nuclear (default = 1000)
                "n_el_steps": 1000,    
                }
            
            sim_params = {
                # initial time   
                "quantum_time": traj_params["time"],
                # time step
                "timestep": traj_params["timestep"],
                # final simulation time
                "max_quantum_time": traj_params["maxtime"],
                # initial qm amplitudes
                "qm_amplitudes": np.ones(1, dtype=np.complex128),
                # energy shift used in quantum propagation
                "qm_energy_shift": 0.0,
                # cloning probability threshold
                "p_threshold": p_thresh,
                # maximum overlap for successful cloning
                "olapmax": olapmax,
                # cloning minimum population parameter
                "pop_threshold": pop_thresh,
                # type of cloning procedure:
                # "toastate" : cloning on to a state energy of which is different from average
                # "pairwise" : considering each pair, transferring population between them
                "cloning_type": "toastate",
            }
            
            # import routines needed for propagation
            exec("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
            exec("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
            exec("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
            exec("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")
            
            # check for the existence of files from a past run
            pyspawn.general.check_files()    
            
            # set up first trajectory
            traj1 = pyspawn.traj()
            # traj1.set_numstates(numstates)
            # traj1.set_numdims(numdims)
            traj1.set_parameters(traj_params)
            
            # set up simulation 
            sim = pyspawn.simulation()
            sim.add_traj(traj1)
            sim.set_parameters(sim_params)
            # begin propagation
            sim.propagate()
    	    os.system('python analysis.py')
