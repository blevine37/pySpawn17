# this analysis script processes the sim.hdf5 file into various human-readable 
# formats.  This script can be run while the simulation is in progress.
import pyspawn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# open sim.hdf5 for processing
an = pyspawn.fafile("sim.hdf5")

# create N.dat and store the data in times and N
times, N = an.compute_norms(column_filename = "N.dat")

# make population (N.dat) plot in png format
plt.plot(times,N[0,:],"ro",times,N[1,:],"bs",markeredgewidth=0.0)
plt.xlabel('Time')
plt.ylabel('Population')
plt.savefig('N.png')
# uncomment to show the plot in a window
#plt.show()

# write xyz files for each trajectory
an.write_xyzs()

# write files with energy data for each trajectory
an.write_trajectory_energy_files()

# write file with time derivative couplings for each trajectory
an.write_trajectory_tdc_files()

# write files with geometric data for each trajectory
bonds = np.array([[0,1],
                  [1,2],
                  [1,3],
                  [0,4],
                  [0,5]])

angles = np.array([[0,1,2],
                   [0,1,3],
                   [1,0,4],
                   [1,0,5]])

an.write_trajectory_bond_files(bonds)
an.write_trajectory_angle_files(angles)
