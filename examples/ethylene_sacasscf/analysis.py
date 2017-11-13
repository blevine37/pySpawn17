# this analysis script processes the sim.hdf5 file into various human-readable 
# formats.  This script can be run while the simulation is in progress.
import pyspawn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# open sim.hdf5 for processing
an = pyspawn.fafile("sim.hdf5")

# create N.dat and store the data in times and N
an.fill_electronic_state_populations(column_filename = "N.dat")

times = an.datasets["quantum_times"]
N = an.datasets["electronic_state_populations"]

# make population (N.dat) plot in png format
plt.plot(times,N[:,0],"ro",times,N[:,1],"bs",markeredgewidth=0.0)
plt.xlabel('Time')
plt.ylabel('Population')
plt.savefig('N.png')
# uncomment to show the plot in a window
#plt.show()

# write xyz files for each trajectory
an.write_xyzs()

# write files with energy data for each trajectory
an.fill_trajectory_energies(column_file_prefix="E")

# write file with time derivative couplings for each trajectory
an.fill_trajectory_tdcs(column_file_prefix="tdc")

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

an.fill_trajectory_bonds(bonds,column_file_prefix="bonds")
an.fill_trajectory_angles(angles,column_file_prefix="angles")

an.list_datasets()

