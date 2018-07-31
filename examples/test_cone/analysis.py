# this analysis script processes the sim.hdf5 file into various human-readable 
# formats.  This script can be run while the simulation is in progress.
import pyspawn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

traj = "00"

# open sim.hdf5 for processing
an = pyspawn.fafile("sim.hdf5")
work = pyspawn.fafile("working.hdf5")
# create N.dat and store the data in times and N
an.fill_electronic_state_populations(column_filename = "N.dat")
an.fill_trajectory_populations(column_file_prefix = "Pop")

times = an.datasets[traj + "_time"]
# N = an.datasets["electronic_state_populations"]
# make population (N.dat) plot in png format
# plt.plot(times,N[:,0],"ro",times,N[:,1],"bs",markeredgewidth=0.0)
# plt.xlabel('Time')
# plt.ylabel('Population')
# plt.savefig('N.png')
# uncomment to show the plot in a window
# plt.show()

# write files with energy data for each trajectory
an.fill_trajectory_energies(column_file_prefix="E")

# list all datasets
an.list_datasets()

e = an.datasets[traj + "_poten"]

# Plotting total energy
f = plt.figure(3)
tot = an.datasets[traj + "_toten"]
plt.plot(times, tot, "g")
plt.xlabel('Time, au')
plt.ylabel('Total Energy, au')

# Plotting kinetic + potential energy
h = plt.figure(2)
aven = an.datasets[traj +"_aven"]
ke = an.datasets[traj + "_kinen"]
plt.plot(times, ke, "b", markeredgewidth=0.0, label = "Kin")
plt.plot(times, aven, "r", markeredgewidth=0.0, label = "Pot")
plt.xlabel('Time, au')
plt.ylabel('Energy, au')
plt.legend()

# Plotting population
g = plt.figure(1)
pop = an.datasets[traj + "_pop"]
plt.plot(times, pop[:, 0], label = "GS ")
plt.plot(times, pop[:, 1], label = "1 exc")
plt.xlabel('Time, au')
plt.ylabel('Population, au')
plt.legend()
plt.show()
