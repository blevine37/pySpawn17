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
times_b0 = an.datasets["00b0_time"]
# times_b1 = an.datasets["00b1_time"]
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
pop = an.datasets[traj + "_pop"]
tot = an.datasets[traj + "_toten"]
aven = an.datasets[traj +"_aven"]
ke = an.datasets[traj + "_kinen"]

tot_b0 = an.datasets["00b0_toten"]
pop_b0 = an.datasets["00b0_pop"]
# pop_b1 = an.datasets["00b1_pop"]

# Plotting total energy
f = plt.figure(3)
t = pop[:,0].ravel()
t1 = pop_b0[:,0].ravel()
plt.scatter(times.ravel(), tot.ravel(), c=t, label = traj)
plt.scatter(times_b0.ravel(), tot_b0.ravel(), c=t1, label = "00b0")
plt.xlabel('Time, au')
plt.ylabel('Total Energy, au')
plt.ylim([min(tot.ravel()), max(tot.ravel())])
# Plotting kinetic + potential energy
h = plt.figure(2)

plt.plot(times, ke, "b", markeredgewidth=0.0, label = "Kin")
plt.plot(times, aven, "r", markeredgewidth=0.0, label = "Pot")
plt.xlabel('Time, au')
plt.ylabel('Energy, au')
plt.legend()

# Plotting population
g = plt.figure(1)
plt.plot(times, pop[:, 0], "b", label = "TBF1: GS ")
plt.plot(times, pop[:, 1], "r", label = "TBF2: 1exc")
plt.plot(times_b0, pop_b0[:, 0], "--b", markersize=1, label = "TBF2: GS ")
plt.plot(times_b0, pop_b0[:, 1], "--r", markersize=1, label = "TBF2: 1exc")
# plt.plot(times_b1, pop_b1[:, 0], "b", linewidth=3, label = "TBF3: GS ")
# plt.plot(times_b1, pop_b1[:, 1], "r", linewidth=3, label = "TBF3: 1exc")

plt.xlabel('Time, au')
plt.ylabel('Population, au')
plt.legend()
plt.show()
