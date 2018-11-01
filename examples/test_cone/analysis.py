# this analysis script processes the sim.hdf5 file into various human-readable 
# formats.  This script can be run while the simulation is in progress.
import pyspawn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_nuclear_populations(keys, numstates, colors, linetypes):
    
    N = an.datasets["nuclear_bf_populations"]
    qm_time = an.datasets["quantum_times"]
    for n in range(numstates):
        plt.plot(qm_time, N[:, n+1], color=colors[n],\
                     label = str((n+1)) + "TBF",)
    plt.xlabel('Time')
    plt.ylabel('Nuclear Population')
    plt.legend()
    plt.savefig("Nuc_pop.png")
    
def plot_el_population(keys, numstates, colors, linestyles):
    l = 0
    for key in keys:
        for n in range(numstates):
            print key
            plt.plot(time[key], pop[key][:, n], color=colors[n],\
                     label = key + ": " + str((n+1)) + "state",
                     linestyle=linestyles[l])
        l += 1
    plt.xlabel('Time, au')
    plt.ylabel('Population, au')
    plt.legend()
    plt.savefig("Elec_pop.png")

def plot_energies(keys, numstates, colors, linetypes):
    
    l = 0
    for key in keys:
        for n in range(numstates):
            plt.plot(time[key], poten[key][:, n], color=colors[n],\
                     label = key + ": " + str((n+1)) + "state",
                     linestyle=linestyles[l])
        
        plt.plot(time[key], aven[key][:], color=colors[n+1],\
                     label = key + ": " + "Ehrenfest",
                     linestyle=linestyles[l])
        l += 1    
    plt.xlabel('Time, au')
    plt.ylabel('Population, au')
    plt.legend()
    plt.savefig("Energies.png")
    
# open sim.hdf5 for processing

an = pyspawn.fafile("sim.hdf5")
work = pyspawn.fafile("working.hdf5")
# create N.dat and store the data in times and N
an.fill_nuclear_bf_populations(column_filename = "N.dat")
an.fill_trajectory_populations(column_file_prefix = "Pop")
an.fill_labels()
# write files with energy data for each trajectory
an.fill_trajectory_energies(column_file_prefix="E")
# list all datasets
an.list_datasets()

ntraj = 6
nstates = 3
colors = ("r", "g", "b", "m", "y", "k")
linestyles = ("-", "--", "-.")
arrays = ("poten", "pop", "toten", "aven", "kinen", "time")
labels = an.datasets["labels"][0:2]

for array in arrays:
    exec(array +"= dict()")

for traj in an.datasets["labels"]:
    
    poten[traj] = an.datasets[traj + "_poten"]
    pop[traj] = an.datasets[traj + "_pop"]
    toten[traj] = an.datasets[traj + "_toten"]
    aven[traj] = an.datasets[traj +"_aven"]
    kinen[traj] = an.datasets[traj + "_kinen"]
    time[traj] = an.datasets[traj + "_time"]
   
g1 = plt.figure("Electronic Populations")
plot_el_population(labels, nstates, colors, linestyles)
plt.show()


g2 = plt.figure("Energies")
plot_energies(labels, nstates, colors, linestyles)
plt.show()


g3 = plt.figure("Nuclear Populations")
plot_nuclear_populations(labels, ntraj, colors, linestyles)   
plt.show()


f = plt.figure(3)
plt.scatter(time["00"], toten["00"], label = "00")
# plt.scatter(times_b0.ravel(), tot_b0.ravel(), c=t1, label = "00b0")
plt.xlabel('Time, au')
plt.ylabel('Total Energy, au')
plt.ylim([min(toten["00"]), max(toten["00"])])
plt.legend()
plt.show()

