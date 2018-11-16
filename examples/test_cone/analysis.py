# this analysis script processes the sim.hdf5 file into various human-readable 
# formats.  This script can be run while the simulation is in progress.
import pyspawn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_nuclear_populations(keys, ntraj, colors, linetypes, labels):

    g3 = plt.figure("Nuclear Populations")
    N = an.datasets["nuclear_bf_populations"]
    qm_time = an.datasets["quantum_times"]
    for n in range(ntraj):
        plt.plot(qm_time, N[:, n+1],\
                     label = labels[n])
    plt.xlabel('Time')
    plt.ylabel('Nuclear Population')
    plt.legend()
    plt.show()
    g3.savefig("Nuc_pop.png")
    
def plot_el_population(keys, numstates, colors, linestyles):
    
    g1 = plt.figure("Electronic Populations")
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
    plt.show()
    g1.savefig("Elec_pop.png")
    
def plot_energies(keys, numstates, colors, linetypes):
    
    g2 = plt.figure("Energies")
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
    plt.ylabel('Energy, au')
    plt.legend()
    plt.show()
    g2.savefig("Energies.png")

def plot_total_energies(keys, ntraj, colors, linetypes):
    
    g4 = plt.figure("Total Energies")
    l = 0
    min_E = min(toten["00"])
    max_E = max(toten["00"])
    for key in keys:
        plt.plot(time[key], toten[key],\
                     label = key)
        if min(toten[key]) < min_E: min_E = min(toten[key])
        if max(toten[key]) > max_E: max_E = max(toten[key])
        l += 1    
    plt.xlabel('Time, au')
    plt.ylabel('Total Energy, au')
    plt.ylim([min_E - 0.05 * (max_E-min_E), max_E + 0.05 * (max_E-min_E)])
    plt.legend()
    plt.show()
    g4.savefig("Total_Energies.png")
    
an = pyspawn.fafile("sim.hdf5")
work = pyspawn.fafile("working.hdf5")
# create N.dat and store the data in times and N
an.fill_nuclear_bf_populations(column_filename = "N.dat")
an.fill_trajectory_populations(column_file_prefix = "Pop")
an.fill_labels()
# write files with energy data for each trajectory
an.fill_trajectory_energies(column_file_prefix="E")
# list all datasets
# an.list_datasets()

ntraj = len(an.datasets["labels"])
nstates = 5
colors = ("r", "g", "b", "m", "y", "k", "k")
linestyles = ("-", "--", "-.", ":")
arrays = ("poten", "pop", "toten", "aven", "kinen", "time")
labels = an.datasets["labels"][0:4]

for array in arrays:
    exec(array +"= dict()")

for traj in an.datasets["labels"]:
    
    poten[traj] = an.datasets[traj + "_poten"]
    pop[traj] = an.datasets[traj + "_pop"]
    toten[traj] = an.datasets[traj + "_toten"]
    aven[traj] = an.datasets[traj +"_aven"]
    kinen[traj] = an.datasets[traj + "_kinen"]
    time[traj] = an.datasets[traj + "_time"]
   
# Plotting
plot_el_population(labels, nstates, colors, linestyles)
plot_energies(labels, nstates, colors, linestyles)
plot_nuclear_populations(labels, ntraj, colors, linestyles, an.datasets["labels"])   
plot_total_energies(an.datasets["labels"], len(an.datasets["labels"]), colors, linestyles)
print "Done"