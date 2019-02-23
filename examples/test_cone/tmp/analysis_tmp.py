# this analysis script processes the sim.hdf5 file into various human-readable 
# formats.  This script can be run while the simulation is in progress.
import pyspawn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import os
import glob
def plot_nuclear_populations(keys, ntraj, colors, linetypes, labels, markers):

    g3 = plt.figure("Nuclear Populations")
    N = an.datasets["nuclear_bf_populations"]
    qm_time = an.datasets["quantum_times"]
    for n in range(ntraj):
        plt.plot(qm_time, N[:, n+1],linestyle=linestyles[n],\
		marker=markers[n], markersize=3, markevery=15,\
                 label = labels[n])
    plt.xlabel('Time')
    plt.ylabel('Nuclear Population')
    plt.legend()
    #plt.show()
    g3.savefig("Nuc_pop.png")
    
def plot_el_population(keys, numstates, colors, linestyles, markers):
    
    g1 = plt.figure("Electronic Populations")
    l = 0
    for key in keys:
        for n in range(numstates):
            #print key
            plt.plot(time[key], pop[key][:, n], color=colors[n],\
                     label = key + ": " + str((n+1)) + "state",
                     linestyle=linestyles[l],marker=markers[l],markersize=2, markevery=20)
        l += 1
    plt.xlabel('Time, au')
    plt.ylabel('Population, au')
    plt.legend()
    #plt.show()
    g1.savefig("Elec_pop.png")

def plot_approx_el_population(keys, numstates, colors, linestyles, markers):
    
    g1 = plt.figure("Approximate Electronic Populations")
    l = 0
    for key in keys:
        key='00b0'
        approx_pop = trajfile['traj_'+key]['approx_pop']
        for n in range(numstates):
            #print key
            plt.plot(time[key], approx_pop[:, n], color=colors[n],\
                     label = key + ": " + str((n+1)) + "state",
                     linestyle=linestyles[l],marker=markers[l],markersize=2, markevery=20)
        l += 1
    plt.xlabel('Time, au')
    plt.ylabel('Population, au')
    plt.legend()
    #plt.show()
    g1.savefig("Approx_elec_pop.png")
    
def plot_energies(keys, numstates, colors, linetypes, markers):
    
    g2 = plt.figure("Energies")
    l = 0
    for key in keys:
        for n in range(numstates):
	        if l == 0:
		        cur_label = str((n+1)) + "state"
	        else:
		        cur_label = None
	        plt.plot(time[key][::10], poten[key][::10, n], color=colors[n],\
                     label = cur_label,
                     linestyle=linestyles[l], marker=markers[l],alpha=0.5,markersize=3, markevery=2)
        
        l += 1    
    
    l = 0
    
    for key in keys:
        plt.plot(time[key], aven[key], color='black',\
                     label = key,
                     linestyle=linestyles[l], marker=markers[l], markersize=3, markevery=20)    
        l += 1
    
    plt.xlabel('Time, au')
    plt.ylabel('Energy, au')
    plt.text(time['00'][-1]-20, poten['00'][-1,0]+0.03, str(round(pop_band[0]*100))+"%")
#     plt.text(time['00'][-1]-20, poten['00'][-1,4]+0.01, str(round(pop_band[1]*100))+"%")
#     plt.text(time['00'][-1]-20, poten['00'][-1,5]-0.03, str(round(pop_band[2]*100))+"%")
    plt.legend()
    #plt.show()
    g2.savefig("Energies.png")

def plot_approx_energies(keys, numstates, colors, linetypes, markers):
    
    g2 = plt.figure(" Approximate Eigenstates' Energies")
    l = 0
    print trajfile['traj_00'].keys()
    for key in keys:
        approx_e = trajfile['traj_'+key]['approx_energies']
        for n in range(numstates):
            if l == 0:
                cur_label = str((n+1)) + "state"
            else:
                cur_label = None
            plt.plot(time[key][::10], approx_e[::10, n], color=colors[n],\
                     label = cur_label,
                     linestyle=linestyles[l], marker=markers[l],alpha=0.5,markersize=3, markevery=2)
        
        l += 1    
    
    l = 0
    
    for key in keys:
        plt.plot(time[key], aven[key], color='black',\
                     label = key,
                     linestyle=linestyles[l], marker=markers[l], markersize=3, markevery=20)    
        l += 1
    
    plt.xlabel('Time, au')
    plt.ylabel('Approximate Energy, au')
    plt.text(time['00'][-1]-20, poten['00'][-1,0]+0.03, str(round(pop_band[0]*100))+"%")
#     plt.text(time['00'][-1]-20, poten['00'][-1,4]+0.01, str(round(pop_band[1]*100))+"%")
#     plt.text(time['00'][-1]-20, poten['00'][-1,5]-0.03, str(round(pop_band[2]*100))+"%")
    plt.legend()
    #plt.show()
    g2.savefig("Approx_energies.png")

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
    #plt.show()
    g4.savefig("Total_Energies.png")

def plot_total_pop():
    
    g5 = plt.figure("Total Electronic Populations")
    for n in range(nstates):
        plt.plot(time["00"][:], el_pop[:,n], color=colors[n], label=str(n+1) + " state")
    plt.xlabel('Time, au')
    plt.ylabel('Electronic Population, au')
    plt.text(time['00'][-1]-10,el_pop[-1,0]-0.05, str(round(el_pop[-1,0]*100))+"%")
    #plt.title('Ehrenfest (no cloning)')
    plt.legend()
    #plt.show()
    g5.savefig("Total_El_pop.png")
    
an = pyspawn.fafile("sim.hdf5")
work = pyspawn.fafile("working.hdf5")
# create N.dat and store the data in times and N
an.fill_nuclear_bf_populations(column_filename = "N.dat")
an.fill_trajectory_populations(column_file_prefix = "Pop")
an.fill_labels()
# write files with energy data for each trajectory
an.fill_trajectory_energies(column_file_prefix="E")
# an.fill_approx_el_populations()
# list all datasets
# an.list_datasets()

ntraj = len(an.datasets["labels"])
nstates = 9
colors = ("r", "g", "b", "m", "y", "tab:purple", 'xkcd:sky blue',"xkcd:teal blue", 'xkcd:puce', 'k')
linestyles = ("-", "--", "-.", ":","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-")
markers=("None","None","None","None","d","o","v","^","s","p","d","o","v","^","s","p", "d","o","v","^","s","p","d","o","v","^","s","p")
arrays = ("poten", "pop", "toten", "aven", "kinen", "time")
labels = an.datasets["labels"]
el_pop = an.datasets["el_pop"]

h5filename = "sim.hdf5"
trajfilename = "working.hdf5"
trajfile = h5py.File(trajfilename, "r")   
full_H = trajfile["traj_00"].attrs["full_H"]
krylov_sub_n = trajfile["traj_00"].attrs["krylov_sub_n"]
h5file = h5py.File(h5filename, "r")
print h5file["sim"].attrs.keys()

p_threshold = h5file["sim"].attrs['p_threshold']
pop_threshold = h5file["sim"].attrs['pop_threshold']
olapmax = h5file["sim"].attrs["olapmax"]


for array in arrays:
    exec(array +"= dict()")

for traj in an.datasets["labels"]:
    
    poten[traj] = an.datasets[traj + "_poten"]
    pop[traj] = an.datasets[traj + "_pop"]
    toten[traj] = an.datasets[traj + "_toten"]
    aven[traj] = an.datasets[traj +"_aven"]
    kinen[traj] = an.datasets[traj + "_kinen"]
    time[traj] = an.datasets[traj + "_time"]

pop_band = np.zeros(3)
pop_band[0] = el_pop[-1, 0]
pop_band[1] = sum(el_pop[-1, 1:5])
pop_band[2] = sum(el_pop[-1, 5:9])
print el_pop[-1,0]
print sum(el_pop[-1, 1:5])
print sum(el_pop[-1, 5:9])
np.savetxt('pop_band.dat', pop_band)


# Plotting
if time['00'].shape[0] == el_pop[:,0].shape[0]:
    plot_total_pop()
plot_el_population(labels, nstates, colors, linestyles, markers)
plot_approx_el_population(labels, krylov_sub_n, colors, linestyles, markers)
plot_energies(labels, nstates, colors, linestyles, markers)
plot_approx_energies(labels, krylov_sub_n, colors, linestyles, markers)
plot_nuclear_populations(labels, ntraj, colors, linestyles, an.datasets["labels"], markers)   
plot_total_energies(an.datasets["labels"], len(an.datasets["labels"]), colors, linestyles)
print trajfile['traj_00'].keys()
print "Done"
