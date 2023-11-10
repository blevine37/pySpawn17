# this analysis script processes the sim.hdf5 file into various human-readable
# formats.  This script can be run while the simulation is in progress.
import copy
import os
import pyspawn
import matplotlib.pyplot as plt
import numpy as np


def plot_total_pop(times):
    """ This plots the total electronic population on each
    electronic state (over all basis functions)"""

    g5 = plt.figure("Total Electronic Populations")
    for n_state in range(nstates):
        plt.plot(times, el_pop[:, n_state], color=colors[n_state],
                 label='S' + str(n_state))
    plt.xlabel('Time, au', fontsize=medium_size)
    plt.ylabel('Population', fontsize=medium_size)
#    plt.text(time['00'][-1] - 13, el_pop[-1, 0] - 0.1,
#             str(round(el_pop[-1, 0] * 100)) + "%", fontsize=medium_size)
    plt.title('Total Electronic Population', fontsize=large_size)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.legend(fontsize=medium_size)
    plt.tight_layout()
    g5.savefig("Total_El_pop.png", dpi=300)


def plot_energies(keys, numstates, istates, colors,
                  linetypes, xlimits, ylimits):

    g2 = plt.figure("Potential Energies")
    line = 0
    col = 0
    for key in keys:
        adj_colors = copy.copy(colors)
#         adj_colors[istates[l]]="k"
        for n in range(numstates):
            if n == istates[line]:
                plt.plot(time[key], poten[key][:, n], color=colors[col],
                         label=key + ": " + str((n+1)) + "state")
#                          linestyle=linestyles[line])
        line += 1
        col += 1
    plt.xlabel('Time, au')
    plt.ylabel('Pot. Energy, au')
    plt.legend()
    plt.xlim(xlimits)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.title('Potential Energies', fontsize=large_size)
#    plt.ylim(ylimits)
#     plt.show()
    plt.tight_layout()
    g2.savefig("Energies.png")


def plot_total_energies(time, toten, keys):
    """Plots total classical energies for each trajectory,
    useful to look at energy conservation"""

    g4 = plt.figure("Total Energies")
    min_E = min(toten["00"])
    max_E = max(toten["00"])
    col = 0
    for key in keys:
        plt.plot(time[key], toten[key],
        label=key,color=colors[col])
        if min(toten[key]) < min_E:
            min_E = min(toten[key])
        if max(toten[key]) > max_E:
            max_E = max(toten[key])
        col += 1

    plt.xlabel('Time, au', fontsize=medium_size)
    plt.ylabel('Total Energy, au', fontsize=medium_size)
    plt.ylim([min_E - 0.05 * (max_E - min_E), max_E + 0.05 * (max_E - min_E)])
    plt.legend(fontsize=medium_size)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.title('Total Energies', fontsize=large_size)
    plt.tight_layout()
    g4.savefig("Total_Energies.png", dpi=300)


def plot_nuclear_populations(ntraj, linestyles, labels, markers):
    """Plots nuclear basis functions' contributions to the total nuclear wf"""

    g3 = plt.figure("Nuclear Populations")
    N = an.datasets["nuclear_bf_populations"]
    qm_time = an.datasets["quantum_times"]
    for n in range(ntraj):
        plt.plot(qm_time, N[:, n + 1], linestyle=linestyles[n],
                 marker=markers[n], markersize=3, markevery=15,
                 label=labels[n])
    plt.xlabel('Time, au', fontsize=medium_size)
    plt.ylabel('Nuclear Population', fontsize=medium_size)
    plt.legend(fontsize=medium_size)
    plt.tick_params(axis='both', which='major', labelsize=small_size)
    plt.title('Nuclear Population', fontsize=large_size)
#    plt.tight_layout()
    g3.savefig("Nuc_pop.png", dpi=300)


# open sim.hdf5 for processing
simfile = os.path.join(os.pardir, "sim.hdf5")
an = pyspawn.fafile(simfile)

# create N.dat and store the data in times and N
an.fill_electronic_state_populations(column_filename="N.dat")
an.fill_labels()
an.fill_istates()
an.get_numstates()

times = an.datasets["quantum_times"]
el_pop = an.datasets["electronic_state_populations"]
istates = an.datasets["istates"]
labels = an.datasets["labels"]
ntraj = len(an.datasets["labels"])
nstates = an.datasets['numstates']

an.fill_nuclear_bf_populations()

# write files with energy data for each trajectory
an.fill_trajectory_energies(column_file_prefix="E")

# write file with time derivative couplings for each trajectory
an.fill_trajectory_tdcs(column_file_prefix="tdc")

# compute Mulliken population of each trajectory
an.fill_mulliken_populations(column_filename="mull.dat")
mull_pop = an.datasets["mulliken_populations"]

# write files with temperature for each trajectory
an.fill_trajectory_temperature(column_file_prefix="Temperature")

# list all datasets
an.list_datasets()

colors = ["r", "g", "b", "m", "y", "c", "k", "gray", "indigo", "fuchsia", "lime", "darkred", "gold", 
"olive", "cyan", "teal", "orange", "limegreen", "firebrick", "olive"]
linestyles = ("-", "--", "-.", ":", "-", "--", "-.", ":","-", "--", "-.","-", "--", "-.", ":", "-", "--", "-.", ":","-", "--", "-.")
small_size = 12
medium_size = 14
large_size = 16
xlimits = [0, 16000]
ylimits = [-0.12, 0.08]
markers = ("None", "None", "None", "None", "d", "o", "v", "^", "s", "p", "d",
           "o", "v", "^", "s", "p", "d", "o", "v", "^", "s", "p", "d", "o",
           "v", "^", "s", "p")

poten = {}
toten = {}
kinen = {}
time = {}

for traj in an.datasets["labels"]:

    poten[traj] = an.datasets[traj + "_poten"]
    toten[traj] = an.datasets[traj + "_toten"]
    kinen[traj] = an.datasets[traj + "_kinen"]
    time[traj] = an.datasets[traj + "_time"]

plot_total_pop(times)
plot_energies(labels, nstates, istates, colors, linestyles, xlimits, ylimits)
plot_total_energies(time, toten, labels)
plot_nuclear_populations(ntraj, linestyles, an.datasets["labels"], markers)

# write files with geometric data for each trajectory
#bonds = np.array([[16, 17],
#                  [17, 23],
#                  [18, 23],
#                  [4, 16]])
#
#angles = np.array([[5, 18, 23],
#                   [5, 4, 16]])
#
#diheds = np.array([[4, 5, 18, 23],
#                   [0, 5, 18, 19],
#                   [5, 4, 16, 17]])
#
# pyrs = np.array([[0, 1, 4, 5],
#                 # [1, 0, 2, 3]])
#
#twists = np.array([[18, 5, 4, 0, 23, 19]])
#
#an.fill_trajectory_bonds(bonds, column_file_prefix="bonds")
#an.fill_trajectory_angles(angles, column_file_prefix="angles")
#an.fill_trajectory_diheds(diheds, column_file_prefix="diheds")
# an.fill_trajectory_pyramidalizations(pyrs, column_file_prefix="pyrs")
#an.fill_trajectory_twists(twists, column_file_prefix="twists")
an.write_xyzs()
print "\nDone"
