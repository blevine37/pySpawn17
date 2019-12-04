import copy
import pyspawn
import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob


def plot_total_energies(time, toten, keys, istates, colors, markers, linestyles):
    """Plots total classical energies for each trajectory and saves it to png file.
    This plot is very useful to check if energy is conserved.
    Color is represented by the istate of a trajectory
    Linestyle and Markers represent different trajectories
    """

    ax = plt.figure("Total Energies")
    min_energy = min(toten["00"])
    max_energy = max(toten["00"])
    for index, key in enumerate(keys):
        if len(keys) > 1:
            plt.plot(time[key], toten[key], label=key, color=colors[int(istates[index])],
                     linestyle=linestyles[index], marker=markers[index])
        else:
            plt.plot(time[key], toten[key], label=key, color=colors[int(istates)],
                     linestyle=linestyles[index], marker=markers[index])
        if min(toten[key]) < min_energy:
            min_energy = min(toten[key])
        if max(toten[key]) > max_energy:
            max_energy = max(toten[key])

    plt.xlabel('Time, au')
    plt.ylabel('Total Energy, au')
    plt.ylim([min_energy - 0.05 * (max_energy - min_energy), max_energy + 0.05 * (max_energy - min_energy)])
    plt.legend()
    plt.tick_params(axis='both', which='major')
    plt.title('Total Energies')
    plt.tight_layout()
    ax.savefig("Total_Energies.png", dpi=300)


def plot_total_pop(time, el_pop, nstates, colors):
    """ This plots the total electronic population on each
    electronic state (over all basis functions)"""

    g5 = plt.figure("Total Electronic Populations")
    for n_state in range(nstates):
        plt.plot(time, el_pop[:, n_state], color=colors[n_state],
                 label='S' + str(n_state))

    plt.xlabel('Time, au')
    plt.ylabel('Population')
    plt.title('Total Electronic Population')
    plt.legend()
    plt.tight_layout()
    g5.savefig("Total_El_pop.png", dpi=300)


def plot_e_gap(time, poten, keys, state1, state2, istates, colors, linestyles, markers):
    """Plots gaps between the specified states for all trajectories
    istates order needs to be fixed!
    """

    g2 = plt.figure("Energy gap")

    for index, key in enumerate(keys):
        if len(keys) > 1:
            plt.plot(time[key], poten[key][:, state2] - poten[key][:, state1], linestyle=linestyles[index],
                     marker=markers[index], color=colors[int(istates[index])],
                     label=key + ": " + r'$S_{}$'.format(state2) + "-"
                           + r'$S_{}$'.format(state1))
        else:
            plt.plot(time[key], poten[key][:, state2] - poten[key][:, state1], linestyle=linestyles[index],
                     marker=markers[index], color=colors[int(istates)],
                     label=key + ": " + r'$S_{}$'.format(state2) + "-"
                           + r'$S_{}$'.format(state1))
    plt.xlabel('Time, au')
    plt.title('Energy gaps, au')
    plt.ylabel('Energy gap, au')
    plt.legend()
    g2.savefig("E_gap.png", dpi=300)


def plot_energies(keys, time, poten, numstates, colors, linestyles):

    g3 = plt.figure("Energies")

    for index_key, key in enumerate(keys):
        for index_state, n in enumerate(range(numstates)):
            plt.plot(time[key], poten[key][:, n],
                     label=key + ": " + str((n + 1)) + "state", linestyle=linestyles[index_key],
                     color=colors[index_state])
        plt.xlabel('Time, au')
        plt.ylabel('Energy, au')
        plt.legend()
        g3.savefig("Energies.png", dpi=300)


def plot_tdc(time, tdc, labels, numstates, spawnthresh):

    plt.figure("Time-derivative couplings")
    for key in labels:
        for n in range(numstates):
            plt.plot(time[key], np.abs(tdc[key][:, n]))

    plt.axhline(y=spawnthresh, alpha=0.5, color='r', linewidth=1.0,
                linestyle='--')
    plt.xlabel('Time, au')
    plt.ylabel('Coupling, au')
#     for m in range(nstates):
#         plt.text(spawn_times[m], plt.ylim()[1]-0.05, labels[m], fontsize=10)

    plt.title('Time-derivative couplings, thresh='+str(spawnthresh))
#     plt.tight_layout
    plt.subplots_adjust(right=0.8)
    plt.savefig("all_tdc.png", dpi=300)
