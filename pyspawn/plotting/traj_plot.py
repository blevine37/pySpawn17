import copy
import pyspawn
import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob
au_to_fs = 0.02418884254
au_to_ev = 13.6
au_to_ang = 0.529177

def plot_total_energies(time, toten, keys, istates_dict, colors, markers, linestyles):
    """Plots total classical energies for each trajectory and saves it to png file.
    This plot is very useful to check if energy is conserved.
    Color is represented by the istate of a trajectory
    Linestyle and Markers represent different trajectories
    """

    ax = plt.figure("Total Energies", figsize=(4.8, 3.6))
    min_energy = min(toten["00"]) * au_to_ev
    max_energy = max(toten["00"]) * au_to_ev
    for index, key in enumerate(keys):
        if min(toten[key]*au_to_ev) < min_energy:
                min_energy = min(toten[key]*au_to_ev)
        if max(toten[key]*au_to_ev) > max_energy:
                max_energy = max(toten[key]*au_to_ev)
        plt.plot(time[key]*au_to_fs, toten[key]*au_to_ev-min_energy, label=key, color=colors[int(istates_dict[key])],
                 linestyle=linestyles[index], marker=markers[index])

    plt.xlabel('Time, fs')
    plt.ylabel('Total Energy, eV')
    # plt.ylim([min_energy - 0.05 * (max_energy - min_energy), max_energy + 0.05 * (max_energy - min_energy)])
    plt.legend()
    plt.tick_params(axis='both', which='major')
    plt.title('Total Energies')
    plt.tight_layout()
    ax.savefig("Total_Energies.png", dpi=300)


def plot_total_pop(time, el_pop, nstates, colors):
    """ This plots the total electronic population on each
    electronic state (over all basis functions)"""

    g5 = plt.figure("Total Electronic Populations", figsize=(4.8, 3.6))
    for n_state in range(nstates):
        plt.plot(time*au_to_fs, el_pop[:, n_state], color=colors[n_state],
                 label='S' + str(n_state))
    plt.xlabel('Time, fs')
    plt.ylabel('Population')
    plt.title('Total Electronic Population')
    plt.legend()
    plt.tight_layout()
    g5.savefig("Total_El_pop.png", dpi=300)


def plot_e_gap(time, poten, keys, state1, state2, istates_dict, colors, linestyles, markers):
    """Plots gaps between the specified states for all trajectories
    istates order needs to be fixed!
    """

    g2 = plt.figure("Energy gap", figsize=(4.8, 3.6))

    for index, key in enumerate(keys):
        plt.plot(time[key]*au_to_fs, poten[key][:, state2]*au_to_ev - poten[key][:, state1]*au_to_ev,
                 linestyle=linestyles[index], marker=markers[index], color=colors[int(istates_dict[key])],
                 label=key + ": " + r'$S_{}$'.format(state2) + "-"
                 + r'$S_{}$'.format(state1))
    plt.xlabel('Time, fs')
    # plt.title('Energy gaps, au')
    plt.ylabel('Energy gap, eV')
    plt.legend()
    plt.tight_layout()
    g2.savefig("E_gap.png", dpi=300)


def plot_energies(keys, time, poten, numstates, colors, linestyles):

    g3 = plt.figure("Energies", figsize=(4.8, 3.6))

    for index_key, key in enumerate(keys):
        for index_state, n in enumerate(range(numstates)):
            plt.plot(time[key]*au_to_fs, poten[key][:, n]*au_to_ev,
                     label=key + ": " + 'S' + str((n + 1)), linestyle=linestyles[index_key],
                     color=colors[index_state])
    plt.xlabel('Time, fs')
    plt.ylabel('Energy, eV')
    plt.legend()
    plt.tight_layout()
    g3.savefig("Energies.png", dpi=300)


def plot_tdc(time, tdc, keys, numstates, istates_dict, spawnthresh, colors, linestyles, markers):

    plt.figure("Time-derivative couplings", figsize=(4.8, 3.6))

    for index_key, key in enumerate(keys):
        # we only plot subset of trajectories but need all of them to match istates with labels
        for n in range(numstates):
            if n != int(istates_dict[key]):
                # we don't plot coupling with itself which is zero
                plt.plot(time[key][:len(tdc[key])]*au_to_fs, np.abs(tdc[key][:, n]),
                         color=colors[istates_dict[key]], linestyle=linestyles[index_key],
                         marker=markers[index_key])

    plt.axhline(y=spawnthresh, alpha=0.5, color='r', linewidth=1.0,
                linestyle='--')
    plt.xlabel('Time, fs')
    plt.ylabel('Coupling, au')
#     for m in range(nstates):
#         plt.text(spawn_times[m], plt.ylim()[1]-0.05, all_keys[m], fontsize=10)

    plt.title('Time-derivative couplings, thresh='+str(spawnthresh))
#     plt.tight_layout
    plt.subplots_adjust(right=0.8)
    plt.savefig("all_tdc.png", dpi=300)


def plot_bonds(time, keys, bonds_list, bonds_array, colors, linestyles):

    bond_labels = []
    for n in range(np.shape(bonds_list)[0]):
        bond_labels.append(str(bonds_list[n, 0]) + "-" + str(bonds_list[n, 1]))
    plt.figure("Bonds", figsize=(4.8, 3.6))

    for index_key, key in enumerate(keys):
        for n in range(np.shape(bonds_list)[0]):
            plt.plot(time[key]*au_to_fs, bonds_array[key][:, n]*au_to_ang, color=colors[index_key],
                         linestyle=linestyles[n], label=key) #+ ":" + bond_labels[n])

    plt.xlabel('Time, fs')
    plt.gca().set_ylabel('Distance, ' + r'$\AA$')
    #plt.ylabel('Bond length, ang')
    plt.legend()
    # plt.title('Bond lengths')
    plt.tight_layout()
    plt.savefig("bonds.png", dpi=300)


def plot_diheds(time, keys, diheds_list, diheds_array, colors, linestyles):

    diheds_labels = []
    for n in range(np.shape(diheds_list)[0]):
        diheds_labels.append(str(diheds_list[n, 0]) + "-" + str(diheds_list[n, 1]) + "-"
                             + str(diheds_list[n, 2]) + "-" + str(diheds_list[n, 3]))
    plt.figure("Dihedral angles", figsize=(4.8, 3.6))

    for index_key, key in enumerate(keys):
        for n in range(np.shape(diheds_list)[0]):
            plt.plot(time[key]*au_to_fs, diheds_array[key][:, n], color=colors[index_key],
                         linestyle=linestyles[n], label=key) #+ ":" + diheds_labels[n])

    plt.xlabel('Time, fs')
    plt.ylabel('Angle, ' + u'\N{DEGREE SIGN}')
    plt.legend()
    # plt.title('Dihedral Angles')
    plt.tight_layout()
    plt.savefig("dihedral_angs.png", dpi=300)
