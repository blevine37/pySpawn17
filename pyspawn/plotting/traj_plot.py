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
