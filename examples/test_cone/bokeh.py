import inspect
import bokeh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import glob
import pyspawn
import h5py
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display
from glob import glob
from bokeh.io import curdoc, output_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import widgetbox, row, column
from bokeh.models import Slider, CheckboxGroup
from bokeh.models.glyphs import MultiLine
from bokeh.models import CategoricalColorMapper, ColorMapper
from bokeh.palettes import Spectral11, Paired, Category20


h5filename = "sim.hdf5"
trajfilename = "working.hdf5"
trajfile = h5py.File(trajfilename, "r")   
#full_H = trajfile["traj_00"].attrs["full_H"]
#krylov_sub_n = trajfile["traj_00"]["krylov_sub_n"]
h5file = h5py.File(h5filename, "r")
#print h5file["sim"].attrs.keys()

an = pyspawn.fafile("sim.hdf5")
work = pyspawn.fafile("working.hdf5")
# create N.dat and store the data in times and N
an.fill_nuclear_bf_populations(column_filename = "N.dat")
an.fill_trajectory_populations(column_file_prefix = "Pop")
an.fill_labels()
# write files with energy data for each trajectory
an.fill_trajectory_energies(column_file_prefix="E")
labels = an.datasets["labels"]
ntraj = len(an.datasets["labels"])
total_el_pop = an.datasets["el_pop"]
N = an.datasets["nuclear_bf_populations"]
qm_time = an.datasets["quantum_times"]

def print_classes():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            print(obj)



nstates = 9
arrays = ("poten", "pop", "toten", "aven", "kinen", "time")

for array in arrays:
    exec(array +"= dict()")

for traj in an.datasets["labels"]:
    
    poten[traj] = an.datasets[traj + "_poten"]
    pop[traj] = an.datasets[traj + "_pop"]
    toten[traj] = an.datasets[traj + "_toten"]
    aven[traj] = an.datasets[traj +"_aven"]
    kinen[traj] = an.datasets[traj + "_kinen"]
    time[traj] = an.datasets[traj + "_time"]

mypalette=Category20[nstates]
qm_time_array = []
total_el_pop_array = []
for n in range(nstates):
    total_el_pop_array.append(total_el_pop[:,n].tolist())
    qm_time_array.append(qm_time.tolist()) 
data_qm = {'qm_time': qm_time_array,
	   'total_el_pop':total_el_pop_array,
	   'colors': mypalette,
	   'states':[str(n)+ ' state' for n in range(nstates)]}
source_qm = ColumnDataSource(data_qm)
colors = ["red", "green", "blue", "magenta", "yellow", "purple", 'darkmagenta',"darlsalmon", 'gold', 'black']
color_mapper = CategoricalColorMapper(factors=labels, palette=Category20[20])
plot = figure(plot_height=450, plot_width=600, title="Total Energy")
plot2 = figure(plot_height=450, plot_width=600, title="Ehrenfest Energy")
plot3 = figure(plot_height=450, plot_width=600, title="Electronic population")
plot4 = figure(plot_height=450, plot_width=600, title="Energies Test")
times = []
e = []
av_en = []
elec_pop = []
pot_en = []
labels_full = []
labels_array = []
nstates_array = []
for key in labels:
    for nstate in range(nstates):
	times.append(time[key])
	e.append(toten[key])
        pot_en.append(poten[key][:,nstate])
        av_en.append(aven[key])
        elec_pop.append(pop[key])
	labels_full.append(key +":"+ str(nstate)+'state'),
	labels_array.append(key)
	nstates_array.append(str(nstate))
data = {
    'time': times,
    'tot_e': e,
    'av_en': av_en,
    'pot_en': pot_en,
    'labels' : labels_array,
    'labels_full': labels_full,
    'elec_pop' : elec_pop,
    'states': nstates_array
}
color_mapper_qm = CategoricalColorMapper(factors= [str(n) for n in range(nstates)], palette=Category20[nstates])
source = ColumnDataSource(data)
def update_plot(attr, old, new): 
    new_labels = []
    for n in checkbox.active:
	new_labels.append(checkbox.labels[n])    
    times = []
    tot_energies = []
    av_en = []
    elec_pop = []
    pot_en = []
    labels_full = []
    new_labels_array = []
    nstates_array = []
    for key in new_labels:
	for nstate in range(nstates):
	    times.append(time[key][:,0].tolist())
	    tot_energies.append(toten[key][:,0].tolist())
            av_en.append(aven[key][:,0].tolist())
    	    pot_en.append(poten[key][:,nstate])
	    elec_pop.append(pop[key][:,0].tolist())
	    new_labels_array.append(key),
	    labels_full.append(key +":"+ str(nstate)+'state')
	    nstates_array.append(str(nstate))
    new_data = {
        'time': times,
        'tot_e': tot_energies,
	'av_en': av_en,
	'pot_en': pot_en,
	'labels': new_labels_array,
	'labels_full': labels_full,
	'elec_pop' : elec_pop,
	'states': nstates_array 
    }
    source.data = new_data

plot.multi_line(xs='time', ys='tot_e', source=source, color=dict(field='labels', transform=color_mapper), legend='labels', line_width=2)
plot2.multi_line(xs='time', ys='av_en', source=source, color=dict(field='labels', transform=color_mapper), legend='labels', line_width=2)

plot3.multi_line(xs='qm_time', ys='total_el_pop', source=source_qm, line_width=2, color='colors', legend='states')
plot2.legend.location = 'bottom_left'
plot.x_range = plot2.x_range
plot3.x_range = plot.x_range
print 'time shape = ', len(data['time'])
#plot4.multi_line(xs='time', ys='pot_en', source=source, color=dict(field='states', transform=color_mapper_qm))
#print inspect.getmembers(bokeh.models.markers[__name__], inspect.isclass)
# Add the plot to the current document
checkbox = CheckboxGroup(labels=list(labels), active=list([n for n in range(labels.shape[0])]))

checkbox.on_change('active', update_plot)
doc = column(widgetbox(checkbox), plot, plot2, plot3, plot4)
curdoc().add_root(doc)
#show(doc)
