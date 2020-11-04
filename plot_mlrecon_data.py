#!/usr/bin/env python3

# author: Michael A. Perlin (github.com/perlinm)

import sys, glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

orientation = "horiz" if len(sys.argv) < 2 else sys.argv[1]
assert(orientation in ["horiz","vert"])

max_width = 8.6/2.54 # maximum width of single-column figure
data_dir = "./mlrecon_data/"

circuit_type = "clustered"
qubit_min = 8
qubit_max = 20

labels = [ "full", "direct", "MLFT" ]
colors = [ "#1f77b4", "#ff7f0e", "#2ca02c" ]
markers = [ "o", "s", "^" ]

params = { "font.size" : 9,
           "text.usetex" : True }
plt.rcParams.update(params)

def get_figure_axes(frag_nums):
    if orientation == "horiz":
        fig_rows = 2
        fig_cols = frag_nums
        fig_width = 1.6 * fig_cols
        fig_height = 3.4
    else: # if orientation == "vert"
        fig_rows = frag_nums
        fig_cols = 2
        fig_width = max_width
        fig_height = 1.5 * fig_rows
    return plt.subplots(fig_rows, fig_cols, sharex = True, sharey = True,
                        figsize = (fig_width, fig_height))

def get_log_ticks(axis_limits):
    base, subs = 10, np.arange(0,1.1,0.1)
    major_locator = mpl.ticker.LogLocator(base = base)
    minor_locator = mpl.ticker.LogLocator(base = base, subs = subs)
    min_val, max_val = axis_limits
    def filter(values):
        return [ val for val in values if min_val <= val <= max_val ]
    major_tick_values = filter(major_locator.tick_values(min_val, max_val))
    minor_tick_values = filter(minor_locator.tick_values(min_val, max_val))
    return major_tick_values, minor_tick_values

##########################################################################################
# get data file info

data_files = glob.glob(data_dir + "*.txt")

def info(file):
    file_tags = file.split("/")[-1][:-4].split("_")
    return { "circuit" : file_tags[0],
             "qubits" : file_tags[1],
             "frags" : file_tags[2],
             "log10_shots" : file_tags[3] }

def select_files(**circuit_info):
    return [ file for file in data_files
             if all( tag in file for tag in circuit_info.values() ) ]

def all_values(info_key, **selection_args):
    return sorted(set([ info(file)[info_key]
                        for file in select_files(**selection_args) ]))

##########################################################################################
# make fidelity plots with variable qubit numbers

circuit_info = { "circuit" : circuit_type,
                 "log10_shots" : "S6.0" }
circuit_frags = all_values("frags", **circuit_info)

figure, axes = get_figure_axes(len(circuit_frags))

for frag_idx, frags in enumerate(circuit_frags):
    circuit_info["frags"] = frags

    def file(qubits):
        file_info = dict(circuit_info, **{ "qubits" : qubits })
        files = select_files(**file_info)
        assert(len(files) == 1)
        return files[0]

    # identify qubit numbers for which we have data
    qubit_strs = sorted(all_values("qubits", **circuit_info))
    qubit_nums = [ int(qq[1:]) for qq in qubit_strs ]

    # filter out by qubit minima / maxima
    keep = [ qubit_min <= qubit_num <= qubit_max
             for qubit_num in qubit_nums ]
    qubit_strs = np.array(qubit_strs)[keep]
    qubit_nums = np.array(qubit_nums)[keep]

    # indexed by [ qubit_num, circuit_instance, simulation_method ]
    data = np.array([ np.loadtxt(file(qubit_str)) for qubit_str in qubit_strs ])

    for fidelities, label, color, marker in zip(data.T, labels, colors, markers):
        fidelity_avg = 1-np.mean(fidelities, axis = 0)
        fidelity_std = np.std(fidelities, axis = 0)
        plot_args = dict( color = color, label = label,
                          fillstyle = "none", markersize = 4 )
        if orientation == "horiz":
            ax_avg = axes[0,frag_idx]
            ax_std = axes[1,frag_idx]
        else: # if orientation == "vert"
            ax_avg = axes[frag_idx,0]
            ax_std = axes[frag_idx,1]
        ax_avg.semilogy(qubit_nums, fidelity_avg, marker, **plot_args)
        ax_std.semilogy(qubit_nums, fidelity_std, marker, **plot_args)

# fix axis ticks
for idx in np.ndindex(axes.shape):
    axes[idx].tick_params(top = True)
    axes[idx].tick_params(right = True)

# set titles and axis labels
if orientation == "horiz":
    axes[0,0].set_ylabel(r"$1-f$")
    axes[1,0].set_ylabel(r"$\sigma_f$")

    for frag_idx, frags in enumerate(circuit_frags):
        axes[0,frag_idx].set_title(f"$F={frags[1:]}$", pad = 8)

else: # if orientation == "vert"
    axes[0,0].set_title(r"$1-f$")
    axes[0,1].set_title(r"$\sigma_f$")

    for frag_idx, frags in enumerate(circuit_frags):
        axes[frag_idx,1].yaxis.set_label_position("right")
        axes[frag_idx,1].set_ylabel(f"$F={frags[1:]}$", labelpad = 8)

for axis in axes[-1,:]:
    axis.set_xlabel("$Q$")

# set horizontal axis ticks and labels
xticks = list(range(qubit_min, qubit_max+1, 2))
xticklabels = [ tick if tick % 4 == 0 else "" for tick in xticks ]
axes[0,0].set_xticks(xticks)
axes[0,-1].set_xticklabels(xticklabels)

# set vertical axis ticks
major_tick_values, minor_tick_values = get_log_ticks(axes[0,0].get_ylim())
axes[0,0].yaxis.set_ticks(major_tick_values)
axes[0,0].yaxis.set_ticks(minor_tick_values, minor = True)

if orientation == "horiz":
    axes[1,0].legend(loc = "best")
else: # if orientation == "vert"
    axes[0,1].legend(loc = "best")

plt.tight_layout(pad = 0.2)
plt.savefig(f"fidelities_qubits_{orientation}.pdf", bbox_inches = "tight")

##########################################################################################
# make fidelity plots with variable shot numbers

circuit_info = { "circuit" : circuit_type,
                 "qubits" : "Q12" }
circuit_frags = all_values("frags", **circuit_info)

figure, axes = get_figure_axes(len(circuit_frags))

for frag_idx, frags in enumerate(circuit_frags):
    circuit_info["frags"] = frags

    def file(log10_shots):
        file_info = dict(circuit_info, **{ "log10_shots" : log10_shots })
        files = select_files(**file_info)
        assert(len(files) == 1)
        return files[0]

    # identify qubit numbers for which we have data
    log10_shot_strs = sorted(all_values("log10_shots", **circuit_info))
    shot_nums = [ 10**float(ss[1:]) for ss in log10_shot_strs ]

    # indexed by [ qubit_num, circuit_instance, simulation_method ]
    data = np.array([ np.loadtxt(file(log10_shot_str))
                      for log10_shot_str in log10_shot_strs ])

    for fidelities, label, color, marker in zip(data.T, labels, colors, markers):
        fidelity_avg = 1-np.mean(fidelities, axis = 0)
        fidelity_std = np.std(fidelities, axis = 0)
        plot_args = dict( color = color, label = label,
                          fillstyle = "none", markersize = 4 )
        if orientation == "horiz":
            ax_avg = axes[0,frag_idx]
            ax_std = axes[1,frag_idx]
        else: # if orientation == "vert"
            ax_avg = axes[frag_idx,0]
            ax_std = axes[frag_idx,1]
        ax_avg.loglog(shot_nums, fidelity_avg, marker, **plot_args)
        ax_std.loglog(shot_nums, fidelity_std, marker, **plot_args)

# fix axis ticks
for idx in np.ndindex(axes.shape):
    axes[idx].tick_params(top = True)
    axes[idx].tick_params(right = True)

# set titles and axis labels
if orientation == "horiz":
    axes[0,0].set_ylabel(r"$1-f$")
    axes[1,0].set_ylabel(r"$\sigma_f$")

    for frag_idx, frags in enumerate(circuit_frags):
        axes[0,frag_idx].set_title(f"$F={frags[1:]}$", pad = 8)

else: # if orientation == "vert"
    axes[0,0].set_title(r"$1-f$")
    axes[0,1].set_title(r"$\sigma_f$")

    for frag_idx, frags in enumerate(circuit_frags):
        axes[frag_idx,1].yaxis.set_label_position("right")
        axes[frag_idx,1].set_ylabel(f"$F={frags[1:]}$", labelpad = 8)

for axis in axes[-1,:]:
    axis.set_xlabel("$S$")

# set horizontal axis ticks
major_tick_values, minor_tick_values = get_log_ticks(axes[0,0].get_xlim())
axes[0,0].xaxis.set_ticks(major_tick_values)
axes[0,0].xaxis.set_ticks(minor_tick_values, minor = True)

# set vertical axis ticks
major_tick_values, minor_tick_values = get_log_ticks(axes[0,0].get_ylim())
axes[0,0].yaxis.set_ticks(major_tick_values)
axes[0,0].yaxis.set_ticks(minor_tick_values, minor = True)

if orientation == "horiz":
    axes[0,0].legend(loc = "best")
else: # if orientation == "vert"
    axes[2,0].legend(loc = "best")

plt.tight_layout(pad = 0.2)
plt.savefig(f"fidelities_shots_{orientation}.pdf", bbox_inches = "tight")
