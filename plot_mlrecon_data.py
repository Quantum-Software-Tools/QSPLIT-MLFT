#!/usr/bin/env python3

# author: Michael A. Perlin (github.com/perlinm)

import sys, glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

data_dir = "./mlrecon_data/"

circuit_type = "clustered"
qubit_min = 8
qubit_max = 20

# labels, colors, and markers for simulation data
labels = [ "full", "direct", "MLFT" ]
colors = [ "tab:blue", "tab:orange", "tab:green" ]
markers = [ "o", "s", "^" ]
data_info = list(zip(labels, colors, markers))

# labels, colors, and markers for analytical estimates
labels = [ "full (est.)", "direct (est.)" ]
colors = [ "k", "tab:red" ]
markers = [ "+", "1" ]
est_info = list(zip(labels, colors, markers))

# misc. plot parameters
params = { "font.size" : 9,
           "text.usetex" : True }
plt.rcParams.update(params)

# sum of fragment infidelities
def frag_infidelity(qubits, frags, total_shots):
    frag_num = int(frags[1:]) # total number of fragments

    # total number of fragment variants
    # 4*3 for first/last fragment, plus 4**2 * 3**2 for middle fragments
    variants = 24 + 144 * (frag_num-2)
    shots = total_shots // variants

    # classical outputs on each fragment
    cls_output_nums = np.array([ qubits // frag_num ] * frag_num)
    for jj in range(frag_num):
        if sum(cls_output_nums) == qubits: break
        cls_output_nums[jj] += 1

    # quantum outputs on each fragment
    qnt_output_nums = np.array([ 2 ] * frag_num)
    qnt_output_nums[0] = qnt_output_nums[-1] = 1

    output_nums = cls_output_nums + qnt_output_nums
    return sum( 2**output_nums ) / shots

frag_infidelity = np.vectorize(frag_infidelity)

##########################################################################################
# extract info from data files

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
# set up figure template

def get_figure_axes(frag_nums = 3):
    fig_rows = 2
    fig_cols = frag_nums
    fig_width = 1.5 * fig_cols + 0.8
    fig_height = 3.3
    return plt.subplots(fig_rows, fig_cols, sharey = True,
                        figsize = (fig_width, fig_height))

figure, axes = get_figure_axes()

##########################################################################################
# plot fidelity as a function of qubit number

log10_shots = 6
shots = 10**log10_shots
circuit_info = { "circuit" : circuit_type,
                 "log10_shots" : f"S{log10_shots:.2f}" }
circuit_frags = all_values("frags", **circuit_info)

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

    # numerical results
    for fidelity, ( label, color, marker ) in zip(data.T, data_info):
        infidelity = 1-np.mean(fidelity, axis = 0)
        plot_args = dict( color = color, label = label,
                          fillstyle = "none", markersize = 5 )
        axes[0,frag_idx].semilogy(qubit_nums, infidelity, marker, **plot_args)

    # analytical estimates of infidelity
    inf_full = 2**qubit_nums/(4*shots)
    inf_cuts = frag_infidelity(qubit_nums, frags, shots)
    for infidelity, ( label, color, marker ) in zip([inf_full, inf_cuts], est_info):
        axes[0,frag_idx].semilogy(qubit_nums, infidelity, marker,
                                  color = color, label = label, zorder = 0)

##########################################################################################
# plot fidelity as a function of shot number

qubits = 16
circuit_info = { "circuit" : circuit_type,
                 "qubits" : f"Q{qubits}" }
circuit_frags = all_values("frags", **circuit_info)

for frag_idx, frags in enumerate(circuit_frags):
    circuit_info["frags"] = frags

    def file(log10_shots):
        file_info = dict(circuit_info, **{ "log10_shots" : log10_shots })
        files = select_files(**file_info)
        assert(len(files) == 1)
        return files[0]

    # identify qubit numbers for which we have data
    log10_shot_strs = sorted(all_values("log10_shots", **circuit_info))
    shot_nums = np.array([ 10**float(ss[1:]) for ss in log10_shot_strs ])

    # indexed by [ qubit_num, circuit_instance, simulation_method ]
    data = np.array([ np.loadtxt(file(log10_shot_str))
                      for log10_shot_str in log10_shot_strs ])

    # numerical results
    for fidelity, ( label, color, marker ) in zip(data.T, data_info):
        infidelity = 1-np.mean(fidelity, axis = 0)
        plot_args = dict( color = color, label = label,
                          fillstyle = "none", markersize = 5 )
        axes[1,frag_idx].loglog(shot_nums, infidelity, marker, **plot_args)

    # analytical estimates of infidelity
    inf_full = 2**qubits/(4*shot_nums)
    inf_cuts = frag_infidelity(qubits, frags, shot_nums)
    for infidelity, ( label, color, marker ) in zip([inf_full, inf_cuts], est_info):
        axes[1,frag_idx].semilogy(shot_nums, infidelity, marker,
                                  color = color, label = label, zorder = 0)

##########################################################################################
# miscellaneous cleanup

# add tick marks to top / right of axes
for idx in np.ndindex(axes.shape):
    axes[idx].tick_params(top = True)
    axes[idx].tick_params(right = True)

# set axis labels and titles
axes[0,0].set_ylabel(r"$1-f$")
axes[1,0].set_ylabel(r"$1-f$")
for axis in axes[0,:]: axis.set_xlabel("$Q$", labelpad = 1)
for axis in axes[1,:]: axis.set_xlabel("$S$")
for frag_idx, frags in enumerate(circuit_frags):
    axes[0,frag_idx].set_title(f"$F={frags[1:]}$", pad = 8)

axes[0,-1].yaxis.set_label_position("right")
axes[1,-1].yaxis.set_label_position("right")
axes[0,-1].set_ylabel(r"$S=10^6$", labelpad = 10)
axes[1,-1].set_ylabel(r"$Q=16$", labelpad = 10)

# set horizontal axis ticks and labels
xticks = list(range(qubit_min, qubit_max+1, 2))
xticklabels = [ tick if tick % 4 == 0 else "" for tick in xticks ]
for axis in axes[0,:]:
    axis.set_xticks(xticks)
    axis.set_xticklabels(xticklabels)

# get major/minor tick marks for a logarithmic axis
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

# set horizontal axis ticks
major_tick_values, minor_tick_values = get_log_ticks(axes[1,0].get_xlim())
for axis in axes[1,:]:
    axis.xaxis.set_ticks(major_tick_values)
    axis.xaxis.set_ticks(minor_tick_values, minor = True)

# set vertical axis ticks
axes[0,0].set_ylim(top = 1)
major_tick_values, minor_tick_values = get_log_ticks(axes[0,0].get_ylim())
for axis in axes[:,0]:
    axis.yaxis.set_ticks(major_tick_values)
    axis.yaxis.set_ticks(minor_tick_values, minor = True)

# place legend outside of plot
handles, labels = axes[0,0].get_legend_handles_labels()
figure.legend(handles, labels, loc = "center right", bbox_to_anchor = (1.18,0.52))

plt.tight_layout(pad = 0.2, h_pad = 0.5)
plt.savefig("infidelities.pdf", bbox_inches = "tight")
