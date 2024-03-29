#!/usr/bin/env python3
"""
Script to plot the data plotted in the maximum likelihood fragment tomography paper.

WARNING: this script is a huge mess.
It was written for the sole purpose of making two figures for a single paper.

Author: Michael A. Perlin (github.com/perlinm)
"""
import glob
import os
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data_dir = "./data/"

num_qubit_min = 8
num_qubit_max = 20

log10_repetitions_default = 6
num_qubits_default = 18

# labels, colors, and markers for simulation data
labels = ["full", "direct", "MLFT"]
colors = ["tab:blue", "tab:orange", "tab:green"]
markers = ["o", "s", "^"]
data_info = list(zip(labels, colors, markers))

# labels, colors, and markers for analytical estimates
labels = ["full (est.)", "direct (est.)"]
colors = ["k", "tab:red"]
markers = ["+", "2"]
est_info = list(zip(labels, colors, markers))

# misc. plot parameters
params = {"font.size": 9, "text.usetex": True}
plt.rcParams.update(params)


# sum of fragment infidelities
def frag_infidelity(num_qubits: int, frag_str: str, total_repetitions: int) -> float:
    frag_num = int(frag_str[1:])  # total number of fragments

    # total number of fragment variants
    # 4*3 for first/last fragment, plus 4**2 * 3**2 for middle fragments
    variants = 24 + 144 * (frag_num - 2)
    repetitions_per_variant = total_repetitions // variants

    # classical outputs on each fragment
    output_nums = np.array([num_qubits // frag_num] * frag_num)
    for jj in range(frag_num):
        if sum(output_nums) == num_qubits:
            break
        output_nums[jj] += 1

    return sum(2**output_nums) / repetitions_per_variant


##########################################################################################
# extract info from data files

data_files = glob.glob(os.path.join(data_dir, "fidelities_*.txt"))


def info(file: str) -> Dict[str, Any]:
    file_tags = file.split("/")[-1][:-4].split("_")
    return {
        "qubits": file_tags[1],
        "frags": file_tags[2],
        "log10_reps": file_tags[3],
    }


def select_files(**circuit_info: Any) -> List[str]:
    return [file for file in data_files if all(tag in file for tag in circuit_info.values())]


def all_values(info_key: str, **selection_args: str) -> List[str]:
    return sorted(set([info(file)[info_key] for file in select_files(**selection_args)]))


##########################################################################################
# set up figure template


def get_figure_axes(frag_nums: int = 3) -> Tuple[mpl.figure.Figure, np.ndarray]:
    fig_rows = 2
    fig_cols = frag_nums
    fig_width = 1.5 * fig_cols + 0.8
    fig_height = 3.3
    return plt.subplots(fig_rows, fig_cols, sharey="row", figsize=(fig_width, fig_height))


figure_avg, axes_avg = get_figure_axes()
figure_std, axes_std = get_figure_axes()

#########################################################################################
# plot fidelity as a function of repetition (shot) number

num_qubits = num_qubits_default
circuit_info = {"qubits": f"Q{num_qubits}"}
circuit_frags = all_values("frags", **circuit_info)

for frag_idx, frag_str in enumerate(circuit_frags):
    circuit_info["frags"] = frag_str

    def get_file_reps(log10_rep_str: str) -> str:
        file_info = dict(circuit_info, **{"log10_reps": log10_rep_str})
        files = select_files(**file_info)
        assert len(files) == 1
        return files[0]

    # identify qubit numbers for which we have data
    log10_rep_strs = sorted(all_values("log10_reps", **circuit_info))
    rep_nums = np.array([10 ** float(ss[1:]) for ss in log10_rep_strs])

    # indexed by [qubit_num, circuit_instance, simulation_method]
    data = np.array([np.loadtxt(get_file_reps(log10_rep_str)) for log10_rep_str in log10_rep_strs])

    # numerical results
    for fidelity, (label, color, marker) in zip(data.T, data_info):
        infidelity_avg = 1 - np.mean(fidelity, axis=0)
        infidelity_std = np.std(fidelity, axis=0)
        plot_args = dict(color=color, label=label, fillstyle="none", markersize=6)
        axes_avg[0, frag_idx].loglog(rep_nums, infidelity_avg, marker, **plot_args)
        axes_std[0, frag_idx].loglog(rep_nums, infidelity_std, marker, **plot_args)

    # analytical estimates of infidelity
    inf_full = 2**num_qubits / (4 * rep_nums)
    inf_cuts = [frag_infidelity(num_qubits, frag_str, rep_num) for rep_num in rep_nums]
    for infidelity, (label, color, marker) in zip([inf_full, inf_cuts], est_info):
        axes_avg[0, frag_idx].semilogy(
            rep_nums, infidelity, marker, color=color, label=label, zorder=0
        )

##########################################################################################
# plot fidelity as a function of qubit number

log10_reps = log10_repetitions_default
repetitions = 10**log10_reps
circuit_info = {"log10_reps": f"S{log10_reps:.2f}"}
circuit_frags = all_values("frags", **circuit_info)

for frag_idx, frag_str in enumerate(circuit_frags):
    circuit_info["frags"] = frag_str

    def get_file_qubits(qubits_tag: str) -> str:
        file_info = dict(circuit_info, **{"qubits": qubits_tag})
        files = select_files(**file_info)
        assert len(files) == 1
        return files[0]

    # identify qubit numbers for which we have data
    qubit_strs = np.array(sorted(all_values("qubits", **circuit_info)))
    qubit_nums = np.array([int(qq[1:]) for qq in qubit_strs])

    # filter out by qubit minima / maxima
    keep = [num_qubit_min <= qubit_num <= num_qubit_max for qubit_num in qubit_nums]
    qubit_strs = qubit_strs[keep]
    qubit_nums = qubit_nums[keep]

    # indexed by [qubit_num, circuit_instance, simulation_method]
    data = np.array([np.loadtxt(get_file_qubits(qubit_str)) for qubit_str in qubit_strs])

    # numerical results
    for fidelity, (label, color, marker) in zip(data.T, data_info):
        infidelity_avg = 1 - np.mean(fidelity, axis=0)
        infidelity_std = np.std(fidelity, axis=0)
        plot_args = dict(color=color, label=label, fillstyle="none", markersize=6)
        axes_avg[1, frag_idx].semilogy(qubit_nums, infidelity_avg, marker, **plot_args)
        axes_std[1, frag_idx].semilogy(qubit_nums, infidelity_std, marker, **plot_args)

    # analytical estimates of infidelity
    inf_full = 2**qubit_nums / (4 * repetitions)
    inf_cuts = [frag_infidelity(qubit_num, frag_str, repetitions) for qubit_num in qubit_nums]
    for infidelity, (label, color, marker) in zip([inf_full, inf_cuts], est_info):
        axes_avg[1, frag_idx].semilogy(
            qubit_nums, infidelity, marker, color=color, label=label, zorder=0
        )

##########################################################################################
# miscellaneous cleanup


# get major/minor tick marks for a logarithmic axis
def get_log_ticks(axis_limits: Tuple[float, float]) -> Tuple[List[float], List[float]]:
    base, subs = 10, np.arange(0, 1.1, 0.1)
    major_locator = mpl.ticker.LogLocator(base=base)
    minor_locator = mpl.ticker.LogLocator(base=base, subs=subs)
    min_val, max_val = axis_limits

    def filter(values: List[float]) -> List[float]:
        return [val for val in values if min_val <= val <= max_val]

    major_tick_values = filter(major_locator.tick_values(min_val, max_val))
    minor_tick_values = filter(minor_locator.tick_values(min_val, max_val))
    return major_tick_values, minor_tick_values


for figure, axes, ylabel in [
    (figure_avg, axes_avg, r"$\mathcal{I}$"),
    (figure_std, axes_std, r"$\sigma(\mathcal{I})$"),
]:
    # add tick marks to top / right of axes
    for idx in np.ndindex(axes.shape):
        axes[idx].tick_params(top=True)
        axes[idx].tick_params(right=True)

    # set axis labels and titles
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    for axis in axes[0, :]:
        axis.set_xlabel("$S$")
    for axis in axes[1, :]:
        axis.set_xlabel("$Q$", labelpad=1)
    for frag_idx, frag_str in enumerate(circuit_frags):
        axes[0, frag_idx].set_title(f"$F={frag_str[1:]}$", pad=8)

    axes[0, -1].yaxis.set_label_position("right")
    axes[1, -1].yaxis.set_label_position("right")
    axes[0, -1].set_ylabel(f"$Q={num_qubits_default}$", labelpad=10)
    axes[1, -1].set_ylabel(f"$S=10^{log10_repetitions_default}$", labelpad=10)

    # set horizontal axis ticks and labels
    major_tick_values, minor_tick_values = get_log_ticks(axes[0, 0].get_xlim())
    for axis in axes[0, :]:
        axis.xaxis.set_ticks(major_tick_values)
        axis.xaxis.set_ticks(minor_tick_values, minor=True)

    xticks = list(range(num_qubit_min, num_qubit_max + 1, 2))
    xticklabels = [tick if tick % 4 == 0 else "" for tick in xticks]
    for axis in axes[1, :]:
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels)

    # set vertical axis limits and ticks
    for axis in axes[:, 0]:
        axis.set_ylim(top=1)
        major_tick_values, minor_tick_values = get_log_ticks(axis.get_ylim())
        axis.yaxis.set_ticks(major_tick_values)
        axis.yaxis.set_ticks(minor_tick_values, minor=True)

    # label individual panels
    bbox = dict(boxstyle="round", facecolor="lightgray", alpha=1)
    kwargs = dict(bbox=bbox, fontweight="bold")
    axes[0, 0].text(0.1, 0.1, "a", transform=axes[0, 0].transAxes, **kwargs, va="bottom")
    axes[0, 1].text(0.1, 0.1, "b", transform=axes[0, 1].transAxes, **kwargs, va="bottom")
    axes[0, 2].text(0.1, 0.1, "c", transform=axes[0, 2].transAxes, **kwargs, va="bottom")
    axes[1, 0].text(0.1, 0.9, "d", transform=axes[1, 0].transAxes, **kwargs, va="top")
    axes[1, 1].text(0.1, 0.9, "e", transform=axes[1, 1].transAxes, **kwargs, va="top")
    axes[1, 2].text(0.1, 0.9, "f", transform=axes[1, 2].transAxes, **kwargs, va="top")

# place legend outside of plot and save
handles, labels = axes_avg[0, 0].get_legend_handles_labels()
figure_avg.legend(handles, labels, loc="center left", bbox_to_anchor=(0.96, 0.52))
figure_avg.tight_layout(pad=0.2, h_pad=0.5)
figure_avg.savefig("infidelities_avg.pdf", bbox_inches="tight")

handles, labels = axes_std[0, 0].get_legend_handles_labels()
figure_std.legend(handles, labels, loc="center left", bbox_to_anchor=(0.96, 0.52))
figure_std.tight_layout(pad=0.2, h_pad=0.5)
figure_std.savefig("infidelities_std.pdf", bbox_inches="tight")
