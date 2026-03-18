"""
TODO.

Bring single_speed_kinematics_comparison (in clustering repo) and run_single_condition_stats (in dimred repo) together because they do pretty much the same

Pipeline_fatigue and pipeline_clustering are the same for 0D, multicondition (segments and speeds respectively) stats, so can also be merged together.
The only difference is the plotting, which can be done with if statements.

Then the second part of multispeed_kinematics_comparison is the same as run_SPM_ANOVA2onerm.
You need to rearrange the data in the clustering script to be in the same format as the fatigue script,
which is more aligned with SPM, and then you can run the same SPM code for both.

TODO. Isolate data preparation then do stats, then plotting. Most of the plotting for the paper figures is done at the start
TODO. so it can be extracted into its own function, you can have the additional figures as an optional?

"""

# %% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spm1d
from itertools import combinations
# from research_utils.statistics import write_spm_stats_str, add_sig_spm_cluster_patch

# %% Functions


def run_SPM_ANOVA2onerm(datadict, designfactors, figargs):
    """ """

    rmlabels = figargs["rmlabels"] if "rmlabels" in figargs else np.unique(designfactors["rm"])

    stat_comparison = SPM_ANOVA2onerm(datadict, designfactors, rmlabels=rmlabels)

    # TODO. Keep this here for now for debugging
    # stat_comparison = np.load(f"Fatigue_kinematics_SPM_ANOVA2onerm.npy", allow_pickle=True).item()

    figs = vis_SPM_ANOVA2onerm_between_and_x_effects(
        datadict,
        designfactors,
        stat_comparison,
        figargs,
    )

    rmfig = vis_SPM_ANOVA2onerm_within_effect(
        datadict,
        designfactors,
        stat_comparison,
        figargs,
    )

    return stat_comparison, figs, rmfig


def SPM_ANOVA2onerm(datadict, designfactors, rmlabels=None):

    # Labels of repeated measures factor
    if rmlabels is None:
        rmlabels = np.unique(designfactors["rm"])

    stat_comparison = {}

    for vari, var in enumerate(datadict.keys()):
        stat_comparison[var] = {}

        # Replace string labels in segments and pt with integers
        rmcodes = pd.Categorical(designfactors["rm"], categories=rmlabels, ordered=True).codes
        ptcodes = pd.Categorical(designfactors["ptids"]).codes

        # Conduct SPM analysis
        spmlist = spm1d.stats.nonparam.anova2onerm(datadict[var], designfactors["group"], rmcodes, ptcodes)

        stat_comparison[var]["ANOVA2onerm"] = spmlist.inference(alpha=0.05, iterations=1000)

        # Post hoc tests and figures
        stat_comparison[var]["posthocs"] = {}

        # Follow up with post-hoc tests if group effects are found
        if stat_comparison[var]["ANOVA2onerm"][0].h0reject:
            stat_comparison[var]["posthocs"]["group"] = {}

            # For each repeated measure
            for rmfi, rmfactor in enumerate(rmlabels):
                stat_comparison[var]["posthocs"]["group"][rmlabels[rmfi]] = {}

                # Get data
                Y = []
                for group in np.unique(designfactors["group"]):
                    # Get indices of clust at current measure
                    gridcs = np.where((designfactors["rm"] == rmfactor) & (designfactors["group"] == group))[0]

                    # Append data to groups
                    Y.append(datadict[var][gridcs, :])

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Y[0], Y[1])
                snpmi = snpm.inference(alpha=0.05 / len(rmlabels), two_tailed=True, iterations=1000)

                # Add snpmi to dictionary
                stat_comparison[var]["posthocs"]["group"][rmfactor]["snpm_ttest2"] = snpmi

        # RM factor effect
        if stat_comparison[var]["ANOVA2onerm"][1].h0reject:
            stat_comparison[var]["posthocs"]["rm"] = {}

            # Get all possible combinations of segments
            rmcombos = list(combinations(range(len(rmlabels)), 2))

            # Calculate change in conditions
            for rmcombo in rmcombos:
                # Get data
                Y = []
                for rmf in rmcombo:
                    # Get indices of clust at current segment
                    rmfidcs = np.where(rmcodes == rmf)[0]

                    # Append data to Y
                    Y.append(datadict[var][rmfidcs, :])

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Y[0], Y[1])
                snpmi = snpm.inference(alpha=0.05 / len(rmcombos), two_tailed=True, iterations=1000)

                # Add snpmi to dictionary
                stat_comparison[var]["posthocs"]["rm"][f"{rmcombo[0]}_v_{rmcombo[1]}"] = {}
                stat_comparison[var]["posthocs"]["rm"][f"{rmcombo[0]}_v_{rmcombo[1]}"]["snpm_ttest2"] = snpmi

        # Interaction effect
        if stat_comparison[var]["ANOVA2onerm"][2].h0reject:
            stat_comparison[var]["posthocs"]["interaction"] = {}

            # Calculate change in conditions
            for rmfi in range(len(rmlabels) - 1):
                # Get data
                Ydiff = []
                for group in np.unique(designfactors["group"]):
                    gridcs = np.where((designfactors["rm"] == rmlabels[rmfi]) & (designfactors["group"] == group))[0]
                    gridcsnext = np.where(
                        (designfactors["rm"] == rmlabels[rmfi + 1]) & (designfactors["group"] == group)
                    )[0]

                    # Append data to groups
                    Ydiff.append(datadict[var][gridcsnext, :] - datadict[var][gridcs, :])

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Ydiff[0], Ydiff[1])
                snpmi = snpm.inference(alpha=0.05 / (len(rmlabels) - 1), two_tailed=True, iterations=1000)

                # Add snpmi to dictionary
                stat_comparison[var]["posthocs"]["interaction"][f"{rmlabels[rmfi + 1]}_wrt_{rmlabels[rmfi]}"] = {}
                stat_comparison[var]["posthocs"]["interaction"][f"{rmlabels[rmfi + 1]}_wrt_{rmlabels[rmfi]}"][
                    "snpm_ttest2"
                ] = snpmi

    return stat_comparison


# def vis_SPM_ANOVA2onerm(datadict, designfactors, statsdict, figargs, rmlabels=None):
#
#     # TODO. You're here. finish isolating all the plotting and let's see what happens
#
#     rmffigrows = figargs['rmffigrows']
#     rmffigcols = figargs['rmffigcols']
#     rmfcolours = figargs['rmfcolours']
#     ylabels = figargs['rmfylabels']
#     grnames = figargs['group_names']
#     grcolours = figargs['grcolours']
#     vlinevar = figargs['vlinevar']
#
#
#     # Labels of repeated measures factor
#     if rmlabels is None:
#         rmlabels = np.unique(designfactors['rm'])
#
#     # Repeated measures
#     rmffig, rmfaxs = plt.subplots(rmffigrows, rmffigcols, figsize=(11, 4.5))
#     rmfaxs = rmfaxs.flatten()
#
#     for vari, var in enumerate(datadict.keys()):
#
#         # Prepare data for SPM and SPM mean and std plots
#         fig = plt.figure()
#         fig.set_size_inches(10, 5)
#         basegrid = fig.add_gridspec(2, 1)
#         topgrid = basegrid[0].subgridspec(1, len(np.unique(designfactors['rm'])))
#         bottomgrid = basegrid[1].subgridspec(1, len(np.unique(designfactors['rm'])) - 1)
#
#         upperaxs = []
#         loweraxs = []
#
#         for rmfi, rmfactor in enumerate(rmlabels):
#
#             # Repeated measures figure
#             rmfidcs = np.where(designfactors['rm'] == rmfactor)[0]
#             spm1d.plot.plot_mean_sd(datadict[var][rmfidcs, :],
#                                     x=np.linspace(0, 100, datadict[var].shape[1]),
#                                     linecolor=rmfcolours[rmfi],
#                                     facecolor=rmfcolours[rmfi],
#                                     ax=rmfaxs[vari])
#
#             # Create axis in group and interaction figure TODO. SEE IF YOU CAN EXTRACT THIS OUTSIDE HERE SO ITS CLEARER
#             upperaxs.append(fig.add_subplot(topgrid[0, rmfi]))
#
#             # Plot mean and std curves
#             for group in np.unique(designfactors['group']):
#                 gridcs = np.where((designfactors['group'] == group) & (designfactors['rm'] == rmfactor))[0]
#
#                 spm1d.plot.plot_mean_sd(datadict[var][gridcs, :],
#                                         x=np.linspace(0, 100, datadict[var].shape[1]),
#                                         ax=upperaxs[rmfi],
#                                         linecolor=grcolours[group],
#                                         facecolor=grcolours[group])
#
#             # Add vertical line at avge toe off for each cluster (outside loop so it doesn't mess the ylims)
#             for group in np.unique(designfactors['group']):
#                 gridcs = np.where((designfactors['group'] == group) & (designfactors['rm'] == rmfactor))[0]
#
#                 upperaxs[rmfi].axvline(x=np.mean(vlinevar[gridcs]) * 100,
#                                        color=grcolours[group],
#                                        linestyle=':')
#
#             # Title
#             upperaxs[rmfi].set_title(rmfactor)
#
#             # xlabel
#             upperaxs[rmfi].set_xlabel('Time (%)', fontsize=10)
#
#             if rmfi > 0:
#
#                 # Plot change in variable by group
#                 loweraxs.append(fig.add_subplot(bottomgrid[0, rmfi - 1]))
#
#                 # Add horizontal line at 0
#                 loweraxs[-1].axhline(0, color='black', linestyle='-', linewidth=0.5, zorder=1)
#
#                 for group in np.unique(designfactors['group']):
#                     # Get indices of clust at current segment
#                     gridcs = np.where((designfactors['group'] == group) & (designfactors['rm'] == rmfactor))[0]
#
#                     # Get indices of clust at previous repeated measure
#                     gridcsprev = np.where((designfactors['rm'] == rmlabels[rmfi - 1]) & (designfactors['group'] == group))[0]
#
#                     # Calculate difference in variable between groups
#                     Ydiff = datadict[var][gridcs, :] - datadict[var][gridcsprev, :]
#
#                     spm1d.plot.plot_mean_sd(Ydiff,
#                                             x=np.linspace(0, 100, Ydiff.shape[1]),
#                                             linecolor=grcolours[group], facecolor=grcolours[group],
#                                             ax=loweraxs[-1])
#
#                 # Add vertical lines at avge toe off for each cluster (outside loop so it doesn't mess the ylims)
#                 for group in np.unique(designfactors['group']):
#                     gridcs = np.where((designfactors['rm'] == rmfactor) & (designfactors['group'] == group))[0]
#
#                     loweraxs[-1].axvline(x=np.mean(vlinevar[gridcs]) * 100,
#                                          color=grcolours[group],
#                                          linestyle=':')
#
#                 # Title
#                 loweraxs[-1].set_title(f'{rmfactor} with respect to {rmlabels[rmfi - 1]}')
#
#                 # xlabel
#                 loweraxs[-1].set_xlabel('Time (%)', fontsize=10)
#
#         # Add vertical line to rm figures at avge toe off (outside loop so it doesn't mess the ylims)
#         for rmfi, rmfactor in enumerate(rmlabels):
#             rmfaxs[vari].axvline(x=np.mean(vlinevar[designfactors['rm'] == rmfactor]) * 100,
#                                  color=rmfcolours[rmfi],
#                                  linestyle=':')
#
#         # Legend
#         # Add '_nolegend_' to the start of the legend labels to avoid duplicate legends from multiple subplots
#         loweraxs[-1].legend(['_nolegend_'] + grnames,
#                             loc='lower center',
#                             bbox_to_anchor=(0.5, 0),
#                             ncol=2,
#                             bbox_transform=fig.transFigure,
#                             frameon=False)
#         plt.subplots_adjust(bottom=0.11)
#
#         # ylabels
#         upperaxs[0].set_ylabel(ylabels[var])
#
#         # Get units and add them to ylabels of loweraxs
#         unitstr = ylabels[var].split('(')[1].split(')')[0]
#         loweraxs[0].set_ylabel(f'$\Delta$ ({unitstr})')
#
#         # Get ylims for all loweraxs
#         ylims = [ax.get_ylim() for ax in upperaxs]
#
#         # Set ylims for all loweraxs
#         for ax in upperaxs:
#             ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])
#
#         # Get ylims for all loweraxs
#         ylims = [ax.get_ylim() for ax in loweraxs]
#
#         # Set ylims for all loweraxs
#         for ax in loweraxs:
#             ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])
#
#         # Tight layout
#         plt.tight_layout()
#
#         # Ylabel
#         rmfaxs[vari].set_ylabel(ylabels[var])
#
#     return fig, rmffig


def vis_SPM_ANOVA2onerm_between_and_x_effects(datadict, designfactors, stat_comparison, figargs):

    rmlabels = figargs["rmlabels"] if "rmlabels" in figargs else np.unique(designfactors["rm"])
    ylabels = figargs["rmfylabels"]
    grnames = figargs["group_names"]
    grcolours = figargs["grcolours"]
    vlinevar = figargs["vlinevar"]
    vartitles = figargs["vartitles"]
    between_label = figargs["between_label"] if "between_label" in figargs else "B"
    within_label = figargs["within_label"] if "within_label" in figargs else "W"

    figs = {}

    for vari, var in enumerate(datadict.keys()):
        # Prepare data for SPM mean and std plots
        figs[var] = plt.figure()
        figs[var].set_size_inches(10, 5)
        basegrid = figs[var].add_gridspec(2, 1)
        topgrid = basegrid[0].subgridspec(1, len(np.unique(designfactors["rm"])))
        bottomgrid = basegrid[1].subgridspec(1, len(np.unique(designfactors["rm"])) - 1)

        upperaxs = []
        loweraxs = []

        # Extract relevant stats from stat_comparison for current variable
        for rmfi, rmfactor in enumerate(rmlabels):
            # Create axis in group and interaction figure
            upperaxs.append(figs[var].add_subplot(topgrid[0, rmfi]))

            # Plot mean and std curves
            for group in np.unique(designfactors["group"]):
                idcs = np.where((designfactors["group"] == group) & (designfactors["rm"] == rmfactor))[0]

                spm1d.plot.plot_mean_sd(
                    datadict[var][idcs, :],
                    x=np.linspace(0, 100, datadict[var].shape[1]),
                    ax=upperaxs[rmfi],
                    linecolor=grcolours[group],
                    facecolor=grcolours[group],
                )

            # Add vertical line at avge toe off for each cluster (outside loop so it doesn't mess the ylims)
            for group in np.unique(designfactors["group"]):
                idcs = np.where((designfactors["group"] == group) & (designfactors["rm"] == rmfactor))[0]

                upperaxs[rmfi].axvline(x=np.mean(vlinevar[idcs]) * 100, color=grcolours[group], linestyle=":")

            # xlabel
            upperaxs[rmfi].set_xlabel("Time (%)", fontsize=10)

            # Title
            upperaxs[rmfi].set_title(rmfactor)

            # Add patches to upperaxs if significant diffs were found
            if stat_comparison[var]["ANOVA2onerm"][0].h0reject:
                between_posthocs = stat_comparison[var]["posthocs"]["group"]
                if between_posthocs[rmfactor]["snpm_ttest2"].h0reject:
                    # Scaler for sigcluster endpoints
                    tscaler = upperaxs[rmfi].get_xlim()[1] / (datadict[var].shape[1] - 1)

                    # Add significant pathces to upperaxs
                    add_sig_spm_cluster_patch(
                        upperaxs[rmfi], between_posthocs[rmfactor]["snpm_ttest2"], tscaler=tscaler
                    )

                    # Add stats to title
                    curr_title = upperaxs[rmfi].get_title()
                    statstr = f"t* = {write_spm_stats_str(between_posthocs[rmfactor]['snpm_ttest2'], mode='full')}"
                    upperaxs[rmfi].set_title(f"{curr_title}\n{statstr}", fontsize=10)

            if rmfi > 0:
                # Plot change in variable by group
                loweraxs.append(figs[var].add_subplot(bottomgrid[0, rmfi - 1]))

                # Add horizontal line at 0
                loweraxs[-1].axhline(0, color="black", linestyle="-", linewidth=0.5, zorder=1)

                for group in np.unique(designfactors["group"]):
                    # Get indices of group at current measure
                    idcs = np.where((designfactors["group"] == group) & (designfactors["rm"] == rmfactor))[0]

                    # Get indices of group at previous repeated measure
                    idcsprev = np.where(
                        (designfactors["rm"] == rmlabels[rmfi - 1]) & (designfactors["group"] == group)
                    )[0]

                    # Calculate difference in variable between groups
                    Ydiff = datadict[var][idcs, :] - datadict[var][idcsprev, :]

                    spm1d.plot.plot_mean_sd(
                        Ydiff,
                        x=np.linspace(0, 100, Ydiff.shape[1]),
                        linecolor=grcolours[group],
                        facecolor=grcolours[group],
                        ax=loweraxs[-1],
                    )

                # Add vertical lines at avge toe off for each cluster (outside loop so it doesn't mess the ylims)
                for group in np.unique(designfactors["group"]):
                    gridcs = np.where((designfactors["rm"] == rmfactor) & (designfactors["group"] == group))[0]

                    loweraxs[-1].axvline(x=np.mean(vlinevar[gridcs]) * 100, color=grcolours[group], linestyle=":")

                # Title
                loweraxs[-1].set_title(f"{rmfactor} with respect to {rmlabels[rmfi - 1]}")

                # xlabel
                loweraxs[-1].set_xlabel("Time (%)", fontsize=10)

                # Add patches to loweraxs if significant diffs are found
                if stat_comparison[var]["ANOVA2onerm"][2].h0reject:
                    interaction_posthocs = stat_comparison[var]["posthocs"]["interaction"]
                    if interaction_posthocs[f"{rmlabels[rmfi + 1]}_wrt_{rmlabels[rmfi]}"]["snpm_ttest2"].h0reject:
                        # Scaler for sigcluster endpoints
                        tscaler = loweraxs[rmfi].get_xlim()[1] / (Ydiff[0].shape[1] - 1)

                        # Add significant pathces to upperaxs
                        add_sig_spm_cluster_patch(
                            loweraxs[rmfi],
                            interaction_posthocs[f"{rmlabels[rmfi + 1]}_wrt_{rmlabels[rmfi]}"]["snpm_ttest2"],
                            tscaler=tscaler,
                        )

                        # Add stats to xlabel
                        statstr = f"t* = {write_spm_stats_str(interaction_posthocs[f'{rmlabels[rmfi + 1]}_wrt_{rmlabels[rmfi]}']['snpm_ttest2'], mode='full')}"
                        loweraxs[rmfi].set_xlabel(statstr, fontsize=10)

        # Legend
        loweraxs[-1].legend(
            ["_nolegend_"] + grnames,
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=2,
            bbox_transform=figs[var].transFigure,
            frameon=False,
        )
        plt.subplots_adjust(bottom=0.11)

        # ylabels
        upperaxs[0].set_ylabel(ylabels[var])

        # Get units and add them to ylabels of loweraxs
        unitstr = ylabels[var].split("(")[1].split(")")[0]
        loweraxs[0].set_ylabel(f"$\Delta$ ({unitstr})")

        # Get ylims for all loweraxs
        ylims = [ax.get_ylim() for ax in upperaxs]

        # Set ylims for all loweraxs
        for ax in upperaxs:
            ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])

        # Get ylims for all loweraxs
        ylims = [ax.get_ylim() for ax in loweraxs]

        # Set ylims for all loweraxs
        for ax in loweraxs:
            ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])

        # Supttitle
        # Write between effect string for suptitle
        statstr = f"{between_label}: F* = {write_spm_stats_str(stat_comparison[var]['ANOVA2onerm'][0], mode='full')}"

        # Write interaction effect string for suptitle
        statstr += f"; {between_label}x{within_label}: F* = {write_spm_stats_str(stat_comparison[var]['ANOVA2onerm'][2], mode='full')}"

        figs[var].suptitle(f"{vartitles[var]}\n{statstr}")
        figs[var].tight_layout()

    return figs


def vis_SPM_ANOVA2onerm_within_effect(datadict, designfactors, stat_comparison, figargs):
    """ """
    rmlabels = figargs["rmlabels"] if "rmlabels" in figargs else np.unique(designfactors["rm"])
    rmffigrows = figargs["rmffigrows"]
    rmffigcols = figargs["rmffigcols"]
    rmfcolours = figargs["rmfcolours"]
    ylabels = figargs["rmfylabels"]
    vartitles = figargs["vartitles"]
    vlinevar = figargs["vlinevar"]

    # Repeated measures
    rmffig, rmfaxs = plt.subplots(rmffigrows, rmffigcols, figsize=(11, 4.5))
    rmfaxs = rmfaxs.flatten()

    for vari, var in enumerate(datadict.keys()):
        for rmfi, rmfactor in enumerate(rmlabels):
            # Repeated measures figure
            rmfidcs = np.where(designfactors["rm"] == rmfactor)[0]
            spm1d.plot.plot_mean_sd(
                datadict[var][rmfidcs, :],
                x=np.linspace(0, 100, datadict[var].shape[1]),
                linecolor=rmfcolours[rmfi],
                facecolor=rmfcolours[rmfi],
                ax=rmfaxs[vari],
            )

            # x and y labels
            rmfaxs[vari].set_xlabel("Time (%)", fontsize=10)
            rmfaxs[vari].set_ylabel(ylabels[var])

        # Add vertical line to rm figures at avge toe off (outside loop so it doesn't mess the ylims)
        for rmfi, rmfactor in enumerate(rmlabels):
            rmfaxs[vari].axvline(
                x=np.mean(vlinevar[designfactors["rm"] == rmfactor]) * 100, color=rmfcolours[rmfi], linestyle=":"
            )

        # Add title to with within ANOVA effect in the title
        statsstr = f"F* = {np.round(stat_comparison[var]['ANOVA2onerm'][1].zstar, 2)}"
        rmfaxs[vari].set_title(f"{vartitles[var]}\n{statsstr}")

        # Add patches to if significant diffs are found
        for comparison in stat_comparison[var]["posthocs"]["rm"].values():
            if comparison["snpm_ttest2"].h0reject:
                # Scaler for sigcluster endpoints
                tscaler = rmfaxs[vari].get_xlim()[1] / (datadict[var].shape[1] - 1)

                for sigcluster in comparison["snpm_ttest2"].clusters:
                    ylim = rmfaxs[vari].get_ylim()
                    rmfaxs[vari].add_patch(
                        plt.Rectangle(
                            (sigcluster.endpoints[0] * tscaler, ylim[0]),
                            (sigcluster.endpoints[1] - sigcluster.endpoints[0]) * tscaler,
                            ylim[1] - ylim[0],
                            color="grey",
                            alpha=0.5,
                            linestyle="",
                        )
                    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    rmfaxs[-2].legend(
        rmlabels, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, bbox_transform=rmffig.transFigure, frameon=False
    )

    return rmffig


def add_sig_spm_cluster_patch(ax, spmobj, tscaler=1):
    """
    Add patches to a plot to indicate significant clusters from SPM (Statistical Parametric Mapping) analysis.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object to which the patches will be added.
    spmobj (object): The SPM object containing the significant clusters.
    tscaler (float, optional): A scaling factor for the time axis. Defaults to 1.
    """

    for sigcluster in spmobj.clusters:
        ax.axvspan(
            sigcluster.endpoints[0] * tscaler, sigcluster.endpoints[1] * tscaler, color="grey", alpha=0.5, linestyle=""
        )


def write_spm_stats_str(spmobj, mode="full"):
    """
    Generate a string representation of SPM (Statistical Parametric Mapping) statistics.

    Parameters:
    spmobj (object): The SPM object containing the statistical results.
    mode (str, optional): The mode of the output string. Must be one of 'full', 'stat', or 'p'. Defaults to 'full'.

    Returns:
    str: A string representation of the SPM statistics.

    Raises:
    ValueError: If the mode is not one of 'full', 'stat', or 'p'.
    """

    # Make sure mode is full, stat or p
    if mode not in ["full", "stat", "p"]:
        raise ValueError("mode must be either full, stat or p")

    # Initialise statsstr
    statsstr = ""

    # Add stat value
    if mode == "full" or mode == "stat":
        statsstr = f"{np.round(spmobj.zstar, 2)}"

    # Add p value
    if mode == "full" or mode == "p":
        if len(spmobj.p) == 1:
            if spmobj.p[0] < 0.001:
                statsstr += ", p < 0.001"
            else:
                statsstr += f", p = {np.round(spmobj.p[0], 3)}"
        elif len(spmobj.p) > 1:
            statsstr += ", p = ["
            for i, p in enumerate(spmobj.p):
                if i > 0:
                    statsstr += ", "
                if p < 0.001:
                    statsstr += "< 0.001"
                else:
                    statsstr += f"{np.round(p, 3)}"
            statsstr += "]"

    return statsstr
