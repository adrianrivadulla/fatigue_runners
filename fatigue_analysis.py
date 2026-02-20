# -*- coding: utf-8 -*-
"""
Look at the kinematics and physiology data from the fatiguing session.
@author: arr43
"""

# TODO. Start moving functions to research_utils and utils.analysis
# TODO. All functions in red are currently in clustering utils. once you move them to research_utils,
# TODO. clustering utils can be wiped from here



# %% Imports
import config
from utils.data_processing import prep_mastersheet, prep_physdata
from utils.analysis import interpolate_dict
from utils.vis import visualise_gas_data
from itertools import combinations
import copy
import natsort
import os
import glob
import pandas as pd
import numpy as np
import math
from scipy.interpolate import CubicSpline
import scipy.io as spio
from scipy.signal import correlate, resample
import matplotlib.pyplot as plt
import openpyxl
import tkinter as tk
from tkinter import filedialog
import copy
import matplotlib
import matplotlib.pyplot as plt
import datetime
import sys
import scipy.stats as stats
import pingouin as pg
import spm1d
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
# from clustering_utils import *
from utils.analysis import calculate_coordvar
from research_utils.statistics import demoanthrophys_analysis

# %% Utils

#%% 2-way ANOVA SPM for the coordination variability variables

def SPM_ANOVA2onerm(datadict, designdict, figargs, rmlabels=None):

    reportdir = figargs['reportdir']
    savingkw = figargs['savingkw']
    rmffigrows = figargs['rmffigrows']
    rmffigcols = figargs['rmffigcols']
    rmfcolours = figargs['rmfcolours']
    ylabels = figargs['rmfylabels']
    grcolours = figargs['grcolours']
    vlinevar = figargs['vlinevar']
    vartitles = figargs['vartitles']
    varkw = figargs['varkw']

    # Labels of repeated measures factor
    if rmlabels is None:
        rmlabels = np.unique(designdict['rm'])

    stat_comparison = {}

    # Repeated measures
    rmffig, rmfaxs = plt.subplots(rmffigrows, rmffigcols, figsize=(11, 4.5))
    rmfaxs = rmfaxs.flatten()

    for vari, var in enumerate(datadict.keys()):
        stat_comparison[var] = {}

        # Initialise data holder
        # group = []
        # trialseg = []
        # subject = []
        # Y = []
        # Ydiff = []

        # Prepare data for SPM and SPM mean and std plots
        fig = plt.figure()
        fig.set_size_inches(10, 5)
        basegrid = fig.add_gridspec(2, 1)
        topgrid = basegrid[0].subgridspec(1, len(np.unique(designdict['rm'])))
        bottomgrid = basegrid[1].subgridspec(1, len(np.unique(designdict['rm'])) - 1)

        upperaxs = []
        loweraxs = []

        for rmfi, rmfactor in enumerate(rmlabels):

            # Get segment idcs
            rmfidcs = np.where(designdict['rm'] == rmfactor)[0]

            # Segment figure
            spm1d.plot.plot_mean_sd(datadict[var][rmfidcs, :],
                                    x=np.linspace(0, 100, datadict[var].shape[1]),
                                    linecolor=rmfcolours[rmfi],
                                    facecolor=rmfcolours[rmfi],
                                    ax=rmfaxs[vari])

            # Create axis in group and interaction figure
            upperaxs.append(fig.add_subplot(topgrid[0, rmfi]))

            # Plot mean and std curves
            for group in np.unique(designdict['group']):
                gridcs = np.where((designdict['group'] == group) & (designdict['rm'] == rmfactor))[0]

                spm1d.plot.plot_mean_sd(datadict[var][gridcs, :],
                                        x=np.linspace(0, 100, datadict[var].shape[1]),
                                        ax=upperaxs[rmfi],
                                        linecolor=grcolours[group],
                                        facecolor=grcolours[group])

            # Add vertical line at avge toe off for each cluster (outside loop so it doesn't mess the ylims)
            for group in np.unique(designdict['group']):
                gridcs = np.where((designdict['group'] == group) & (designdict['rm'] == rmfactor))[0]

                upperaxs[rmfi].axvline(x=np.mean(vlinevar[gridcs]) * 100,
                                       color=grcolours[group],
                                       linestyle=':')

            # Title
            upperaxs[rmfi].set_title(rmfactor)

            # xlabel
            upperaxs[rmfi].set_xlabel('Time (%)', fontsize=10)

            if rmfi > 0:

                # Plot change in variable by group
                loweraxs.append(fig.add_subplot(bottomgrid[0, rmfi - 1]))

                # Add horizontal line at 0
                loweraxs[-1].axhline(0, color='black', linestyle='-', linewidth=0.5, zorder=1)

                for group in np.unique(designdict['group']):

                    # Get indices of clust at current segment
                    gridcs = np.where((designdict['group'] == group) & (designdict['rm'] == rmfactor))[0]

                    # Get indices of clust at previous segment
                    gridcsprev = np.where((designdict['rm'] == rmlabels[rmfi - 1]) & (designdict['group'] == group))[0]

                    # Calculate difference in variable between groups
                    Ydiff = datadict[var][gridcs, :] - datadict[var][gridcsprev, :]

                    spm1d.plot.plot_mean_sd(Ydiff,
                                            x=np.linspace(0, 100, Ydiff.shape[1]),
                                            linecolor=grcolours[group], facecolor=grcolours[group],
                                            ax=loweraxs[-1])

                # Add vertical lines at avge toe off for each cluster (outside loop so it doesn't mess the ylims)
                for group in np.unique(designdict['group']):
                    gridcs = np.where((designdict['rm'] == rmfactor) & (designdict['group'] == group))[0]

                    loweraxs[-1].axvline(x=np.mean(vlinevar[gridcs]) * 100,
                                         color=grcolours[group],
                                         linestyle=':')

                # Title
                loweraxs[-1].set_title(f'{rmfactor} with respect to {rmlabels[rmfi - 1]}')

                # xlabel
                loweraxs[-1].set_xlabel('Time (%)', fontsize=10)

        # Add vertical line to segment figures at avge toe off (outside loop so it doesn't mess the ylims)
        for rmfi, rmfactor in enumerate(rmlabels):
            rmfaxs[vari].axvline(x=np.mean(vlinevar[designdict['rm'] == rmfactor]) * 100,
                                 color=rmfcolours[rmfi],
                                 linestyle=':')

        # Legend
        loweraxs[-1].legend(['_nolegend_', 'Neutral', 'Tilted'],
                            loc='lower center',
                            bbox_to_anchor=(0.5, 0),
                            ncol=2,
                            bbox_transform=fig.transFigure,
                            frameon=False)
        plt.subplots_adjust(bottom=0.11)

        # ylabels
        upperaxs[0].set_ylabel(ylabels[var])

        # Get units of ylabels if in brackets
        unitstr = ylabels[var].split('(')[1].split(')')[0]

        # Add units to ylabels of loweraxs
        loweraxs[0].set_ylabel(f'$\Delta$ ({unitstr})')

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

        # Tight layout
        plt.tight_layout()

        # Ylabel
        if vari == 0 or vari > 0 and ylabels[var] != ylabels[list(datadict.keys())[vari - 1]]:
            rmfaxs[vari].set_ylabel(ylabels[var])

        # Replace string labels in segments and pt with integers
        rmcodes = pd.Categorical(designdict['rm'], categories=rmlabels, ordered=True).codes
        ptcodes = pd.Categorical(designdict['ptids']).codes

        # Conduct SPM analysis
        spmlist = spm1d.stats.nonparam.anova2onerm(datadict[var],
                                                   designdict['group'],
                                                   rmcodes,
                                                   ptcodes)

        stat_comparison[var]['ANOVA2onerm'] = spmlist.inference(alpha=0.05, iterations=1000)

        # Post hoc tests and figures
        stat_comparison[var]['posthocs'] = {}

        # Follow up with post-hoc tests if cluster effects are found
        if stat_comparison[var]['ANOVA2onerm'][0].h0reject:

            stat_comparison[var]['posthocs']['cluster'] = {}

            # For each segment
            for rmfi, rmfactor in enumerate(rmlabels):

                stat_comparison[var]['posthocs']['cluster'][rmlabels[rmfi]] = {}

                # Get data
                Y = []
                for group in np.unique(designdict['group']):

                    # Get indices of clust at current segment
                    gridcs = np.where((designdict['rm'] == rmfactor) & (designdict['group'] == group))[0]

                    # Append data to groups
                    Y.append(datadict[var][gridcs, :])

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Y[0], Y[1])
                snpmi = snpm.inference(alpha=0.05 / len(rmlabels), two_tailed=True, iterations=1000)

                # Add snpmi to dictionary
                stat_comparison[var]['posthocs']['cluster'][rmfactor]['snpm_ttest2'] = snpmi

                # Plot
                plt.figure()
                snpmi.plot()
                snpmi.plot_threshold_label(fontsize=8)
                snpmi.plot_p_values(size=10)
                plt.gcf().suptitle(f'{var}_posthoc_{rmfactor}')

                # Save figure and close it
                plt.savefig(os.path.join(reportdir, f'{savingkw}_{var}_posthoc_{rmfactor}.png'))
                plt.close(plt.gcf())

                # Add patches to upperaxs if significant diffs are found
                if snpmi.h0reject:
                    # Scaler for sigcluster endpoints
                    tscaler = upperaxs[rmfi].get_xlim()[1] / (Y[0].shape[1] - 1)

                    # Add significant pathces to upperaxs
                    add_sig_spm_cluster_patch(upperaxs[rmfi], snpmi, tscaler=tscaler)

                # Add stats to title
                statstr = f't* = {write_spm_stats_str(snpmi, mode="full")}'
                curr_title = upperaxs[rmfi].get_title()
                upperaxs[rmfi].set_title(f'{curr_title}\n{statstr}', fontsize=10)

        # Supttitle
        # Write cluster effect string for suptitle
        statstr = f'C: F* = {write_spm_stats_str(stat_comparison[var]["ANOVA2onerm"][0], mode="full")}'

        # Write interaction effect string for suptitle
        statstr += f'; CxE: F* = {write_spm_stats_str(stat_comparison[var]["ANOVA2onerm"][2], mode="full")}'

        fig.suptitle(f'{vartitles[var]}\n{statstr}')
        fig.tight_layout()

        # Save and close
        fig.savefig(os.path.join(reportdir, f'{savingkw}_{var}_ANOVA2onerm.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        # RM factor effect
        if stat_comparison[var]['ANOVA2onerm'][1].h0reject:

            stat_comparison[var]['posthocs']['rm'] = {}

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
                stat_comparison[var]['posthocs']['rm'][f'{rmcombo[0]}_v_{rmcombo[1]}'] = {}
                stat_comparison[var]['posthocs']['rm'][f'{rmcombo[0]}_v_{rmcombo[1]}']['snpm_ttest2'] = snpmi

                # Plot
                plt.figure()
                snpmi.plot()
                snpmi.plot_threshold_label(fontsize=8)
                snpmi.plot_p_values(size=10)
                plt.gcf().suptitle(f'{var}_posthoc_{rmcombo[0]}_v_{rmcombo[1]}')

                # Save figure and close it
                plt.savefig(os.path.join(reportdir, f'{savingkw}_{var}_fatigue_posthoc_{rmcombo[0]}_v_{rmcombo[1]}.png'))
                plt.close(plt.gcf())

                # Add patches to if significant diffs are found
                if snpmi.h0reject:

                    # Get the average pattern of both segments being compared
                    Yavg = [np.mean(Y[0], axis=0), np.mean(Y[1], axis=0)]

                    # Calculate difference in variable between groups
                    delta = Yavg[0] - Yavg[1]

                    # Scaler for sigcluster endpoints
                    tscaler = rmfaxs[vari].get_xlim()[1] / (Y[0].shape[1] - 1)

                    for sigcluster in snpmi.clusters:
                        ylim = rmfaxs[vari].get_ylim()
                        rmfaxs[vari].add_patch(plt.Rectangle((sigcluster.endpoints[0] * tscaler, ylim[0]),
                                                                (sigcluster.endpoints[1] - sigcluster.endpoints[0]) * tscaler,
                                                                ylim[1] - ylim[0], color='grey', alpha=0.5,
                                                                linestyle=''))

                        # Print avge change in variable at the area of interest
                        aoidelta = np.round(np.mean(delta[int(sigcluster.endpoints[0]):int(sigcluster.endpoints[1])]), 2)
                        aoiref = np.round(np.mean(Yavg[0][int(sigcluster.endpoints[0]):int(sigcluster.endpoints[1])]), 2)
                        print(f'{var} $\Delta$ at {rmlabels[rmcombo[0]]} vs {rmlabels[rmcombo[1]]} = '
                              f'{aoidelta} CV ({int(sigcluster.endpoints[0] * tscaler)}-{int(sigcluster.endpoints[1] * tscaler)}% stride)')

        # Add title to segment figure
        statsstr = f'F* = {np.round(stat_comparison[var]["ANOVA2onerm"][1].zstar, 2)}'
        rmfaxs[vari].set_title(f'{vartitles[var]}\n{statsstr}')

        # xlabel
        rmfaxs[vari].set_xlabel('Time (%)', fontsize=10)

        # Interaction effect
        if stat_comparison[var]['ANOVA2onerm'][2].h0reject:

            stat_comparison[var]['posthocs']['interaction'] = {}

            # Calculate change in conditions
            for rmfi in range(len(rmlabels) - 1):

                # Get data
                Ydiff = []
                for group in np.unique(designdict['group']):
                    gridcs = np.where((designdict['rm'] == rmlabels[rmfi]) & (designdict['group'] == group))[0]
                    gridcsnext = np.where((designdict['rm'] == rmlabels[rmfi + 1]) & (designdict['group'] == group))[0]

                    # Append data to groups
                    Ydiff.append(datadict[var][gridcsnext, :] - datadict[var][gridcs, :])

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Ydiff[0], Ydiff[1])
                snpmi = snpm.inference(alpha=0.05 / (len(rmlabels) - 1), two_tailed=True, iterations=1000)

                # Add snpmi to dictionary
                stat_comparison[var]['posthocs']['interaction'][f'{rmlabels[rmfi + 1]}_wrt_{rmlabels[rmfi]}'] = {}
                stat_comparison[var]['posthocs']['interaction'][f'{rmlabels[rmfi + 1]}_wrt_{rmlabels[rmfi]}']['snpm_ttest2'] = snpmi

                # Plot
                plt.figure()
                snpmi.plot()
                snpmi.plot_threshold_label(fontsize=8)
                snpmi.plot_p_values(size=10)
                plt.gcf().suptitle(f'{var}_posthoc_{rmlabels[rmfi + 1]}_v_{rmlabels[rmfi]}')

                # Save figure and close it
                plt.savefig(os.path.join(reportdir, f'{savingkw}_{var}_interact_posthoc_{rmlabels[rmfi + 1]}_v_{rmlabels[rmfi]}.png'))
                plt.close(plt.gcf())

                # Add patches to loweraxs if significant diffs are found
                if snpmi.h0reject:

                    # Scaler for sigcluster endpoints
                    tscaler = loweraxs[rmfi].get_xlim()[1] / (Ydiff[0].shape[1] - 1)

                    for sigcluster in snpmi.clusters:
                        ylim = loweraxs[rmfi].get_ylim()
                        loweraxs[rmfi].add_patch(plt.Rectangle((sigcluster.endpoints[0] * tscaler, ylim[0]),
                                                                (sigcluster.endpoints[1] - sigcluster.endpoints[0]) * tscaler,
                                                                ylim[1] - ylim[0], color='grey', alpha=0.5,
                                                                linestyle=''))

                    # Add stats to xlabel
                    statstr = f't* = {write_spm_stats_str(snpmi, mode="full")}'
                    loweraxs[rmfi].set_xlabel(statstr, fontsize=10)

    # Legend
    rmffig.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    rmfaxs[-2].legend(rmlabels,
                      loc='lower center',
                      bbox_to_anchor=(0.5, 0),
                      ncol=3,
                      bbox_transform=rmffig.transFigure,
                      frameon=False)

    # Save and close
    rmffig.savefig(os.path.join(reportdir, f'{savingkw}_{varkw}_rm_effect.png'), dpi=300, bbox_inches='tight')
    plt.close(rmffig)

    return stat_comparison


# %% Default

# Saving kword
savingkw = 'Fatigue'

# %% Data arranging

# Load biomechanical data
segments = np.load(config.datapath, allow_pickle=True).item()
pts = np.unique(segments['misc']['pt'])

# Load mastersheet
master = prep_mastersheet(config.masterdatapath)
master = master.loc[pts]

# Load clustlabels
clustlabels = pd.read_csv(config.clustlabelspath, index_col='ptcode')
uniqueclustlabels = clustlabels.groupby('clustlabel', as_index=False)['colourcode'].first()

# Keep only pts in mastersheet that are in clustlabels
master = master.join(clustlabels, how='inner')

# Session 2 times to seconds and measure covered distance
master['Sess2_times'] = master['Sess2_time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
speedms = (master['LT'] + 0.05 * master['LT']) * 1000 / 3600
master['Sess2_dist'] = speedms * master['Sess2_times']

# Load physiological data
physdata = prep_physdata(config.physdatapath, wantedsignals=['smooth'], wantedpts=pts)
normphysdata = interpolate_dict(physdata['smooth'], new_length=101)

# Normalise VO by VO2peakkg to get relative VO and express it as a percentage of VO2peak
assert list(physdata['smooth'].keys()) == list(master.index), 'Participants in physdata and master do not match'
normphysdata['VO2'] /= np.reshape(master['Mass'].values, (len(master), 1))
normphysdata['VO2'] = normphysdata['VO2'] / np.reshape(master['VO2peakkg'].values, (len(master), 1)) * 100

# Speeds
speeds = [11, 12, 13]

# Get unique clustlabels and corresponding colour TODO. Can be deleted once you finish placing the colours where they need to be
# uniqclustlabels = natsort.natsorted(np.unique(clustlabels['clustlabel']))
# uniqclustcolours = [clustlabels['colourcode'].loc[
#                         clustlabels['clustlabel'] == x].iloc[0] for x in uniqclustlabels]

stat_comparison = {'demoanthrophys': {}, 'kinematics': {}, 'cv': {}}


#%% Gas data visualisation

gasfig = visualise_gas_data(normphysdata, config.wantedgasvars, config.gas_titles, config.gas_ylabels)
gasfig.savefig(os.path.join(config.reportdir, f'{savingkw}_gasdata_norm.png'), dpi=300, bbox_inches='tight')
plt.close(gasfig)


#%% Demoanthrophys comparisons

# TODO. SEE WHAT YOU DO WITH ALL THESE PRINTS
# Print avge and std temperature and humidity for Sess2
print(f'Avge temp: {np.mean(master["Sess2_Temperature"])}C, std: {np.std(master["Sess2_Temperature"])}C')
print(f'Avge humidity: {np.mean(master["Sess2_Humidity"])}%, std: {np.std(master["Sess2_Humidity"])}%')

# Report mean and std La
print(f'Mean La: {np.nanmean(master["Sess2_La"])}')
print(f'Std La: {np.nanstd(master["Sess2_La"])}')

# Report median RPE and interquartile range
print(f'Median RPE: {np.nanmedian(master["Sess2_RPE"])}')
print(f'IQR RPE: {np.nanpercentile(master["Sess2_RPE"], 75) - np.nanpercentile(master["Sess2_RPE"], 25)}')

# Get 5 and 95 percentiles of sess2_time
time5pctile = np.nanpercentile(master['Sess2_times'], 5)
time95pctile = np.nanpercentile(master['Sess2_times'], 95)

# prin them as mm:ss
print(f'5th percentile: {str(datetime.timedelta(seconds=time5pctile))}')
print(f'95th percentile: {str(datetime.timedelta(seconds=time95pctile))}')

# Set figargs
figargs = {'reportdir': config.reportdir,
           'savingkw': savingkw,
           'demoanthrophysvars_titles': config.demoanthrophysvars_titles,
           'demoanthrophysvars_ylabels': config.demoanthrophysvars_ylabels,
           'grouplabels': uniqueclustlabels['clustlabel'].tolist(),
           'groupcolours': uniqueclustlabels['colourcode'].tolist(),
           'custom_groupnames': ['Neutral', 'Tilted'],
           'savingkw': savingkw}

# TODO. potentially rename to compare_demoanthrophys
stat_comparison['demoanthrophys'] = demoanthrophys_analysis(master, 'clustlabel', speeds, figargs)

#%% Kinematics TODO. YOU ARE HERE

# Preallocate data holders
rowsn = len(pts) * len(seglabels)
designfactors = {'ptids': np.empty(rowsn, dtype=object),
                 'rm': np.empty(rowsn, dtype=object),
                 'group': np.empty(rowsn, dtype=int)}
cv = {f'{coupling[0]}__{coupling[1]}': np.ones((rowsn, segments['vars'][coupling[0]]['linreg'].shape[1])) * np.nan for coupling in couplings}
avgesegments = {}
for kinvar in segments['vars'].keys():
    if isinstance(segments['vars'][kinvar], np.ndarray):
        avgesegments[kinvar] = np.ones((rowsn)) * np.nan
    else:
        avgesegments[kinvar] = np.ones((rowsn, segments['vars'][kinvar]['linreg'].shape[1])) * np.nan

rowi = 0
for pt in pts:

    # Get indices of pt and segment
    ptstartsegidcs = np.where((segments['misc']['pt'] == pt) & (segments['misc']['segment'] == 'start'))[0]

    # Store every segment data in an easy format for SPM analysis
    for seg in seglabels:
        ptsegidcs = np.where((segments['misc']['pt'] == pt) & (segments['misc']['segment'] == seg))[0]

        for kinvar in segments['vars'].keys():
            if isinstance(segments['vars'][kinvar], np.ndarray):
                avgesegments[kinvar][rowi] = np.mean(segments['vars'][kinvar][ptsegidcs], axis=0)
            elif isinstance(segments['vars'][kinvar], dict):
                avgesegments[kinvar][rowi, :] = np.mean(segments['vars'][kinvar]['linreg'][ptsegidcs, :], axis=0)

        # Calculate coordination variability
        for coupling in couplings:
            couplingname = f'{coupling[0]}__{coupling[1]}'
            prox = segments['vars'][coupling[0]]['linreg'][ptsegidcs, :]
            dist = segments['vars'][coupling[1]]['linreg'][ptsegidcs, :]
            cv[couplingname][rowi, :] = calculate_coordvar(prox, dist)

        # Store segment label
        designfactors['rm'][rowi] = seg

        # Store pt and clust
        designfactors['ptids'][rowi] = pt
        designfactors['group'][rowi] = clustlabels[clustlabels['ptcode'] == pt]['clustlabel'].values[0]

        # add row
        rowi += 1


#%% Disc var analysis

discsegfig, discsegaxs = plt.subplots(1, 2, figsize=(11, 4))
discsegaxs = discsegaxs.flatten()

for vari, varname in enumerate(discvars):

    discvarfig, discvaraxs = plt.subplots(1, 3, figsize=(11, 3))
    discvaraxs = discvaraxs.flatten()

    stat_comparison['kinematics'][varname] = {}

    # Get data
    df = pd.DataFrame()
    df[varname] = avgesegments[varname]
    df['segment'] = designfactors['rm']
    df['clustlabel'] = designfactors['group']
    df['ptcode'] = designfactors['ptids']

    # Run 2 way ANOVA with one RM factor (segment) and one between factor (cluster)
    stat_comparison['kinematics'][varname] = anova2onerm_0d_and_posthocs(df,
                                                                         dv=varname,
                                                                         within='segment',
                                                                         between='clustlabel',
                                                                         subject='ptcode')

    # Plot results
    for segi, seg in enumerate(seglabels):

        # Violin plot
        sns.violinplot(ax=discvaraxs[segi],
                       x='clustlabel',
                       y=varname,
                       data=df.loc[df['segment'] == seg],
                       palette=uniqclustcolours,
                       legend=False)
        # Xticks
        discvaraxs[segi].set_xticks([0, 1], ['Neutral', 'Tilted'])

        # Add stats in xlabel
        if stat_comparison['kinematics'][varname]['ANOVA2onerm']['p-unc'].loc[
            stat_comparison['kinematics'][varname]['ANOVA2onerm']['Source'] == 'clustlabel'].values < 0.05:
            statsstr = write_0Dposthoc_statstr(stat_comparison['kinematics'][varname]['posthocs'],
                                               'segment * clustlabel', 'segment', seg)
            discvaraxs[segi].set_xlabel(f'C: {statsstr}', fontsize=10)

        else:
            discvaraxs[segi].set_xlabel(' ', fontsize=10)

        # y label
        if segi == 0:
            discvaraxs[segi].set_ylabel(kinematics_ylabels[varname])
        else:
            discvaraxs[segi].set_ylabel('')

        # Add title
        discvaraxs[segi].set_title(seg)

    # Same y limits
    ylims = [ax.get_ylim() for ax in discvaraxs]
    for ax in discvaraxs:
        ax.set_ylim([min([ylim[0] for ylim in ylims]), max([ylim[1] for ylim in ylims])])

    # Write stats string
    statsstr = write_0DmixedANOVA_statstr(stat_comparison['kinematics'][varname]['ANOVA2onerm'],
                                          between='clustlabel',
                                          within='segment',
                                          betweenlabel='C',
                                          withinlabel='E')

    # Remove the within stat. not very elegant but works
    statsstr_parts = statsstr.split(';')
    statsstr = ';'.join([part for part in statsstr_parts if not part.strip().startswith('E:')])

    # Set title
    discvarfig.suptitle(f'{kinematics_titles[varname]}\n{statsstr}')

    # Set ylabel
    discvaraxs[0].set_ylabel(kinematics_ylabels[varname])

    # Save and close
    plt.tight_layout()
    discvarfig.savefig(os.path.join(reportdir, f'{savingkw}_{varname}_ANOVA2onerm.png'), dpi=300, bbox_inches='tight')
    plt.close(discvarfig)

    # Create segment figure
    sns.violinplot(x='segment', y=varname, data=df, palette=segcolours, ax=discsegaxs[vari])

    # xlabeloff
    discsegaxs[vari].set_xlabel('')

    # Make ylims 20% bigger, 7% on each side
    ylim = discsegaxs[vari].get_ylim()
    discsegaxs[vari].set_ylim([ylim[0] - (ylim[1] - ylim[0]) * 0.20, ylim[1] + (ylim[1] - ylim[0]) * 0.07])

    # Annotate graph with posthoc stats if significant effect of segment is found
    if stat_comparison['kinematics'][varname]['ANOVA2onerm']['p-unc'].loc[
        stat_comparison['kinematics'][varname]['ANOVA2onerm']['Source'] == 'segment'].values < 0.05:

        # Get xticks and xticklabels
        xticks = discsegaxs[vari].get_xticks()
        # Go through all contrasts called segment
        for _, contrast in stat_comparison['kinematics'][varname]['posthocs'].loc[stat_comparison['kinematics'][varname]['posthocs']['Contrast'] == 'segment'].iterrows():

            # Get the x position as the mean of the xticks
            x = np.mean([xticks[seglabels.index(contrast['A'])], xticks[seglabels.index(contrast['B'])]])

            # Annotate stats if mid, put it at the bottom and two lines
            if contrast['A'] == 'mid' or contrast['B'] == 'mid':
                y = ylim[0]
                if contrast['p-corr'] < 0.001:
                    strstats = (f't = {np.round(contrast["T"], 2)}, p < 0.001,\n'
                                f'd = {np.round(contrast["cohen"], 2)}[{np.round(contrast["esci95_low"], 2)}, {np.round(contrast["esci95_up"], 2)}]')
                else:
                    strstats = (f't = {np.round(contrast["T"], 2)}, p = {np.round(contrast["p-corr"], 3)},\n'
                                f'd = {np.round(contrast["cohen"], 2)}[{np.round(contrast["esci95_low"], 2)}, {np.round(contrast["esci95_up"], 2)}]')

                # Annotate
                discsegaxs[vari].annotate(strstats, (x, y - y * 0.02), ha='center', va='top', fontsize=10)

            # if end and start, put it at the top in one single line
            else:
                y = ylim[1]
                if contrast['p-corr'] < 0.001:
                    strstats = f't = {np.round(contrast["T"], 2)}, p < 0.001, d = {np.round(contrast["cohen"], 2)}[{np.round(contrast["esci95_low"], 2)}, {np.round(contrast["esci95_up"], 2)}]'
                else:
                    strstats = f't = {np.round(contrast["T"], 2)}, p = {np.round(contrast["p-corr"], 3)}, d = {np.round(contrast["cohen"], 2)}[{np.round(contrast["esci95_low"], 2)}, {np.round(contrast["esci95_up"], 2)}]'

                # Annotate
                discsegaxs[vari].annotate(strstats, (x, y), ha='center', va='bottom', fontsize=10)

            # Hline to indicate the significant difference in post hoc test
            if contrast['A'] == 'mid' and contrast['B'] == 'end' or contrast['B'] == 'mid' and contrast['A'] == 'end':
                x1 = xticks[seglabels.index('mid')] + 0.05
                x2 = xticks[seglabels.index('end')] - 0.05

            elif contrast['A'] == 'mid' and contrast['B'] == 'start' or contrast['B'] == 'mid' and contrast['A'] == 'start':
                x1 = xticks[seglabels.index('mid')] - 0.05
                x2 = xticks[seglabels.index('start')] + 0.05

            else:
                x1 = xticks[seglabels.index('start')] + 0.05
                x2 = xticks[seglabels.index('end')] - 0.05

            # Draw line
            discsegaxs[vari].hlines(y, x1, x2, color='k', linewidth=0.5)

    # Set ylabel
    discsegaxs[vari].set_ylabel(kinematics_ylabels[varname])

    if stat_comparison['kinematics'][varname]['ANOVA2onerm']['p-unc'].loc[stat_comparison['kinematics'][varname]['ANOVA2onerm']['Source'] == 'segment'].values < 0.001:
        strstats = (f'F = {np.round(stat_comparison["kinematics"][varname]["ANOVA2onerm"]["F"].values[1], 2)},'
                    f' p < 0.001')
    else:
        strstats = (f'F = {np.round(stat_comparison["kinematics"][varname]["ANOVA2onerm"]["F"].values[1], 2)},'
                    f' p = {np.round(stat_comparison["kinematics"][varname]["ANOVA2onerm"]["p-unc"].values[1], 3)}')

    # Set title
    discsegaxs[vari].set_title(f'{kinematics_titles[varname]} ({strstats})')

# Save and close
plt.tight_layout()
discsegfig.savefig(os.path.join(reportdir, f'{savingkw}_discvars_fatigue_effect.png'), dpi=300, bbox_inches='tight')
plt.close(discsegfig)

#%% 2-way ANOVA SPM for the continuous variables

figargs = {'reportdir': reportdir,
           'savingkw': savingkw,
           'rmffigrows': 2,
           'rmffigcols': 3,
           'rmfcolours': segcolours,
           'rmfylabels': kinematics_ylabels,
           'grcolours': uniqclustcolours,
           'vlinevar': avgesegments['DF'],
           'vartitles': kinematics_titles,
           'varkw': 'contvars'}

stat_comparison['kinematics'] = SPM_ANOVA2onerm({contvar: avgesegments[contvar] for contvar in contvars},
                                                designfactors,
                                                figargs,
                                                rmlabels=['start', 'mid', 'end'])


#%% 2-way ANOVA SPM for the coordination variability variables

figargs = {'reportdir': reportdir,
           'savingkw': savingkw,
           'rmffigrows': 1,
           'rmffigcols': 3,
           'rmfcolours': segcolours,
           'rmfylabels': coord_labels,
           'grcolours': uniqclustcolours,
           'vlinevar': avgesegments['DF'],
           'vartitles': coord_titles,
           'varkw': 'coordvars'}

stat_comparison['cv'] = SPM_ANOVA2onerm(cv,
                                        designfactors,
                                        figargs,
                                        rmlabels=['start', 'mid', 'end'])
