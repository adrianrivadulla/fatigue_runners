# -*- coding: utf-8 -*-
"""
Look at the kinematics and physiology data from the fatiguing session.
@author: arr43
"""


# TODO. Test run_0D_ANOVA2onerm in the clustering repo and make sure it works there

"""

write a printer for things on the screen like humidity, all that jazz that is needed for the papers.

see if you can separate clustering part and stats
if so, then the stats of the clustering paper and the fatigue paper are the same



"""

# TODO.


# %% Imports
import config
from utils.data_processing import prep_mastersheet, prep_phys_data, prep_kinematic_data
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
from utils.analysis import calculate_coordvar, run_SPM_ANOVA2onerm
from research_utils.statistics import demoanthrophys_analysis, anova2onerm_0d_and_posthocs
from research_utils.pipelines import run_0D_ANOVA2onerm
# from utils.temp import run_0D_ANOVA2onerm
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
physdata = prep_phys_data(config.physdatapath, wantedsignals=['smooth'], wantedpts=pts)
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

#%% Kinematics

# Get average pattern for each segment and participant, coordination variability data and design factors for stats
avgesegments, cv, designfactors = prep_kinematic_data(segments, pts, clustlabels, config.seglabels, config.couplings)

#%% Disc var analysis

figs, stat_comparison['kinematics']['0D'] = run_0D_ANOVA2onerm({discvar: avgesegments[discvar] for discvar in config.discvars},
                                                             designfactors,
                                                             config.kinematics_titles,
                                                             config.kinematics_ylabels,
                                                             uniqueclustlabels['colourcode'].tolist(),
                                                             config.seglabels,
                                                             group_names=['Neutral', 'Tilted'],
                                                             between_factor='clustlabel',
                                                             within_factor='segment',
                                                             between_label='C',
                                                             within_label='E',
                                                             within_vis=True,
                                                             rm_colours=config.segcolours,
                                                             )

for figname, fig in figs.items():
    fig.savefig(os.path.join(config.reportdir, f'{savingkw}_{figname}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

#%% 2-way ANOVA SPM for the continuous variables  TODO. YOU ARE HERE. READ PIPELINES IN THE RESEARCH-UTILS MODULE TO REFACTOR THE NEXT SECTION

figargs = {'reportdir': config.reportdir,
           'savingkw': savingkw,
           'rmffigrows': 2,
           'rmffigcols': 3,
           'rmfcolours': config.segcolours,
           'rmfylabels': config.kinematics_ylabels,
           'grcolours': uniqueclustlabels['colourcode'].tolist(),
           'vlinevar': avgesegments['DF'],
           'vartitles': config.kinematics_titles,
           'varkw': 'contvars'}

stat_comparison['kinematics'] = run_SPM_ANOVA2onerm({contvar: avgesegments[contvar] for contvar in config.contvars},
                                                designfactors,
                                                figargs,
                                                rmlabels=['start', 'mid', 'end'])


#%% 2-way ANOVA SPM for the coordination variability variables

figargs = {'reportdir': config.reportdir,
           'savingkw': savingkw,
           'rmffigrows': 1,
           'rmffigcols': 3,
           'rmfcolours': config.segcolours,
           'rmfylabels': config.coord_labels,
           'grcolours': uniqueclustlabels['colourcode'].tolist(),
           'vlinevar': avgesegments['DF'],
           'vartitles': config.coord_titles,
           'varkw': 'coordvars'}

stat_comparison['cv'] = run_SPM_ANOVA2onerm(cv,
                                        designfactors,
                                        figargs,
                                        rmlabels=['start', 'mid', 'end'])
