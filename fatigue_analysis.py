import config
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from research_utils.pipelines import run_demoanthrophys_two_groups_comparisons, run_0D_ANOVA2onerm, run_SPM_ANOVA2onerm
from utils.analysis import interpolate_dict
from utils.data_processing import prep_mastersheet, prep_phys_data, prep_kinematic_data
from utils.vis import visualise_gas_data


# from utils.temp import run_SPM_ANOVA2onerm

# TODO. Keep adding plot args as kwargs in 0D_ANOVA whatever and SPM_ANOVA
# TODO. Test demoanthrophys and SPM_ANOVA in clustering. if happy, move SPM_ANOVA to research_utils.pipelines


# %% Default

# Saving kword
savingkw = "Fatigue"

# %% Load data

# Load biomechanical data
segments = np.load(config.datapath, allow_pickle=True).item()
pts = np.unique(segments["misc"]["pt"])

# Load mastersheet
master = prep_mastersheet(config.masterdatapath)
master = master.loc[pts]

# Load clustlabels
clustlabels = pd.read_csv(config.clustlabelspath, index_col="ptcode")
uniqueclustlabels = clustlabels.groupby("clustlabel", as_index=False)["colourcode"].first()

# Keep only pts in mastersheet that are in clustlabels
master = master.join(clustlabels, how="inner")

# Session 2 times to seconds and measure covered distance
master["Sess2_times"] = master["Sess2_time"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
speedms = (master["LT"] + 0.05 * master["LT"]) * 1000 / 3600
master["Sess2_dist"] = speedms * master["Sess2_times"]

# Load physiological data
physdata = prep_phys_data(config.physdatapath, wantedsignals=["smooth"], wantedpts=pts)
normphysdata = interpolate_dict(physdata["smooth"], new_length=101)

# Normalise VO by VO2peakkg to get relative VO and express it as a percentage of VO2peak
assert list(physdata["smooth"].keys()) == list(master.index), "Participants in physdata and master do not match"
normphysdata["VO2"] /= np.reshape(master["Mass"].values, (len(master), 1))
normphysdata["VO2"] = normphysdata["VO2"] / np.reshape(master["VO2peakkg"].values, (len(master), 1)) * 100

stat_comparison = {"demoanthrophys": {}, "kinematics": {}, "cv": {}}

# %% Gas data visualisation

gasfig = visualise_gas_data(normphysdata, config.wantedgasvars, config.gas_titles, config.gas_ylabels)
gasfig.savefig(os.path.join(config.reportdir, f"{savingkw}_gasdata_norm.png"), dpi=300, bbox_inches="tight")
plt.close(gasfig)

# %% Print extra info reported in the paper

# Print avge and std temperature and humidity for Sess2
print(f"Avge temp: {np.mean(master['Sess2_Temperature'])}C, std: {np.std(master['Sess2_Temperature'])}C")
print(f"Avge humidity: {np.mean(master['Sess2_Humidity'])}%, std: {np.std(master['Sess2_Humidity'])}%")

# Report mean and std La
print(f"Mean La: {np.nanmean(master['Sess2_La'])}")
print(f"Std La: {np.nanstd(master['Sess2_La'])}")

# Report median RPE and interquartile range
print(f"Median RPE: {np.nanmedian(master['Sess2_RPE'])}")
print(f"IQR RPE: {np.nanpercentile(master['Sess2_RPE'], 75) - np.nanpercentile(master['Sess2_RPE'], 25)}")


# %% Compare demogrpahics, anthropometrics and physiological data between clusters

req_variables = (
    [key for key in config.demoanthrophysvars_titles if key != "RE"]
    + [f"EE{speed}kg" for speed in config.speeds]
    + ["clustlabel"]
)

stat_comparison["demoanthrophys"], demoanthrophysfig, normfigs, refig = run_demoanthrophys_two_groups_comparisons(
    master[req_variables],
    grouping_var="clustlabel",
    re_speeds=config.speeds,
    titles=config.demoanthrophysvars_titles,
    ylabels=config.demoanthrophysvars_ylabels,
    group_names=config.clustnames,
    group_colours=uniqueclustlabels["colourcode"].tolist(),
)

# Save figures
demoanthrophysfig.savefig(
    os.path.join(config.reportdir, f"{savingkw}_demoanthrophys.png"), dpi=300, bbox_inches="tight"
)
plt.close(demoanthrophysfig)
for var, fig in normfigs.items():
    fig.savefig(os.path.join(config.reportdir, f"{savingkw}_{var}_QQplot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
plt.close(refig)

# %% Kinematics

# Get average pattern for each segment and participant, coordination variability data and design factors for stats
avgesegments, cv, designfactors = prep_kinematic_data(segments, pts, clustlabels, config.seglabels, config.couplings)

# Disc var analysis
figs, stat_comparison["kinematics"]["0D"] = run_0D_ANOVA2onerm(
    {discvar: avgesegments[discvar] for discvar in config.discvars},
    designfactors,
    between_factor="clustlabel",
    within_factor="segment",
    titles=config.kinematics_titles,
    ylabels=config.kinematics_ylabels,
    group_names=config.clustnames,
    group_colours=uniqueclustlabels["colourcode"].tolist(),
    rm_names=config.seglabels,
    between_label="C",
    within_label="E",
    within_vis=True,
    rm_colours=config.segcolours,
)

for figname, fig in figs.items():
    fig.savefig(os.path.join(config.reportdir, f"{savingkw}_{figname}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# %% 2-way ANOVA SPM for the continuous variables

# TODO. You are here, reformat run_SPM... to take kwargs and then try in in clustering. if all good, you're done with this repo

stat_comparison["kinematics"], kinspmfigs, kinfigs, kinrmfig = run_SPM_ANOVA2onerm(
    {contvar: avgesegments[contvar] for contvar in config.contvars},
    designfactors,
    spm_random_seed=42,
    titles=config.kinematics_titles,
    ylabels=config.kinematics_ylabels,
    group_names=config.clustnames,
    group_colours=uniqueclustlabels["colourcode"].tolist(),
    rm_names=config.seglabels,
    rm_fig_rows=2,
    rm_fig_cols=3,
    rm_colours=config.segcolours,
    between_label="C",
    within_label="E",
    vline_var=avgesegments["DF"],
)

# Save group and interaction effect figures
for var, fig in kinfigs.items():
    fig.savefig(os.path.join(config.reportdir, f"{savingkw}_{var}_ANOVA2onerm.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# Save rm effect figure
kinrmfig.savefig(os.path.join(config.reportdir, f"{savingkw}_contvars_rm_effect.png"), dpi=300, bbox_inches="tight")
plt.close(kinrmfig)

# Save spm figures
for var, fig in kinspmfigs.items():
    fig.savefig(os.path.join(config.reportdir, f"{savingkw}_{var}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# # %% 2-way ANOVA SPM for the coordination variability variables
stat_comparison["cv"], cvspmfigs, cvfigs, cvrmfig = run_SPM_ANOVA2onerm(
    cv,
    designfactors,
    spm_random_seed=42,
    titles=config.coord_titles,
    ylabels=config.coord_ylabels,
    group_names=config.clustnames,
    group_colours=uniqueclustlabels["colourcode"].tolist(),
    rm_names=config.seglabels,
    rm_fig_rows=1,
    rm_fig_cols=3,
    rm_colours=config.segcolours,
    between_label="C",
    within_label="E",
    vline_var=avgesegments["DF"],
)

for var, fig in cvfigs.items():
    fig.savefig(os.path.join(config.reportdir, f"{savingkw}_{var}_ANOVA2onerm.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

cvrmfig.savefig(os.path.join(config.reportdir, f"{savingkw}_coordvars_rm_effect.png"), dpi=300, bbox_inches="tight")
plt.close(cvrmfig)

# Save spm figures
for var, fig in cvspmfigs.items():
    fig.savefig(os.path.join(config.reportdir, f"{savingkw}_{var}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

a = 56
