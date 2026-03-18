import config
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from research_utils.statistics import demoanthrophys_analysis
from research_utils.pipelines import run_0D_ANOVA2onerm
from utils.analysis import interpolate_dict
from utils.data_processing import prep_mastersheet, prep_phys_data, prep_kinematic_data
from utils.vis import visualise_gas_data


from utils.temp import run_SPM_ANOVA2onerm


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

# Get unique clustlabels and corresponding colour TODO. Can be deleted once you finish placing the colours where they need to be
# uniqclustlabels = natsort.natsorted(np.unique(clustlabels['clustlabel']))
# uniqclustcolours = [clustlabels['colourcode'].loc[
#                         clustlabels['clustlabel'] == x].iloc[0] for x in uniqclustlabels]

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

# Set figargs
figargs = {
    "reportdir": config.reportdir,
    "demoanthrophysvars_titles": config.demoanthrophysvars_titles,
    "demoanthrophysvars_ylabels": config.demoanthrophysvars_ylabels,
    "grouplabels": uniqueclustlabels["clustlabel"].tolist(),
    "groupcolours": uniqueclustlabels["colourcode"].tolist(),
    "custom_groupnames": config.clustnames,
    "savingkw": savingkw,
}

# TODO. potentially rename to compare_demoanthrophys
stat_comparison["demoanthrophys"] = demoanthrophys_analysis(master, "clustlabel", config.speeds, figargs)

# %% Kinematics

# Get average pattern for each segment and participant, coordination variability data and design factors for stats
avgesegments, cv, designfactors = prep_kinematic_data(segments, pts, clustlabels, config.seglabels, config.couplings)

# %% Disc var analysis

figs, stat_comparison["kinematics"]["0D"] = run_0D_ANOVA2onerm(
    {discvar: avgesegments[discvar] for discvar in config.discvars},
    designfactors,
    config.kinematics_titles,
    config.kinematics_ylabels,
    uniqueclustlabels["colourcode"].tolist(),
    config.seglabels,
    group_names=config.clustnames,
    between_factor="clustlabel",
    within_factor="segment",
    between_label="C",
    within_label="E",
    within_vis=True,
    rm_colours=config.segcolours,
)

for figname, fig in figs.items():
    fig.savefig(os.path.join(config.reportdir, f"{savingkw}_{figname}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# %% 2-way ANOVA SPM for the continuous variables

figargs = {
    "rmlabels": config.seglabels,
    "rmffigrows": 2,
    "rmffigcols": 3,
    "rmfcolours": config.segcolours,
    "rmfylabels": config.kinematics_ylabels,
    "grcolours": uniqueclustlabels["colourcode"].tolist(),
    "group_names": config.clustnames,
    "between_label": "C",
    "within_label": "E",
    "vlinevar": avgesegments["DF"],
    "vartitles": config.kinematics_titles,
}

stat_comparison["kinematics"], kinfigs, kinrmfig = run_SPM_ANOVA2onerm(
    {contvar: avgesegments[contvar] for contvar in config.contvars}, designfactors, figargs
)

# Save group and interaction effect figures
for var, fig in kinfigs.items():
    fig.savefig(os.path.join(config.reportdir, f"{savingkw}_{var}_ANOVA2onerm.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# Save rm effect figure
kinrmfig.savefig(os.path.join(config.reportdir, f"{savingkw}_contvars_rm_effect.png"), dpi=300, bbox_inches="tight")


# TODO. you are here. try to run the same you have above for the coordination variability part.
# TODO. If everything works, test on clustering data.
# TODO. If happy, polish the functions as much as you want and then move them to research_utils.pipelines and remove from temp
# # %% 2-way ANOVA SPM for the coordination variability variables

figargs = {
    "rmlabels": config.seglabels,
    "rmffigrows": 1,
    "rmffigcols": 3,
    "rmfcolours": config.segcolours,
    "rmfylabels": config.coord_labels,
    "grcolours": uniqueclustlabels["colourcode"].tolist(),
    "group_names": config.clustnames,
    "between_label": "C",
    "within_label": "E",
    "vlinevar": avgesegments["DF"],
    "vartitles": config.coord_titles,
}


stat_comparison["cv"], cvfigs, cvrmfig = run_SPM_ANOVA2onerm(cv, designfactors, figargs)
