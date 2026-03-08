import pandas as pd
import numpy as np
from utils.analysis import calculate_coordvar

# %% Functions

# TODO. Write proper docstrings

def prep_mastersheet(mastersheetpath, selectedidcs=None):
    """
    Loads in the master sheet and prepares it for analysis. This includes:

    Parameters:
    mastersheetpath (str): path to the master sheet excel file.

    Returns:
    master (pd.DataFrame): master sheet with VO2max renamed to VO2peakkg for simplicity.

    """

    # Load mastersheet
    master = pd.read_excel(mastersheetpath, index_col='Participant', header=1)

    # Have VO2max called VO2peakkg
    master['VO2peakkg'] = master['VO2max']

    # Get time in seconds
    master['Time10Ks'] = master['Time10K'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

    # Normalise EE values by mass to get EE per kg
    for eecol in master.filter(like='EE').columns:
        master[f'{eecol}kg'] = master[eecol] / master['Mass']

    # Filter master sheet to only include rows where filtercol is in filtervals
    if selectedidcs is not None:
        master = master.loc[selectedidcs]

    return master


def prep_phys_data(physdatapath, wantedsignals='all', wantedpts='all'):
    """

    """

    # Load physiological data
    data = np.load(physdatapath, allow_pickle=True).item()

    # if wantedsignals is a string and is 'all', set it to all keys in data. Same for wantedpts and data[key].keys()
    if isinstance(wantedsignals, str) and wantedsignals == 'all':
        wantedsignals = list(data.keys())

    if isinstance(wantedpts, str) and wantedpts == 'all':
        wantedpts = data[list(data.keys())[0]].keys()

    for key in list(data.keys()):
        if key not in wantedsignals:
            data.pop(key)
        else:
            for pt in list(data[key].keys()):
                if pt not in wantedpts:
                    data[key].pop(pt)

    return data


def prep_kinematic_data(segments, pts, clustlabels, seglabels, couplings):
    """
    Prepares kinematic data for SPM analysis by calculating the average segment values for each participant.

    Paremeters:
    segments (dict): dictionary containing the kinematic data for each participant and segment.
    pts (list): list of participant IDs to include in the analysis.
    clustlabels (pd.DataFrame): dataframe containing the cluster labels for each participant.
    seglabels (list): list of segment labels to include in the analysis.
    couplings (list of tuples): list of tuples containing the names of the proximal and distal variables to calculate coordination variability for.
     Each tuple should be in the format (proximal_variable_name, distal_variable_name).

    Returns:
    avgesegments (dict): dictionary containing the average segment values for each participant and segment.
    cv (dict): dictionary containing the coordination variability values for each participant and segment for each coupling.
    designfactors (dict): dictionary containing the design factors for each participant and segment, including participant IDs, segment labels, and cluster labels.
    """


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
            designfactors['group'][rowi] = clustlabels.loc[pt]['clustlabel']

            # add row
            rowi += 1

    return avgesegments, cv, designfactors