import pandas as pd
import numpy as np

# %% Functions


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

def prep_physdata(physdatapath, wantedsignals='all', wantedpts='all'):
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
