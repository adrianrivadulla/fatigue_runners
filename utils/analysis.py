import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import spm1d

# %% Functions

def calculate_coordvar(prox, dist, p=0.95):
    """
    Calculate coordination variability between two signals using method... TODO.
    """

    cv = np.ones(prox.shape[1]) * np.nan

    for t, (prox_at_t, dist_at_t) in enumerate(zip(prox.T, dist.T)):

        # Calculate covariance
        cov = np.cov(prox_at_t, dist_at_t)

        # Get eigenvals and vectors
        eigvals, eigvecs = np.linalg.eig(cov)

        k = np.sqrt(-2 * np.log(1 - p))
        scaledeig = k * np.sqrt(eigvals)
        area = np.pi * np.prod(scaledeig)

        cv[t] = area

    return cv

def interpolate_dict(data, new_length=101):
    """
    Interpolate data in a dictionary to a new length.

    Parameters:
    data (dict): Dictionary containing data for each participant and variable. [participant][variable] = array of data
    new_length (int): New length to interpolate to.

    Returns:
    dict: Dictionary containing interpolated data for each participant and variable. [variable] = array of interpolated data
    """

    datanorm = {var: np.full((len(data.keys()), new_length), np.nan) for var in data[list(data.keys())[0]]}

    for pti, pt in enumerate(data.keys()):
        for vari, var in enumerate(data[pt].keys()):
            interpolator = interp1d(np.linspace(0, 1, len(data[pt][var])), data[pt][var])
            datanorm[var][pti, :] = interpolator(np.linspace(0, 1, new_length))

    return datanorm


def run_SPM_ANOVA2onerm(datadict, designdict, figargs, rmlabels=None):

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