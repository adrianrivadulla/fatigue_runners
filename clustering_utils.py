# -*- coding: utf-8 -*-
"""

This module contains all the functions developed for the clustering study

Created on Tue February 2023

@author: arr43
"""

# %% Imports
import os
import natsort
from itertools import combinations
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scikit_posthocs import posthoc_ttest, posthoc_dunn
import pingouin as pg
import statsmodels.api as sm
import spm1d


# %% Utils


def comparison_0D_contvar_indgroups(datadict, grouping, title_kword, figdir, colours):

    """
    Compare continuous variables between independent groups using various statistical tests.

    Parameters:
    datadict (dict): Dictionary containing the data to be compared.
    grouping (list or np.ndarray): List or array containing the group labels for each data point.
    title_kword (str): Keyword to be used in the title of the plots.
    figdir (str): Directory where the plots will be saved.
    colours (list or np.ndarray): List or array containing the colors for the groups.

    Returns:
    disc_comp (dict): A dictionary containing the results of the statistical tests.
    """

    disc_comp = {}

    for key, values in datadict.items():

        disc_comp[key] = {}

        # Check for nans
        if np.any(np.isnan(values)):
            print(f'NaNs found in {key} and they will be removed.')

        # Get variable in groups
        holder = pd.DataFrame({key: np.squeeze(values)})
        holder['grouping'] = grouping
        groups = [holder.groupby(['grouping']).get_group(x)[key].dropna() for x in
                  np.sort(holder['grouping'].dropna().unique())]

        # Run normality tests
        disc_comp[key]['normality'] = {}
        fig, axes = plt.subplots(1, len(groups))
        fig.set_size_inches([11, 3.3])

        # test trigger
        param_route = 1

        for labi, group in enumerate(groups):
            disc_comp[key]['normality'][str(labi)] = {}
            disc_comp[key]['normality'][str(labi)]['W_stat'], disc_comp[key]['normality'][str(labi)][
                'p'] = stats.shapiro(group)

            # if there were violations of normality or homoscedasticity change trigger for tests later
            if disc_comp[key]['normality'][str(labi)]['p'] <= 0.05:
                param_route = 0

            # Q-Q plots
            sm.qqplot(group, ax=axes[labi], markeredgecolor=colours[labi], markerfacecolor=colours[labi], line='r',
                      fmt='k-')
            axes[labi].get_lines()[1].set_color('black')
            axes[labi].set_xlabel('Cluster ' + str(labi))

            if disc_comp[key]['normality'][str(labi)]['p'] < 0.001:
                axes[labi].set_title(
                    'W: ' + str(np.round(disc_comp[key]['normality'][str(labi)]['W_stat'], 3)) + '; p < 0.001')
            else:
                axes[labi].set_title(
                    'W: ' + str(np.round(disc_comp[key]['normality'][str(labi)]['W_stat'], 3)) + '; p = ' + str(
                        np.round(disc_comp[key]['normality'][str(labi)]['p'], 3)))

        fig.suptitle(title_kword + '_' + key)
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, title_kword + '_' + key + '_' + 'QQplot.png'))
        plt.close(plt.gcf())

        # Parametric route
        if param_route:

            if len(groups) == 2:

                # Run heteroscedasticity tests
                disc_comp[key]['homoscedasticity'] = {}
                disc_comp[key]['homoscedasticity']['Levene_stat'], disc_comp[key]['homoscedasticity']['p'] = stats.levene(*groups)

                if disc_comp[key]['homoscedasticity']['p'] > 0.05:

                    # Independent standard t-test
                    disc_comp[key]['ttest_ind'] = {}
                    disc_comp[key]['ttest_ind']['t'], disc_comp[key]['ttest_ind']['p'] = stats.ttest_ind(*groups)

                else:

                    # Welch's t-test
                    disc_comp[key]['ttest_ind'] = {}
                    disc_comp[key]['ttest_ind']['welch_t'], disc_comp[key]['ttest_ind']['p'] = stats.ttest_ind(*groups, equal_var=False)

                # Get Cohen's d
                disc_comp[key]['ttest_ind']['Cohens_d'] = (np.mean(groups[0]) - np.mean(groups[1])) / \
                                                          np.sqrt(
                                                              (np.std(groups[0], ddof=1) ** 2 + np.std(groups[1], ddof=1) ** 2) / 2)

                # Get Hedge's g
                disc_comp[key]['ttest_ind']['Hedges_g'] = disc_comp[key]['ttest_ind']['Cohens_d'] * (
                        1 - (3 / (4 * (len(groups[0]) + len(groups[1]) - 2) - 1)))

            elif len(groups) > 2:

                # One-way ANOVA
                disc_comp[key]['ANOVA_1'] = {}
                disc_comp[key]['ANOVA_1']['F_stat'], disc_comp[key]['ANOVA_1']['p'] = stats.f_oneway(*groups)

                if disc_comp[key]['ANOVA_1']['p'] <= 0.05:
                    # Bonferroni post hoc tests
                    disc_comp[key]['Bonferroni_post_hoc'] = posthoc_ttest(groups, p_adjust='bonferroni')

        # Non-parametric route
        else:

            if len(groups) == 2:

                # Mann-Whitney U test
                disc_comp[key]['mann_whitney_U'] = {}
                disc_comp[key]['mann_whitney_U']['U_stat'], disc_comp[key]['mann_whitney_U']['p'] = stats.mannwhitneyu(
                    *groups)

            elif len(groups) > 2:

                # Kruskal
                disc_comp[key]['Kruskal'] = {}
                disc_comp[key]['Kruskal']['Hstat'], disc_comp[key]['Kruskal']['p'] = stats.kruskal(*groups)

                if disc_comp[key]['Kruskal']['p'] <= 0.05:
                    # Dunn post hoc tests
                    disc_comp[key]['Dunn_post_hoc'] = posthoc_dunn(groups, p_adjust='bonferroni')

    return disc_comp


def comparison_1D_contvar_indgroups(datadict, grouping, title_kword, figdir, colours):

    """
    Compare continuous variables between independent groups using traditional SPM1D non-parametric tests.

    Parameters:
    datadict (dict): Dictionary containing the data to be compared.
    grouping (list or np.ndarray): List or array containing the group labels for each data point.
    title_kword (str): Keyword to be used in the title of the plots.
    figdir (str): Directory where the plots will be saved.
    colours (list or np.ndarray): List or array containing the colors for the groups.

    Returns:
    cont_comp (dict): A dictionary containing the results of the statistical tests.
    """

    # Conduct traditional SPM1D non-param tests
    cont_comp = {}

    for key, values in datadict.items():

        cont_comp[key] = {}

        # Get variable in groups
        groups = [values[np.where(grouping == x)[0], :] for x in natsort.natsorted(np.unique(grouping))]

        if len(groups) == 2:

            # Non param ttest
            nonparam_ttest2 = spm1d.stats.nonparam.ttest2(groups[0], groups[1])
            cont_comp[key]['np_ttest2'] = nonparam_ttest2.inference(alpha=0.05, two_tailed=True, iterations=500)

            # Vis
            varfig = plt.figure(figsize=(10, 4))

            # Average and std patterns by group
            plt.subplot(1, 2, 1)
            for group, colour in zip(groups, colours):
                spm1d.plot.plot_mean_sd(group, linecolor=colour, facecolor=colour)
            plt.title(key)

            plt.subplot(1, 2, 2)

            cont_comp[key]['np_ttest2'].plot()
            cont_comp[key]['np_ttest2'].plot_threshold_label(fontsize=8)
            cont_comp[key]['np_ttest2'].plot_p_values()
            plt.title(f'np_ttest2 {key}')

            plt.tight_layout()
            varfig.savefig(os.path.join(figdir, f'{title_kword}_{key}_np_ttest2.png'))
            plt.close(varfig)

        elif len(groups) > 2:

            # Non parametric ANOVA
            nonparam_ANOVA = spm1d.stats.nonparam.anova1(values, grouping)
            cont_comp[key]['np_ANOVA'] = nonparam_ANOVA.inference(alpha=0.05, iterations=500)

            # Vis
            varfig = plt.figure(figsize=(10, 4))

            # Average and std patterns by group
            plt.subplot(1, 2, 1)
            for group, colour in zip(groups, colours):
                spm1d.plot.plot_mean_sd(group, linecolor=colour, facecolor=colour)
                plt.title(key)

            plt.subplot(1, 2, 2)
            cont_comp[key]['np_ANOVA'].plot()
            cont_comp[key]['np_ANOVA'].plot_threshold_label(fontsize=8)
            cont_comp[key]['np_ANOVA'].plot_p_values()
            plt.title(f'np_ANOVA {key}')
            plt.tight_layout()
            varfig.savefig(os.path.join(figdir, f'{title_kword}_{key}_np_ANOVA.png'))
            plt.close(varfig)

            if cont_comp[key]['np_ANOVA'].h0reject:

                # Adjust alpha for the number of comparisons to be performed
                ngroups = len(groups)
                alpha = 0.05 / ngroups * (ngroups - 1) / 2

                # Get unique pairwise comparisons
                paircomp = list(combinations(np.unique(grouping), 2))

                # Set number of subplots for comparison
                if len(paircomp) == 3:
                    fig, axes = plt.subplots(2, 3)
                    fig.set_size_inches(11, 6)

                elif len(paircomp) == 6:
                    fig, axes = plt.subplots(4, 3)
                    fig.set_size_inches(11, 12)

                else:
                    print('I am not ready for so many plots. Figure it out.')
                axes = axes.flat
                for pairi, pair in enumerate(paircomp):

                    # Get pair key word
                    pairkw = f'{str(pair[0])}_{str(pair[1])}'

                    # Run post-hoc analysis
                    cont_comp[key]['post_hoc_np_ttest2'] = {}
                    nonparam_ttest2 = spm1d.stats.nonparam.ttest2(groups[pair[0]], groups[pair[1]])
                    cont_comp[key]['post_hoc_np_ttest2'][pairkw] = nonparam_ttest2.inference(alpha=alpha,
                                                                                             two_tailed=True,
                                                                                             iterations=500)

                    # Vis
                    if pairi <= 2:
                        axi = pairi
                    else:
                        axi = pairi + 6

                    # NOTE THIS ASSUMES THAT THE ORDER OF THE COLOURS MATCHES THE ORDER OF THE LABELS
                    spm1d.plot.plot_mean_sd(groups[pair[0]], ax=axes[axi],
                                            linecolor=colours[pair[0]],
                                            facecolor=colours[pair[0]])
                    spm1d.plot.plot_mean_sd(groups[pair[1]], ax=axes[axi],
                                            linecolor=colours[pair[0]],
                                            facecolor=colours[pair[1]])
                    axes[pairi].set_title(str(pair))

                    pairkw = f'{str(pair[0])}_{str(pair[1])}'

                    cont_comp[key]['post_hoc_np_ttest2'][pairkw].plot(ax=axes[axi + 3])
                    cont_comp[key]['post_hoc_np_ttest2'][pairkw].plot_threshold_label(ax=axes[axi + 3], fontsize=8)
                    cont_comp[key]['post_hoc_np_ttest2'][pairkw].plot_p_values(ax=axes[axi + 3])

                fig.suptitle(f'{title_kword}_{key}')
                plt.tight_layout()
                plt.savefig(os.path.join(figdir, f'{title_kword}_{key}_posthoc.png'))
                plt.close(plt.gcf())

    return cont_comp


def anova2onerm_0d_and_posthocs(datadf, dv='', within='', between='', subject=''):

    """
    Perform a two-way ANOVA with one repeated measures factor and one between-subjects factor,
    followed by Bonferroni post-hoc tests.

    Parameters:
    datadf (pd.DataFrame): The data frame containing the data.
    dv (str): The dependent variable.
    within (str): The within-subjects factor.
    between (str): The between-subjects factor.
    subject (str): The subject identifier.

    Returns:
    statsdict (dict): A dictionary containing the ANOVA results and post-hoc test results.
    """

    statsdict = {}

    # Run ANOVA with one RM factor and one between factor
    statsdict['ANOVA2onerm'] = pg.mixed_anova(dv=dv,
                                              within=within,
                                              subject=subject,
                                              between=between,
                                              data=datadf,
                                              effsize='np2',
                                              correction=True)

    # Run Bonferroni post-hoc tests
    statsdict['posthocs'] = pg.pairwise_ttests(dv=dv,
                                               within=within,
                                               subject=subject,
                                               between=between,
                                               data=datadf,
                                               padjust='bonf',
                                               effsize='cohen')

    # Add 95% CI to posthocs
    statsdict['posthocs']['esci95_low'] = np.nan
    statsdict['posthocs']['esci95_up'] = np.nan
    for i, row in statsdict['posthocs'].iterrows():
        if row['Paired'] == True:
            ci = pg.compute_esci(row['cohen'],
                                 nx=len(datadf[subject].unique()),
                                 ny=len(datadf[subject].unique()),
                                 paired=True,
                                 eftype='cohen',
                                 confidence=0.95)
        else:
            subset =datadf.drop_duplicates(subset=[subject], keep='first')

            ci = pg.compute_esci(row['cohen'],
                                 nx=len(subset.loc[subset[between] == row['A']]),
                                 ny=len(subset.loc[subset[between] == row['B']]),
                                 paired=False,
                                 eftype='cohen',
                                 confidence=0.95)

        statsdict['posthocs']['esci95_low'].loc[i] = ci[0]
        statsdict['posthocs']['esci95_up'].loc[i] = ci[1]

    return statsdict


def write_spm_stats_str(spmobj, mode='full'):

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
    if mode not in ['full', 'stat', 'p']:
        raise ValueError('mode must be either full, stat or p')

    # Initialise statsstr
    statsstr = ''

    # Add stat value
    if mode == 'full' or mode == 'stat':
        statsstr = f'{np.round(spmobj.zstar, 2)}'

    # Add p value
    if mode == 'full' or mode == 'p':
        if len(spmobj.p) == 1:
            if spmobj.p[0] < 0.001:
                statsstr += f', p < 0.001'
            else:
                statsstr += f', p = {np.round(spmobj.p[0], 3)}'
        elif len(spmobj.p) > 1:
            statsstr += ', p = ['
            for i, p in enumerate(spmobj.p):
                if i > 0:
                    statsstr += ', '
                if p < 0.001:
                    statsstr += f'< 0.001'
                else:
                    statsstr += f'{np.round(p, 3)}'
            statsstr += ']'

    return statsstr


def add_sig_spm_cluster_patch(ax, spmobj, tscaler=1):

    """
    Add patches to a plot to indicate significant clusters from SPM (Statistical Parametric Mapping) analysis.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object to which the patches will be added.
    spmobj (object): The SPM object containing the significant clusters.
    tscaler (float, optional): A scaling factor for the time axis. Defaults to 1.
    """

    for sigcluster in spmobj.clusters:
        ylim = ax.get_ylim()
        ax.add_patch(plt.Rectangle((sigcluster.endpoints[0] * tscaler, ylim[0]),
                                   (sigcluster.endpoints[1] - sigcluster.endpoints[0]) * tscaler,
                                   ylim[1] - ylim[0], color='grey', alpha=0.5, linestyle=''))


def write_0Dposthoc_statstr(posthoctable, contrastvalue, withinfactor, withinfactorvalue):

    """
    Generate a string representation of post-hoc test statistics for a given contrast and within-factor value.

    Parameters:
    posthoctable (pd.DataFrame): The DataFrame containing the post-hoc test results.
    contrastvalue (str): The contrast value to filter the post-hoc table.
    withinfactor (str): The within-subjects factor to filter the post-hoc table.
    withinfactorvalue (str): The value of the within-subjects factor to filter the post-hoc table.

    Returns:
    str: A string representation of the post-hoc test statistics, including t-value, p-value, and Cohen's d.
    """

    t = np.round(posthoctable['T'].loc[
                     (posthoctable['Contrast'] == contrastvalue) & (
                                 posthoctable[withinfactor] == withinfactorvalue)].values[
                     0], 2)
    d = np.round(posthoctable['cohen'].loc[
                     (posthoctable['Contrast'] == contrastvalue) & (
                                 posthoctable[withinfactor] == withinfactorvalue)].values[
                     0], 2)

    ci = [posthoctable['esci95_low'].loc[
                     (posthoctable['Contrast'] == contrastvalue) & (
                                 posthoctable[withinfactor] == withinfactorvalue)].values[
                     0], posthoctable['esci95_up'].loc[
                     (posthoctable['Contrast'] == contrastvalue) & (
                                 posthoctable[withinfactor] == withinfactorvalue)].values[
                     0]]

    if posthoctable['p-corr'].loc[
        (posthoctable['Contrast'] == contrastvalue) & (posthoctable[withinfactor] == withinfactorvalue)].values[
        0] < 0.001:
        p = '< 0.001'
    else:
        p = np.round(posthoctable['p-corr'].loc[
                         (posthoctable['Contrast'] == contrastvalue) & (
                                     posthoctable[withinfactor] == withinfactorvalue)].values[
                         0], 3)

    return f't = {t}, p = {p}, d = {d}[{np.round(ci[0],2)}, {np.round(ci[1],2)}]'


def write_0DmixedANOVA_statstr(mixed_anovatable, between='', within='', betweenlabel='', withinlabel='', write_between=True, write_within=True, write_interaction=True):

    """
    Write a formatted string summarizing the results of a mixed ANOVA with one between-subjects factor and
     one within-subjects factor.

    Parameters:
    mixed_anovatable (pd.DataFrame): DataFrame containing the ANOVA results. Output of penguoin mixed_anova.
    between (str): Name of the between-subjects factor.
    within (str): Name of the within-subjects factor.
    betweenlabel (str, optional): Label for the between-subjects factor. Defaults to the value of 'between'.
    withinlabel (str, optional): Label for the within-subjects factor. Defaults to the value of 'within'.

    Returns:
    statstr (str): A formatted string summarizing the ANOVA results.
    """


    # Get factor labels or set them to factor names if not provided
    if betweenlabel == '':
        betweenlabel = between
    if withinlabel == '':
        withinlabel = within

    statstr = ''

    if write_between:
        if mixed_anovatable['p-unc'].loc[mixed_anovatable['Source'] == between].values < 0.001:
            statstr += f'{betweenlabel}: F = {np.round(mixed_anovatable["F"].values[0], 2)}, p < 0.001'
        else:
            statstr += (f'{betweenlabel}: F = {np.round(mixed_anovatable["F"].values[0], 2)}, '
                       f'p = {np.round(mixed_anovatable["p-unc"].values[0], 3)}')

    if write_within:
        if mixed_anovatable['p-unc'].loc[mixed_anovatable['Source'] == within].values < 0.001:
            statstr += f'; {withinlabel}: F = {np.round(mixed_anovatable["F"].values[1], 2)}, p < 0.001'
        else:
            statstr += (f'; {withinlabel}: F = {np.round(mixed_anovatable["F"].values[1], 2)}, '
                        f'p = {np.round(mixed_anovatable["p-unc"].values[1], 3)}')

    if write_interaction:
        if mixed_anovatable['p-unc'].loc[mixed_anovatable['Source'] == 'Interaction'].values < 0.001:
            statstr += (f'; {betweenlabel}x{withinlabel}: F = {np.round(mixed_anovatable["F"].values[2], 2)}, '
                        f'p < 0.001')
        else:
            statstr += (f'; {betweenlabel}x{withinlabel}: F = {np.round(mixed_anovatable["F"].values[2], 2)}, '
                        f'p = {np.round(mixed_anovatable["p-unc"].values[2], 2)}')

    return statstr

def demoanthrophys_analysis(datasheet, groupvarname, respeeds, figargs):
    """
    Compare demographics, anthropometrics and physiological variables between clusters.

    :return:
    """

    # Get figargs
    reportdir = figargs['reportdir']
    savingkw = figargs['savingkw']
    demoanthrophysvars_titles = figargs['demoanthrophysvars_titles']
    demoanthrophysvars_ylabels = figargs['demoanthrophysvars_ylabels']
    grlabels = figargs['grouplabels']
    grcolours = figargs['groupcolours']
    custom_groupnames = figargs['custom_groupnames']

    # Keys without RE
    noRE_keys = [key for key in demoanthrophysvars_ylabels.keys() if 'RE' not in key]

    # Demographics, anthropometrics and physiological variables ignoring EE
    demoanthrophys = comparison_0D_contvar_indgroups(
        {key: datasheet[key] for key in noRE_keys if 'Sex' not in key},
        datasheet[groupvarname].values,
        savingkw,
        reportdir,
        grcolours)

    # Make figures
    if len(noRE_keys) == 16:
        demoanthrophysfig, demoanthrophysaxs = plt.subplots(4, 4, figsize=(11, 8))

    elif len(noRE_keys) == 15:
        demoanthrophysfig, demoanthrophysaxs = plt.subplots(3, 5, figsize=(11, 6))

    else:
        print('Number of variables without RE is not 15 or 16.'
              'Figure may look a mess.'
              'Please modify the code accordingly.')
        nrows = int(np.ceil(len(noRE_keys) / 4))
        demoanthrophysfig, demoanthrophysaxs = plt.subplots(nrows, 4, figsize=(11, 3 * nrows))

    # Flatten axes
    demoanthrophysaxs = demoanthrophysaxs.flatten()

    # Go through each variable
    for vari, varname in enumerate([key for key in demoanthrophysvars_ylabels.keys() if key != 'RE']):

        if varname == 'Sex':

            fempctge = []
            sextable = []

            for gri, group in enumerate(grlabels):

                # Get pts in that cluster
                groupmaster = datasheet.loc[datasheet[groupvarname] == group]
                fempctge.append(len(groupmaster.loc[groupmaster['Sex'] == 'Female']) / len(groupmaster) * 100)

                # Get number of women and men in that cluster
                sextable.append([len(groupmaster.loc[groupmaster['Sex'] == 'Female']),
                                 len(groupmaster.loc[groupmaster['Sex'] == 'Male'])])

            # Add chi square test
            demoanthrophys[varname] = {}
            demoanthrophys[varname]['chi_test'] = {}
            demoanthrophys[varname]['chi_test']['chi_sq'], \
                demoanthrophys[varname]['chi_test']['p'], _, _ = stats.chi2_contingency(sextable)

            # Bar plot
            sns.barplot(ax=demoanthrophysaxs[vari],
                        x=grlabels,
                        y=fempctge,
                        hue=grlabels,
                        palette=grcolours,
                        legend=False)

            # Set xticks
            if custom_groupnames:
                demoanthrophysaxs[vari].set_xticks(demoanthrophysaxs[vari].get_xticks(), grlabels)

        elif varname == 'RunningDaysAWeek':

            # Count plot
            sns.countplot(ax=demoanthrophysaxs[vari],
                          x=datasheet[varname],
                          hue=datasheet[groupvarname],
                          palette=grcolours)

            # Remove legend
            demoanthrophysaxs[vari].get_legend().remove()

            # Remove xlabel
            demoanthrophysaxs[vari].set_xlabel('')

        else:

            # Violin plot
            sns.violinplot(ax=demoanthrophysaxs[vari],
                           x=datasheet[groupvarname],
                           y=datasheet[varname],
                           hue=datasheet[groupvarname],
                           palette=grcolours,
                           legend=False)

            # Xticks
            if custom_groupnames:
                demoanthrophysaxs[vari].set_xticks(demoanthrophysaxs[vari].get_xticks(), custom_groupnames)
            else:
                demoanthrophysaxs[vari].set_xticks(demoanthrophysaxs[vari].get_xticks(),
                                                   [f'C{int(x)}' for x in demoanthrophysaxs[vari].get_xticks()])

        # Yticks for Time10Ks
        if varname == 'Time10Ks' or varname == 'Sess2_times':
            # Convert to datetime and keep just mm:ss
            yticks = [str(datetime.timedelta(seconds=x)) for x in demoanthrophysaxs[vari].get_yticks()]
            yticks = [x[x.find(':') + 1:] for x in yticks]

            # Set new ticks
            demoanthrophysaxs[vari].set_yticklabels(yticks)

        # Ylabels
        demoanthrophysaxs[vari].set_ylabel(demoanthrophysvars_ylabels[varname])

        # Xlabel off
        demoanthrophysaxs[vari].set_xlabel('')

        # Title
        if varname in demoanthrophysvars_titles.keys():
            title = demoanthrophysvars_titles[varname]
        elif varname == 'Sex':
            title = 'Sex'

        if varname in demoanthrophys.keys():

            # Get key which is not normality
            stat_test = [key for key in demoanthrophys[varname].keys() if key != 'normality'][0]

            # Add asterisk to indicate significant differences
            if demoanthrophys[varname][stat_test]['p'] < 0.05:
                demoanthrophysaxs[vari].set_title(f'{title} *')
            else:
                demoanthrophysaxs[vari].set_title(title)

        else:
            demoanthrophysaxs[vari].set_title(title)

    plt.tight_layout()

    # Save and close
    demoanthrophysfig.savefig(os.path.join(reportdir, f'{savingkw}_demoantrhophys.png'), dpi=300, bbox_inches='tight')
    plt.close(demoanthrophysfig)

    # RE variables

    # Get EE data into a dataframe FIX PT AND SPEEDS
    redf = pd.DataFrame()
    redf['EE'] = np.concatenate([datasheet[f'EE{speed}kg'].values for speed in respeeds])
    redf['speed'] = np.concatenate([[int(speed)] * len(datasheet.index) for speed in respeeds])
    redf['clustlabel'] = np.tile(datasheet[groupvarname].values, len(respeeds))
    redf['ptcode'] = np.tile(datasheet.index, len(respeeds))

    # Run 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
    demoanthrophys['EE'] = anova2onerm_0d_and_posthocs(redf,
                                                       dv='EE',
                                                       within='speed',
                                                       between='clustlabel',
                                                       subject='ptcode')

    refig, reaxs = plt.subplots(1, 1, figsize=(6, 2))
    sns.violinplot(ax=reaxs, x='speed', y='EE', hue='clustlabel', data=redf, palette=grcolours)

    # Append km/h to each xtick
    reaxs.set_xticklabels([f'{speed} km/h' for speed in respeeds])
    reaxs.set_xlabel('')
    reaxs.set_ylabel(demoanthrophysvars_ylabels['RE'])
    reaxs.set_title('Running Economy')

    # Legend
    reaxs.legend(loc='lower center',
                 bbox_to_anchor=(0.5, 0),
                 ncol=2,
                 bbox_transform=refig.transFigure,
                 frameon=False)
    plt.subplots_adjust(bottom=0.25)

    # Get legend
    legend = reaxs.get_legend()

    # Change legend labels
    if custom_groupnames:
        for gri, (group, groupname) in enumerate(zip(grlabels, custom_groupnames)):
            legend.get_texts()[gri].set_text(groupname)

    # Save and close
    plt.tight_layout()
    refig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_RE.png'), dpi=300, bbox_inches='tight')
    plt.close(refig)

    return demoanthrophys