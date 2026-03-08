
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
