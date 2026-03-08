                                                       subject='ptcode')
        # Plot results
        for speedi, speed in enumerate(speeds):

            # Violin plot
            sns.violinplot(ax=discvaraxs[speedi],
                           x='clustlabel',
                           y=varname,
                           data=df.loc[df['speed'] == speed],
                           palette=groupcolours,
                           legend=False)

            # Xticks
            discvaraxs[speedi].set_xticks(np.arange(len(grouplabels)), grouplabels)

            # Add stats in xlabel
            if stat_comparison['0D'][varname]['ANOVA2onerm']['p-unc'].loc[
                stat_comparison['0D'][varname]['ANOVA2onerm']['Source'] == 'clustlabel'].values < 0.05:
                statsstr = write_0Dposthoc_statstr(stat_comparison['0D'][varname]['posthocs'],
                                                   'speed * clustlabel', 'speed', speed)
                discvaraxs[speedi].set_xlabel(f'C: {statsstr}', fontsize=11)

            else:
                discvaraxs[speedi].set_xlabel(' ', fontsize=11)

            # y label
            if speedi == 0:
                discvaraxs[speedi].set_ylabel(kinematics_ylabels[varname])
            else:
                discvaraxs[speedi].set_ylabel('')

            # Add title
            discvaraxs[speedi].set_title(f'{speed} km/h')

        # Same y limits
        ylims = [ax.get_ylim() for ax in discvaraxs]
        for ax in discvaraxs:
            ax.set_ylim([min([ylim[0] for ylim in ylims]), max([ylim[1] for ylim in ylims])])

        # Set suptitle as the var name and stats
        statsstr = write_0DmixedANOVA_statstr(stat_comparison['0D'][varname]['ANOVA2onerm'],
                                              between='clustlabel',
                                              within='speed',
                                              betweenlabel='C',
                                              withinlabel='S')

        discvarfig.suptitle(f'{kinematics_titles[varname]}\n{statsstr}')

        # Save and close
        plt.tight_layout()
        discvarfig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_{varname}_ANOVA2onerm.png'), dpi=300,
                           bbox_inches='tight')
        plt.close(discvarfig)

