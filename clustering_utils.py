# -*- coding: utf-8 -*-
"""

This module contains all the functions developed for the clustering study

Created on Tue February 2023

@author: arr43
"""

# %% Imports
import os
import warnings
import natsort
from itertools import combinations
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
import sys
projectdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(projectdir, 'additional_modules'))
from gapstat import gapstat
import tkinter as tk
from yellowbrick.cluster import KElbowVisualizer
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, TransformerMixin
import mplcursors
from matplotlib import colors
from scipy import stats
# import statsmodels.api as sm
from scikit_posthocs import posthoc_ttest, posthoc_dunn
import pingouin as pg
import statsmodels.api as sm
import spm1d
# import opensim as osim #TODO create opensim_utils module
import matplotlib.colors as mcolors


# %% Utils



def pca_dimensionality_reduction(datadict, vartracker, stage, figargs, pca_varexpthresh=0.99):

    """
    Perform PCA dimensionality reduction on the given data and assess reconstruction quality.

    Parameters:
    datadict (dict): Dictionary containing the data to be reduced.
    vartracker (list): List of variable names corresponding to the columns in the data.
    stage (str): The stage of the analysis ('multispeed' or 'single speed').
    figargs (dict): Dictionary containing figure information for plotting.
    pca_varexpthresh (float, optional): The variance explained threshold for PCA. Defaults to 0.99.

    Returns:
    pcaed (np.ndarray): The PCA-transformed data.
    dr_scores (pd.DataFrame): DataFrame containing the number of components, MSE, and RMSE for each stage.
    """

    # Get info from figargs
    reportdir = figargs['reportdir']
    savingkw = figargs['savingkw']
    colour = figargs['colour']
    acceptable_errors = figargs['acceptable_errors']
    kinematics_titles = figargs['kinematics_titles']
    short_ylabels = figargs['short_ylabels']
    recbot_ylims = figargs['recbot_ylims']

    # Get wantedvars as the keys of datadict
    wantedvars = list(datadict.keys())

    # Concatenate data
    X = np.hstack([datadict[key] for key in datadict.keys()])

    # Standardise data
    scaler = CustomScaler()
    Xz = scaler.fit_transform(X, vartracker=vartracker)

    # Apply PCA
    pca = PCA(n_components=pca_varexpthresh)
    pcaed = pca.fit_transform(Xz)

    if stage == 'multispeed':

        # Get all the unique strings in vartracker containing wantedvars[0]
        uniqspeedexts = np.unique([f'_{var.split("_")[-1]}' for var in vartracker if wantedvars[0] in var])
        titlekw = 'Multispeed'
        savingkw = 'Multispeed'
        colours = ['C0', 'C6', 'C3']

    else:
        uniqspeedexts = ['']
        titlekw = 'Single speed'
        savingkw = 'Single_speed'

    #%% Reconstruction quality
    yhat = pca.inverse_transform(pcaed)
    mse = metrics.mean_squared_error(Xz, yhat)
    rmse = metrics.mean_squared_error(Xz, yhat, squared=False)

    # Get row wise mse
    ptmse = metrics.mean_squared_error(Xz.T, yhat.T, multioutput='raw_values')

    # Sorted ptmse
    ptmsesorted = np.sort(ptmse)

    # Get index of pt with median mse to be the representative pt
    medianptidx = np.where(ptmse == ptmsesorted[len(ptmsesorted) // 2])[0][0]

    # Back scale the predicted data
    yhat_original = scaler.inverse_transform(yhat)

    # Get errors by variable
    recerrors = yhat_original - X
    recmeanerrors = np.mean(recerrors, axis=0)
    rec25pctileerrors = np.quantile(recerrors, 0.025, axis=0)
    rec975pctileerrors = np.quantile(recerrors, 0.975, axis=0)

    # For each speed extension

    dr_scores = pd.DataFrame(columns=[f'n_components{str(int(pca_varexpthresh*100))}', 'mse', 'rmse'])

    for exti, ext in enumerate(uniqspeedexts):

        # Create grid figure for reconstruction quality assessment
        recfig, recaxs, gridshape = make_splitgrid(2, int(len(wantedvars) / 2), figsize=(11, 5.75))

        # Plot ground truth curve data in top axs
        for vari, varname in enumerate(wantedvars):

            # Get varidx
            varidx = np.where(np.array(vartracker) == f'{varname}{ext}')[0]

            if len(varidx) == 1:

                # Plot ground truth
                recaxs['topaxs'][vari].plot(X[medianptidx, varidx], '-o', color='k')

                # Plot reconstructed with error
                recaxs['topaxs'][vari].plot(yhat_original[medianptidx, varidx], '-o', color=colour[exti])

                # Plot  as a point with errorbar
                recaxs['bottomaxs'][vari].vlines(x=1, ymin=rec25pctileerrors[varidx],
                                                 ymax=rec975pctileerrors[varidx], color=colour[exti])
                recaxs['bottomaxs'][vari].plot(1, recmeanerrors[varidx], 'o', color=colour[exti])


            else:

                # Plot ground truth
                recaxs['topaxs'][vari].plot(np.linspace(0, 100, len(X[medianptidx, varidx])),
                                            X[medianptidx, varidx], color='k')

                # Plot reconstructed with error
                recaxs['topaxs'][vari].plot(np.linspace(0, 100, len(yhat_original[medianptidx, varidx])),
                                            yhat_original[medianptidx, varidx],
                                            color=colour[exti])

                # Plot 95% of the errors
                recaxs['bottomaxs'][vari].plot(np.linspace(0, 100, len(recmeanerrors[varidx])), recmeanerrors[varidx],
                                               color=colour[exti])
                recaxs['bottomaxs'][vari].fill_between(np.linspace(0, 100, len(recmeanerrors[varidx])),
                                                       rec25pctileerrors[varidx], rec975pctileerrors[varidx],
                                                       alpha=0.5, color=colour[exti], edgecolor='none')

                # Set xlims
                recaxs['topaxs'][vari].set_xlim([0, 100])
                recaxs['bottomaxs'][vari].set_xlim([0, 100])

            recaxs['topaxs'][vari].set_title(kinematics_titles[varname])
            recaxs['topaxs'][vari].set_ylabel(short_ylabels[vari])
            recaxs['topaxs'][vari].set_xticks([])

        # Decorate recfig

        # Get ylims of stride frequency and duty factor
        ylims = recaxs['topaxs'][0].get_ylim()
        recaxs['topaxs'][0].set_ylim(ylims[0] - 0.05 * ylims[1], ylims[1] + 0.05 * ylims[1])
        ylims = recaxs['topaxs'][1].get_ylim()
        recaxs['topaxs'][1].set_ylim(ylims[0] - 0.05 * ylims[1], ylims[1] + 0.05 * ylims[1])

        # Round current yticks in duty factor top axs figure
        recaxs['topaxs'][1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        recaxs['bottomaxs'][0].set_xticks([1], ['Direct PCA 99'])
        recaxs['bottomaxs'][0].set_xlim([0.5, 1.5])
        recaxs['bottomaxs'][1].set_xticks([1], ['Direct PCA 99'])
        recaxs['bottomaxs'][1].set_xlim([0.5, 1.5])

        for vari, (var, ax) in enumerate(zip(acceptable_errors.keys(), recaxs['bottomaxs'])):

            # Add horizontal lines to indicate acceptable errors
            ax.hlines(y=acceptable_errors[var],
                      xmin=ax.get_xlim()[0],
                      xmax=ax.get_xlim()[1],
                      color='k', linestyle=':')
            ax.hlines(y=-acceptable_errors[var],
                      xmin=ax.get_xlim()[0],
                      xmax=ax.get_xlim()[1],
                      color='k', linestyle=':')

            # Make top spine visible
            ax.spines['top'].set_visible(True)

            # Set ylims
            ax.set_ylim(recbot_ylims[vari])

            # Set yticks
            ax.set_yticks([-acceptable_errors[var], acceptable_errors[var]])

        # Set ylabels
        recaxs['bottomaxs'][0].set_ylabel('Error')
        recaxs['bottomaxs'][4].set_ylabel('Error')

        # Add legend
        recaxs['topaxs'][-1].legend(['Ground Truth', 'Direct PCA 99'],
                                    loc='lower center',
                                    bbox_to_anchor=(0.5, 0),
                                    ncol=5,
                                    bbox_transform=recfig.transFigure,
                                    frameon=False)
        plt.subplots_adjust(bottom=0.11)

        # title and saving
        if ext == '':
            recfig.suptitle(f'{titlekw} PCA {str(9 + int(stage[-2:]))} km/h')
            figpath = os.path.join(reportdir, f'{savingkw}_{stage}_recquality.png')

        else:
            recfig.suptitle(f'{titlekw} PCA {ext[1:]} km/h')
            figpath = os.path.join(reportdir, f'{savingkw}_{stage}{ext}_recquality.png')

        # Save figure
        recfig.savefig(figpath, dpi=300, bbox_inches='tight')
        print(f'Saving reconstruction analysis figure to {figpath}')

        # Close figure
        plt.close(recfig)

        # Dimensionality reduction scores
        dr_scores.loc[f'{stage}{ext}'] = {f'n_components{str(int(pca_varexpthresh*100))}': pcaed.shape[1],
                                          'mse': mse,
                                          'rmse': rmse}

    return pcaed, dr_scores


class HierarchClusteringAnalysisTool:

    """
    A GUI to choose the number of clusters for hierarchical clustering based on internal validity scores and visualisation.
    """

    def __init__(self, data, **kwargs):

        """
        Initialize the HierarchClusteringAnalysisTool.

        Parameters:
        data (pd.DataFrame or np.ndarray): The data to be clustered.
        kwargs (dict): Additional keyword arguments for customization.
        """

        self.data = data
        self.kwargs = kwargs

        # kwargs
        figtitle = self.kwargs.get('figtitle', 'Hierarchical Clustering Analysis')
        datalabels = self.kwargs.get('labels', None)

        # Instantiate model
        hrcal_model = AgglomerativeClustering()

        # Choose number of clusters GUI
        # Create GUI
        master = tk.Tk()

        # Get the screen width and height
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()

        # Set the window dimensions to cover the full screen
        master.geometry(f"{int(screen_width)}x{int(screen_height)}")
        master.configure(bg='white')
        master.title('Choose number of clusters')
        master.attributes("-topmost", True)
        master.focus_force()

        # Figure with scores to decide n of clusters
        fig_width = min(screen_width/100*0.5, 5)  # Cap the width at 10 inches
        fig_height = min(screen_height/100*0.9, 6)  # Cap the height at 6 inches
        self.scorefig = plt.figure(figsize=(fig_width, fig_height))

        # Silhouette scores
        ax = plt.subplot(4, 2, 1)
        visualiser = KElbowVisualizer(hrcal_model, k=(2, 11), metric='silhouette', timings=False, locate_elbow=False)
        visualiser.fit(data)
        plt.title('')
        plt.ylabel('Silhouette')
        plt.xlabel('')
        plt.grid()
        ylimits = plt.ylim()
        plt.ylim([0.9 * ylimits[0], 1.1 * ylimits[1]])
        plt.text(0.98, 0.98, 'optimum: 1', ha='right', va='top', transform=ax.transAxes)
        ax.spines[['top', 'right']].set_visible(False)

        # Store scores
        self.scores = pd.DataFrame(visualiser.k_scores_, index=visualiser.k_values_, columns=['Silhouette'])

        # Calinski_harabasz
        ax = plt.subplot(4, 2, 3)
        visualiser = KElbowVisualizer(hrcal_model, k=(2, 11), metric='calinski_harabasz', timings=False, locate_elbow=False)
        visualiser.fit(data)
        plt.title('')
        plt.ylabel('Calinski-Harabasz \nIndex')
        ylimits = plt.ylim()
        plt.ylim([0.9 * ylimits[0], 1.1 * ylimits[1]])
        plt.xlabel('')
        plt.grid()
        plt.text(0.98, 0.98, 'optimum: largest', ha='right', va='top', transform=ax.transAxes)
        ax.spines[['top', 'right']].set_visible(False)

        # Store scores
        self.scores['Calinski-Harabasz'] = visualiser.k_scores_

        # Davies_Bouldin
        ax = plt.subplot(4, 2, 5)
        scores = [metrics.davies_bouldin_score(data, AgglomerativeClustering(n_clusters=k).fit_predict(data)) for k
                  in range(2, 11)]
        plt.plot(range(2, 11), scores, linestyle='-', marker='D', color='b')
        plt.title('')
        plt.xlabel('K')
        plt.ylabel('Davies-Bouldin \nIndex')
        plt.grid()
        ylimits = plt.ylim()
        plt.ylim([0.9 * ylimits[0], 1.1 * ylimits[1]])
        plt.text(0.98, 0.98, 'optimum: 0', ha='right', va='top', transform=ax.transAxes)
        ax.spines[['top', 'right']].set_visible(False)

        # Store scores
        self.scores['Davies-Bouldin'] = scores

        # Gap statistic
        _, _, gapstats = gapstat(data, hrcal_model, max_k=10, calcStats=True)
        scores = gapstats['data'][:, list(gapstats['columns']).index('Gap')]
        ax = plt.subplot(4, 2, 7)
        plt.plot(gapstats['index'][1:-1], scores[1:-1], linestyle='-', marker='D', color='b')
        plt.ylabel('Gap')
        ylimits = plt.ylim()
        plt.ylim([0.9 * ylimits[0], 1.1 * ylimits[1]])
        plt.grid()
        plt.text(0.98, 0.98, 'optimum: largest', ha='right', va='top', transform=ax.transAxes)
        plt.title('Internal validity scores')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Store scores
        self.scores['Gap'] = scores[1:-1]

        # Silhouette plots
        plotis = [2, 4, 6, 8]
        ks = [2, 3, 4, 5]

        self.Silhouette_samples = {}

        for ki, ploti in zip(ks, plotis):

            ax = plt.subplot(4, 2, ploti)
            if ploti == 2:
                ax.set_title('Silhouette analysis')
            # Set top and right spines invisible
            ax.spines[['top', 'right']].set_visible(False)

            templabels = AgglomerativeClustering(n_clusters=ki).fit_predict(data)

            # Get avge and sample Silhouette scores
            silh_avge = metrics.silhouette_score(data, templabels)
            silh_samp = metrics.silhouette_samples(data, templabels)

            # Store silhouette scores
            self.Silhouette_samples[ki] = silh_samp

            y_lower = 10

            for i in range(ki):

                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = silh_samp[templabels == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = 'C' + str(i + 1)
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers in the middle
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            # Make it pretty
            if ax.get_xlim()[0] > -0.1:
                ax.set_xlim(-0.1, ax.get_xlim()[1])
            ax.set_ylabel("Label")

            # The vertical line for average silhouette score of all the values
            ax.axvline(x=silh_avge, color="red", linestyle="--")

        # X label silhouette score columns
        ax.set_xlabel("Silhouette scores")
        self.scorefig.suptitle(figtitle)
        plt.tight_layout()

        # Embed figures in tkinter
        # Internal validity scores and Silhouette analysis
        lefttopframe = tk.Frame(master)
        lefttopframe.grid(row=0, column=0)
        canvas = FigureCanvasTkAgg(self.scorefig, master=lefttopframe)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # String with instructions
        # number of cluster choice
        leftbottomframe = tk.Frame(master)
        leftbottomframe.grid(row=1, column=0)
        instructs = ('Set number of clusters in the drop menu on the right. '
                     'Silhouette, Calinski-Harabasz, Davies-Bouldin considered in the paper. '
                     'Priority given to Silhouette. '
                     'See paper for more details.')
        string = tk.Label(leftbottomframe, textvariable=tk.StringVar(leftbottomframe, instructs))
        string.grid(row=0, column=0)

        # Blank dendrogram
        righttopframe = tk.Frame()
        righttopframe.grid(row=0, column=1)
        self.dendrofig = plt.figure(figsize=(fig_width, fig_height))
        clustdataholder = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data)
        _, linkage = plot_dendrogram(clustdataholder,
                                     datalabels,
                                     color_threshold=0,
                                     orientation='left')
        plt.title('Dendrogram')

        canvas2 = FigureCanvasTkAgg(self.dendrofig, master=righttopframe)
        canvas2.draw()
        canvas2.get_tk_widget().pack()

        # number of cluster choice
        midrightframe = tk.Frame(master)
        midrightframe.grid(row=1, column=1)
        string = tk.Label(midrightframe, textvariable=tk.StringVar(midrightframe, 'N clusters:'))
        string.grid(row=0, column=0)
        n_cluster_choice = tk.StringVar(midrightframe)
        menu = tk.OptionMenu(midrightframe, n_cluster_choice, *range(2, 11))
        menu.grid(row=0, column=1)

        # ok button
        bottomrightframe = tk.Frame(master)
        bottomrightframe.grid(row=2, column=1)

        # when accept is clicked
        def accept():

            """
            Accept the selected number of clusters and apply hierarchical clustering
            with the selected number of clusters.
            """

            # Store selected number of clusters
            n_clusters = n_cluster_choice.get()
            self.n_clusters = int(n_clusters)

            # Apply hierarchical clustering with final choice of  n_clusters
            self.dendrofig = plt.figure(figsize=(6.5, 1.5))
            self.clustlabels, self.dendro, self.finalscore_table, self.linkmat, _ = hierarch_clust(self.data, self.n_clusters, self.kwargs['labels'])
            self.dendroax = plt.gca()

            # Get participant labels and colours from dendrogram
            self.colourid = pd.DataFrame(
                {'datalabels': self.dendro['ivl'], 'colourcode': self.dendro['leaves_color_list']}).sort_values(
                by=['datalabels'], ignore_index=True)

            # Get unique colours and create a clustlabel variable for every colour
            self.colours = np.sort(self.colourid['colourcode'].unique())
            self.colourid['clustlabel'] = 0
            for label, colourcode in enumerate(self.colours):
                self.colourid.loc[self.colourid['colourcode'] == colourcode, 'clustlabel'] = int(label)

            # Close the window
            master.quit()
            master.destroy()

        # Create Accept button
        acceptbutton = tk.Button(bottomrightframe, text='Accept', command=accept)
        acceptbutton.grid(row=0, column=0)
        acceptbutton.focus_force()

        master.mainloop()


def hierarch_clust(X, n_clusters, datalabels):

    """
    Perform hierarchical clustering on the given data and plot the dendrogram.

    Parameters:
    X (pd.DataFrame or np.ndarray): The data to be clustered.
    n_clusters (int): The number of clusters to form.
    datalabels (list or np.ndarray): The labels for the data points.

    Returns:
    labels (np.ndarray): Cluster labels for each point.
    dendro (dict): Dendrogram data.
    scores (pd.DataFrame): DataFrame containing silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
    linkage_matrix (np.ndarray): Linkage matrix for the dendrogram.
    branch_height (float): The height of the branches in the dendrogram.
    """

    # Fit model
    hrcal_model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hrcal_model.fit_predict(X)

    # Calculate scores
    scorenames = ['Silhouette', 'Calinski_Harabasz', 'Davies_Bouldin']
    scorelist = [metrics.silhouette_score(X, labels),
                 metrics.calinski_harabasz_score(X, labels),
                 metrics.davies_bouldin_score(X, labels)]
    scores = pd.DataFrame(data=scorelist, index=scorenames)

    # Plot dendrogram
    clustdataholder = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X)
    dendro, linkage_matrix = plot_dendrogram(clustdataholder,
                                             datalabels=datalabels,
                                             color_threshold=0)
    n_dendroclusters = 0
    shrink_factor = 0.05
    branch_height = np.max(linkage_matrix[:, 2])

    while n_dendroclusters != n_clusters:
        branch_height -= np.max(linkage_matrix[:, 2]) * shrink_factor
        clustdataholder = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X)
        dendro, linkage_matrix = plot_dendrogram(clustdataholder,
                                                 datalabels=datalabels,
                                                 color_threshold=branch_height)

        # Due to some weird behaviour in the dendrogram function possibly related to different versions of scipy
        try:
            n_dendroclusters = len(np.unique(dendro['leaves_color_list']))
        except:
            n_dendroclusters = len(np.unique(dendro['color_list']))

        # Keep readjusting shrinking factor until found
        if n_dendroclusters > n_clusters:
            shrink_factor *= 0.2
            branch_height = np.max(linkage_matrix[:, 2]) - np.max(linkage_matrix[:, 2]) *\
                            shrink_factor

    return labels, dendro, scores, linkage_matrix, branch_height


def plot_dendrogram(model, datalabels, color_threshold=None, orientation='top'):

    """
    Plot dendrogram from hierarchical clustering model.

    Parameters:
    model (AgglomerativeClustering): The hierarchical clustering model.
    datalabels (list or np.ndarray): The labels for the data points.
    color_threshold (float, optional): The threshold to apply for coloring clusters.
    orientation (str, optional): The orientation of the dendrogram ('top', 'bottom', 'left', 'right').

    Returns:
    dendrofig (dict): Dendrogram data.
    linkage_matrix (np.ndarray): Linkage matrix for the dendrogram.
    """

    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrofig = dendrogram(linkage_matrix,
                           labels=datalabels,
                           color_threshold=color_threshold,
                           orientation=orientation)

    return dendrofig, linkage_matrix


def append_bottom_leaves_dendrogram(dendroax, labelcolour=[]):

    """
    Append bottom leaves to a dendrogram axis and color them according to labelcolour.

    Parameters:
    dendroax (matplotlib.axes.Axes): The dendrogram axis to modify.
    labelcolour (pd.DataFrame, optional): DataFrame containing 'colourcode' and 'ptcode' for coloring the leaves. Defaults to an empty list.
    """

    ylim = dendroax.get_ylim()
    ylimlen = ylim[1] - ylim[0]

    # Colour bottom leaves according to colour in previous stage
    for xtick, ticklabel in zip(dendroax.get_xticks(), dendroax.get_xticklabels()):
        if not isinstance(labelcolour, pd.DataFrame):
            plt.vlines(xtick, 0 - 0.08 * ylimlen, 0 - 0.01 * ylimlen, color='k')
        else:
            plt.vlines(xtick, 0 - 0.08 * ylimlen, 0 - 0.01 * ylimlen, color=
                       labelcolour['colourcode'].loc[labelcolour['ptcode'] == ticklabel.get_text()].values[0])

    dendroax.set_ylim(ylim[0] - 0.09 * ylimlen, ylim[1])
    dendroax.set_axis_off()
    plt.draw()


def add_dendro_legend(dendroax, colourid):

    """
    Add a legend to the dendrogram axis with the correct color codes and cluster counts.

    Parameters:
    dendroax (matplotlib.axes.Axes): The dendrogram axis to modify.
    colourid (pd.DataFrame): DataFrame containing 'colourcode' and 'ptcode' for coloring the leaves.
    """

    # Unique colourcodes for temporary legend
    # Get unique colourcodes
    templegend = list(colourid['colourcode'].unique())
    templegend.insert(0, '_nolegend')
    legend = dendroax.legend(templegend, frameon=False)

    # Get handles and their colours
    handlecolours = [handle.get_color() for handle in legend.legendHandles]

    # Get labels which should be the same as the CN colourcode in matplotlib
    leglabels = [label.get_text() for label in legend.get_texts()]
    leglabelcolours = [mcolors.to_rgba(leglabel) for leglabel in leglabels]

    # Get indices of leglabelcolours in handlecolours
    idcs = []
    for handlecolour in handlecolours:
        for leglabi, leglabelcolour in enumerate(leglabelcolours):
            if np.all(handlecolour == leglabelcolour):
                idcs.append(leglabi)
                break

    # Get the colourcode again and order them correctly
    finalcolours = list(colourid['colourcode'].unique())

    orderedcolours = [finalcolours[idx] for idx in idcs]

    # Get count of pts in each cluster
    clustcount = [len(colourid.loc[colourid['colourcode'] == colourlabel]) for colourlabel in orderedcolours]

    # Subtract 1 from the digit in finalcolours
    orderedcolours = [orderedcolours[:-1] + str(int(orderedcolours[-1]) - 1) for orderedcolours in orderedcolours]

    # Add count of pts in each cluster
    orderedcolours = [f'{orderedcolour} ({clustcounti})' for orderedcolour, clustcounti in
                      zip(orderedcolours, clustcount)]

    # Set the legend correctly now
    orderedcolours.insert(0, '_nolegend')
    dendroax.legend(orderedcolours, frameon=False)

    plt.tight_layout()


def tsne_plot(X, perplexities, colours, **kwargs):

    """
    Create t-SNE plots for the given data with different perplexities.

    Parameters:
    X (np.ndarray): The data to be transformed and plotted.
    perplexities (list): A list of perplexity values for t-SNE.
    colours (list or np.ndarray): The colors for the data points.
    kwargs (dict): Additional keyword arguments for customization.
        - ringcolours (list or np.ndarray, optional): The colors for the edges of the data points. Defaults to the value of 'colours'.
        - title (str, optional): The title of the plot. Defaults to 't-SNE'.
        - labels (list, optional): The labels for the data points. Defaults to 'No labels added.' repeated for each data point.

    Returns:
    matplotlib.figure.Figure: The figure object containing the t-SNE plots.
    """


    # Get kwargs
    ringcolours = kwargs.get('ringcolours', colours)
    title = kwargs.get('title', 't-SNE')
    labels = kwargs.get('labels', 'No labels added.'*len(X))

    # Create figure
    fig, axs = plt.subplots(1, len(perplexities), figsize=(12.38, 3.52))
    axs = axs.flat

    for plti, p in enumerate(perplexities):

        # TSNE data with perplexity p
        tsne = TSNE(perplexity=p, )
        tsned = tsne.fit_transform(X)

        # Plot data points
        axs[plti].scatter(tsned[:, 0], tsned[:, 1], c=colours, edgecolors=ringcolours, linewidths=2)

        # Make it pretty
        axs[plti].spines['top'].set_visible(False)
        axs[plti].spines['right'].set_visible(False)
        axs[plti].set_xlabel('Emb dim 1(.)')
        axs[plti].set_title(f'p = {p}')

        # Use mplcursors to display labels on hover
        try:
            mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))
        except:
            print('mplcursors not available. Plot will not display labels on hover.')

    fig.suptitle(title)
    axs[0].set_ylabel('Emb dim 2(.)')
    fig.suptitle(title)
    fig.tight_layout()

    return fig

def pca_expvar_plot(pca, threhsolds, colours=['r'], threshlabels=[], highlighted=[], title='PCA explained variance'):

    """
    Visualize the explained variance of PCA and plot thresholds.

    Parameters:
    pca (PCA): The PCA object containing the explained variance ratio.
    threhsolds (list): List of thresholds for explained variance.
    colours (list, optional): List of colors for the thresholds. Defaults to ['r'].
    threshlabels (list, optional): List of labels for the thresholds. Defaults to an empty list.
    highlighted (list, optional): List of thresholds to be highlighted. Defaults to an empty list.
    title (str, optional): The title of the plot. Defaults to 'PCA explained variance'.

    Returns:
    matplotlib.figure.Figure: The figure object containing the explained variance plot.
    """

    # Visualise explained variance
    fig = plt.figure(figsize=[5.5, 4])

    # Plot thresholds first
    if len(threhsolds) != len(colours):
        print('Number of colours and thresholds do not match. Using red for all thresholds.')
        colours = ['r'] * len(threhsolds)

    if threshlabels == []:
        threshlabels = [str(int(thresh * 100)) + '%' for thresh in threhsolds]

    pcns = []
    for threshold, colour, threshlabel in zip(threhsolds, colours, threshlabels):
        pcns.append(np.where(np.cumsum(pca.explained_variance_ratio_) >= threshold)[0][0] + 1)
        plt.axhline(y=threshold, color=colour, linestyle=':', zorder=4)

        if threshold in highlighted:
            plt.vlines(x=pcns[-1], ymin=0, ymax=threshold, label=threshlabel,  color=colour, linestyle='-', linewidth=3, zorder=1)
        else:
            plt.vlines(x=pcns[-1], ymin=0, ymax=threshold, label=threshlabel,  color=colour, linestyle='-', zorder=1)

    # Plot explained variance
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5,
            align='center', label='Individual', zorder=3)
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid',
             label='Cumulative', color='k', zorder=3)

    # Set ylim
    plt.ylim([0, 1.03])

    # Set yticks
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # Scale to 100%
    plt.yticks(yticks, [int(ytick * 100) for ytick in yticks])
    plt.ylabel('Explained variance (%)')
    plt.xticks(pcns)
    plt.xlabel('PC index')
    plt.title(title)
    plt.legend(loc='best', frameon=False)

    plt.tight_layout()

    return fig


def corrmat_plot(array, figsize=(5, 5)):

    """
    Generate and plot a correlation matrix using a heatmap.

    Parameters:
    array (np.ndarray): The input data array for which the correlation matrix is to be computed.
    figsize (tuple, optional): The size of the figure to be created. Defaults to (5, 5).

    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the heatmap.
    ax (matplotlib.axes.Axes): The axes object containing the heatmap.
    """

    # Generate a correlation matrix
    corrmat = np.corrcoef(array, rowvar=False)

    # Plot the correlation matrix using a heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corrmat, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_title("Correlation Matrix")

    return fig, ax


class TransitionAnalysis:


    """
    A class to analyse transitions between two clustering partitions. It allows to recolour the dendrogram based on the
    transitions between the two partitions.
    """

    def __init__(self, prev_colourid, curr_colourid, dendrofig, dendroax):

        """
        Initialize the TransitionAnalysis class.

        Parameters:
        prev_colourid (pd.DataFrame): DataFrame containing the previous color IDs and cluster labels.
        curr_colourid (pd.DataFrame): DataFrame containing the current color IDs and cluster labels.
        dendrofig (matplotlib.figure.Figure): The dendrogram figure.
        dendroax (matplotlib.axes.Axes): The dendrogram axis.
        """

        self.prev_colourid = prev_colourid
        self.curr_colourid = curr_colourid
        self.dendrofig = dendrofig
        self.dendroax = dendroax

        # Get previous colour-label convention
        self.prevcolourlabels = self.prev_colourid[['colourcode', 'clustlabel']].drop_duplicates()

        # Get current colour-label convention
        self.currentcolourlabels = self.curr_colourid[['colourcode', 'clustlabel']].drop_duplicates()

        # Get participants that are in both speeds
        self.reppts = list(set(self.curr_colourid['ptcode']).intersection(set(self.prev_colourid['ptcode'])))

        # Get transitions
        self.jointdf, self.unique_transitions = self.get_transitions()

        # Transition GUI
        transGUI = TransitionAnalysisGUI(self.unique_transitions, self.dendrofig)

        # Get pts with each current label
        currcolourpts = []
        for lab in transGUI.recolour.keys():
            currcolourpts.append(self.curr_colourid['ptcode'].loc[self.curr_colourid['clustlabel'] == int(lab)].to_list())

        # Replace labels and colours
        for colourptsidcs, (oldlab, newlab) in zip(currcolourpts, transGUI.recolour.items()):
            self.curr_colourid['clustlabel'].loc[self.curr_colourid['ptcode'].isin(colourptsidcs)] = int(newlab)

            # Replace colour Cn colour code using old colours and then new colours as exception
            try:
                newcolour = self.prevcolourlabels['colourcode'].loc[self.prevcolourlabels['clustlabel'] == int(newlab)].values[0]
            except:
                newcolour = \
                self.currentcolourlabels['colourcode'].loc[self.currentcolourlabels['clustlabel'] == int(newlab)].values[0]

            self.curr_colourid['colourcode'].loc[self.curr_colourid['ptcode'].isin(colourptsidcs)] = newcolour

            # The n-1 first children in the axes are going to be the clusters
            self.dendroax._children[int(oldlab) + 1].set_color(newcolour)

        # Create updated transitions
        self.updatedjointdf, self.updatedtransitions = self.get_transitions()

        # Calculate AMI scores
        self.ami = metrics.adjusted_mutual_info_score(self.updatedjointdf['prev_clustlabel'], self.updatedjointdf['curr_clustlabel'])

    def get_transitions(self):
        """
         Get the transitions between the previous and current clustering partitions.

         Returns:
         jointdf (pd.DataFrame): DataFrame containing the joint data of previous and current partitions.
         unique_transitions (pd.DataFrame): DataFrame containing the unique transitions and their counts.
         """

        previous = self.prev_colourid.loc[self.prev_colourid['ptcode'].isin(self.reppts)]
        current = self.curr_colourid.loc[self.curr_colourid['ptcode'].isin(self.reppts)]

        # Make ptcode the index
        previous = previous.set_index('ptcode')
        current = current.set_index('ptcode')

        # Add prefix to columns
        previous = previous.add_prefix('prev_')
        current = current.add_prefix('curr_')

        # Merge them on ptcode
        jointdf = pd.merge(previous, current, on='ptcode')

        # Count unique transitions
        unique_transitions = jointdf.value_counts().reset_index(name='count')

        # Relative counts
        unique_transitions['rel_count_prev'] = 0
        unique_transitions['rel_count_post'] = 0
        for i, row in unique_transitions.iterrows():
            unique_transitions['rel_count_prev'].iloc[i] = row['count'] / np.sum(
                unique_transitions['count'].loc[unique_transitions['prev_clustlabel'] == row['prev_clustlabel']])
            unique_transitions['rel_count_post'].iloc[i] = row['count'] / np.sum(
                unique_transitions['count'].loc[unique_transitions['curr_clustlabel'] == row['curr_clustlabel']])

        return jointdf, unique_transitions

class TransitionAnalysisGUI:

    """
    A GUI for the TransitionAnalysis class.
    Display colours and let user choose the ralabelling
    """

    def __init__(self, unique_transitions, dendrofig):

        """
        Initialize the TransitionAnalysisGUI class.

        Parameters:
        unique_transitions (pd.DataFrame): DataFrame containing the unique transitions and their counts.
        dendrofig (matplotlib.figure.Figure): The dendrogram figure.
        """

        self.unique_transitions = unique_transitions
        self.dendrofig = dendrofig

        # Create transitions df for GUI. Make a copy of unique_transitions
        self.brief_transitions = self.unique_transitions.copy()
        # Drop columns containing clustlabel
        self.brief_transitions = self.brief_transitions.drop(columns=['prev_clustlabel', 'curr_clustlabel'])

        # Replace colourcodes with their corresponding colour in matplotlib as a string
        self.brief_transitions['prev_colourcode'] = self.brief_transitions['prev_colourcode'].apply(
            self.convert_cncode_to_colour)
        self.brief_transitions['curr_colourcode'] = self.brief_transitions['curr_colourcode'].apply(
            self.convert_cncode_to_colour)

        # Create GUI
        self.master = tk.Tk()
        self.master.geometry('1000x550')
        self.master.configure(bg='white')
        self.master.title('Colour matching')
        self.master.attributes("-topmost", True)
        self.master.focus_force()

        # Instructions frame
        topframe = tk.Frame(self.master)
        topframe.grid(row=0, column=0)

        # Display instructions
        instructstr = ('Indicate the colours for the new dendrogram based on the transitions and visualisation below. \n'
                       'The dendrogram displayed ignores the previous partition. The vlines underneath represent the \n'
                       'colour of that datapoint in the previous clustering partition.\n'
                       'The aim is to recolour the dendrogram if needed to maximise the colour match between the two partitions. \n')

        instructholder = tk.StringVar(topframe, instructstr)
        instructions = tk.Label(topframe, textvariable=instructholder)
        instructions.grid(row=0, column=0)

        # Display transitions
        top2frame = tk.Frame(self.master)
        top2frame.grid(row=1, column=0)
        transitionstring = '    ' + self.brief_transitions.to_string(index=False)
        transitionholder = tk.StringVar(top2frame, transitionstring)
        dflabel = tk.Label(top2frame, textvariable=transitionholder)
        dflabel.grid(row=0, column=0)

        # Colour matching options
        midframe = tk.Frame(self.master)
        midframe.grid(row=2, column=0)

        # All the possible colours in the transitions
        posscolours = np.unique(pd.concat([self.brief_transitions['prev_colourcode'],
                                           self.brief_transitions['curr_colourcode']]))

        self.currcols = {}
        self.correctcols = []

        for i, currcol in enumerate(np.unique(self.brief_transitions['curr_colourcode'])):

            # Write current colour in a string
            self.currcols[currcol] = tk.Label(midframe, textvariable=tk.StringVar(midframe, currcol))
            self.currcols[currcol].grid(column=0, row=i)

            # Write possible colours to replace current colour
            currcol = tk.StringVar(midframe, currcol)
            menu = tk.OptionMenu(midframe, currcol, *np.unique(posscolours))
            menu.grid(column=1, row=i)
            self.correctcols.append(currcol)

        # Accept frame
        bottomframe = tk.Frame(self.master)
        bottomframe.grid(row=4, column=0)

        # Create Accept button
        acceptbutton = tk.Button(bottomframe, text='Accept', command=self.accept)
        acceptbutton.grid(row=0, column=0)
        acceptbutton.focus_force()

        # dendrogram frame
        dendroframe = tk.Frame(self.master)
        dendroframe.grid(row=5, column=0)
        canvas = FigureCanvasTkAgg(self.dendrofig, master=dendroframe)
        canvas.draw()
        canvas.get_tk_widget().pack()

        tk.mainloop()

    # Retrieve the color from the default color cycle
    def convert_cncode_to_colour(self, colstr, inverse_transform=False):

        """
        Retrieve the color from the default color cycle.

        Parameters:
        colstr (str): The color code string.
        inverse_transform (bool, optional): Whether to perform inverse transformation. Defaults to False.

        Returns:
        str: The corresponding color string.
        """

        tableau_colours = {'C0': ('blue'),
                           'C1': ('orange'),
                           'C2': ('green'),
                           'C3': ('red'),
                           'C4': ('purple'),
                           'C5': ('brown'),
                           'C6': ('pink'),
                           'C7': ('gray'),
                           'C8': ('yellow-green'),
                           'C9': ('cyan')}
        if inverse_transform:
            return {value: key for key, value in tableau_colours.items()}[colstr]
        else:
            return tableau_colours[colstr]


    # when accept is clicked
    def accept(self):

        """
        Accept the selected color matching and update the recolour dictionary.
        """

        # Store current label and correct label (new)
        self.recolour = {}
        for currcol, correctcol in zip(self.currcols, self.correctcols):

            # Get label corresponding to current colour in unique transition
            currcncode = self.convert_cncode_to_colour(currcol, inverse_transform=True)
            correctcncode = self.convert_cncode_to_colour(correctcol.get(), inverse_transform=True)

            # Get corresponding label in unique transition current column
            currlabel = self.unique_transitions['curr_clustlabel'].loc[self.unique_transitions['curr_colourcode'] == currcncode].values[0]
            correctlabel = self.unique_transitions['curr_clustlabel'].loc[self.unique_transitions['curr_colourcode'] == correctcncode].values[0]
            self.recolour[str(currlabel)] = correctlabel

        # Close the window
        self.master.quit()
        self.master.destroy()


class CustomScaler(BaseEstimator, TransformerMixin):

    """
    A custom scaler class for standardizing data based on variable-specific means and standard deviations.
    It uses vartracker to know what columns belong to the same variable to calculate the mean and std for each variable.
    This is useful when scaling e.g., time series.
    """

    def fit(self, X, y=None, vartracker=None):

        """
        Fit the scaler to the data.

        Parameters:
        X (np.ndarray): The data to fit.
        y (None, optional): Ignored, present for compatibility. Defaults to None.
        vartracker (list, optional): List of variable names corresponding to the columns in the data. Defaults to None.

        Returns:
        self: Fitted scaler instance.
        """

        self.vartracker_ = vartracker

        # Get vartracker
        if self.vartracker_ is None:
            self.mean_ = np.mean(X)
            self.std_ = np.std(X)
        else:
            self.vartracker_ = np.array(vartracker)

            # Get mean and std for each key in datadict
            self.mean_ = {var: np.mean(X[:, np.where(self.vartracker_ == var)]) for var in np.unique(self.vartracker_)}
            self.std_ = {var: np.std(X[:, np.where(self.vartracker_ == var)]) for var in np.unique(self.vartracker_)}

        return self

    def transform(self, X):

        """
        Transform the data using the fitted scaler.

        Parameters:
        X (np.ndarray): The data to transform.

        Returns:
        np.ndarray: The standardized data.
        """

        Xz = np.zeros(X.shape)

        # Standardise data
        for var in np.unique(self.vartracker_):
            Xz[:, np.where(self.vartracker_ == var)] = (X[:, np.where(self.vartracker_ == var)] - self.mean_[var]) / \
                                                       self.std_[var]

        return Xz

    def fit_transform(self, X, y=None, vartracker=None):

        """
         Fit the scaler to the data and then transform it.

         Parameters:
         X (np.ndarray): The data to fit and transform.
         y (None, optional): Ignored, present for compatibility. Defaults to None.
         vartracker (list, optional): List of variable names corresponding to the columns in the data. Defaults to None.

         Returns:
         np.ndarray: The standardized data.
         """

        self.fit(X, y, vartracker)
        Xz = self.transform(X)

        return Xz

    def inverse_transform(self, Xz, y=None):

        """
        Inverse transform the standardized data back to the original scale.

        Parameters:
        Xz (np.ndarray): The standardized data to inverse transform.
        y (None, optional): Ignored, present for compatibility. Defaults to None.

        Returns:
        np.ndarray: The data in the original scale.
        """

        X = np.zeros(Xz.shape)

        # Standardise data
        for var in np.unique(self.vartracker_):
            X[:, np.where(self.vartracker_ == var)] = Xz[:, np.where(self.vartracker_ == var)] * self.std_[var] + \
                                                       self.mean_[var]

        return X


def make_splitgrid(nrows, ncols=None, figsize=(19.2, 9.77)):

    """
    Create a grid of subplots with specified number of rows and columns.

    Parameters:
    nrows (int): Number of rows in the grid.
    ncols (int, optional): Number of columns in the grid. Defaults to the value of nrows.
    figsize (tuple, optional): Size of the figure to be created. Defaults to (19.2, 9.77).

    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the grid.
    axs (dict): A dictionary containing the axes objects for the top and bottom subplots.
    rows_x_cols (list): A list containing the number of rows and columns in the grid.
    """

    if ncols is None:
        ncols = nrows

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, 1, hspace=0.3)
    axs = {'topaxs': [], 'bottomaxs': []}

    for row in gs:
        subgs = row.subgridspec(2, ncols, hspace=0, wspace=0.4)
        for subgdi, subgd in enumerate(subgs):
            if subgdi < subgs.ncols:
                axs['topaxs'].append(plt.subplot(subgd))
            else:
                axs['bottomaxs'].append(plt.subplot(subgd))

    rows_x_cols = [gs.nrows, subgs.ncols]

    return fig, axs, rows_x_cols


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


# 3D distance
def dist3D(a, b):

    """
    Calculate the 3D Euclidean distance between two points.

    Parameters:
    a (array-like): The first point, with coordinates [x, y, z].
    b (array-like): The second point, with coordinates [x, y, z].

    Returns:
    float: The Euclidean distance between points a and b.
    """

    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

#  # TODO. Create opensim utils file with all of these because it gives dll imort errors
# def list_to_osim_array_str(list_str):
#     """Convert Python list of strings to OpenSim::Array<string>.Taken from:
#         https://github.com/mitkof6/opensim_automated_pipeline/blob/7822e1520ceb4ce0943b613a58471b6614437b57/simple/scripts/utils.py"""
#     arr = osim.ArrayStr()
#     for element in list_str:
#         arr.append(element)
#
#     return arr
#
#
# def create_opensim_storage(time, data, column_names):
#     """Creates a OpenSim::Storage. Taken from:
#         https://github.com/mitkof6/opensim_automated_pipeline/blob/7822e1520ceb4ce0943b613a58471b6614437b57/simple/scripts/utils.py
#     Parameters
#     ----------
#     time: SimTK::Vector
#     data: SimTK::Matrix
#     column_names: list of strings
#     Returns
#     -------
#     sto: OpenSim::Storage
#     """
#     sto = osim.Storage()
#     sto.setColumnLabels(list_to_osim_array_str(['time'] + column_names))
#     for i in range(data.nrow()):
#         row = osim.ArrayDouble()
#         for j in range(data.ncol()):
#             row.append(data.getElt(i, j))
#
#         sto.append(time[i], row)
#
#     return sto
#
#
# def np_array_to_simtk_matrix(array):
#     """Convert numpy array to SimTK::Matrix"""
#     n, m = array.shape
#     M = osim.Matrix(n, m)
#     for i in range(n):
#         for j in range(m):
#             M.set(i, j, array[i, j])
#
#     return M
#
#
# def list_to_osim_array_str(list_str):
#     """Convert Python list of strings to OpenSim::Array<string>."""
#     arr = osim.ArrayStr()
#     for element in list_str:
#         arr.append(element)
#
#     return arr
#
#
# def create_clust_opensim_models(scalingdata, ptlabels, modelpath, genscalsetuppath, reportdir):
#
#
#     for clust in np.unique(ptlabels['clustlabel']):
#
#         # Get ptcodes from ptcode column for participants in given cluster
#         clustpts = ptlabels['ptcode'].loc[ptlabels['clustlabel'] == clust]
#
#         # Get average measurements
#         height = scalingdata.loc[clustpts]['Height'].mean()
#         exp_pelv_width = scalingdata.loc[clustpts]['PelvWidth'].mean()
#         exp_thighl_r = scalingdata.loc[clustpts]['ThiLgth_r'].mean()
#         exp_shankl_r = scalingdata.loc[clustpts]['ShaLgth_r'].mean()
#
#         # Initialise model
#         model = osim.Model(modelpath)
#         state = model.initSystem()
#
#         # Extract marker positions
#         markers = model.getMarkerSet()
#         markerpos = {}
#
#         for i in range(markers.getSize()):
#             markername = markers.get(i).getName()
#             markerpos[markername] = markers.get(i).getLocationInGround(state)
#
#         # Calculate ukelele distances between markers
#         mod_pelv_width = dist3D(markerpos['RHJC'], markerpos['LHJC'])
#         mod_thighl_r = dist3D(markerpos['RHJC'], markerpos['RKJC'])
#         mod_shankl_r = dist3D(markerpos['RKJC'], markerpos['RAJC'])
#
#         # Divide experimental by model
#         pelvwidth_sf = exp_pelv_width / mod_pelv_width
#         thighl_r_sf = exp_thighl_r / mod_thighl_r
#         shankl_r_sf = exp_shankl_r / mod_shankl_r
#
#         # Initialise scaler tool
#         scaleTool = osim.ScaleTool(genscalsetuppath)
#
#         # Scale height
#         # scaleTool.setSubjectHeight(height)
#
#         # Scale segments
#         scaleTool.getModelScaler().getScaleSet().get('pelvis').setScaleFactors(osim.Vec3(pelvwidth_sf))
#         scaleTool.getModelScaler().getScaleSet().get('femur_r').setScaleFactors(osim.Vec3(thighl_r_sf))
#         scaleTool.getModelScaler().getScaleSet().get('tibia_r').setScaleFactors(osim.Vec3(shankl_r_sf))
#
#         # Set path to generic model file
#         scaleTool.getGenericModelMaker().setModelFileName(r'gait2392_simbody_custom.osim')
#
#         # Set path to scaled model file
#         scaleTool.getModelScaler().setOutputModelFileName(os.path.join(reportdir, f'Clust_{clust}.osim'))
#         scaleTool.getModelScaler().processModel(model)
#
#         scaleTool.printToXML(r'C:\Users\arr43\Documents\OpenSim\4.3\Models\Gait2392_Simbody\Test_Scale_setup.xml')
#
#         # Run scaler tool
#         scaleTool.run()
#
#         # Get Cn colour as RGB
#         colour = ptlabels['colourcode'].loc[ptlabels['clustlabel'] == clust].values[0]
#         colour = mcolors.to_rgb(colour)
#
#         # Read in scaled model as a list of strings
#         with open(os.path.join(reportdir, f'Clust_{clust}.osim'), 'r') as file:
#             filedata = file.read().split('\n')
#
#         # Get model segment names
#         bodies = model.get_BodySet()
#         segnames = [bodies.get(i).getName() for i in range(bodies.getSize())]
#
#         # Get color lines
#         colorlines = [i for i, x in enumerate(filedata) if '<color>' in x]
#
#         for body in segnames:
#
#             # Find the line with the segment name
#             bodystart = [i for i, x in enumerate(filedata) if f'<Body name="{body}"' in x][0]
#             bodyend = [i for i, x in enumerate(filedata) if f'</Body>' in x and i > bodystart][0]
#
#             # Get colorlines within bodystart and bodyend
#             bodycolorlines = [i for i in colorlines if bodystart < i < bodyend]
#
#             for line in bodycolorlines:
#
#                 # Find where <color> starts in the line
#                 start = filedata[line].find('<color>')
#
#                 # Replace the line with the new colour
#                 filedata[line] = filedata[line][:start] + f'<color>{colour[0]} {colour[1]} {colour[2]}</color>'
#
#         # Merge all list items into a string with new lines
#         filedata = '\n'.join(filedata)
#         with open(os.path.join(reportdir, f'Clust_{clust}.osim'), 'w') as file:
#             file.write(filedata)

def single_speed_kinematics_comparison(datadict, discvars, contvars, figargs):

    """
    Compare kinematic variables between two groups at a single speed and visualize the results.

    Parameters:
    datadict (dict): Dictionary containing the data to be compared.
    discvars (list): List of discrete variables to be compared.
    contvars (list): List of continuous variables to be compared.
    figargs (dict): Dictionary containing figure arguments for plotting.

    Returns:
    stat_comparison (dict): A dictionary containing the results of the statistical tests.
    """

    # Get figure arguments
    reportdir = figargs['reportdir']
    savingkw = figargs['savingkw']
    study_title = figargs['study_title']
    kinematics_titles = figargs['kinematics_titles']
    kinematics_ylabels = figargs['kinematics_ylabels']

    # Get group labels and corresponding colours from ['ptlabels'] within datadict
    grouplabels = natsort.natsorted(np.unique(datadict['ptlabels']['clustlabel']))
    groupcolours = [datadict['ptlabels']['colourcode'].loc[
                            datadict['ptlabels']['clustlabel'] == g].iloc[0] for g in grouplabels]

    # if len of grouplabels is different from 2, return and write a warning
    if len(grouplabels) != 2:
        warnings.warn(f'{len(grouplabels)} clusters were identified for {study_title}.\n '
                      f'This function is limited to 2 clusters to replicate the results in Rivadulla et al. (2024).\n '
                      f'If you identified more clusters, you will have to extend the functionality of '
                      f'this code by yourself ;)', category=UserWarning)
        return None

    # Perform 0D and 1D statistical comparisons
    stat_comparison = {'0D': {}, '1D': {}}
    stat_comparison['0D'] = comparison_0D_contvar_indgroups({key: datadict[key] for key in discvars},
                                                                   datadict['ptlabels']['clustlabel'],
                                                                   savingkw,
                                                                   reportdir,
                                                                   groupcolours)

    stat_comparison['1D'] = comparison_1D_contvar_indgroups({key: datadict[key] for key in contvars},
                                                                   datadict['ptlabels']['clustlabel'],
                                                                   savingkw,
                                                                   reportdir,
                                                                   groupcolours)

    # Plot all variables with significant differences indicated
    kinfig, kinaxs = plt.subplots(2, 4, figsize=(11, 4.5))
    kinaxs = kinaxs.flatten()

    # Get avge toe off for each cluster to show in the  SPM plots
    avgeto = []
    for grouplabel in grouplabels:
        groupidcs = np.where(datadict['ptlabels']['clustlabel'] == grouplabel)[0]
        avgeto.append(np.round(np.mean(datadict['DUTYFACTOR'][groupidcs, :]) * 100, 1))

    for vari, varname in enumerate(discvars + contvars):

        # 0D variables
        if varname in discvars:

            # Violin plot
            sns.violinplot(ax=kinaxs[vari],
                           x=datadict['ptlabels']['clustlabel'].values,
                           y=datadict[varname].flatten(),
                           hue=datadict['ptlabels']['clustlabel'].values,
                           palette=groupcolours,
                           legend=False)

            # Xticks
            kinaxs[vari].set_xticks(kinaxs[vari].get_xticks(),
                                    [f'C{int(x)}' for x in kinaxs[vari].get_xticks()])

            # Get key which is not normality
            stat_test = [key for key in stat_comparison['0D'][varname].keys() if key != 'normality'][0]

            # Add asterisk to the title to indicate significant differences
            if stat_comparison['0D'][varname][stat_test]['p'] < 0.05:
                kinaxs[vari].set_title(f'{kinematics_titles[varname]} *')
            else:
                kinaxs[vari].set_title(f'{kinematics_titles[varname]}')

        # 1D variables
        elif varname in contvars:

            groups = []

            for clusti, grouplabel in enumerate(grouplabels):
                clustidcs = np.where(datadict[f'ptlabels']['clustlabel'] == grouplabel)[0]
                groups.append(datadict[varname][clustidcs, :])

                # SPM plot
                spm1d.plot.plot_mean_sd(groups[-1], x=np.linspace(0, 100, groups[-1].shape[1]),
                                        linecolor=groupcolours[clusti], facecolor=groupcolours[clusti],
                                        ax=kinaxs[vari])

            # Add vertical line at avge toe off (outside the previous loop so we can get the final ylimits)
            for groupi, groupavgeto in enumerate(avgeto):
                kinaxs[vari].axvline(x=groupavgeto, color=groupcolours[groupi], linestyle=':')

            # Add patch to indicate significant differences
            spmtest = list(stat_comparison['1D'][varname].keys())[0]
            if stat_comparison['1D'][varname][spmtest].h0reject:

                # Scaler for sigcluster endpoints
                tscaler = kinaxs[vari].get_xlim()[1] / (groups[0].shape[1] - 1)

                # Add significant patches
                add_sig_spm_cluster_patch(kinaxs[vari], stat_comparison['1D'][varname][spmtest],
                                          tscaler=tscaler)

            # title
            kinaxs[vari].set_title(kinematics_titles[varname])

        # Add ylabel
        kinaxs[vari].set_ylabel(kinematics_ylabels[varname])

    # Legend
    # create cluster labels as C and the number
    kinaxs[-1].legend([f'C{int(grouplabel)}' for grouplabel in grouplabels], loc='lower right', frameon=False)

    # Suptitle
    kinfig.suptitle(f'{study_title} kinematics')
    plt.tight_layout()

    # Save and close
    kinfig.savefig(os.path.join(reportdir, f'{savingkw}_kinematics.png'), dpi=300, bbox_inches='tight')
    plt.close(kinfig)

    return stat_comparison


def multispeed_kinematics_comparison(datadict, stages, speeds, discvars, contvars, figargs):

    """
    Compare kinematic variables between multiple speeds and visualize the results.

    Parameters:
    datadict (dict): Dictionary containing the data to be compared.
    stages (list): List of stages for the analysis.
    speeds (list): List of speeds corresponding to each stage.
    discvars (list): List of discrete variables to be compared.
    contvars (list): List of continuous variables to be compared.
    figargs (dict): Dictionary containing figure arguments for plotting.

    Returns:
    stat_comparison (dict): A dictionary containing the results of the statistical tests.
    """

    # Get figargs
    reportdir = figargs['reportdir']
    savingkw = figargs['savingkw']
    speedcolours = figargs['speedcolours']
    kinematics_ylabels = figargs['kinematics_ylabels']
    kinematics_titles = figargs['kinematics_titles']
    stg_titles = figargs['stg_titles']

    # Get group labels and corresponding colours from ['ptlabels'] within datadict
    grouplabels = natsort.natsorted(np.unique(datadict['multispeed']['ptlabels']['clustlabel']))
    groupcolours = [datadict['multispeed']['ptlabels']['colourcode'].loc[
                            datadict['multispeed']['ptlabels']['clustlabel'] == g].iloc[0] for g in grouplabels]

    # Initialise stat_comparison
    stat_comparison = {'0D': {}, '1D': {}}

    # 0D variables: 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
    for vari, varname in enumerate(discvars):
        discvarfig, discvaraxs = plt.subplots(1, 3, figsize=(11, 3))
        discvaraxs = discvaraxs.flatten()

        stat_comparison['0D'][varname] = {}

        # Get data
        df = pd.DataFrame()
        df[varname] = np.concatenate(datadict['multispeed'][varname].T)
        df['speed'] = np.concatenate(
            [[int(speeds[stgi])] * datadict['multispeed'][varname].shape[0] for stgi, stage in enumerate(stages)])
        df['clustlabel'] = np.tile(datadict['multispeed']['ptlabels']['clustlabel'].values, len(stages))
        df['ptcode'] = np.tile(datadict['multispeed']['ptlabels']['ptcode'].values, len(stages))

        # Run 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
        stat_comparison['0D'][varname] = anova2onerm_0d_and_posthocs(df,
                                                                     dv=varname,
                                                                     within='speed',
                                                                     between='clustlabel',
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

    # 1D variables: SPM 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
    speedfig, speedaxs = plt.subplots(2, 3, figsize=(11, 4.5))
    speedaxs = speedaxs.flatten()

    # Get avge toe off for each speed and for each group based on duty factor for the plots
    avgeto = {}
    speedavgeto = []
    for stage in stages:
        avgeto[stage] = []
        for group in grouplabels:
            groupidcs = np.where(datadict[stage]['ptlabels']['clustlabel'] == group)[0]
            avgeto[stage].append(np.round(np.mean(datadict[stage]['DUTYFACTOR'][groupidcs, :]) * 100, 1))

        speedavgeto.append(np.round(np.mean(datadict[stage]['DUTYFACTOR']) * 100, 1))

    for vari, contvar in enumerate(contvars):

        stat_comparison['1D'][contvar] = {}

        # Initialise data holders
        group = []
        speed = []
        subject = []
        Y = []
        Ydiff = []

        # Prepare data for SPM and SPM mean and std plots
        fig = plt.figure()
        fig.set_size_inches(10, 5)
        basegrid = fig.add_gridspec(2, 1)
        topgrid = basegrid[0].subgridspec(1, len(stages))
        bottomgrid = basegrid[1].subgridspec(1, len(stages) - 1)

        upperaxs = []
        loweraxs = []

        for stgi, stage in enumerate(stages):

            # Append group speed and subject
            group.append(datadict['multispeed']['ptlabels']['clustlabel'].values)
            speed.append(np.ones(datadict['multispeed'][varname].shape[0]) * stgi)
            subject.append(np.arange(len(datadict['multispeed']['ptlabels']['clustlabel'].values)))
            Y.append(datadict[stage][contvar])

            # Create axis
            upperaxs.append(fig.add_subplot(topgrid[0, stgi]))

            # Plot mean and std curves
            for labi, lab in enumerate(np.sort(np.unique(group[-1]))):
                # Top row: group by group for each speed
                spm1d.plot.plot_mean_sd(Y[-1][np.where(group[-1] == lab)[0], :],
                                        x=np.linspace(0, 100, Y[-1].shape[1]),
                                        linecolor=groupcolours[labi], facecolor=groupcolours[labi],
                                        ax=upperaxs[stgi])

            # Add vertical line at avge toe off (outside the previous loop so we can get the final ylimits)
            for labi, lab in enumerate(np.sort(np.unique(group[-1]))):
                upperaxs[stgi].axvline(x=avgeto[stage][labi], color=groupcolours[labi], linestyle=':')

            # xlabel. This ensures they are all the same size and will get filled with stats if post-hocs were performed
            upperaxs[stgi].set_xlabel(' ')

            # Speed figure
            spm1d.plot.plot_mean_sd(Y[-1], x=np.linspace(0, 100, Y[-1].shape[1]),
                                    linecolor=speedcolours[stgi], facecolor=speedcolours[stgi],
                                    ax=speedaxs[vari])

            if stgi > 0:

                # Calculate change from one speed to another
                Ydiff.append(Y[-1] - Y[-2])

                # Plot it by group
                loweraxs.append(fig.add_subplot(bottomgrid[0, stgi - 1]))

                # Add horizontal line at 0
                loweraxs[-1].axhline(0, color='black', linestyle='-', linewidth=0.5, zorder=1)

                for uni in np.sort(np.unique(group)):
                    spm1d.plot.plot_mean_sd(Ydiff[-1].T[:, group[stgi] == uni].T,
                                            x=np.linspace(0, 100, Ydiff[-1].shape[1]),
                                            linecolor=groupcolours[uni], facecolor=groupcolours[uni],
                                            ax=loweraxs[-1])

                # Add vline at avge toe off between speeds (outside the previous loop so we can get the final ylimits)
                for labi, lab in enumerate(np.sort(np.unique(group[-1]))):
                    loweraxs[-1].axvline(x=np.mean([avgeto[stages[stgi - 1]][labi], avgeto[stage][labi]]),
                                         color=groupcolours[labi], linestyle=':')

                # Set title
                loweraxs[-1].set_title(f'{speeds[stgi]} wrt {speeds[stgi - 1]} km/h')

                # xlabel. This ensures they are all the same size and
                #  will get filled with stats if post-hocs were performed
                loweraxs[-1].set_xlabel(' ')

                # Legend
                loweraxs[-1].legend(['_nolegend_', 'C0', 'C1'],
                                    loc='lower center',
                                    bbox_to_anchor=(0.5, 0),
                                    ncol=2,
                                    bbox_transform=fig.transFigure,
                                    frameon=False)
                plt.subplots_adjust(bottom=0.11)

            # Title
            upperaxs[stgi].set_title(stg_titles[stgi])

        # ylabels
        upperaxs[0].set_ylabel(kinematics_ylabels[contvar])
        loweraxs[0].set_ylabel('${\Delta}$')

        # Get ylims for all upperaxs
        ylims = [ax.get_ylim() for ax in upperaxs]

        # Set ylims for all upperaxs
        for ax in upperaxs:
            ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])

        # Get ylims for all loweraxs
        ylims = [ax.get_ylim() for ax in loweraxs]

        # Set ylims for all loweraxs
        for ax in loweraxs:
            ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])

        # add vertical lines at avge toe off to speed figure
        for spavgetoi, spavgeto in enumerate(speedavgeto):
            speedaxs[vari].axvline(x=spavgeto, color=speedcolours[spavgetoi], linestyle=':')

        # title and ylabel for speed figure
        speedaxs[vari].set_title(kinematics_titles[contvar])
        speedaxs[vari].set_ylabel(kinematics_ylabels[contvar])

        # Conduct SPM analysis
        spmlist = spm1d.stats.nonparam.anova2onerm(np.concatenate(Y, axis=0),
                                                   np.concatenate(group),
                                                   np.concatenate(speed),
                                                   np.concatenate(subject))
        stat_comparison['1D'][contvar]['ANOVA2onerm'] = spmlist.inference(alpha=0.05, iterations=1000)

        # Post hoc tests and figures
        stat_comparison['1D'][contvar]['posthocs'] = {}

        # Add patches to speed figure if there is an effect of speed
        if stat_comparison['1D'][contvar]['ANOVA2onerm'][1].h0reject:
            # Scaler for sigcluster endpoints
            tscaler = speedaxs[vari].get_xlim()[1] / (Y[0].shape[1] - 1)

            # Add patches to speed figure
            add_sig_spm_cluster_patch(speedaxs[vari], stat_comparison['1D'][contvar]['ANOVA2onerm'][1],
                                      tscaler=tscaler)

        # Add title to speed figure
        statstr = f'F* = {write_spm_stats_str(stat_comparison["1D"][contvar]["ANOVA2onerm"][1], mode="stat")}'

        speedaxs[vari].set_title(f'{kinematics_titles[contvar]}\n{statstr}')

        # Follow up with post-hoc tests if cluster effects are found
        if stat_comparison['1D'][contvar]['ANOVA2onerm'][0].h0reject:

            stat_comparison['1D'][contvar]['posthocs']['cluster'] = {}

            # For each speed
            for spi, (groupi, Yi) in enumerate(zip(group, Y)):

                stat_comparison['1D'][contvar]['posthocs']['cluster'][stages[spi]] = {}

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Yi[groupi == 0, :], Yi[groupi == 1, :], )
                snpmi = snpm.inference(alpha=0.05 / len(Y), two_tailed=True, iterations=1000)

                # Add snpmi to dictionary
                stat_comparison['1D'][contvar]['posthocs']['cluster'][stages[spi]]['snpm_ttest2'] = snpmi

                # Add stats to xlabel
                statstr = f't* = {write_spm_stats_str(snpmi, mode="full")}'
                upperaxs[spi].set_xlabel(statstr, fontsize=10)

                # Plot
                plt.figure()
                snpmi.plot()
                snpmi.plot_threshold_label(fontsize=8)
                snpmi.plot_p_values(size=10)
                plt.gcf().suptitle(f'{contvar}_posthoc_{stages[spi]}')

                # Save figure and close it
                plt.savefig(os.path.join(reportdir, f'{savingkw}_{contvar}_posthoc_{stages[spi]}.png'))
                plt.close(plt.gcf())

                # Add patches to upperaxs if significant diffs are found
                if snpmi.h0reject:
                    # Scaler for sigcluster endpoints
                    tscaler = upperaxs[spi].get_xlim()[1] / (Y[0].shape[1] - 1)

                    # Add significant pathces to upperaxs
                    add_sig_spm_cluster_patch(upperaxs[spi], snpmi, tscaler=tscaler)

        # Interaction effect
        if stat_comparison['1D'][contvar]['ANOVA2onerm'][2].h0reject:

            stat_comparison['1D'][contvar]['posthocs']['interaction'] = {}

            # Calculate change in conditions
            for condi in range(len(stages) - 1):

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Ydiff[condi][group[condi] == 0, :],
                                                   Ydiff[condi][group[condi] == 1, :], )
                snpmi = snpm.inference(alpha=0.05 / len(range(len(stages) - 1)), two_tailed=True, iterations=1000)

                # Add snpmi to dictionary
                stat_comparison['1D'][contvar]['posthocs']['interaction'][
                    f'{speeds[condi + 1]}_wrt_{speeds[condi]}'] = {}
                stat_comparison['1D'][contvar]['posthocs']['interaction'][
                    f'{speeds[condi + 1]}_wrt_{speeds[condi]}']['snpm_ttest2'] = snpmi

                # Add stats to xlabel
                statstr = f't* = {write_spm_stats_str(snpmi, mode="full")}'
                loweraxs[condi].set_xlabel(statstr, fontsize=10)

                # Plot
                plt.figure()
                snpmi.plot()
                snpmi.plot_threshold_label(fontsize=8)
                snpmi.plot_p_values(size=10)
                plt.gcf().suptitle(f'{contvar}_posthoc_{speeds[condi + 1]}_v_{speeds[condi]}')

                # Save figure and close it
                plt.savefig(os.path.join(reportdir,
                                         f'{savingkw}_multispeed_{contvar}_interact_posthoc_{speeds[condi + 1]}_v_{speeds[condi]}.png'))
                plt.close(plt.gcf())

                # Add patches to loweraxs if significant diffs are found
                if snpmi.h0reject:
                    # Scaler for sigcluster endpoints
                    tscaler = loweraxs[condi].get_xlim()[1] / (Ydiff[0].shape[1] - 1)

                    # Add significant pathces to loweraxs
                    add_sig_spm_cluster_patch(loweraxs[condi], snpmi, tscaler=tscaler)

        # Write cluster effect string for suptitle
        statstr = f'C: F* = {write_spm_stats_str(stat_comparison["1D"][contvar]["ANOVA2onerm"][0], mode="full")}'

        # Write interaction effect string for suptitle
        statstr += f'; CxS: F* = {write_spm_stats_str(stat_comparison["1D"][contvar]["ANOVA2onerm"][2], mode="full")}'

        # Set suptitle
        fig.suptitle(f'{kinematics_titles[contvar]}\n{statstr}')

        # Save and close
        plt.subplots_adjust(bottom=0.13)
        plt.tight_layout()
        fig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_{contvar}_ANOVA2onerm.png'), dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    # Add legend to last plot in speed figure
    speedaxs[-1].legend(speeds, frameon=False)

    # Save speed figure
    plt.tight_layout()
    speedfig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_kinematics_by_speed.png'), dpi=300,
                     bbox_inches='tight')
    plt.close(speedfig)

    return stat_comparison


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


def write_0DmixedANOVA_statstr(mixed_anovatable, between='', within='', betweenlabel='', withinlabel=''):

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

    if mixed_anovatable['p-unc'].loc[mixed_anovatable['Source'] == between].values < 0.001:
        statstr = f'{betweenlabel}: F = {np.round(mixed_anovatable["F"].values[0], 2)}, p < 0.001'
    else:
        statstr = (f'{betweenlabel}: F = {np.round(mixed_anovatable["F"].values[0], 2)}, '
                   f'p = {np.round(mixed_anovatable["p-unc"].values[0], 3)}')

    if mixed_anovatable['p-unc'].loc[mixed_anovatable['Source'] == 'speed'].values < 0.001:
        statstr += f'; {withinlabel}: F = {np.round(mixed_anovatable["F"].values[1], 2)}, p < 0.001'
    else:
        statstr += (f'; {withinlabel}: F = {np.round(mixed_anovatable["F"].values[1], 2)}, '
                    f'p = {np.round(mixed_anovatable["p-unc"].values[1], 3)}')

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