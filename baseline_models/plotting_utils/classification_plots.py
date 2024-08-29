"""
Collection of functions to plot results from a classification model.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()
sns.set_context('talk')
sns.set_style('darkgrid', {'axes.edgecolor': '0.2',
                           'xtick.bottom': True,
                           'ytick.left': True
                          })


def make_confusion_matrix(cm,
                          group_names=['True Neg', 'False Pos', 'False Neg', 'True Pos'],
                          categories='auto',
                          display_count=True, 
                          display_percent=True, 
                          cbar=True, 
                          display_xyticks=True,
                          display_xyplotlabels=True,
                          display_sum_stats=True,
                          cmap= 'Blues',
                          title=None,
                          ):
    """
    Plots a confusion matrix generated from sklearn.

    Args:
        cm: Confusion matrix to be passed in (i.e., 2D numpy array of values).

        group_names: List of strings that represent the labels row by row to be shown in each square.
                     By default, assumes 2x2 confusion matrix from binary classification.

        categories: List of strings containing the categories to be displayed on the x,y axis. 
        
        display_count: Boolean indicating whether to show the raw number in the confusion matrix.

        display_percent: Boolean indicating whether to show the perctanges for each category.

        cbar: Boolean indicating whether to show the color bar.

        display_xyticks: Boolean indicating whether show x and y ticks. 
        
        display_xyplotlabels: Boolean indicating whether to show 'True Label' and 'Predicted Label' on the figure.

        display_sum_stats: Boolean indicating whether to display summary statistics below the figure.

        сmaр: Colormap of the values displayed from matplotlib.pyplot.cm.
              See choices at http://matplotlib.org/examples/color/colormaps_reference.html

        title: Optional title for the heatmap.

    Returns:
        fig: the figure object
        ax: the ax object
    """

    # generate text inside each square
    blanks = ["" for i in range(cm.size)]

    if group_names and len(group_names) == cm.size:
        group_labels = [f"{value}\n" for value in group_names]
    else:
        group_labels = blanks
    
    if display_count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
    else:
        group_counts = blanks

    if display_percent:
        group_percentages = ["{0:.1%}".format(value) for value in cm.flatten() / np.sum(cm)]
    else:
        group_percentages = blanks
    
    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0], cm.shape[1])

    # summary statistics
    if display_sum_stats:
        # accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cm) / float(np.sum(cm))
        
        # show more stats if confusion matrix is for binary classification
        if len(cm) == 2:
            tp, fp, fn, tp = cm.ravel()
            
            # metrics for binary confusion matrices
            precision = tp / (tp + fp)      # cm[1,1] / sum(cm[:,1])
            recall = tp / (tp + fn)         # cm[1,1] / sum(cm[1, :])
            f1_score = 2 * tp / (2 * tp + fn + fp)  # 2*precision*recall / (precision + recall)
            stats_text = f"\n\nAccuracy={accuracy:0.2f}\nPrecision={precision:0.2f}\nRecall={recall:0.2f}\nF1 Score={f1_score:0.2f}"
        else:
            stats_text = f"\n\nAccuracy={accuracy: 0.2f}"
    else:
        stats_text = ""

    if display_xyticks == False:
        categories = False

    fig, ax = plt.subplots()
    ax = sns.heatmap(cm, ax=ax, 
                     annot=box_labels, fmt="",
                     cmap=cmap, cbar=cbar, 
                     xticklabels=categories,
                     yticklabels=categories,
                     )
    
    cmin, cmax = ax.collections[0].get_clim()
    ax.collections[0].set_clim(0, cmax)

    if display_xyplotlabels:
        ax.set_xlabel('Predicted Label' + stats_text, labelpad=10)
        ax.set_ylabel('True Label', labelpad=10)
    else:
        ax.set_xlabel(stats_text)

    if title:
        ax.set_title(title)
    
    return fig, ax


def make_enrichment_plot(y_true,
                         y_pred,
                         cutoff_1=0.75,
                         cutoff_2=1.5,
                         title='',
                         show_stats=True,
                        ):

    if len(y_true) != len(y_pred):
        msg = f'Length of y_true should equal length of y_pred. '
        msg += f'Length of y_true is {len(y_true)} while length of y_pred is {len(y_pred)}.'
        raise ValueError(msg)

    # combine into a df for easy querying
    df_combined = pd.DataFrame({'experimental': y_true, 'pred': y_pred})

    df = pd.DataFrame(
        {f'x <= {cutoff_1}':[
            len(df_combined.query(f'experimental <= {cutoff_1}').query(f'pred <= {cutoff_1}') ),
            len(df_combined.query(f'experimental <= {cutoff_1}').query(f'pred > {cutoff_1}').query(f'pred <= {cutoff_2}') ),
            len(df_combined.query(f'experimental <= {cutoff_1}').query(f'pred > {cutoff_2}')),
        ],
        f'{cutoff_1} < x <= {cutoff_2}': [
            len(df_combined.query(f'experimental > {cutoff_1}').query(f'experimental <= {cutoff_2}').query(f'pred <= {cutoff_1}')),
            len(df_combined.query(f'experimental > {cutoff_1}').query(f'experimental <= {cutoff_2}').query(f'pred > {cutoff_1}').query(f'pred <= {cutoff_2}') ),
            len(df_combined.query(f'experimental > {cutoff_1}').query(f'experimental <= {cutoff_2}').query(f'pred > {cutoff_2}') ),
        ],
        f'x > {cutoff_2}': [
            len(df_combined.query(f'experimental > {cutoff_2}').query(f'pred <= {cutoff_1}') ),
            len(df_combined.query(f'experimental > {cutoff_2}').query(f'pred > {cutoff_1}').query(f'pred <= {cutoff_2}') ),
            len(df_combined.query(f'experimental > {cutoff_2}').query(f'pred > {cutoff_2}') )
        ],
        },
        index=[f'x <= {cutoff_1}', 
               f'{cutoff_1} < x <= {cutoff_2}',
               f'x > {cutoff_2}',
               ]
    )
    # make sure all values are accounted for
    assert df.sum().sum() == len(y_true)

    # plot results
    COLORS = ['crimson', 'gold', 'green']
    if descending:
      COLORS = list(reversed(COLORS))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = df.plot(kind='bar',
                 stacked=True,
                 color=COLORS,
                 ax=ax)

    ax.legend(title='Experimental Values', fontsize=14)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel('Predicted Value', labelpad=10)
    ax.set_ylabel('Counts', labelpad=10)

    if title:
        ax.set_title(title, pad=8)

    if show_stats:
        # get total for each row
        total = df.sum(axis=1)

        # calculate percent for each row
        per = df.div(total, axis=0).mul(100).round(1)

        # iterate through containers
        for c in ax.containers:
            # get current segment label
            label = c.get_label()

            # create custom labels
            labels = [f'{int(v.get_height())}\n({row}%)' if row > 10 else '' for v, row in zip(c, per[label])]

            # add new annotation
            ax.bar_label(c,
                         labels=labels,
                         label_type='center',
                         fontsize=16,
                         color='k',
                        )

    return fig, ax
