'''
Author: Zachery Berger <zackeberger@g.ucla.edu>, Parth Agrawal <parthagrawal24@g.ucla.edu>, Tian Yu Liu <tianyu139@g.ucla.edu>, Alex Wong <alexw@cs.ucla.edu>
If you use this code, please cite the following paper:

Z. Berger, P. Agrawal, T. Liu, S. Soatto, and A. Wong. Stereoscopic Universal Perturbations across Different Architectures and Datasets.
https://arxiv.org/pdf/2112.06116.pdf

@inproceedings{berger2022stereoscopic,
  title={Stereoscopic Universal Perturbations across Different Architectures and Datasets},
  author={Berger, Zachery and Agrawal, Parth and Liu, Tian Yu and Soatto, Stefano and Wong, Alex},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


TITLE_FONT_SIZE = 18
LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 15
TICK_FONT_SIZE = 12
LINE_WIDTH = 3
LINE_STYLE = ':'  # ['-', '--', '-.', ':']

COLORS = [
    'blue',
    'green',
    'red',
    'orange',
    'magenta',
    'cyan',
    'indigo',
    'yellow'
]


def create_plot(plot_title,
                x_values,
                y_values,
                y_errors,
                legend_names,
                filename,
                x_label=r'$\epsilon$',
                y_label='D1-all error (%)',
                x_lim_min=None,
                x_lim_max=None,
                y_lim_min=None,
                y_lim_max=None,
                show_errorbars=True,
                legend_loc='best',
                long_title=False):

    x_values = np.array(x_values)
    y_values = np.array(y_values)
    y_errors = np.array(y_errors)

    x_lim_min = np.min(np.array(x_values)) if x_lim_max is None else x_lim_min
    x_lim_max = np.max(np.array(x_values)) if x_lim_max is None else x_lim_max

    y_lim_min = np.min(np.array(y_values)) if y_lim_min is None else y_lim_min
    y_lim_max = np.max(np.array(y_values)) if y_lim_max is None else y_lim_max

    if len(x_values.shape) == 1 and len(y_values.shape) > 1:
        x_values = np.expand_dims(x_values, axis=0)
        x_values = np.tile(x_values, (y_values.shape[0], 1))

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(1, 1, 1)
    legends = []

    for i, (x, y, y_err) in enumerate(zip(x_values, y_values, y_errors)):

        if len(legend_names) > 0:
            legend_name = legend_names[i]

        if np.unique(y).shape[0] == 1:
            ax.plot(x[0], y[0], 'k*', markersize=10)
        else:
            if show_errorbars:
                ax.errorbar(x, y, y_err,
                    linewidth=LINE_WIDTH, color=COLORS[i], linestyle=LINE_STYLE)
            else:
                ax.plot(x, y,
                    linewidth=LINE_WIDTH, color=COLORS[i], linestyle=LINE_STYLE)

            plt.fill_between(x, y - y_err, y + y_err,
                color=COLORS[i], linestyle=LINE_STYLE, alpha=0.2)

        # Set x and y label
        plt.ylabel(y_label, fontsize=LABEL_FONT_SIZE)
        plt.xlabel(x_label, fontsize=LABEL_FONT_SIZE)

        if len(legend_names) > 0:
            legends.append('{}'.format(legend_name))

    if long_title:
        plt.title('%s' % (plot_title), fontsize=TITLE_FONT_SIZE, loc='center', wrap=True)
    else:
        plt.title('%s' % (plot_title), fontsize=TITLE_FONT_SIZE)
    if len(legend_names) > 0:
        plt.legend(legends, loc=legend_loc, fontsize=LEGEND_FONT_SIZE)

    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)

    plt.ylim(y_lim_min, y_lim_max)
    plt.xlim(x_lim_min, x_lim_max)
    #     ax.set(facecolor = "white")
    ax.get_xaxis().set_major_formatter(FormatStrFormatter('%0.0f'))
    ax.grid(color='gray', linestyle='dashdot', linewidth=1)

    #     plt.show()

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    fig.savefig(filename)
    plt.close(fig)


'''
Sampling Efficiency of the Perturbations
'''
plot_title = 'Sampling Efficiency'

legend_names = [
    'AANet',
    'DeepPruner',
    'PSMNet'
]

x_values = [47890, 43000, 33000, 23000, 17000, 10000]

y_values = [
    [48.0119,
    48.0501,
    48.2143,
    48.3924,
    48.4256,
    47.3633],
    [52.703,
    52.5007,
    51.5707,
    47.6536,
    39.6906,
    24.6476],
    [87.4181,
    87.3595,
    87.1546,
    86.8565,
    86.3623,
    83.7932]
]

y_errors = [
    [16.7623, 16.793, 16.8458, 16.8983, 16.8737, 16.7438],
    [23.89, 23.8277, 22.6217, 20.491, 16.4811, 9.6638],
    [11.9089, 11.917, 12.0054, 12.0545, 12.192, 13.3243]
]

x_label = 'Number of Training Samples'

y_label = 'D1-all error (%)'

filename = 'plots/sampling_efficiency.png'

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=x_label,
    y_label=y_label,
    x_lim_min=10000,
    x_lim_max=50000,
    y_lim_min=0.0,
    y_lim_max=110,
    show_errorbars=False)
