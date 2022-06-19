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

import matplotlib.pyplot as plt

TITLE_FONT_SIZE = 18
LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 15
TICK_FONT_SIZE = 12
LINE_WIDTH = 3
LINE_STYLE = ':'  # ['-', '--', '-.', ':']

COLORS = [
    'red',
    'blue',
    'green',
    'orange',
    'magenta',
    'cyan',
    'indigo',
    'yellow'
]


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=COLORS[i])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(), facecolor='white', loc='best', fontsize=LEGEND_FONT_SIZE)


nmap = {
    0: 'PSMNet',
    1: 'PSMNet w/ Our Design'
}

data = {x : [] for x in range(2)}

type_defenses = ['JPEG Compression',
'Gaussian Blur',
'Brightness',
'Contrast',
'Gaussian Noise',
'Shot Noise',
'Pixelate',
'Defocus Blur',
'Motion Blur']

data[0] = [3.958927015,
26.53022774,
6.412943642,
14.19891194,
19.12814112,
50.1519041,
12.79291585,
17.50312051,
28.77935046]

data[1] = [0.6784022457,
1.239838587,
1.31001813,
9.917450788,
4.690332768,
9.988888239,
-0.4912567986,
0.7485817884,
25.9488859]

data = {nmap[k] : v for k, v in data.items()}

fig, ax = plt.subplots(dpi=100, facecolor='white', figsize=(10, 8))
ax.set_facecolor('white')
ax.patch.set_facecolor('white')
ax.tick_params(axis='y', colors='black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['right'].set_color('black')
ax.grid(color='gray', linestyle='dashdot', linewidth=1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1)

ax.tick_params(axis='both', which='both', length=5)

plot_title = 'Error under Image Corruptions'

x_label = 'Image Corruption'

y_label = '% Change Relative to Error for Clean Images'

filename = 'plots/img_corruption.png'

bar_plot(ax, data)
plt.xticks(range(len(type_defenses)), [f"{type_defense}" for type_defense in type_defenses], fontsize=15, rotation=70)
plt.ylim(-5, 60)
plt.xlabel(x_label, labelpad=-12, fontsize=15)
plt.ylabel(y_label, fontsize=15)
plt.title(plot_title, fontsize=TITLE_FONT_SIZE)

fig.savefig(filename, bbox_inches="tight")
