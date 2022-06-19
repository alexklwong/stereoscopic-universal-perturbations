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
from matplotlib import pyplot as plt
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

    fig = plt.figure()
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
    plt.xlim(y_lim_min, x_lim_max)
    ax.get_xaxis().set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.grid(color='gray', linestyle='dashdot', linewidth=1)

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    fig.savefig(filename)
    plt.close(fig)


'''
KITTI 2012, Tile64: AANet to X
'''
plot_title = 'KITTI 2012, Tile64: AANet to X'

filename = os.path.join('plots/perturbation_error/kitti2012/', 'kitti_2012_tile64_aanet_to_X.png')
legend_names = [
    'AANet -> PSMNet',
    'AANet -> DeepPruner',
    'AANet -> AANet'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [2.5647, 4.0995, 13.7416, 36.6901, 65.1774],
    [1.5086, 2.2732, 8.1611, 29.79, 63.1389],
    [1.7012, 1.9954, 8.7458, 30.0043, 58.0412]
]
y_errors = [
    [1.4057, 2.8003, 9.1834, 10.4002, 10.5993],
    [1.1324, 2.9364, 7.6811, 12.6723, 14.3369],
    [1.442, 1.4455, 7.2048, 9.329, 12.5103]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)


'''
KITTI 2012, Tile64: DeepPruner to X
'''
plot_title = 'KITTI 2012, Tile64: DeepPruner to X'

filename = os.path.join('plots/perturbation_error/kitti2012/', 'kitti_2012_tile64_deeppruner_to_X.png')
legend_names = [
    'DeepPruner -> PSMNet',
    'DeepPruner -> DeepPruner',
    'DeepPruner -> AANet'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [2.5647, 4.5292, 14.4132, 31.9704, 52.0898],
    [1.5086, 2.3739, 9.6819, 27.5951, 60.6098],
    [1.7012, 2.6736, 8.199, 23.3335, 40.1987]
]
y_errors = [
    [1.4057, 3.0852, 8.5128, 9.7007, 11.5504],
    [1.1324, 1.6908, 7.2166, 11.9874, 15.764],
    [1.442, 2.6822, 6.6997, 8.3295, 10.2487]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)


'''
KITTI 2012, Tile64: PSMNet to X
'''
plot_title = 'KITTI 2012, Tile64: PSMNet to X'

filename = os.path.join('plots/perturbation_error/kitti2012/', 'kitti_2012_tile64_psmnet_to_X.png')
legend_names = [
    'PSMNet -> PSMNet',
    'PSMNet -> DeepPruner',
    'PSMNet -> AANet'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [2.5647, 4.8034, 7.7224, 18.6483, 44.9259],
    [1.5086, 2.277, 3.2028, 9.9328, 32.4902],
    [1.7012, 2.2376, 2.8888, 8.0723, 22.3738]
]
y_errors = [
    [1.4057, 3.0607, 5.4755, 8.1368, 11.1443],
    [1.1324, 1.5641, 3.0693, 8.6236, 14.5687],
    [1.442, 1.6378, 2.147, 5.1486, 6.9787]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)


'''
KITTI 2015, Tile64: AANet to X
'''
plot_title = 'KITTI 2015, Tile64: AANet to X'

filename = os.path.join('plots/perturbation_error/kitti2015/', 'kitti_2015_tile64_aanet_to_X.png')
legend_names = [
    'AANet -> PSMNet',
    'AANet -> DeepPruner',
    'AANet -> AANet'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [4.2461, 5.8267, 14.767, 33.5133, 61.6551],
    [1.2752, 1.7088, 7.0071, 23.7899, 52.6561],
    [1.4687, 1.9766, 7.6245, 22.93, 48.4298]
]
y_errors = [
    [2.4059, 3.6107, 10.5028, 14.5481, 14.3948],
    [0.8152, 1.5024, 8.0219, 14.7297, 19.4433],
    [1.6175, 2.3444, 7.3166, 11.9861, 16.8842]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)


'''
KITTI 2015, Tile64: DeepPruner to X
'''
plot_title = 'KITTI 2015, Tile64: DeepPruner to X'

filename = os.path.join('plots/perturbation_error/kitti2015/', 'kitti_2015_tile64_deeppruner_to_X.png')
legend_names = [
    'DeepPruner -> PSMNet',
    'DeepPruner -> DeepPruner',
    'DeepPruner -> AANet'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [4.2461, 6.5704, 19.4328, 34.6456, 64.1793],
    [1.2752, 1.8854, 8.9017, 24.1559, 52.7353],
    [1.4687, 1.8301, 7.1604, 19.317, 31.6396]
]
y_errors = [
    [2.4059, 4.2599, 12.0831, 14.9762, 15.3673],
    [0.8152, 1.4900, 8.1121, 14.4015, 23.6079],
    [1.6175, 1.9456, 6.9521, 11.5247, 12.4734]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)


'''
KITTI 2015, Tile64: PSMNet to X
'''
plot_title = 'KITTI 2015, Tile64: PSMNet to X'

filename = os.path.join('plots/perturbation_error/kitti2015/', 'kitti_2015_tile64_psmnet_to_X.png')
legend_names = [
    'PSMNet -> PSMNet',
    'PSMNet -> DeepPruner',
    'PSMNet -> AANet'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [4.2461, 7.4076, 28.9683, 67.2886, 87.7183],
    [1.2752, 1.8169, 1.7737, 7.1459, 31.1279],
    [1.4687, 1.7788, 2.0752, 6.3324, 19.7222]
]
y_errors = [
    [2.4059, 5.2931, 17.148, 19.3448, 11.5573],
    [0.8152, 1.4314, 1.0968, 7.1585, 18.497],
    [1.6175, 1.9361, 2.3460, 5.6114, 10.5772]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)


'''
KITTI 2015, Comparing Perturbation Sizes: AANet to AANet
'''
plot_title = 'KITTI 2015, Perturbation Size Comparison:\nAANet to AANet'

filename = os.path.join('plots/perturbation_error/kitti2015/', 'kitti_2015_comparing_perturbation_sizes_aanet_to_aanet.png')
legend_names = [
    'Tile16',
    'Tile32',
    'Tile64',
    'Full',
    'Stereopagnosia'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.4687, 2.1786, 6.6459, 16.9776, 34.1414],
    [1.4687, 2.01, 6.7863, 21.2933, 45.4093],
    [1.4687, 1.9766, 7.6245, 22.93, 48.4298],
    [1.4687, 1.6899, 10.0698, 27.1286, 50.1459],
    [1.334, 4.1785, 9.8354, 23.0663, 42.0942]
]
y_errors = [
    [1.6175, 2.4941, 5.1265, 7.1892, 11.9434],
    [1.6175, 2.2657, 6.8021, 11.4263, 14.268],
    [1.6175, 2.3444, 7.3166, 11.9861, 16.8842],
    [1.6175, 1.5813, 9.8252, 12.9208, 15.3348],
    [1.4429, 3.9179, 7.0505, 9.5121, 11.25]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False,
    legend_loc='upper left',
    long_title=True)


'''
KITTI 2015, Comparing Perturbation Sizes: AANet to PSMNet
'''
plot_title = 'KITTI 2015, Perturbation Size Comparison:\nAANet to PSMNet'

filename = os.path.join('plots/perturbation_error/kitti2015/', 'kitti_2015_comparing_perturbation_sizes_aanet_to_psmnet.png')
legend_names = [
    'Tile16',
    'Tile32',
    'Tile64',
    'Full',
    'Stereopagnosia'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [4.2461, 5.649, 13.528, 31.622, 59.8827],
    [4.2461, 5.5768, 14.663, 28.0221, 57.7261],
    [4.2461, 5.8267, 14.767, 33.5133, 61.6551],
    [4.2461, 6.0132, 17.5284, 38.6221, 63.2849],
    [3.2699, 7.6937, 19.9692, 35.9104, 53.5496]
]
y_errors = [
    [2.4059, 3.4047, 7.1911, 10.7205, 15.2079],
    [2.4059, 4.0009, 10.4221, 11.9244, 12.2524],
    [2.4059, 3.6107, 10.5028, 14.5481, 14.3948],
    [2.4059, 4.3008, 12.5845, 14.8398, 15.1377],
    [1.6243, 5.1494, 10.2842, 12.9997, 14.651]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False,
    legend_loc='upper left',
    long_title=True)


'''
KITTI 2015, Comparing Perturbation Sizes: AANet to DeepPruner
'''
plot_title = 'KITTI 2015, Perturbation Size Comparison:\nAANet to DeepPruner'

filename = os.path.join('plots/perturbation_error/kitti2015/', 'kitti_2015_comparing_perturbation_sizes_aanet_to_deeppruner.png')
legend_names = [
    'Tile16',
    'Tile32',
    'Tile64',
    'Full',
    'Stereopagnosia'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.2752, 2.001, 5.0405, 15.5677, 26.9014],
    [1.2752, 1.8377, 6.6374, 18.2013, 42.7941],
    [1.2752, 1.7088, 7.0071, 23.7899, 52.6561],
    [1.2752, 1.3499, 9.0986, 25.1169, 46.1693],
    [1.1481, 1.8445, 4.6762, 12.9083, 30.9755]
]
y_errors = [
    [0.8152, 1.7318, 4.3396, 6.8329, 12.9271],
    [0.8152, 2.0626, 6.8171, 11.6491, 13.482],
    [0.8152, 1.5024, 8.0219, 14.7297, 19.4433],
    [0.8152, 0.8882, 9.9238, 14.1687, 18.5678],
    [0.7146, 1.4171, 3.7663, 8.2971, 12.6008]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False,
    legend_loc='upper left',
    long_title=True)


'''
Generalization to FlyingThings3D
'''
plot_title = 'Generalization to FlyingThings3D'

filename = os.path.join('plots/perturbation_error/flyingthings3d/', 'flyingthings3d_generalization_to_other_datasets.png')
legend_names = [
    'Tile64: AANet -> PSMNet',
    'Full: AANet -> PSMNet',
    'Tile64: AANet -> DeepPruner',
    'Full: AANet -> DeepPruner',
    'Tile64: AANet -> AANet',
    'Full: AANet -> AANet',
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [5.5685, 6.0741, 9.6439, 13.7688, 31.9326],
    [5.5685, 5.7968, 7.6098, 12.1797, 25.3499],
    [6.5264, 6.7316, 9.6106, 13.868, 34.8665],
    [6.5264, 6.6294, 8.1728, 12.0696, 23.2758],
    [5.3526, 6.612, 11.9876, 23.2211, 46.135],
    [5.3526, 5.7364, 9.8962, 19.9932, 36.0869],
]
y_errors = [
    [6.7231, 6.3887, 7.8343, 8.7592, 12.6927],
    [6.7231, 6.8207, 7.5031, 8.7407, 11.6732],
    [5.805, 5.6471, 7.7574, 9.36, 15.5795],
    [5.805, 5.752, 6.7982, 8.5009, 12.4277],
    [5.6142, 6.4044, 8.4688, 10.8376, 12.8868],
    [5.6142, 6.1582, 7.7072, 9.8818, 11.7735],
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)


'''
FlyingThings3D, Tile64: AANet to X
'''
plot_title = 'FlyingThings3D, Tile64: AANet to X'

filename = os.path.join('plots/perturbation_error/flyingthings3d/', 'flyingthings3d_tile64_aanet_to_X.png')
legend_names = [
    'AANet -> PSMNet',
    'AANet -> DeepPruner',
    'AANet -> AANet'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.2697, 1.353, 1.9779, 2.6542, 6.5987],
    [1.2482, 1.2631, 1.7703, 2.4351, 6.8562],
    [1.2981, 1.4878, 2.3348, 4.0531, 9.4717]
]
y_errors = [
    [1.7228, 1.7138, 2.4852, 2.6053, 4.6308],
    [1.9829, 1.6623, 3.2008, 3.0162, 6.2488],
    [2.2925, 2.6374, 2.3401, 3.0362, 4.8831]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='EPE',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=20,
    show_errorbars=False)


'''
FlyingThings3D, Tile64: DeepPruner to X
'''
plot_title = 'FlyingThings3D, Tile64: DeepPruner to X'

filename = os.path.join('plots/perturbation_error/flyingthings3d/', 'flyingthings3d_tile64_deeppruner_to_X.png')
legend_names = [
    'DeepPruner -> PSMNet',
    'DeepPruner -> DeepPruner',
    'DeepPruner -> AANet'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.2697, 1.3821, 1.9439, 3.4048, 14.9213],
    [1.2482, 1.3171, 1.8039, 2.9063, 14.7735],
    [1.2981, 1.4144, 1.7876, 2.9743, 4.4912]
]
y_errors = [
    [1.7228, 2.5242, 2.1884, 3.2665, 16.7652],
    [1.9829, 2.0263, 2.8716, 3.0202, 12.3985],
    [2.2925, 3.4156, 1.7864, 2.2325, 2.8947]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='EPE',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=20,
    show_errorbars=False)


'''
FlyingThings3D, Tile64: PSMNet to X
'''
plot_title = 'FlyingThings3D, Tile64: PSMNet to X'

filename = os.path.join('plots/perturbation_error/flyingthings3d/', 'flyingthings3d_tile64_psmnet_to_X.png')
legend_names = [
    'PSMNet -> PSMNet',
    'PSMNet -> DeepPruner',
    'PSMNet -> AANet'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.2697, 1.3388, 2.2631, 4.0999, 18.875],
    [1.2482, 1.3032, 1.8614, 2.5533, 7.0341],
    [1.2981, 1.3585, 1.49, 1.9068, 3.2496]
]
y_errors = [
    [1.7228, 1.6657, 3.6673, 6.0984, 19.2027],
    [1.9829, 1.9213, 4.459, 4.3175, 8.0537],
    [2.2925, 1.9973, 2.2911, 2.1695, 3.1578]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='EPE',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=20,
    show_errorbars=False)


'''
KITTI 2015: Adversarial Fine-tuning
'''
plot_title = 'KITTI 2015: Fine-tuned Models'

filename = os.path.join('plots/perturbation_error/defenses/', 'kitti_2015_fine_tuned_models.png')
legend_names = [
    'AANet v. Original',
    'AANet v. New',
    'DeepPruner v. Original',
    'DeepPruner v. New',
    'PSMNet v. Original',
    'PSMNet v. New'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.5554, 1.5917, 1.6572, 1.9533, 3.6208],
    [1.5554, 1.5343, 2.1973, 7.5387, 25.539],
    [1.4134, 1.4044, 1.4457, 1.7961, 2.8336],
    [1.4134, 1.5266, 4.4609, 14.9016, 31.5399],
    [1.4419, 1.5872, 1.5009, 1.7062, 2.957],
    [1.4419, 2.3441, 7.9375, 18.1719, 35.7476]
]

y_errors = [
    [1.8929, 1.9772, 2.2139, 2.6529, 3.7609],
    [1.8929, 1.9552, 2.8123, 7.8186, 11.7012],
    [0.8858, 0.8015, 0.858, 1.1071, 2.0822],
    [0.8858, 1.12, 3.8834, 9.8826, 13.0629],
    [1.1792, 1.3361, 1.1623, 1.1278, 1.9328],
    [1.1792, 2.1979, 4.9695, 11.3407, 14.2075]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=100,
    show_errorbars=False)


'''
KITTI 2015: Defending PSMNet
'''
plot_title = 'KITTI 2015: Defending PSMNet'

filename = os.path.join('plots/perturbation_error/defenses/', 'kitti_2015_defending_psmnet.png')
legend_names = [
    'PSMNet',
    'PSMNet Fine-tuned',
    'PSMNet +6 D.C.',
    'PSMNet +25 D.C.',
    'PSMNet +25 D.C. w/ E.M.',
    'PSMNet +25 D.C. w/ E.M., Fine-tuned'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [4.2461, 7.4076, 28.9683, 67.2886, 87.7183],
    [1.4419, 2.3441, 7.9375, 18.1719, 35.7476],
    [1.6947, 3.4942, 14.1743, 34.8571, 66.1015],
    [1.7099, 3.0066, 12.4841, 28.731, 52.0955],
    [1.2945, 2.6431, 6.5245, 17.6915, 39.4659],
    [1.4542, 1.5205, 2.0454, 7.2829, 33.8493]
]

y_errors = [
    [2.4059, 5.2931, 17.148, 19.3448, 11.5573],
    [1.1792, 2.1979, 4.9695, 11.3407, 14.2075],
    [1.0275, 2.8719, 7.6374, 14.7332, 15.8579],
    [1.1256, 2.5254, 7.9448, 11.353, 13.1073],
    [0.8379, 1.1178, 2.6536, 9.6124, 20.7936],
    [0.8596, 0.8829, 1.4142, 5.0246, 17.0127]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=100,
    show_errorbars=False)


'''
KITTI 2015: DeepPruner to X
'''
plot_title = 'KITTI 2015: DeepPruner to X'

filename = os.path.join('plots/perturbation_error/defenses/', 'kitti_2015_deeppruner_to_x_defenses.png')
legend_names = [
    'AANet',
    'DeepPruner',
    'PSMNet',
    'PSMNet +6 D.C.',
    'PSMNet +25 D.C.',
    'PSMNet +25 D.C. w/ E.M.'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.4687, 1.8301, 7.1604, 19.317, 31.6396],
    [1.2752, 1.8854, 8.9017, 24.1559, 52.7353],
    [4.2461, 6.5704, 19.4328, 34.6456, 64.1793],
    [1.6947, 2.5653, 11.5646, 27.2709, 46.2783],
    [1.7099, 2.4633, 9.0197, 20.1896, 33.7108],
    [1.2945, 1.91, 7.096, 17.0624, 30.1855]
]

y_errors = [
    [1.6175, 1.9456, 6.9521, 11.5247, 12.4734],
    [0.8152, 1.49, 8.1121, 14.4015, 23.6079],
    [2.4059, 4.2599, 12.0831, 14.9762, 15.3673],
    [1.0275, 1.9796, 9.1342, 13.2402, 13.2555],
    [1.1256, 1.9524, 8.6121, 12.9713, 13.9854],
    [0.8379, 1.6044, 7.1477, 12.7492, 16.5984]

]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=100,
    show_errorbars=False)


'''
KITTI 2015, Comparing Tile Sizes: AANet to AANet
'''
plot_title = 'KITTI 2015, Tile Size Comparison:\nAANet to AANet'

filename = os.path.join('plots/supp_mat/kitti2015/', 'kitti_2015_comparing_tile_sizes_aanet_to_aanet.png')
legend_names = [
    'Tile16',
    'Tile32',
    'Tile64',
    'Tile128'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.4687, 2.1786, 6.6459, 16.9776, 34.1414],
    [1.4687, 2.01, 6.7863, 21.2933, 45.4093],
    [1.4687, 1.9766, 7.6245, 22.93, 48.4298],
    [1.4687, 2.0621, 9.4713, 24.8834, 49.9605]
]
y_errors = [
    [1.6175, 2.4941, 5.1265, 7.1892, 11.9434],
    [1.6175, 2.2657, 6.8021, 11.4263, 14.268],
    [1.6175, 2.3444, 7.3166, 11.9861, 16.8842],
    [1.6175, 2.2073, 7.9931, 12.495, 16.1708]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False,
    legend_loc='upper left',
    long_title=True)


'''
KITTI 2015, Comparing Tile Sizes: AANet to PSMNet
'''
plot_title = 'KITTI 2015, Tile Size Comparison:\nAANet to PSMNet'

filename = os.path.join('plots/supp_mat/kitti2015/', 'kitti_2015_comparing_tile_sizes_aanet_to_psmnet.png')
legend_names = [
    'Tile16',
    'Tile32',
    'Tile64',
    'Tile128'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [4.2461, 5.649, 13.528, 31.622, 59.8827],
    [4.2461, 5.5768, 14.663, 28.0221, 57.7261],
    [4.2461, 5.8267, 14.767, 33.5133, 61.6551],
    [4.2461, 6.2302, 14.1994, 35.1459, 61.7239]
]
y_errors = [
    [2.4059, 3.4047, 7.1911, 10.7205, 15.2079],
    [2.4059, 4.0009, 10.4221, 11.9244, 12.2524],
    [2.4059, 3.6107, 10.5028, 14.5481, 14.3948],
    [2.4059, 4.1059, 9.135, 14.4131, 15.1804]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False,
    legend_loc='upper left',
    long_title=True)


'''
KITTI 2015, Comparing Tile Sizes: AANet to DeepPruner
'''
plot_title = 'KITTI 2015, Tile Size Comparison:\nAANet to DeepPruner'

filename = os.path.join('plots/supp_mat/kitti2015/', 'kitti_2015_comparing_tile_sizes_aanet_to_deeppruner.png')
legend_names = [
    'Tile16',
    'Tile32',
    'Tile64',
    'Tile128'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.2752, 2.001, 5.0405, 15.5677, 26.9014],
    [1.2752, 1.8377, 6.6374, 18.2013, 42.7941],
    [1.2752, 1.7088, 7.0071, 23.7899, 52.6561],
    [1.2752, 1.4809, 6.9483, 24.8218, 52.3382]
]
y_errors = [
    [0.8152, 1.7318, 4.3396, 6.8329, 12.9271],
    [0.8152, 2.0626, 6.8171, 11.6491, 13.482],
    [0.8152, 1.5024, 8.0219, 14.7297, 19.4433],
    [0.8152, 0.9161, 6.4742, 14.1294, 19.2398]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False,
    legend_loc='upper left',
    long_title=True)


'''
FlyingThings3D, Comparing Tile Sizes
'''
plot_title = 'FlyingThings3D, Tile Size Comparison'

filename = os.path.join('plots/supp_mat/flyingthings3d/', 'flyingthings3d_tile_size_comparison.png')

legend_names = [
    'Tile64: AANet -> PSMNet',
    'Tile128: AANet -> PSMNet',
    'Tile64: AANet -> DeepPruner',
    'Tile128: AANet -> DeepPruner',
    'Tile64: AANet -> AANet',
    'Tile128: AANet -> AANet',
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [5.5685, 6.0741, 9.6439, 13.7688, 31.9326],
    [5.5685, 5.9759, 8.2501, 13.4119, 29.3307],
    [6.5264, 6.7316, 9.6106, 13.868, 34.8665],
    [6.5264, 6.6683, 8.5984, 13.3626, 27.8879],
    [5.3526, 6.612, 11.9876, 23.2211, 46.135],
    [5.3526, 6.0569, 10.454, 22.0406, 40.8442],
]
y_errors = [
    [6.7231, 6.3887, 7.8343, 8.7592, 12.6927],
    [6.7231, 6.6804, 7.2811, 8.4235, 11.6799],
    [5.805, 5.6471, 7.7574, 9.36, 15.5795],
    [5.805, 5.7874, 6.7918, 8.64, 13.6369],
    [5.6142, 6.4044, 8.4688, 10.8376, 12.8868],
    [5.6142, 5.9355, 7.6091, 9.3133, 11.6011],
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)


'''
KITTI 2015: On Deformable Convolution
'''
plot_title = 'KITTI 2015: On Deformable Convolution'

filename = os.path.join('plots/supp_mat/kitti2015/', 'kitti_2015_on_deformable_convolution.png')
legend_names = [
    'PSMNet',
    'PSMNet +25 D.C.',
    'DeepPruner',
    'DeepPruner +25 D.C.',
    'AANet',
    'AANet w/out D.C.'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [4.2461, 7.4076, 28.9683, 67.2886, 87.7183],
    [1.7099, 3.0066, 12.4841, 28.731, 52.0955],
    [1.2752, 1.8854, 8.9017, 24.1559, 52.7353],
    [1.2945, 2.6431, 6.5245, 17.6915, 39.4659],
    [1.4687, 1.9766, 7.6245, 22.93, 48.4298],
    [1.5802, 2.3669, 10.3844, 26.431, 54.3178]
]
y_errors = [
    [2.4059, 5.2931, 17.148, 19.3448, 11.5573],
    [1.1256, 2.5254, 7.9448, 11.353, 13.1073],
    [0.8152, 1.4900, 8.1121, 14.4015, 23.6079],
    [0.8379, 1.1178, 2.6536, 9.6124, 20.7936],
    [1.6175, 2.3444, 7.3166, 11.9861, 16.8842],
    [1.8972, 2.6258, 9.4577, 13.4215, 16.5938]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False,
    legend_loc='upper left',
    long_title=True)


'''
KITTI 2015, Fine-tuned Models: AANet to X
'''
plot_title = 'KITTI 2015, Fine-tuned Models: AANet to X'

filename = os.path.join('plots/supp_mat/finetuned/', 'kitti_2015_fine_tuned_models_aanet_to_x.png')
legend_names = [
    'AANet v. Original',
    'AANet v. New',
    'DeepPruner v. Original',
    'DeepPruner v. New',
    'PSMNet v. Original',
    'PSMNet v. New'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.5554, 1.5917, 1.6572, 1.9533, 3.6208],
    [1.5554, 1.5343, 2.1973, 7.5387, 25.539],
    [1.4134, 1.426, 1.5013, 2.1617, 11.318],
    [1.4134, 1.4711, 1.7864, 4.0782, 19.3857],
    [1.4419, 1.4695, 3.1788, 11.0308, 31.5425],
    [1.4419, 1.6398, 3.4532, 9.4297, 28.1044]
]

y_errors = [
    [1.8929, 1.9772, 2.2139, 2.6529, 3.7609],
    [1.8929, 1.9552, 2.8123, 7.8186, 11.7012],
    [0.8858, 0.9023, 0.94, 1.6046, 10.7047],
    [0.8858, 0.9159, 1.1544, 3.8524, 11.4796],
    [1.1792, 1.1967, 3.6037, 9.388, 14.0726],
    [1.1792, 1.2751, 3.2152, 8.1954, 12.1239]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)


'''
KITTI 2015, Fine-tuned Models: DeepPruner to X
'''
plot_title = 'KITTI 2015, Fine-tuned Models: DeepPruner to X'

filename = os.path.join('plots/supp_mat/finetuned/', 'kitti_2015_fine_tuned_models_deeppruner_to_x.png')
legend_names = [
    'AANet v. Original',
    'AANet v. New',
    'DeepPruner v. Original',
    'DeepPruner v. New',
    'PSMNet v. Original',
    'PSMNet v. New'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.5554, 1.5902, 1.7062, 2.2544, 6.3959],
    [1.5554, 1.5875, 2.3462, 7.6921, 19.3579],
    [1.4134, 1.4044, 1.4457, 1.7961, 2.8336],
    [1.4134, 1.5266, 4.4609, 14.9016, 31.5399],
    [1.4419, 1.5857, 2.8435, 8.0204, 11.4362],
    [1.4419, 1.6003, 4.0344, 14.57, 28.048]
]

y_errors = [
    [1.8929, 1.9734, 2.0102, 3.102, 5.4725],
    [1.8929, 1.8871, 2.6544, 6.1278, 11.8868],
    [0.8858, 0.8015, 0.858, 1.1071, 2.0822],
    [0.8858, 1.12, 3.8834, 9.8826, 13.0629],
    [1.1792, 1.4202, 2.9295, 7.2035, 8.965],
    [1.1792, 1.427, 3.671, 9.6648, 12.632]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)


'''
KITTI 2015, Fine-tuned Models: PSMNet to X
'''
plot_title = 'KITTI 2015, Fine-tuned Models: PSMNet to X'

filename = os.path.join('plots/supp_mat/finetuned/', 'kitti_2015_fine_tuned_models_psmnet_to_x.png')
legend_names = [
    'AANet v. Original',
    'AANet v. New',
    'DeepPruner v. Original',
    'DeepPruner v. New',
    'PSMNet v. Original',
    'PSMNet v. New'
]

x_values = [0.00, 0.002, 0.005, 0.010, 0.020]

y_values = [
    [1.5554, 1.5887, 1.6794, 2.342, 7.3723],
    [1.5554, 1.6806, 2.7891, 9.4403, 21.3492],
    [1.4134, 1.4455, 1.4693, 1.6596, 2.9687],
    [1.4134, 1.4646, 2.851, 7.7588, 19.0568],
    [1.4419, 1.5872, 1.5009, 1.7062, 2.957],
    [1.4419, 2.3441, 7.9375, 18.1719, 35.7476]
]

y_errors = [
    [1.8929, 1.9315, 2.2402, 2.9462, 6.296],
    [1.8929, 2.0852, 2.7831, 7.4442, 11.5261],
    [0.8858, 0.8382, 0.8147, 0.8879, 1.8523],
    [0.8858, 0.8451, 2.5808, 5.4032, 10.8212],
    [1.1792, 1.3361, 1.1623, 1.1278, 1.9328],
    [1.1792, 2.1979, 4.9695, 11.3407, 14.2075]
]

create_plot(
    plot_title=plot_title,
    x_values=x_values,
    y_values=y_values,
    y_errors=y_errors,
    legend_names=legend_names,
    filename=filename,
    x_label=r'$\epsilon$',
    y_label='D1-all error (%)',
    x_lim_min=0.0,
    x_lim_max=None,
    y_lim_min=0.0,
    y_lim_max=90,
    show_errorbars=False)
