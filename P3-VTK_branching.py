# Use vtp files: pore.diameter, throat.diameter, throat.length
# Use porespy data: pore size distribution

import os

import matplotlib.pyplot as plt  # Hunter J., 2007
import matplotlib
import matplotlib.patches as patches

import pandas as pd  # McKinney, 2010
import numpy as np  # VanDerWalt S. et al., 2011

import openpnm as op
from itertools import product as iterproduct

DISPLAY = True

# Figure
size = 8
matplotlib.rcParams.update({'font.size': size})
h_fig = 3  # in
w_fig = 6  # in
cmaps = plt.cm.get_cmap('viridis')

scale = 40 / 1000
voxel_size = 40  # um

# Data
data_dir = '/mnt/data/UAF-data/paper/4/'
fig_dir = os.path.join('/home/megavolts/UAF/paper/Chapter4/figures/pathway')

# Sample list:
samples = {}
vtk_dir = os.path.join(data_dir, 'VTK_pathway')
for file in os.listdir(vtk_dir):
    if file.endswith(".vtp"):
        samples[file.split('-bin')[0]] = file

def plot_throat(Lrow, data_df, ax=None, pn=None, throat_colors=None, scale=None, pore=True, Diameter=True):
    """
    :param Lrow:
    :param data_df:
    :param ax:
    :param pn:
    :param throat_colors:
    :param scale:
    :return:
    """

    # Create dictionnary name:
    pi_dict = {Pi: ii for ii, Pi in enumerate(data_df.Pi.unique())}
    po_dict = {Po: ii+len(pi_dict) for ii, Po in enumerate(data_df.Po.unique())}

    from itertools import combinations_with_replacement as cwr
    import string
    alphabet = string.ascii_lowercase
    length = np.ceil(len(data_df.Po.unique())/26).astype(int)
    po_names = ["".join(comb) for comb in cwr(alphabet, length)]
    po_dict = {Po: po_names[ii] for ii, Po in enumerate(data_df.Po.unique())}

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots(1, 1)

    # plot branching if needed
    L_throat = 0
    if len(Lrow) > 0:
        row = Lrow[-1]
        L_throat = data_df.loc[data_df.index == row[0], range(0, row[2])]
        if L_throat.empty:
            L_throat = 0
        else:
            L_throat = L_throat.apply(lambda _n: pn['throat.total_length'][int(_n)]).sum()*scale
        if row[0] is not None:
            ax.plot([L_throat, L_throat], [row[4], row[5]], 'k')
        _ts = row[2]
        _y = row[5]
    else:
        _ts = 0
        _y = 0

    path_id = Lrow[-1][1]
    while data_df.loc[data_df.index == path_id, _ts].notna().values[0]:
        _tn = data_df.loc[data_df.index == path_id, _ts].values[0].astype(int)  #throat number
        _L = pn['throat.total_length'][_tn] * scale  # throat length
        _R = pn['throat.diameter'][_tn] / 2 * scale  # throat radius
        if Diameter:
            _R = 0.5 * _R
            ax.fill_between([L_throat, L_throat + _L], [_y - _R, _y- _R], [_y + _R, _y + _R], color=throat_colors[_tn])
        else:
            ax.fill_between([L_throat, L_throat + _L], [_y, _y], [_y + _R, _y + _R], color=throat_colors[_tn])
        ax.set_yticks(np.arange(0, len(Lrow), 1)+Lrow[0][5])  # y + offset
        ax.set_yticklabels(np.array(Lrow)[:, 1])
        L_throat += _L
        _ts += 1
        if _ts not in data_df.columns:
            break

    if pore:
        if len(Lrow) == 1:
            Pi = data_df.loc[data_df.index == path_id, 'Pi'].astype(int).values[0]
            yPi = Lrow[np.where(np.array(Lrow)[:, 1] == path_id)[0][0]][5]
            xPi = -0.2

            ax.text(xPi, yPi, pi_dict[Pi], horizontalalignment='right', verticalalignment='center')

        Po = data_df.loc[data_df.index == path_id, 'Po'].astype(int).values[0]
        yPo = Lrow[np.where(np.array(Lrow)[:, 1] == path_id)[0][0]][5]
        xPo = L_throat + 0.2
        ax.text(xPo, yPo, po_dict[Po], verticalalignment='center')
    return ax


def branch(trunk_df, p_trunk, Lrow=[], ax=None, pn=None, throat_colors=None, scale=None, update=False, Diameter=True):
    # Look for branch with longest common trunk, originating in the same pore
    branch_df = trunk_df.copy()
    branch_df_prev = branch_df.copy()

    Lrow_ar = np.array(Lrow)
    if len(Lrow) > 0:
        discovered = Lrow_ar[:, 1]
        y_branch = max(Lrow_ar[:, 5]) + 1
    else:
        discovered = []
        y_branch = 0

    throat_l = 0
    while not branch_df.loc[~branch_df.index.isin(discovered)].empty:
        branch_df_prev = branch_df.copy()  # status fo branch_df on previous iteration
        throat_l_id = branch_df.loc[branch_df.index == p_trunk, throat_l].astype(int).values[0]
        branch_df = branch_df.loc[branch_df[throat_l] == throat_l_id]
        throat_l += 1
    branch_df = branch_df_prev

    # Classify by length
    Nbranch = len(branch_df.loc[~branch_df.index.isin(np.array(Lrow)[:, 1])])
    while Nbranch > 0:
        p_branch = branch_df.loc[~branch_df.index.isin(np.array(Lrow)[:, 1])].sort_values(by='L').index.values[0]
        p_throat_end = trunk_df.loc[p_branch, columns].notnull().sum()
        def get_y_trunk(_p_trunk, throat_l, Lrow_ar):
            _p_trunk0 = _p_trunk
            if throat_l-1 > Lrow_ar[np.where(Lrow_ar[:, 1] == _p_trunk)[0][0], 2]:
                return _p_trunk0
            if Lrow_ar[np.where(Lrow_ar[:, 1] == _p_trunk)[0][0], 0] is None:
                return _p_trunk0
            else:
                _p_trunk = Lrow_ar[np.where(Lrow_ar[:, 1] == _p_trunk)[0][0], 0]
                _p_t = Lrow_ar[np.where(Lrow_ar[:, 1] == _p_trunk)[0][0], 2]  # throat on new trunk
                while throat_l-1 <= _p_t:
                    return get_y_trunk(_p_trunk, throat_l, Lrow_ar)
            return _p_trunk
        _p_trunk = get_y_trunk(p_trunk, throat_l, Lrow_ar)
        y_trunk = Lrow_ar[np.where(Lrow_ar[:, 1] == _p_trunk)[0][-1], -1]
        Lrow.append([_p_trunk, p_branch, throat_l-1, p_throat_end, y_trunk, y_branch])  # trunk, branch, throat level for trunk, throat level for branch, y_trunk, y_branch
        # plot branch
        ax = plot_throat(Lrow, trunk_df, ax=ax, pn=pn, throat_colors=throat_colors, scale=scale)
        if update:
            plt.ion()
            plt.draw()

        # For testing recursive function
        Lrow, ax = branch(trunk_df, p_branch, Lrow=Lrow, ax=ax, pn=pn, throat_colors=throat_colors, scale=scale)
        Nbranch = len(branch_df.loc[~branch_df.index.isin(np.array(Lrow)[:, 1])])
    return Lrow, ax


#%% produce plot
for sample in samples.keys():
     # Load data
    SNOW_fn = samples[sample]
    print(sample, SNOW_fn)
    SNOW_fp = os.path.join(vtk_dir, SNOW_fn)
    net = op.io.VTK.load(SNOW_fp)
    pn = net.network

    # Extract data
    data = []
    max_n_throat = 0
    throat_ids = []
    for b_pore, t_pore in iterproduct(pn.pores(labels=['bottom']), pn.pores(labels=['top'])):
        pathway = op.topotools.find_path(pn, [b_pore, t_pore])
        if pathway['throats'][0].__len__() > 0:
            throats = pathway['throats'][0]
            throat_ids.extend(throats)
            if len(throats) > max_n_throat:
                max_n_throat = len(throats)
            data.append([b_pore, t_pore, pathway['pores'][0], pathway['throats'][0]])

    max_n_pore = max_n_throat + 1
    throat_ids = np.unique(throat_ids)
    throat_colors = {tn: cmaps(ii_t / len(throat_ids)) for ii_t, tn in enumerate(throat_ids)}

    throat_array = np.nan * np.ones([len(data), max_n_throat])
    throat_array_l = np.nan * np.ones([len(data), max_n_throat])
    Pi = []
    Po = []
    for ii_p, pathway in enumerate(data):
        throat_array[ii_p, 0:len(pathway[3])] = pathway[3]
        Pi.append(pathway[0])
        Po.append(pathway[1])
        for ii_t in range(0, len(pathway[3])):
            throat_array_l[ii_p, ii_t] = pn['throat.total_length'][int(throat_array[ii_p, ii_t])]
    pathway_length = np.atleast_2d(np.nansum(throat_array_l, axis=1))

    Pi = np.atleast_2d(Pi).transpose()
    Po = np.atleast_2d(Po).transpose()
    throat_data = np.hstack([throat_array, pathway_length.transpose(), Pi, Po])
    columns = [ii for ii in range(0, max_n_throat)]
    columns_L = columns + ['L', 'Pi', 'Po']
    throat_df = pd.DataFrame(throat_data, columns=columns_L)

    h_sample = (int(SNOW_fn.split('_')[-1].split('-')[0]) - 75) * 40 / 1000

    fig = plt.figure()
    ax = fig.subplots(1, 1)
    Lrow_all = []
    ax.set_xlabel('Distance along the connected pathway,\n'
                  'measured from sample bottom (mm)')
    ax.set_title(sample + ' h$_{sample}$=' + str('%.1f (mm)' % h_sample))

    for Pi in throat_df.sort_values(by='L')['Pi'].unique():
        p_shortest_id = throat_df.loc[throat_df.Pi == Pi].sort_values(by='L').index.values[0]
        print(int(Pi), p_shortest_id)
        # Plot shortest pathway
        p_shortest_Nt = throat_df.loc[p_shortest_id, columns].notnull().sum()
        if len(Lrow_all) == 0:
            Lrow = [[None, p_shortest_id, 0, p_shortest_Nt, 0, 0]]
        else:
            y0 = max(np.array(Lrow)[:, 5])+1
            Lrow = [[None, p_shortest_id, 0, p_shortest_Nt, 0, y0]]
        ax = plot_throat(Lrow, throat_df, ax=ax, pn=pn, throat_colors=throat_colors, scale=scale, Diameter=True)
        plt.draw()
        p_pore_ids = throat_df.loc[throat_df.Pi == Pi].index.values
        p_trunk = p_shortest_id

        trunk_df = throat_df.loc[p_pore_ids].copy()
        Lrow, ax = branch(trunk_df, p_trunk, Lrow=Lrow, ax=ax, pn=pn, throat_colors=throat_colors,
                          scale=scale, Diameter=True)
        Lrow_all.extend(Lrow)

    # Label
    #y_label = np.array(Lrow_all)[:, 1]
    #ax.set_yticks(np.arange(0, len(y_label), 1))
    #ax.set_yticklabels(y_label)

    xlim = ax.get_xlim()
    ax.set_xlim([0, 40])

    y_max = np.max(np.array(Lrow_all)[:, 5])
    y_lim = [-2, y_max+2]
    ax.plot([h_sample, h_sample], y_lim, ls=':', c=(0.5, 0.5, 0.5))
    ax.set_ylim(y_lim)

    print(sample, y_max)

    # Scale

    L_throat = throat_df.loc[throat_df.index == Lrow_all[0][1], range(0, Lrow_all[0][3])]
    L_throat = L_throat.apply(lambda _n: pn['throat.total_length'][int(_n)]).sum() * scale
    x0 = L_throat + 4  # offset 2 milimeers to the right
    rect = patches.Rectangle((x0, -0.4), 1, 1, lw=0, facecolor='k')
    ax.add_patch(rect)
    ax.text(x0+1.5, 0, '1 (mm)', verticalalignment='center')

    # Set xtick
    NTmax = np.max(np.array(Lrow_all)[:, 3])
    NTmax_pid = np.where(np.array(Lrow_all)[:, 3] == NTmax)[0][0]
    L_max = throat_df.loc[throat_df.index == NTmax_pid, range(0, NTmax)].dropna(axis=1)
    L_max = L_max.apply(lambda _n: pn['throat.total_length'][int(_n)]).sum()*scale
    xticks = np.arange(0, L_max+2.5, 5)
    xticks = ax.set_xticks(xticks)

    # Hide
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(labelleft=False, left=False)
    ax.set_aspect('equal', adjustable='box')
    fig.set_figwidth(6)
    fig.set_figheight(1+len(data)*0.5)
    plt.savefig(os.path.join(fig_dir, sample + '-ps.png'), dpi=500)
    plt.show()

