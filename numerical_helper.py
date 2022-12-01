import numpy as np
from matplotlib.cm import viridis as cmap
import matplotlib.pyplot as plt
import pysic

try:
    from project_helper import *
except ImportError:
    import sys
    if '/home/megavolts/git/SimpleOilModel' in sys.path:
        raise
    sys.path.append('/home/megavolts/git/SimpleOilModel')
    from project_helper import *

def nearest_interp(xi, x, y):
    xi = np.array(xi)
    x = np.array(x)
    y = np.array(y)
    idx = np.abs(x - xi[:, None])
    return y[idx.argmin(axis=1)]

# Tortuosity data from Winter granualar and columnar
def tortuosity(N, texture='granular'):
        tau_dict = {'granular':np.array([1.39993489, 1.23344433, 1.48926926, 2.20221257, 1.10855222,
           1.76314127, 1.39246857, 1.05618429, 1.03011024, 1.02138197,
           1.36508071, 1.19056833, 1.15145993, 1.18594146, 1.16133761,
           1.67786026, 1.5766089 , 2.57046652, 1.53577507, 1.22512555,
           1.0612967 , 2.50943327, 1.15764403, 1.10653043, 1.83399332,
           3.37539339, 1.54267502, 2.05272651, 1.31681573, 1.12150037,
           1.10836124, 1.70066142, 1.41111255, 1.36437559, 1.13010406,
           1.81235313, 1.23472726, 1.29396462, 1.18125248, 1.1750654 ,
           1.37240982, 1.20658469, 1.24340129, 1.20299208, 1.17949045,
           1.3350879 , 1.14031529, 1.12983656, 1.39201677, 1.17935741,
           1.25159335, 1.12243056, 1.22740078, 1.17327547, 1.1498493 ,
           1.22381759, 1.18097615, 1.24117112, 1.32126176, 1.15329361,
           1.16412842, 1.42632127, 1.29992712, 1.19248629, 1.31752765,
           1.09446871, 1.21202731, 1.23496449, 1.32010889, 1.2639159 ,
           1.1473124 , 1.38417435, 1.30039549, 1.11877012, 1.54807472,
           1.40581191, 1.12757838, 1.11788487, 1.55356932, 1.18625379,
           1.09892821, 1.10047483, 1.06075704, 1.22529125, 1.12370896,
           1.20937777, 1.23226345, 1.16172349, 1.18790412, 1.26397204,
           1.28303051, 1.19683433, 1.34947073, 1.23248887, 1.25172746,
           1.23796463, 1.10233772, 1.64187682, 2.11275744, 1.45640516,
           1.35173583, 1.25572276, 1.33834732, 1.15598071, 1.14534187,
           1.20165718, 1.20023048, 1.15806735, 1.11650646, 1.06232715,
           1.07441103, 1.03885496, 2.29863191, 1.18094003, 1.04141521,
           1.06595969, 1.15776086, 1.09852004, 1.09265399, 1.0235554 ,
           1.02750182, 1.03846991, 1.11563909, 1.11692429, 1.10189807,
           1.12882113, 1.09861386, 1.20460725, 1.10649383, 1.12242746,
           1.57543969, 1.25195885, 1.05212688, 1.03586066, 1.04685235,
           1.12349296, 1.0758239 , 1.13123822, 1.92416823, 1.23559952,
           1.06077218, 1.06527448, 1.21199369, 1.13727117]),
                  'columnar': np.array([1.03528392, 1.2653271 , 1.06601834, 1.03100038, 1.09156609,
           1.05788922, 1.04239261, 1.08230126, 1.05718791, 1.11408985,
           1.07491434, 1.10996246, 1.13983011, 1.03493249, 1.24646759,
           1.02439475, 1.09795511, 1.06137204, 1.08054602, 1.03729653,
           1.05761707, 1.01298547, 1.02246737, 1.02835763, 1.02374983,
           1.11100781, 1.08845437, 1.06495833, 1.08502352, 1.18052697,
           1.00971675, 1.01540279, 1.02704358, 1.06573319, 1.01644516,
           1.01322091, 1.03521371, 1.03799593, 1.03385246, 1.02440965,
           1.03005993, 1.02625966, 1.01061606, 1.01967394, 1.01875615,
           1.02408636, 1.01163125, 1.01451564, 1.00769067, 1.01490879,
           1.03469062, 1.07179558, 1.00983787, 1.01603222, 1.03218794,
           1.01367486, 1.01822758, 1.00748217, 1.02862787, 1.01384461,
           1.03565943, 1.00978863, 1.05455613, 1.24064493, 1.0555737,
           1.01207387, 1.0143925 , 1.03499925, 1.01108873, 1.01289666,
           1.01529682, 1.03601944])}
        N_tau = np.random.randint(0, len(tau_dict[texture]), N)
        tau = tau_dict[texture][N_tau]
        return tau


def radius(N=100, texture='columnar'):
    if texture == 'granular':
        mu = 0.025
        sigma = 1.27
        antilog = 0.0240
    elif texture == 'columnar':
        mu =0.013
        sigma = 0.85
        antilog = 0.0132

    from scipy.stats import lognorm
    # import numpy as np
    # import matplotlib.pyplot as plt

    A = lognorm.rvs(sigma, size=N)
    A = A * np.exp(mu)
    R = np.sqrt(A/np.pi)

    # hx, hy, _ = plt.hist(R, bins=50, color="lightblue")
    # plt.ylim(0.0, max(hx) + 0.05)
    # plt.xscale('log')
    # plt.show()

    R = R*1e-3  # in m
    return R


def generate_TS_porespace(porespace, os_dict):

    if os_dict['config']['SI_PHYSIC'] == 'override':
        porespace['temperature'] = os_dict['mat']['si']['T']
    elif not os_dict['config']['SI_PHYSIC']:
        porespace['temperature'] = porespace.temperature.mean()

    if os_dict['config']['SI_PHYSIC'] == 'override':
        porespace['salinity'] = os_dict['mat']['si']['S']
    elif not os_dict['config']['SI_PHYSIC']:
        porespace['salinity'] = porespace.salinity.mean()

    porespace['texture'] = 'c'
    if os_dict['config']['SI_STRAT']:
        if os_dict['BC']['HG'] >0:
            HGC = os_dict['BC']['HG']
        else:
            HGC = os_dict['BC']['HI'] - os_dict['BC']['HG']
        porespace.loc[porespace.h >= HGC, 'texture'] = 'g'

    if os_dict['config']['TORT'] == 'override':
        if os_dict['BC']['HG'] > 0:
            HGC = os_dict['BC']['HG']
        else:
            HGC = os_dict['BC']['HI'] - os_dict['BC']['HG']
        porespace.loc[porespace.h >= HGC, 'texture'] = 'g'
        porespace = porespace.assign(tau=os_dict['mat']['si']['tau_c'])
        porespace.loc[porespace.texture == 'g', 'tau'] = os_dict['mat']['si']['tau_g']
        porespace['texture'] = 'c'
    elif os_dict['config']['TORT']:
        porespace = porespace.assign(tau=tortuosity(len(porespace), 'columnar'))
        if os_dict['config']['SI_STRAT']:
            porespace.loc[porespace.texture == 'g', 'tau'] = tortuosity(len(porespace.loc[porespace.texture == 'g', 'tau']), 'granular')
    else:
        porespace = porespace.assign(tau=1)

    if os_dict['config']['R'] == True:
        porespace = porespace.assign(r=radius(porespace.__len__()))
        porespace.loc[porespace.texture == 'g', 'r'] = radius(porespace.loc[porespace.texture == 'g', 'r'].size, texture='granular')
    else:
        porespace['r'] = os_dict['config']['R']

    # Compute physical properties
    porespace['S_b'] = pysic.property.brine.salinity(porespace['temperature'], extend_t_0=True)
    porespace['rho_b'] = pysic.property.brine.density(porespace['temperature'], extend_t_0=True)
    porespace['mu_b'] = pysic.property.brine.viscosity(porespace['S_b'], porespace['temperature'], override_s=True)

    porespace['rho_o'] = rho_oil(porespace['temperature'])
    porespace['mu_o'] = mu_oil(porespace['temperature'])

    if os_dict['cst']['MU_CST']:
        porespace['mu_b'] = pysic.property.brine.viscosity(os_dict['mat']['b']['S'], os_dict['mat']['b']['Ti'], override_s=True)[0]
        porespace['mu_o'] = mu_oil(os_dict['mat']['o']['Ti'])
    if os_dict['cst']['RHO_CST']:
        porespace['rho_b'] = pysic.property.brine.density(os_dict['mat']['b']['Ti'], extend_t_0=True)[0]
        porespace['rho_o'] = rho_oil(os_dict['mat']['o']['Ti'])
    if os_dict['cst']['OIL_CST']:
        porespace['mu_o'] = mu_oil(os_dict['mat']['o']['Ti'])
        porespace['rho_o'] = rho_oil(os_dict['mat']['o']['Ti'])
    if os_dict['cst']['BRINE_CST']:
        porespace['rho_b'] = pysic.property.brine.density(os_dict['mat']['b']['Ti'], extend_t_0=True)[0]
        porespace['mu_b'] = pysic.property.brine.viscosity(os_dict['mat']['b']['S'], os_dict['mat']['b']['Ti'], override_s=True)[0]

    porespace['vf_b'] = pysic.property.si.brine_volume_fraction(porespace['salinity'], porespace['temperature'])
    porespace['kv'] = pysic.property.si.permeability_from_porosity(porespace['vf_b'])
    porespace['kh1'] = 0.1 * porespace['kv']
    porespace['kh2'] = 0.01 * porespace['kv']

    # Fix permeability for granular ice, 0 at the bottom
    if os_dict['config']['SI_STRAT']:
        porespace.loc[porespace.texture == 'g', 'kv'] = 0.1 * porespace.loc[porespace.texture == 'g', 'kv']
        porespace.loc[porespace.texture == 'g', 'kh1'] = 0.1 * porespace.loc[porespace.texture == 'g', 'kv']
        porespace.loc[porespace.texture == 'g', 'kh2'] = 0.1 * porespace.loc[porespace.texture == 'g', 'kv']

    # Pore space data are reference from the top. We inverse to the bottom:
    porespace['l'] = porespace['dh']*porespace['tau']  # length: length of channel
    porespace['alpha'] = porespace[['dh', 'l']].apply(lambda x: np.arccos(x[0]/x[1] if x[1] > 0 else np.nan), axis=1)  # alpha : angle of channel with vertical
    porespace['V'] = porespace[['r', 'l']].apply(lambda x: cell_volume(x), axis=1)

    return porespace


def plot_porespace(porespace, ax=None, figsize=(10, 4)):
    HI = porespace.h.max()

    if ax is None or len(ax) != 7:
        fig, ax = plt.subplots(1, 7, sharey=True, figsize=figsize)
    ax[0].plot(porespace.temperature, porespace.h, c=cmap(0.1), label='temperature')

    ax[1].step(porespace.salinity, porespace.h, c=cmap(0.1), label="S$_{si}$")
    ax1 = ax[1].twiny()
    ax1.step(porespace.S_b, porespace.h, c=cmap(0.5), label="S$_b$")

    ax[2].step(porespace.mu_o, porespace.h, c=cmap(0.95), label="$\mu_o$")
    ax[2].step(porespace.mu_b, porespace.h, c=cmap(0.5), label='$\mu_b$')
    ax[2].set_xlim([1e-3, 1])
    ax[2].set_xticks([1e-3, 1e-1])
    ax[2].set_xticklabels([1e-2, 5e-1])
    ax[2].set_xscale('log')


    ax[3].step(porespace.rho_o, porespace.h, c=cmap(0.95), label='$\\rho_o$')
    ax[3].step(porespace.rho_b, porespace.h, c=cmap(0.5), label='$\\rho_b$')
    ax[4].step(porespace.vf_b * 100, porespace.h, c=cmap(0.5), label='$V_{f,b}$')

    ax[5].step(porespace.kv, porespace.h, c=cmap(0.1), label='$k_v$')
    ax[5].step(porespace.kh1, porespace.h, '--', c=cmap(0.1), label='$k_{h1}}$')
    ax[5].step(porespace.kh2, porespace.h, ':', c=cmap(0.1), label='$k_{h2}}$')
    ax[5].set_xlim([1e-15, 1e-9])
    ax[5].set_xticks([1e-15, 1e-9])
    ax[5].set_xticklabels([1e-14, 1e-9])
    ax[5].set_xscale('log')

    ax[6].step(porespace.r, porespace.h, c=cmap(0.1), label="r")
    ax[6].set_xlim([1e-4, 1e-2])
    ax6 = ax[6].twiny()
    ax6.step(porespace.tau, porespace.h, c=cmap(0.5), label="$\\tau$")
    ax6.set_xlim([0.95, 3])

    ax[0].set_ylim([HI, 0])
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_xlabel('Temperature\n(°C)')
    ax[0].tick_params(bottom=True, labelbottom=True, top=True, labeltop=True)

    ax[1].set_xlabel('Salinity\n(‰)')
    h_, l_ = ax[1].get_legend_handles_labels()
    h2_, l2_ = ax1.get_legend_handles_labels()
    h_.extend(h2_)
    l_.extend(l2_)
    ax[1].legend(h_, l_, frameon=False)
    ax1.spines['top'].set_color(cmap(0.5))
    ax1.tick_params(axis='x', colors=cmap(0.5))
    ax[1].spines['top'].set_color(cmap(0.1))
    ax[1].tick_params(axis='x', colors=cmap(0.1))

    ax[2].set_xlabel('Viscosity\n(Pa s)')
    ax[2].tick_params(bottom=True, labelbottom=True, top=True, labeltop=True)
    ax[2].legend(frameon=False)


    ax[3].set_xlabel('Density\n(kg m$^{-3}$)')
    ax[3].tick_params(bottom=True, labelbottom=True, top=True, labeltop=True)
    ax[3].legend(frameon=False)
    ax[3].set_xticks([900, 1100])
    ax[3].set_xticklabels([900, 1100])

    ax[4].set_xlabel('Brine volume\nfraction (%)')
    ax[4].tick_params(bottom=True, labelbottom=True, top=True, labeltop=True)
    ax[4].legend(frameon=False)

    ax[5].set_xlabel('Permeability\n(m$^2$)')
    ax[5].tick_params(bottom=True, labelbottom=True, top=True, labeltop=True)
    ax[5].legend(frameon=False)

    ax[6].set_xlabel('Throat\n(m)')
    ax6.set_xlabel('Tortuosity\n')
    h_, l_ = ax[6].get_legend_handles_labels()
    h2_, l2_ = ax6.get_legend_handles_labels()
    h_.extend(h2_)
    l_.extend(l2_)
    ax[6].legend(h_, l_, frameon=False)
    ax6.spines['top'].set_color(cmap(0.5))
    ax6.tick_params(axis='x', colors=cmap(0.5))
    ax[6].spines['top'].set_color(cmap(0.1))
    ax[6].tick_params(axis='x', colors=cmap(0.1))
    ax[6].set_xscale('log')

    plt.ylim([0, 1])
    plt.tight_layout()

    return ax


def brine_hyraulic_head_dm(porespace, os_dict):
    """

    :param porespace: pd.DataFrame()
        Porespace definition including. a least the height, h, and the radius 'r' and the brine density 'rho_b' as function
        of the height.
    :param GAMMA_b: 'float'
        Surface tension in Pa m. Default surface tension of brine on ice in air
    :param THETA_b:
        Contact angle in degree. Default contact angle of brine on ice in air
    :return:
        Brine hydraulic head in the porespace
    """
    hi = porespace.h.max()

    data_ = porespace[['dh', 'h', 'l', 'r', 'rho_b', 'alpha']]

    # Height of brine from buoyancy
    rho_sw = float(pysic.property.sw.density_p0(os_dict['mat']['sw']['S'], os_dict['mat']['sw']['T'])[0])
    data_ = data_.assign(hd_b=data_[['h', 'rho_b']].apply(lambda x: rho_sw / x[1] * x[0], axis=1))
    data_ = data_.loc[data_.index >= data_.loc[data_.h <= os_dict['BC']['HD']].index.max()].dropna()

    # Compute hydraulic head of brine
    data_ = data_.assign(pcb=data_[['r', 'rho_b']].apply(lambda x: 2*GAMMA_b*np.cos(np.deg2rad(THETA_b))/(x[0] * x[1] * g), axis=1))
    pcb_h0 = data_.iloc[0].pcb
    hd_b0 = data_.iloc[0].hd_b
    data_ = data_.loc[~(data_.hd_b >= data_.hd_b.min() + pcb_h0)]
    data_ = data_.assign(dpcb=data_[['pcb', 'hd_b']].apply(lambda x: -(x[1] - hd_b0) + x[0], axis=1))
    if (data_.index < data_.loc[data_.dpcb < 0].index.max()).any():
        data_ = data_.loc[data_.index < data_.loc[data_.dpcb < 0].index.max(skipna=True)]
    hb = data_.hd_b.max() + data_['dpcb'].iloc[-1]

    ii = porespace.loc[porespace.h < hb].index.max()
    vf_bc = (hb - porespace.loc[porespace.index == ii, 'h'].values[0]) / porespace.loc[porespace.index == ii, 'dh'].values[0]

    if hb > hi:
        hb = hi
        vf_bc = 1
    return hb, ii, vf_bc


def p0(hr, os_dict):
    """
    Return pressure at the bottom of the channel z=0, as function of the oil lens thickness

    :param hr: 'float'
        Oil lens thickness in m
    :return: 'float'
        Pressure at the bottom of the channel z=0
    """
    rho_o_R = rho_oil(os_dict['mat']['o']['Tsw'])  # oil density in sea water
    rho_sw = float(pysic.property.sw.density_p0(os_dict['mat']['sw']['S'], os_dict['mat']['sw']['T'])[0])
    p_0 = rho_sw * g * os_dict['BC']['HD'] + (rho_sw - rho_o_R) * g * hr
    return p_0


def pc_b(r_hb):
    """"
    Return capillary pressure for brine

    :param r_hb: 'float'
        Radius at top of the channel in m
    :return: 'float'
        Capillary pressure of brine
    """
    p_hb = 2 * GAMMA_b * np.cos(np.deg2rad(GAMMA_b)) / r_hb
    return p_hb


def pc_o(r_hb):
    """"
    Return capillary pressure for oil

    :param r_hb: 'float'
        Radius at top of the channel in m
    :return: 'float'
        Capillary pressure of oil
    """
    p_hb = 2 * GAMMA_o * np.cos(np.deg2rad(GAMMA_o)) / r_hb
    return p_hb


def cell_volume(x):
    """
    Return the cell volume or fraction volume as function of the radius (r), length (dh) and fraction (f)
    :param x:
    :return:
    """
    if len(x) == 2:
        r = x[0]
        l = x[1]
        f = 1
    elif len(x) == 3:
        r = x[0]
        l = x[1]
        f = x[2]
    return np.pi * r ** 2 * l * f


def p_rho_i_f(x):
    """
    Return the pressure term driven by buoyancy as function of the vertical extent (dh), density (rho), and fraction of
    the occupied cell (f)

    :param x:
    :return:
    """
    if len(x) == 2:
        rho = x[0]
        dh = x[1]
        f = 1
    elif len(x) == 3:
        rho = x[0]
        dh = x[1]
        f = x[2]
    return rho * dh * g * f


def q_mu_i_f(x):
    """
    Return the flow as function of viscosity (mu), radius (r), length (l) and fraction (f)

    :param x:
    :return:
    """
    if len(x) == 3:
        mu = x[0]
        r = x[1]
        l = x[2]
        f = 1
    elif len(x) == 4:
        mu = x[0]
        r = x[1]
        l = x[2]
        f = x[3]
    return 8*mu / (np.pi*r**4) * (l * f)


def fraction_cell(data, phase):
    data = np.atleast_2d(data)
    if phase == 'brine' or phase == 'b':
        fraction = np.abs(data[0, :] * (data[0, :] < 0))

        # Look for oil/brine mixed cell (0 < f < 1)
        ii_cell = np.argwhere((0 < data[0, :]) & (data[0, :] < 1))
        if len(ii_cell) > 0:
            for ii_ in ii_cell[0]:
                fraction[ii_] = 1 - np.abs(np.abs(data[0, ii_] * (data[0, ii_] > 0)))
    elif phase == 'oil' or phase == 'o':
        fraction = np.abs(data[0, :] * (data[0, :] > 0))
    return fraction


def fraction_cell_old(data, phase):
    data = np.atleast_2d(data)
    if phase == 'brine' or phase == 'b':
        fraction = np.abs(data[0, :] * (data[0, :] < 0))

        # Look for oil/brine mixed cell (0 < f < 1)
        ii_cell = np.argwhere((0 < data[0, :]) & (data[0, :] < 1))
        if len(ii_cell) > 0:
            for ii_ in ii_cell[0]:
                fraction[ii_] = 1 - np.abs(np.abs(data[0, ii_] * (data[0, ii_] > 0)))
    elif phase == 'oil' or phase == 'o':
        fraction = np.abs(data[0, :] * (data[0, :] > 0))

    # Fill surface data
    if fraction[-1] == -1:
        fraction = np.concatenate([fraction, [-1]])
    elif fraction[-1] == 1:
        fraction = np.concatenate([fraction, [1]])
    else:
        fraction = np.concatenate([fraction, [0]])

    return fraction


def q_brine_i_f(x, dL):
    k1 = x[0]
    k2 = x[1]
    mu = x[2]
    r = x[3]
    dh = x[4]
    f_b = x[5]
    if dL == None or dL == False:
        return np.nan
    else:
        # Darcy Law
        Q = ((k1 + k2) / mu) * (1/dL) * (4 * r * dh * f_b)
    return Q


def ice_draft_profile(t_profile, s_profile):
    return None


def ice_draft_nm(porespace, os_dict):
    ps = porespace.copy()
    rho_sw = pysic.property.sw.density_p0(os_dict['mat']['sw']['S'], os_dict['mat']['sw']['T'])

    ps['rho_si'] = pysic.property.si.density(ps.salinity.values, ps.temperature.values)
    m_ice = np.sum(ps[['rho_si', 'dh']].apply(lambda x: p_rho_i_f(x), axis=1))
    ps['b_ice_i'] = ps['dh'].apply(lambda x: x*rho_sw * g)
    ps['b_ice'] = ps['dh'].apply(lambda x: x*rho_sw * g).cumsum()
    hd_index = ps.loc[ps.b_ice <= m_ice].index[-1]

    hd = ps['h'].iloc[hd_index] + ps['dh'].iloc[hd_index] * (m_ice - ps['b_ice'].iloc[hd_index]) / ps['b_ice_i'].iloc[hd_index]

    return hd
