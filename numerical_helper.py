import numpy as np
from matplotlib.cm import viridis as cmap
import matplotlib.pyplot as plt
import pysic

try:
    from project_helper import *
except ImportError:
    import sys
    if '/home/megavolts/git/1DVerticalOilMigration' in sys.path:
        raise
    sys.path.append('/home/megavolts/git/1DVerticalOilMigration')
    from project_helper import *

# Microstructure sample data, based on Oggier and Eicken, 2022
# For development purpose, we used only BRW_CS-20130331
tex_dict = {'granular': {'20130331-g': [9.9, -11.7, 0.05]},  # 2-8 section
            'columnar': {'20130331-c': [4.6, -9.2, 0.42]}}  # 41-47 section
# Additional available data (Oggier and Eicken, 2022)
# TODO: extract data for all samples in Oggier and Eicken, 2022 to populate both texture dictionnary table, and tortuosity
# tex_dict = {'granular': {'20130331-g': [9.9, -11.7, 0.05],  #  2 -  8 section
#                          '20130513-g': [5.7, -4.7, 0.03],  #  0 -  6 section
#                          '20130609-g': [2.8, -4.1, 0.27],  # 24 - 30 section
#                          '20140327-g': [8.8, -9.0, 0.07],  #  4 - 10 section
#                          '20140513-g': [2.7, -2.5, 0.08],  #  5 - 11 section
#                          '20140610-g': [0.9, -0.1, 0.20]},  # 17 - 23 section
#             'columnar': {'20130331-g': [4.6, -9.2, 0.42],  # 39 - 45 section
#                          '20130513-c': [6.7, -4.5, 0.23],  # 20 - 26 section
#                          '20130513-c': [4.8, -4.1, 0.73],  # 70 - 76 section
#                          '20130609-c': [4.3, -0.9, 0.87],  # 84 - 90 section
#                          '20140327-c': [5.4, -5.5, 0.65],  # 62 - 68 section
#                          '20140513-c': [4.1, -2.9, 0.65],  # 62 - 68 section
#                          '20140513-c': [3.7, -2.8, 0.83],  # 80 - 86 section
#                          '20140610-c': [4.2, -0.8, 0.71]}}  # 68 -74 section



def nearest_interp(xi, x, y):
    xi = np.array(xi)
    x = np.array(x)
    y = np.array(y)
    idx = np.abs(x - xi[:, None])
    return y[idx.argmin(axis=1)]

def lookup_si_endpoint(S, T, y_mid, texture):
    """
    #TODO: inverse y_mid and texture importance order
    :param S: target salinity
    :param T: target temperature
    :param texture: target texture
    :param y_mid: target depth (optional)
    :return:
    """

    distance = False
    endpoint = False
    for sample in tex_dict[texture]:
        d_temp = np.sqrt((S-tex_dict[texture][sample][0])**2 + (T-tex_dict[texture][sample][1])**2 + (y_mid-tex_dict[texture][sample][1])**2)
        if not distance:
            distance = d_temp
            endpoint = sample
        elif d_temp < distance:
            distance = d_temp
            endpoint = sample
    return endpoint, tex_dict[texture][endpoint]

def tortuosity(endpoint, N=1):
    # TODO: extract data for other samples tortuosity dictionnary
    tau_dict = {'20130331-g': np.array([1.39993489, 1.23344433, 1.48926926, 2.20221257, 1.10855222,
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
              '20130331-c': np.array([1.03528392, 1.2653271 , 1.06601834, 1.03100038, 1.09156609,
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
    N_tau = np.random.randint(0, len(tau_dict[endpoint]), N)
    tau = tau_dict[endpoint][N_tau]
    return tau

def radius(endpoint, N=1):
    """
    :param endpoint: sea ice sample name
    :param N: number of layer
    :return:
    """
    # TODO: extract data for other samples radius dictionnary
    from scipy.stats import lognorm

    # microstrucutre dictionnary for throat
    # sample: [mu, sigma, antilog]
    throat_dict = {'20130331-c': [0.025, 1.27, 0.0240],
                   '20130331-g': [0.013, 0.85, 0.0132]}

    mu = throat_dict[endpoint][0]
    sigma = throat_dict[endpoint][1]

    A = lognorm.rvs(sigma, size=N)
    A = A * np.exp(mu)
    R = np.sqrt(A/np.pi)

    # hx, hy, _ = plt.hist(R, bins=50, color="lightblue")
    # plt.ylim(0.0, max(hx) + 0.05)
    # plt.xscale('log')
    # plt.show()

    R = R*1e-3  # in m
    return R

# tortuosity function old version for V5

def tortuosityV0(N, texture='granular'):
    tau_dict = {'granular': np.array([1.39993489, 1.23344433, 1.48926926, 2.20221257, 1.10855222,
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

# Radius data from Winter granular (T= , S= ) and columnar (T=, S= )
# Based on BRW_CS-20130331
# Granular 11-17 cm, T= -13.2 ,S= 4.9  , Vfb= 2.198, Vp =
# Columanr 64 70 cm, T=  -9.7 ,S= 4.6  , Vfb= 2.76, Vp =
# radius function old version for V5
def radiusV0(N=100, texture='columnar'):
    #TODO:
    # radius
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


# generate_TS_porespace function old version for V5
def generate_TS_porespaceV0(porespace, os_dict):

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
        if os_dict['BC']['HG'] > 0:
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
        porespace = porespace.assign(tau=tortuosityV0(len(porespace), 'c'))
        if os_dict['config']['SI_STRAT']:
            porespace.loc[porespace.texture == 'g', 'tau'] = tortuosityV0(len(porespace.loc[porespace.texture == 'g', 'tau']), 'granular')
    else:
        porespace = porespace.assign(tau=1)


    # Correction factor for brine volume fraction K_r = Vfb_local /Vfb_sample
    porespace['vf_b'] = pysic.property.si.brine_volume_fraction(porespace['salinity'], porespace['temperature'])
    Vfb_g = pysic.property.si.brine_volume_fraction(4.9, -13.2)  # For BRW_CS-20130331
    Vfb_c = pysic.property.si.brine_volume_fraction(4.6, -9.7)  # For BRW_CS-20130331

    porespace['K_r'] = [1] * len(porespace)
    porespace.loc[porespace.texture == 'g', 'K_r'] = porespace.loc[porespace.texture == 'g', 'vf_b']/Vfb_g
    porespace.loc[porespace.texture == 'c', 'K_r'] = porespace.loc[porespace.texture == 'c', 'vf_b']/Vfb_c

    if os_dict['config']['R'] == True:
        porespace = porespace.assign(r=radiusV0(porespace.__len__()))
        porespace.loc[porespace.texture == 'g', 'r'] = radiusV0(porespace.loc[porespace.texture == 'g', 'r'].size, texture='granular')
    else:
        porespace['r'] = os_dict['config']['R']

    porespace['r_corrected'] = porespace['r'] * np.sqrt(porespace['K_r'])

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
    porespace['kv_corrected'] = porespace['kv'] * porespace['K_r']**3

    porespace['kh1'] = 0.1 * porespace['kv']
    porespace['kh2'] = 0.01 * porespace['kv']
    porespace['kh1_corrected'] = 0.1 * porespace['kv_corrected']
    porespace['kh2_corrected'] = 0.01 * porespace['kv_corrected']

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

def generate_porespace_geometry(porespace, os_dict):
    if os_dict['config']['SI_PHYSIC'] == 'override':
        porespace['temperature'] = os_dict['mat']['si']['T']
    elif not os_dict['config']['SI_PHYSIC']:
        porespace['temperature'] = porespace.temperature.mean()

    if os_dict['config']['SI_PHYSIC'] == 'override':
        porespace['salinity'] = os_dict['mat']['si']['S']
    elif not os_dict['config']['SI_PHYSIC']:
        porespace['salinity'] = porespace.salinity.mean()

    porespace['texture'] = 'columnar'
    if os_dict['config']['SI_STRAT']:
        HGC = os_dict['BC']['HI'] - os_dict['BC']['HG']
        porespace.loc[porespace.h >= HGC, 'texture'] = 'granular'

    # SI endpoint lookup
    endpoint = []
    for ii in porespace.index:
        S, T, y_mid, texture = porespace[['salinity', 'temperature', 'h_mid', 'texture']].iloc[ii]
        endpoint.append(lookup_si_endpoint(S, T, y_mid, texture)[0])
    porespace['endpoint'] = endpoint

    # tortuosity
    if os_dict['config']['TORT'] == 'override':
        if os_dict['BC']['HG'] > 0:
            HGC = os_dict['BC']['HG']
        else:
            HGC = os_dict['BC']['HI'] - os_dict['BC']['HG']
        porespace.loc[porespace.h >= HGC, 'texture'] = 'granular'
        porespace = porespace.assign(tau=os_dict['mat']['si']['tau_c'])
        porespace.loc[porespace.texture == 'g', 'tau'] = os_dict['mat']['si']['tau_g']
        porespace['texture'] = 'ccolumnar'
    elif os_dict['config']['TORT']:
        for si_endpoint in porespace.endpoint:
            N = len(porespace[porespace.endpoint == si_endpoint])
            tau_endpoint = tortuosity(si_endpoint, N)
            porespace.loc[porespace.endpoint == si_endpoint, 'tau'] = tau_endpoint
    else:
        porespace = porespace.assign(tau=1)

    # throat radius
    if os_dict['config']['R']:
        for si_endpoint in porespace.endpoint:
            N = len(porespace[porespace.endpoint == si_endpoint])
            radius_endpoint = radius(si_endpoint, N)
            porespace.loc[porespace.endpoint == si_endpoint, 'r'] = radius_endpoint
    else:
        porespace['r'] = os_dict['config']['R']
    porespace['r0'] = porespace['r']
    return porespace

def correct_porespace_radius(porespace):
    """
    Correct porespace radius as function of local T, S relatively to sample T, S

    :param porespace:
    :return:
    """
    # Compute brine volume fraction for local TS and endpoint
    porespace['Vf_b'] = pysic.property.si.brine_volume_fraction(porespace['salinity'], porespace['temperature'])
    _temp = [porespace.texture.tolist(), porespace.endpoint.tolist()]
    Vf_b_endpoint = []
    for ii in range(0, len(_temp[0])):
        Vf_b_endpoint.append(pysic.property.si.brine_volume_fraction(tex_dict[_temp[0][ii]][_temp[1][ii]][0], tex_dict[_temp[0][ii]][_temp[1][ii]][1])[0])

    # Correction factor for brine volume fraction K_r = Vfb_local /Vfb_sample
    porespace['K_r'] = porespace['Vf_b'] / Vf_b_endpoint

    # Corrected porespace radius:
    porespace['r'] = porespace['r0'] * np.sqrt(porespace['K_r'])
    return porespace

def generate_porespace_physics(porespace, os_dict):
    """
    compute physical property of pore space according to T,S
    :param porespace:
    :return:
    """
    # Compute physical properties
    porespace['S_b'] = pysic.property.brine.salinity(porespace['temperature'], extend_t_0=True)
    porespace['rho_b'] = pysic.property.brine.density(porespace['temperature'], extend_t_0=True)
    porespace['mu_b'] = pysic.property.brine.viscosity(porespace['S_b'], porespace['temperature'], override_s=True)

    porespace['mu_o'] = mu_oil(porespace['temperature'])
    porespace['rho_o'] = rho_oil(porespace['temperature'])

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

    # permeability
    porespace['vf_b'] = pysic.property.si.brine_volume_fraction(porespace['salinity'], porespace['temperature'])

    # set permeability for columnar ice
    porespace['kv'] = pysic.property.si.permeability_from_porosity(porespace['vf_b'])
    porespace['kh1'] = 0.1 * porespace['kv']
    porespace['kh2'] = 0.01 * porespace['kv']

    # Fix permeability for granular ice, 0 at the bottom, according to Oggier and Eicken, 2022
    if os_dict['config']['SI_STRAT']:
        porespace.loc[porespace.texture == 'granular', 'kv'] = 0.1 * porespace.loc[porespace.texture == 'granular', 'kv']
        porespace.loc[porespace.texture == 'granular', 'kh1'] = 0.1 * porespace.loc[porespace.texture == 'granular', 'kv']
        porespace.loc[porespace.texture == 'granular', 'kh2'] = 0.1 * porespace.loc[porespace.texture == 'granular', 'kv']

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

    plt.ylim([0, HI])
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
    return rho * dh * g * f  #(kg m-3) (m)  (m s-2) (-) =


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
    return 8*mu / (np.pi*r**4) * (l * f)  # (Pa s-1) m-4 m = Pa s-1 m-3


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

### THIS SHOULD BE IN project_helper.py but it stopped working
#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
import numpy as np  # VanDerWalt S. et al., 2011
import configparser
try:
    from numerical_helper import *
except ImportError:
    import sys
    if '/home/megavolts/git/SimpleOilModel' in sys.path:
        raise
    sys.path.append('/home/megavolts/git/SimpleOilModel')
    from numerical_helper import *

try:
    import pysic
except ImportError:
    import sys
    if '/home/megavolts/git/pysic' in sys.path:
        raise
    sys.path.append('/home/megavolts/git/pysic' )
    import pysic
g = 9.80665  # m s^-2, Earth Gravitational Constant

# Oil properties
GAMMA_o = 20e-3  # kg s^-2 or 1e3 dyne/cm, interfacial tension between crude oil and water and ice, Malcom et al., 1979
THETA_o = 0  # deg, contact angel between crude oil and ice, in water Malcom et al., 1979

# Brine properties
GAMMA_b = 75.65e-3  # kg s^-2 interfacial tension between brine and sea ice in air
THETA_b = 0  # deg, contact angle between brine and sea ice in air


def t_sw_f(S):
    """
    Return freezing point of sea water as function of salinity
    :param S: 'float' or array-like
        Salinity in per thousands
    :return: T: 'float' or array-like
        Temperature in degree C
    """

    t_sw = pysic.property.sw.freezingtemp(S)

    if t_sw.size == 1:
        return t_sw[0]
    else:
        return t_sw


def rho_seawater(S, T):
    """
    Return density of sea water as function of salinity and temperature
    :param S: 'float' or array-like
        Salinity in per thousands
    :param T: 'float' or array-like
        Temperature in degree C
    :return:  'float' or array-like
        Density of sea-water in kg/m3
    """

    r_sw = pysic.property.sw.density_p0(S, T)

    if r_sw.size == 1:
        return r_sw[0]
    else:
        return r_sw


def rho_brine(T):
    """
    Return brine density as function of T
    :param T: 'float' or array-like
        Temperature in degree C
    :return: rho: 'float' or array-like
        Brine density in kg / m3
    """
    rho_b = pysic.property.brine.density(T)

    if rho_b.size == 1:
        return rho_b[0]
    else:
        return rho_b


def mu_oil(temp, oil_type='ANS'):
    """
    Return the viscosity of crude oil as function of temperature.

    :param temp: 'float', temperature in degreeC
    :param oil_type: 'string', oil type. Defaults: Alaska North Slope (ANS)
    :return: mu, 'float', oil viscosity in Pa S
    """
    if oil_type == 'ANS':
        # coefficient fitted to Scott Loranger data in mPa s
        a = 8.71884712e-14
        b = -2.95030660e+04
        si_scaling = 1e-3
    else:
        return None

    # Arrehnius Law
    mu = a*np.exp(-b/(np.pi*(temp+273.15)))
    mu = mu * si_scaling

    return mu


def mu_brine(S, T, override_s=True, override_t=True):
    """
    Return density of sea water as function of salinity and temperature
    :param S: 'float' or array-like
        Salinity in per thousands
    :param T: 'float' or array-like
        Temperature in degree C
    :return:  'float' or array-like
        Dynamic viscosity of brine in Pa s
    """

    mu_b = pysic.property.brine.viscosity(S, T, override_s=override_s, override_t=override_t)

    if mu_b.size == 1:
        return mu_b[0]
    else:
        return mu_b


def pc_b(r, oriented=True):
    """
    Return capillary pressure for brine with respect to known wettability
    :param oriented: 'boolean', Default True
        True for known wettabiliyt, False otherway
    :return:
    """
    if oriented:
        pc = np.abs(2 * GAMMA_b * np.abs(np.cos(np.deg2rad(THETA_b))) / r)
    else:
        pc = 2 * GAMMA_b * np.cos(np.deg2rad(THETA_b)) / r
    return pc


def pc_o(r, oriented=True):
    """
    Return capillary pressure for brine with respect to known wettability
    :param oriented: 'boolean', Default True
        True for known wettabiliyt, False otherway
    :return:
    """
    if oriented:
        pc = np.abs(2 * GAMMA_o * np.abs(np.cos(np.deg2rad(THETA_o))) / r)
    else:
        pc = 2 * GAMMA_o * np.cos(np.deg2rad(THETA_o)) / r
    return pc


def rho_oil(temp, oil_type='ANS'):
    """
    Return the density of crude oil as function of temperature.

    :param temp: 'float', temperature in degreeC
    :param oil_type: 'string', oil type. Defaults: Alaska North Slope (ANS)
    :return: mu, 'float', oil viscosity in Pa S
    """
    if oil_type == 'ANS':
        # coefficient fitted to Pegau et al. (2016) in g cm-3
        a = -7.63513516e-04
        b = 8.89247748e-01
        si_scaling = 1e3
    else:
        return None

    # Linear Law
    rho = a * temp + b
    rho = rho * si_scaling

    return rho


def lookup_TS_si_cover(TS, HI):
    """
    :param T: target ice surface temperature
    :param HI: target ice thickness temperature
    :return:

    Lookup for target TS and HI within simulated hindcast of ice growth and decay seasonal evolution using CICE model. Forced with
    reanalysis data from 1979 - 2018 (Oggier et al., 2020, submitted https://tc.copernicus.org/preprints/tc-2020-52/)
    """

    import pandas as pd
    data = pd.read_csv('CICE_data-UTQ/modelout-mod2.csv')
    data['date'] = pd.to_datetime(data[['year', 'month','day']])

    # compute minimal distance between (TS, HS) and hindcast
    data['distance'] = np.sqrt((HI*100-data['hi'])**2 + (TS-data['T_1'])**2)

    # look for minimal distance
    profile = data[data.distance == data.distance.min()].iloc[0]

    start_date = profile.date
    if profile.month <= 12:
        end_date = pd.to_datetime(str(profile.year+1)+'-9-15')
    else:
        end_date = pd.to_datetime(str(profile.year)+'-9-15')

    hi = profile.hi
    if np.abs(hi - HI*100)/(HI*100) < 0.2:
        s_header = [h for h in profile.index if 'S' in h]
        s_profile = np.array(profile[s_header]).astype(float)
        t_header = [h for h in profile.index if 'T' in h]
        t_profile = np.array(profile[t_header]).astype(float)
        TS_season = data.loc[(start_date <= data.date) & (data.date <= end_date)]
        return s_profile, t_profile, TS_season
    else:
        print("ERROR: difference in ice thicknesses are too large")
        return None

def extract_TS_profile(TS_season, date, y):
    t_header = [c for c in TS_season.columns if 'T' in c]
    s_header = [c for c in TS_season.columns if 'S' in c]

    profile = TS_season.loc[(TS_season.year == date.year) & (TS_season.month == date.month) & (TS_season.day == date.day)]
    s_profile = np.array(profile[s_header].iloc[0])
    t_profile = np.array(profile[t_header].iloc[0])
    y_profile = np.linspace(0, profile.hi.iloc[0], 21)/100
    y_profile = np.diff(y_profile)/2 + y_profile[:-1]

    y = np.array(y)
    y_mid = np.diff(y)/2 + y[:-1]

    s_nearest = nearest_interp(y, y_profile, s_profile)[::-1]
    t_interp = np.interp(y_mid, y_profile, t_profile)[::-1]

    return s_nearest, t_interp

def load_case(case_fp):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(case_fp)

    bc_dict = {}
    bc_dict['HI'] = config['ICE'].getfloat('HI')  # m, ice thickness
    bc_dict['HD'] = config['ICE'].getfloat('HD')  # m, ice draft
    if bc_dict['HI'] is not None and bc_dict['HD'] is not None:
        bc_dict['HF'] = bc_dict['HI'] - bc_dict['HD']  # ice freeboard
    else:
        print("Freeboard HD not defined")
    bc_dict['HG'] = config['ICE'].getfloat('HG')  # m, granular ice thickness
    bc_dict['HC'] = bc_dict['HI'] - bc_dict['HG']   # m, columnar ice thickness

    bc_dict['HR'] = config['OIL'].getfloat('HR')  # m, initial oil lens thickness
    bc_dict['VR'] = config['OIL'].getfloat('VR')  # L, initial oil volume
    return bc_dict


def brine_hydraulic_head(r, hi, hd, rho_sw, rho_b):
    """
        Return brine hydraulic head in sea-ice cover measured from ice/ocean interface for vertical cylindrical pore of
        given diameter.

    :param r: 'float'
        Pore radius
    :param hi: 'float'
        Ice thickness
    :param h_d: 'float'
        Ice draft
    :param gamma: 'float'
        Interfacial tension in N m-1
    :param theta: 'float'
        Contact angle degree
    :param rho_sw: 'float'
        Density of sea-water in kg m-3
    :param rho_b:
        Density of brine in kg m-3

    :return: 'float'
        Brine hydraulic head in meter
    """
    hb = rho_sw / rho_b * hd + pc_b(r) / (rho_b * g)
    hb = np.atleast_1d(hb)

    hb_mask = [hb > hi]
    hb[hb > hi] = hi

    if hb.size == 1:
        return hb[0]
    else:
        return hb


def ice_draft(t_si, s_si, s_sw, hi):
    # t_si = -5
    # s_si = 5
    # s_sw = 32
    t_sw = pysic.property.sw.freezingtemp(s_sw)
    rho_si = pysic.property.si.density(s_si, t_si)
    rho_sw = pysic.property.sw.density_p0(s_sw, t_sw)

    hd = rho_si / rho_sw * hi
    return hd


def oil_penetration_depth(ho, r, os_dict, DEBUG=False):
    """
    Return the time required by the oil to reach a given penetration depth as function of problem geometry
    (bc_dict) and material properties (mat_dict)

    :param ho: 'float' or array-like
        Oil penetration depth to reach in m
    :param r: 'float' or array-like
        Pore diameter in m
    :param os_dict: dict
        Oil spill condition with at least ice thickness, ice draft or freeboard in m
    :param p_top:
        Boundary condition at the top of the channel. Default is 'brine'
        Options are :
            'brine':    brine capillary pressure as function of sea water/air surface tension, contact angle and pore
                        radius. Behavior change to 'atm' if ho + hb > hi
            'atm' :     atmospheric pressure
            'float' :   user defined pressure value
    :param DEBUG:
    :return:
    """

    # Extract boundary condition:
    hi = os_dict['HI']  # m, initial ice thickness
    hd = os_dict['HD']  # m, initial ice draft
    hr = os_dict['HR']  # m, initial oil lens thickness
    vr = os_dict['VR']  # l, initial oil lens volume

    # Extract material parameters:
    rho_sw = rho_seawater(os_dict['sw']['S'], t_sw_f(os_dict['sw']['S']))

    rho_b = rho_brine(os_dict['b']['Ti'])
    mu_b = mu_brine(os_dict['b']['S'], os_dict['b']['Ti'])

    rho_o_R = rho_oil(os_dict['o']['Tsw'])  # oil density in sea water
    rho_o_c = rho_oil(os_dict['o']['Ti'])  # oil density in sea ice
    mu_o = mu_oil(os_dict['o']['Ti'])

    hb = brine_hydraulic_head(r, hi, hd, rho_sw, rho_b)
    a1 = mu_b / mu_o * hb
    b1 = rho_sw * g * r * hd + (rho_sw - rho_o_R) * g * r * hr - pc_o(1) + pc_b(1) - rho_b * g * r * hb
    c1 = np.pi * (rho_sw - rho_o_R) * g * r ** 3 * hr / vr + rho_o_c * g * r

    a2 = mu_b / (mu_o - mu_b) * hi
    b2 = rho_sw * g * r * hd - rho_b * g * r * hi + (rho_sw - rho_o_R) * g * r * hr - pc_o(1)
    c2 = np.pi * (rho_sw - rho_o_R) * g * r ** 3 * hr / vr + (rho_b - rho_o_c) * g * r

    if isinstance(a2, float):
        a2 = a2 * np.ones_like(b2)

    def f1(ho_):
        t = 8 / r * ((a1 * c1 + b1) * np.log(b1 / (b1 - c1 * ho_)) - c1 * ho_) / c1 ** 2

        if not isinstance(t, float):
            t = np.array(t)
            tnan = np.nan * np.ones_like(t)
            maskhb = (hb == hi)
            t[maskhb] = tnan[maskhb]
        else:
            if hb == hi:
                t = np.nan
        return t

    def f2A(ho_):
        t = ((a2 * c2 + b2) * np.log((b2 - c2 * (hi - hb)) / (b2 - c2 * ho_)) + c2 * ho_ - c2 * (hi - hb)) / c2 ** 2
        t += ((a1 * c1 + b1) * np.log(b1 / (b1 - c1 * (hi - hb))) - c1 * (hi - hb)) / c1 ** 2
        t = 8 / r * t
        if not isinstance(t, float):
            t = np.array(t)
            tnan = np.nan * np.ones_like(t)
            maskhb = (hb == hi)
            t[maskhb] = tnan[maskhb]
        else:
            if hb == hi:
                t = np.nan
        return t


    def f2B(ho_):
        t = 8 / r * ((a2 * c2 + b2) * np.log(b2 / (b2 - c2 * ho_)) + c2 * ho_) / c2 ** 2
        if not isinstance(t, float):
            t = np.array(t)
            tnan = np.nan*np.ones_like(t)
            maskhb = (hb < hi)
            t[maskhb] = tnan[maskhb]

        else:
            if hb < hi:
                t = np.nan
        return t


    def f(ho_):
        t = f2A(ho_)
        mask = (ho_ < hi - hb)
        t[mask] = f1(ho_)[mask]

        mask = (ho_ == 0)
        t[mask] = 0

        maskhb = (hi == hb)
        t[maskhb] = f2B(ho_)[maskhb]

        if not isinstance(t, float):
            t = np.array(t)
        return t

    if DEBUG:
        const = {'a1': a1, 'b1': b1, 'c1': c1,
                 'a2': a2, 'b2': b2, 'c2': c2}
        return f(ho), f1(ho), f2A(ho), f2B(ho), const
    else:
        return f(ho)


def r_lim_C1(hr, ho, os_dict, oriented=True):
    """
    Return the minimal pore radius that could be invaded by oil under capillary forces of both brine and oil as
    function of oil lens thickness and oil penetration depth

    :param hr: 'float', array-like
        Oil lens thickness, in m
    :param ho: 'float', array-like
        Oil penetration depth, in m
    :param os_dict:
        Dictionary containing oil spill condition
    :param oriented: 'boolean', default True
        Oriented contact angle. If true, 0 < theta < 90

    :return:  'float', array-like
        Pore radius, in m.
        If r = -1, radius is negative: r < r_c
        If r = -2, ho+hb > hi
    """
    rho_sw = rho_seawater(os_dict['sw']['S'], t_sw_f(os_dict['sw']['S']))
    rho_o_R = rho_oil(os_dict['o']['Tsw'])  # oil density in sea water
    rho_o_c = rho_oil(os_dict['o']['Ti'])  # oil density in sea ice
    rho_b = rho_brine(os_dict['b']['Ti'])

    try:
        hi = os_dict['HI']
    except KeyError:
        hi = None
    try:
        hd = os_dict['HD']
    except KeyError:
        hd = None

    if oriented:
        r = pc_o(1)
    else:
        r = pc_o(1, oriented)

    # Valid only if hi < hi + hb
    if hi is None and hd is None:
        nanmask = (r <= 0)
        r_nan = np.nan * np.ones_like(r)
        r[nanmask] = r_nan[nanmask]
        if r.size == 1:
            return r[0]
        else:
            return r
    elif hi is None and hd is not None:
        hi = hd / 0.9
    elif hd is None and hi is not None:
        hd = hi * 0.9

    r = r / ((rho_sw - rho_o_R) * g * hr - rho_o_c * ho * g)
    r = np.atleast_1d(r)

    hb = brine_hydraulic_head(r, hi, hd, rho_sw, rho_b)

    # nanmask = (ho + hb > hi)
    # r_nan = -2 * np.ones_like(r)
    # r[nanmask] = r_nan[nanmask]
    #
    # nanmask = (r < 0)
    # r_nan = np.nan * np.ones_like(r)
    # r[nanmask] = r_nan[nanmask]

    if r.size == 1:
        return r[0]
    else:
        return r


def r_lim_initial(hr, os_dict, oriented=True):
    """
    Return the minimal pore radius that could be invaded by oil under capillary forces of both brine and oil as
    function of oil lens thickness and oil penetration depth

    :param hr: 'float', array-like
        Oil lens thickness, in m
    :param os_dict:
        Dictionary containing oil spill condition
    :param oriented: 'boolean', default True
        Oriented contact angle. If true, 0 < theta < 90

    :return:  'float', array-like
        Pore radius, in m.
        If r = -1, radius is negative: r < r_c
        If r = -2, ho+hb > hi
    """

    r = r_lim_C1(hr, 0, os_dict, oriented=oriented)
    return r


def hb_C1(r, os_dict, oriented=True):
    """
    Return the minimal pore radius that could be invaded by oil under capillary forces of oil as function of oil lens
    thickness and oil penetration depth

    :param r: 'float', array-like
        Pore radius, in m
    :param hr: 'float', array-like
        Oil lens thickness in m
    :param os_dict: 'float', array-like
        Dictionnary containing oil spill condition
    :param oriented: 'boolean', default True
        Oriented contact angle. If true, 0 < theta < 90

    :return:  'float', array-like
        Oil penetration depth, ho,  in m
        If hi is defined  in os_dict, then:
            ho = hi, if ho + hb > hi
            ho = np.nan if hb > hi
    """
    rho_sw = rho_seawater(os_dict['sw']['S'], t_sw_f(os_dict['sw']['S']))
    rho_b = rho_brine(os_dict['b']['Ti'])

    try:
        hi = os_dict['HI']
    except KeyError:
        hi = None
    try:
        hd = os_dict['HD']
    except KeyError:
        hd = None

    hb = brine_hydraulic_head(r, hi, hd, rho_sw, rho_b)

    hb = np.atleast_1d(hb)

    if hb.size == 1:
        return hb[0]
    else:
        return hb


def ho_lim_C1(r, hr, os_dict, oriented=True):
    """
    Return the minimal pore radius that could be invaded by oil under capillary forces of oil as function of oil lens
    thickness and oil penetration depth

    :param r: 'float', array-like
        Pore radius, in m
    :param hr: 'float', array-like
        Oil lens thickness in m
    :param os_dict: 'float', array-like
        Dictionnary containing oil spill condition
    :param oriented: 'boolean', default True
        Oriented contact angle. If true, 0 < theta < 90

    :return:  'float', array-like
        Oil penetration depth, ho,  in m
        If hi is defined  in os_dict, then:
            ho = hi, if ho + hb > hi
            ho = np.nan if hb > hi
    """
    rho_sw = rho_seawater(os_dict['sw']['S'], t_sw_f(os_dict['sw']['S']))
    rho_o_R = rho_oil(os_dict['o']['Tsw'])  # oil density in sea water
    rho_o_c = rho_oil(os_dict['o']['Ti'])  # oil density in sea ice
    rho_b = rho_brine(os_dict['b']['Ti'])

    try:
        hi = os_dict['HI']
    except:
        hi = None

    try:
        hd = os_dict['HD']
    except:
        hd = None

    # Valid only if hi < hi + hb
    if os_dict is None:
        ho_ = (rho_sw - rho_o_R) / rho_o_c * hr
        if oriented:
            ho_ = ho_ - pc_o(r) / (rho_o_c * g)
        else:
            ho_ = ho_ + pc_o(r, oriented=oriented) / (rho_o_c * g)
        ho_ = np.atleast_1d(ho_)
        nanmask = ho_ < 0
        r_nan = +1 * np.atleast_1d(np.ones_like(r))
        ho_[nanmask] = r_nan[nanmask]
        if ho_.size == 1:
            return ho_[0]
        else:
            return ho_
    elif hi is None and hd is not None:
        hi = hd / 0.9
    elif hd is None and hi is not None:
        hd = hi * 0.9

    hb = brine_hydraulic_head(r, hi, hd, rho_sw, rho_b)

    ho = (rho_sw * hd + (rho_sw - rho_o_R) * hr - rho_b * hb) / rho_o_c
    if oriented:
        ho += (pc_b(r) - pc_o(r)) / (rho_o_c * g)
    else:
        ho += (pc_b(r, oriented) + pc_o(r, oriented)) / (rho_o_c * g)

    ho = np.atleast_1d(ho)
    nanmask = (ho + hb > hi)
    h_hi = hi * np.ones_like(ho)
    ho[nanmask] = h_hi[nanmask]

    r_lim = r_lim_initial(hr, os_dict)
    nanmask = (r < r_lim) & (hb >= hi)
    ho[nanmask] = 0

    nanmask = (r >= r_lim) & (ho + hb > hi)
    h_hi = hi * np.ones_like(ho)
    ho[nanmask] = h_hi[nanmask]

    if ho.size == 1:
        return ho[0]
    else:
        return ho


def pressure_equilibrium(ho, r, os_dict, oriented=True):
    """    if r.size == 1:
        return r[0]
    else:

    Return force balance for a given penetration depth

    :param r: 'float', array-like
        Pore radius, r, in m
    :param ho: 'float', array-like
        Oil penetration depth, ho, in m
    :param os_dict:
        Dictionary containing oil spill condition
    :param oriented: 'boolean', default True
        Oriented contact angle. If true, 0 < theta < 90
    :return:
        Pressure equilibrium, pE.
        If pE < 0 oil moves up
        If pE > 0 oil stop moving

    """
    rho_sw = rho_seawater(os_dict['sw']['S'], t_sw_f(os_dict['sw']['S']))
    rho_b = rho_brine(os_dict['b']['Ti'])
    rho_o_R = rho_oil(os_dict['o']['Tsw'])  # oil density in sea water
    rho_o_c = rho_oil(os_dict['o']['Ti'])  # oil density in sea ice

    try:
        hr = os_dict['HR']
    except KeyError:
        hr = None

    try:
        hi = os_dict['HI']
    except KeyError:
        hi = None
    try:
        hd = os_dict['HD']
    except KeyError:
        hd = None

    if hi is None and hd is None:
        return np.nan
    elif hi is None and hd is not None:
        hi = hd / 0.9
    elif hd is None and hi is not None:
        hd = hi * 0.9

    try:
        hb = os_dict['HB']
    except KeyError:
        hb = brine_hydraulic_head(r, hi, hd, rho_sw=rho_sw, rho_b=rho_b)

    m_b = - rho_b * g * hb
    m_o = -  (rho_o_R * g * hr + rho_o_c * g * ho)
    b_b = rho_sw * g * (hd - ho)
    b_o = rho_sw * g * (ho + hr)
    pco = pc_o(r, oriented)
    pcb = pc_b(r, oriented)

    if oriented:
        equilibrium = m_b + m_o + b_b + b_o + pcb - pco
    else:
        equilibrium = + m_b + m_o + b_b + b_o + pco

    return equilibrium


def r_lim_hb(r, os_dict):
    """
    Return the pore radius when Hb = Hi as function of Hi and HD

    :param os_dict:
    :return:
    Pore radius r in m
    """
    hi = os_dict['HI']
    hd = os_dict['HD']
    rho_sw = rho_seawater(os_dict['sw']['S'], t_sw_f(os_dict['sw']['S']))
    rho_b = rho_brine(os_dict['b']['Ti'])

    return max(r[brine_hydraulic_head(r, hi, hd, rho_sw, rho_b) == hi])


def hr_lim_hb(r, os_dict):
    """
    Return the critical oil lens thickness, at the critical pore radius when Hb=Hi as function
    of Hi and Hd
    :param os_dict:
    :return:
    """
    hi = os_dict['HI']

    r_lim = r_lim_hb(r, os_dict)

    def hr_f(x):
        return r_lim_initial(x, os_dict, brine_capillary=False) - r_lim

    from scipy.optimize import root
    sol = root(hr_f, 0.01)

    return sol.x