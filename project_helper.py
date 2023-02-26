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