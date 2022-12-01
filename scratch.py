#!/usr/bin/python3
# -*- coding: utf-8 -*-
#

import matplotlib.pyplot as plt  # Hunter J., 2007
from matplotlib.cm import viridis as cmap
import pandas as pd  # McKinney, 2010
import matplotlib.gridspec as gridspec
import pysic
import numpy as np
import yaml

import os

try:
    from project_helper import *
except ImportError:
    import sys
    if '/home/megavolts/git/paper3' in sys.path:
        raise
    sys.path.append('/home/megavolts/git/paper3')
    from project_helper import *



##
T_si = -5  # C
S_si = 5  # ppt
S_sw = 32
V = 10  # ml
dT = 3  # C
dh =1

L = pysic.property.si.latent_heat(S_si, T_si, 'fusion') # J kg-1
cp = pysic.property.si.heat_capacity(S_si, T_si)  # J kg-1 K-1

rho_b = pysic.property.brine.density(T_si)
rho_sw = pysic.property.sw.density_p0(S_sw, pysic.property.sw.freezingtemp(S_sw))
rho_o = rho_oil(pysic.property.sw.freezingtemp(S_sw))
rho_i = pysic.property.si.density(S_si, T_si)

V_SI = V * 1e-6   # m3
m = rho_i * V_SI  # kg
E = L * m + cp * m * dT  # J

Ep = (rho_sw - rho_o) * V_SI/10 * g * dh

print(E, Ep)

# PEclet number
T_si = -5
S_si = 5
u = 1 / (2.5 * 3600)
L = 1

s = S_si
t = T_si
rho = pysic.property.si.density(S_si, T_si)  #kg m-3
k = pysic.property.si.thermal_conductivity(S_si, T_si)  # W m-1 K-1
cp = pysic.property.si.heat_capacity(S_si, T_si)  # J kg-1

alpha = k / (rho * cp)

u = 1 / (3600)
L = 0.1
Pe = (u * L * cp * rho) / k

u * L / alpha