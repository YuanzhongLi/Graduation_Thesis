# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from scipy.spatial.transform import Rotation
from scipy.linalg import expm
import numpy as np
from Functions import inputdata, get_abaqus_data

inp_path = 'test02.txt'

originCoordsG, ae, npoin, nele, delta_t, n_t, VX, VY, VZ, RVX, RVY, RVZ, head_mass, gamma, omega, HHT_alpha \
= inputdata(inp_path)

v_data = get_abaqus_data('test05V.txt', 264, 23)


def make_amp(v_data):
    step = v_data
    with open


