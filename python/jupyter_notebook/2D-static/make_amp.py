# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np


def get_acc(sec, deg):
    rad = np.deg2rad(deg)
    return 2*rad/(sec**2)


def make_rv_amp(sec=1.0, n_t=100, deg=180):
    delta_t = sec / n_t
    acc_rad = get_acc(sec, deg)
    with open('amp_rvz_等角加速度_{0}sec_{1}step_{2}deg.txt'.format(sec, n_t, deg), 'w', encoding='ascii') as f:
        for i in range(n_t):
            if i < n_t-1:
                f.write('{0:.5f} {1}\n'.format(i*delta_t, (i+1)*delta_t*acc_rad))
            else:
                f.write('{0:.5f} {1}'.format(i*delta_t, (i+1)*delta_t*acc_rad))
    return


make_rv_amp(0.6, 600, 180)


def make_circle_v_amp(sec=0.3, n_t=300, deg=180, r=0.2):
    delta_t = sec / n_t
    rad = np.deg2rad(deg)
    a_rad = 2 * rad / (sec**2)
    with open('amp_vx_半径{3}_{0}sec_{1}step_{2}deg.txt'.format(sec, n_t, deg, r), 'w', encoding='ascii') as f:
        for i in range(n_t):
            v_rad = a_rad*((i+1)*delta_t)
            u_rad = 1/2*a_rad*(((i+1)*delta_t)**2)
            if i < n_t-1:
                f.write('{0:.5f} {1}\n'.format(i*delta_t, -r*np.sin(u_rad)*v_rad))
            else:
                f.write('{0:.5f} {1}'.format(i*delta_t, -r*np.sin(u_rad)*v_rad))

    with open('amp_vy_半径{3}_{0}sec_{1}step_{2}deg.txt'.format(sec, n_t, deg, r), 'w', encoding='ascii') as f:
        for i in range(n_t):
            v_rad = a_rad*((i+1)*delta_t)
            u_rad = 1/2*a_rad*(((i+1)*delta_t)**2)
            if i < n_t-1:
                f.write('{0:.5f} {1}\n'.format(i*delta_t, r*np.cos(u_rad)*v_rad))
            else:
                f.write('{0:.5f} {1}'.format(i*delta_t, r*np.cos(u_rad)*v_rad))
    return


make_circle_v_amp(0.3, 300, 180, 0.6)

make_circle_v_amp(1, 1000, 180, 0.6)

make_circle_v_amp(3, 3000, 180, 0.6)


