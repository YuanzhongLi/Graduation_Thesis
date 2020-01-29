# -*- coding: utf-8 -*-
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


def make_circle_v_amp(sec=1.0, n_t=100, deg=180):
    delta_t = sec / n_t
    acc_rad = get_acc(sec, deg)
    with open('amp_rvz_等角加速度_{0}sec_{1}step_{2}deg.txt'.format(sec, n_t, deg), 'w', encoding='ascii') as f:
        for i in range(n_t):
            if i < n_t-1:
                f.write('{0:.5f} {1}\n'.format(i*delta_t, (i+1)*delta_t*acc_rad))
            else:
                f.write('{0:.5f} {1}'.format(i*delta_t, (i+1)*delta_t*acc_rad))
    return
