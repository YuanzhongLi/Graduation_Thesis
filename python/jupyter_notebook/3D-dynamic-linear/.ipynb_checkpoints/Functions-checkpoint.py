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

import numpy as np


def getNodePart(f):
    array = []
    while True:
        line = f.readline()
        if '*' in line:
            break
        line_array = line.strip().split(',')
        x = np.float64(line_array[1])
        y = np.float64(line_array[2])
        z = np.float64(line_array[3])
        array.append([x, y, z])        
    return np.array(array, dtype=np.float64).T


def getBeemGeneralSection(line, f):
    l_array = line.strip().split(',')
    density = np.float64(l_array[2].replace('density=', ''))
    l1 = f.readline()
    l1_array = l1.strip().split(',')
    A = np.float64(l1_array[0])
    I11 = np.float64(l1_array[1])
    I12 = np.float64(l1_array[2])
    I22 = np.float64(l1_array[3])
    J = np.float64(l1_array[4])
    
    l2 = f.readline()
    
    l3 = f.readline()
    l3_array = l3.strip().split(',')
    E = np.float64(l3_array[0])
    G = np.float64(l3_array[1])
    
    l4 = f.readline()
    l4_array = l4.strip().split(',')
    alpha = np.float64(l4_array[1].replace('alpha=', ''))
    beta = np.float64(l4_array[2].replace('beta=', ''))
#     print(density, A, I11, I12, I22, J, E, G, alpha, beta)
    return density, A, I11, I12, I22, J, E, G, alpha, beta


def getHeadMass(f):
    line = f.readline()
    line.strip()
    line = line.replace(',', '')
    return np.float64(line)


def getAmplitude(f):
    array = [[0.0, 0.0]]
    isBreak = False
    while True:
        line = f.readline()
        if '**' in line:
            break
        l_array = line.strip().split(',')
        for i in range(len(l_array) // 2):
            array.append([np.float64(l_array[i*2]), np.float64(l_array[i*2+1])])
    return np.array(array, dtype=np.float64)


def getDelta_t(f):
    line = f.readline()
    l_array = line.strip().split(',')
    return np.float64(l_array[0])


def inputdata(file_path):
    with open(file_path, 'r') as f:
        originCoordsG = None
        ae = None
        npoin = None
        nele = None
        delta_t = None
        n_t = None
        VX = None
        VY = None
        VZ = None 
        RVX = None
        RVY = None
        RVZ = None
        head_mass = None
        gamma = None
        omega = None
        HHT_alpha = None
        i = 0
        while True:
            line = f.readline()                       
            if '*END STEP' in line.upper():
                break
            if '*NODE\n' == line.upper():
                originCoordsG = getNodePart(f)
                npoin = originCoordsG.shape[1]
                nele = npoin-1
                ae = np.zeros((8, nele), dtype=np.float64)
            elif '*Beam General Section' in line:
                density, A, I11, I12, I22, J, E, G, alpha, beta = getBeemGeneralSection(line, f)
                ae[0, i] = A
                ae[1, i] = I11
                ae[2, i] = I12
                ae[3, i] = I22
                ae[4, i] = J
                ae[5, i] = E
                ae[6, i] = G
                L = np.sqrt(np.sum((originCoordsG.T[i+1]-originCoordsG.T[i])**2))
                ae[7, i] = A * L * density
                gamma = alpha
                omega = beta
                i += 1
            elif '*MASS' in line.upper():
                head_mass = getHeadMass(f)
            elif '*Amplitude' in line:
                if 'name=VX' in line:
                    VX = getAmplitude(f)
                    n_t = VX.shape[0]-1
                elif 'name=VY' in line:
                    VY = getAmplitude(f)
                    n_t = VY.shape[0]-1
                elif 'name=VZ' in line:
                    VZ = getAmplitude(f)
                    n_t = VZ.shape[0]-1
                elif 'name=RVX' in line:
                    RVX = getAmplitude(f)
                    n_t = RVX.shape[0]-1
                elif 'name=RVY' in line:
                    RVY = getAmplitude(f)
                    n_t = RVY.shape[0]-1
                elif 'name=RVZ' in line:
                    RVZ = getAmplitude(f)
                    n_t = RVZ.shape[0]-1
            elif ('*Dynamic' in line) and ('direct' in line):
                l_array = line.strip().split(',')
                HHT_alpha = np.float64(l_array[1].replace('alpha=', ''))
                delta_t = getDelta_t(f)
        return originCoordsG, ae, npoin, nele, delta_t, n_t, VX, VY, VZ, RVX, RVY, RVZ, head_mass, gamma, omega, HHT_alpha                                


# +
# originCoordsG, ae, npoin, nele, delta_t, n_t, VX, VY, VZ, RVX, RVY, RVZ, head_mass, gamma, omega, HHT_alpha = inputdata('test08.txt')
# -

def get_abaqus_data(file_path, step, node_num):
    ret = np.zeros((step, node_num*6), dtype=np.float64)
    with open(file_path, 'r') as f:
        for i in range(step):
            line = '-'
            while (len(line) < 20 or line[0] != '-'):
                line = f.readline()
            for j in range(node_num):
                l = f.readline().strip().split()
                for index,ele in enumerate(l):
                    if index > 0:
                        ret[i, j*6+index-1] = float(ele)
    return ret
