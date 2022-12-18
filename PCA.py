import numpy as np
import matplotlib.pyplot as plt

global updated_Z

def compute_covariance_matrix(Z):
    global updated_Z
    # Centering
    x0_avg = round(sum(Z[:, 0]) / (len(Z[:, 0]) * 1.0), 2)
    x1_avg = round(sum(Z[:, 1]) / (len(Z[:, 1]) * 1.0), 2)
    new_x0 = []
    for i, val in enumerate(Z[:, 0]):
        stand = val - x0_avg
        new_x0.append([stand])
    for i, val in enumerate(Z[:, 1]):
        stand = val - x1_avg
        new_x0[i].append(stand)

    new_Z = np.array(new_x0)
    # Standard Deviation
    std = np.std(new_Z, axis=0)
    new_x0 = []
    for i, val in enumerate(new_Z[:, :]):
        x0 = round(val[0] / std[0], 2)
        x1 = round(val[1] / std[1], 2)
        new_x0.append([x0, x1])
    Z = np.array(new_x0)
    updated_Z = np.array(new_x0)
    Z_T = np.array(new_x0).T
    z_cov = np.matmul(Z_T,Z)
    return z_cov


def find_pcs(cov):
    pcs = np.linalg.eig(cov)
    return pcs


def project_data(Z, PCS, L):
    global updated_Z
    print('Z:', updated_Z)
    print('pcs:', PCS)
    print('L:', L)
    z_proj = np.matmul(updated_Z, L[0]*-1)
    print('proj:', z_proj)
    return z_proj


def show_plot(Z, Z_star):
    global updated_Z
    fig, axs = plt.subplots(2)
    fig.suptitle('Principle Component Analysis')
    axs[0].plot(updated_Z[:, 0], updated_Z[:, 1], 'ro')
    proj_shape = Z_star.shape
    y_zeros = np.zeros(proj_shape)
    axs[1].plot(Z_star, y_zeros, marker='o')
    plt.show()
