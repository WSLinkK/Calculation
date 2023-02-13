import hf_aux
import scipy.linalg
import numpy as np
import matplotlib
from numpy import linalg as la


def hf_loop(n_ao, R, nuc_rep, N):

    h_core_ao,v_ao,s_ao = hf_aux.get_integrals(n_ao)
    x_mat = hf_aux.calculate_s12_transformation_matrix(s_ao)
    p_mat = np.zeros((n_ao, n_ao), dtype=float)
    fock_mat = np.zeros((n_ao, n_ao), dtype=float)
    print(f'h_core{h_core_ao}')
    crit = 1e-15
    k = 0
    E = 0.0

    while (k < 2000):
        k += 1
        print(k)
        p_mat_prev = p_mat
        g_mat = np.zeros((n_ao, n_ao))

        for mu in range(n_ao):
            for nu in range(n_ao):
                for sigma in range(n_ao):
                    for delta in range(n_ao):
                        g_mat[mu, nu] = g_mat[mu, nu] + p_mat_prev[delta, sigma] * (
                                v_ao[mu, nu, sigma, delta] - 0.5 * v_ao[mu, delta, sigma, nu])
        print(g_mat)
        fock_mat[:, :] = h_core_ao[:, :] + g_mat[:, :]
        print(fock_mat)

        fock_p = np.dot(np.dot(x_mat.T, fock_mat),  x_mat)
        eps, c_p = np.linalg.eigh(fock_p)
        #print(f'eps:{eps}')
        #dx = eps.argsort()
        #eps = eps[idx]
        #c_p = c_p[:, idx]
        c = np.dot(x_mat, c_p)

        p_mat = np.zeros([n_ao, n_ao])

        for i in range(n_ao):
            for j in range(n_ao):
                for a in range(int(N / 2)):
                    p_mat[i, j] = p_mat[i, j] + 2 * c[i, a] * c[j, a]


        q = np.dot(p_mat, h_core_ao + fock_mat)
        E = np.trace(q) * 0.5

        #for i in range(n_ao):
        #    for j in range(n_ao):
        #            E += 0.5 * p_mat[j, i] * (h_core_ao[i,j] + fock_mat[i,j])
        print('Electronic energy =', E)

        #print(f'new{p_mat}')
        error = np.zeros((n_ao, n_ao))
        error = p_mat - p_mat_prev
        error_p = np.sqrt(np.sum(error ** 2)) / 4.0
        print(error_p)

        #check = np.trace(np.dot(p_mat, s_ao))
        #print(f'N_elec:{check}')

        if (error_p < crit):
            E_tot = E + nuc_rep
            print("Calculation converged with electronic energy:", E)
            print("Calculation converged with total energy:", E_tot)
            print("Density matrix", p_mat)
            print("Coeffients", c)
            break

hf_loop(2, 0.7, 1.428571428571, 2)















