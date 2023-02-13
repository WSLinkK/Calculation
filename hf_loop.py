import hf_aux
import scipy.linalg
import numpy as np
import matplotlib
from numpy import linalg as la


def hf_loop(n_ao, nuc_rep, N):

    h_core_ao,v_ao,s_ao = hf_aux.get_integrals(n_ao)
    x_mat = hf_aux.calculate_s12_transformation_matrix(s_ao)
    p_mat = np.zeros((n_ao, n_ao), dtype=float)
    fock_mat = np.zeros((n_ao, n_ao), dtype=float)
    c = np.zeros((n_ao, n_ao), dtype=float)
    conv_criteria = 1e-11
    k = 0
    E = 0.0

    while k < 2000:
        k += 1
        print(f'______Convergence step {k}______')
        p_mat_prev = p_mat
        g_mat = np.zeros((n_ao, n_ao))

        for mu in range(n_ao):
            for nu in range(n_ao):
                for sigma in range(n_ao):
                    for delta in range(n_ao):
                        g_mat[mu, nu] = g_mat[mu, nu] + p_mat_prev[delta, sigma] * (
                                v_ao[mu, nu, sigma, delta] - 0.5 * v_ao[mu, delta, sigma, nu])
        #print(g_mat)
        fock_mat[:, :] = h_core_ao[:, :] + g_mat[:, :]
        #print(fock_mat)

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
        print(f'Convergence error:{error_p}')

        #check = np.trace(np.dot(p_mat, s_ao))
        #print(f'N_elec:{check}')

        if error_p < conv_criteria:
            E_tot = E + nuc_rep
            print(f'______Met Convergence Criteria______')
            print(f'Calculation converged with electronic energy:{E}')
            print(f'Calculation converged with total energy:{E_tot}')
            print(f'Density matrix:\n{p_mat}')
            print(f'Coeffients:\n{c}')
            return E_tot, v_ao, c, h_core_ao
...
E_save = []
E, v_ao, c, fock_mat = hf_loop(2, 1.42857, 2)
print(E)
E_save += [E]

fock_mat_mo = hf_aux.fock_transform(fock_mat, c)

print(f'fock_ao:{fock_mat}')
print(f'fock_mo:{fock_mat_mo}')

#print(E_save)
#y = [x + 2*0.5459 for x in E_save]
#print(y)
...
#test_1 = hf_aux.mo_eri_transformation_n8(2, v_ao, c)
#test_2 = hf_aux.mo_eri_transformation_n4(2, v_ao, c)
#test_3 = hf_aux.transform_v(v_ao, c)
#test_4 = hf_aux.mo_eri_transformation_n5(2, v_ao, c)

#print(f'method 1:{type(test_1)}\n{np.round(test_1, decimals=8)}')
#print(f'method 2:{type(test_2)}\n{np.round(test_2, decimals=8)}')
#print(f'method 3:{type(test_3)}\n{np.round(test_3, decimals=8)}')
#print(f'method 4:{type(test_4)}\n{np.round(test_4, decimals=8)}')














