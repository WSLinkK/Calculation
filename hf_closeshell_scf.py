# Jan 04/2023
import numpy as np
from numpy import *


def repeat_loop(dim: int, nested: int) -> np.ndarray:
    return np.array(
        [np.tile(np.repeat(np.arange(dim), dim ** (nested - 1 - i)), dim ** i) for i in range(nested)]).T


def calculate_s12_transformation_matrix(s_mat):
    # Diagonalize Matrix
    # SL = Ls
    # X = S^(1/2) = L  s^(-1/2)  L^T
    # Create L and s from eigenvalue and eigenvector from matrix S
    s, l_vec = np.linalg.eigh(s_mat)
    inv_s_12 = (np.diag(s ** (-0.5)))
    x_mat = np.dot(l_vec, np.dot(inv_s_12, np.transpose(l_vec)))
    return x_mat


class Molecule:
    def __init__(self, n_ao):
        self.n_ao = n_ao
        h_core_ao, v_ao, s_ao = self.get_integrals()
        self.h_core_ao = h_core_ao
        self.v_ao = v_ao
        self.s_ao = s_ao

    def get_t_int(self, file_t_int):
        # Generate  Kinetic Integral for H_core
        f = open(file_t_int, 'r')
        t_int = np.zeros((self.n_ao, self.n_ao))
        for line in f:
            t_oks = line.split()
            i = int(t_oks[0])
            j = int(t_oks[1])
            t_int[i - 1, j - 1] = float(t_oks[2].replace("D", "E"))
            t_int[j - 1, i - 1] = t_int[i - 1, j - 1]
        f.close()
        return t_int

    def get_v_int(self, file_integrals):
        # Generate Two-Electron Integral
        v_int = np.zeros((self.n_ao, self.n_ao, self.n_ao, self.n_ao))
        f = open(file_integrals, 'r')
        for line in f:
            t_oks = line.split()
            p = int(t_oks[0])
            q = int(t_oks[1])
            r = int(t_oks[2])
            s = int(t_oks[3])
            v_int[p - 1, q - 1, r - 1, s - 1] = float(t_oks[4].replace("D", "E"))
        for mu in range(self.n_ao):
            for nu in range(mu + 1):
                for lam in range(self.n_ao):
                    for sig in range(lam + 1):
                        if abs(v_int[mu, nu, lam, sig]) >= 1.e-12:
                            v_int[nu, mu, lam, sig] = v_int[mu, nu, lam, sig]
                            v_int[mu, nu, sig, lam] = v_int[mu, nu, lam, sig]
                            v_int[nu, mu, sig, lam] = v_int[mu, nu, lam, sig]
                            v_int[nu, mu, lam, sig] = v_int[mu, nu, lam, sig]
                            v_int[lam, sig, mu, nu] = v_int[mu, nu, lam, sig]
                            v_int[sig, lam, mu, nu] = v_int[mu, nu, lam, sig]
                            v_int[lam, sig, nu, mu] = v_int[mu, nu, lam, sig]
                            v_int[sig, lam, nu, mu] = v_int[mu, nu, lam, sig]
        f.close()
        return v_int

    def get_overlap_matrix(self, file_overlap):
        # Generate Overlap Integral
        f = open(file_overlap, 'r')
        s_mat = np.zeros((self.n_ao, self.n_ao))
        for line in f:
            toks = line.split()
            i = int(toks[0])
            j = int(toks[1])
            s_mat[i - 1, j - 1] = float(toks[2].replace("D", "E"))
            s_mat[j - 1, i - 1] = s_mat[i - 1, j - 1]
        f.close()
        return s_mat

    def get_integrals(self):
        # Generate Molecular Integral
        file_v_int = "fort.3001"
        file_h_core_int = "fort.3005"
        # file_eigenvalues = "fort.2024"
        # file_eigenvectors = "fort.2999"
        file_overlap = "fort.3003"

        h_core_ao = self.get_t_int(file_h_core_int)
        v_ao = self.get_v_int(file_v_int)
        s_ao = self.get_overlap_matrix(file_overlap)
        return h_core_ao, v_ao, s_ao

    def mo_eri_transformation_n5(self, coefficient):
        eri_mo_n5 = np.zeros(self.v_ao.shape, dtype=float)
        eri_mo_tmp1 = np.zeros(self.v_ao.shape, dtype=float)
        eri_mo_tmp2 = np.zeros(self.v_ao.shape, dtype=float)
        eri_mo_tmp3 = np.zeros(self.v_ao.shape, dtype=float)
        loop_indices = repeat_loop(self.n_ao, 4)

        for p in range(self.n_ao):
            for mu, nu, sigma, delta in loop_indices:
                eri_mo_tmp1[p, nu, sigma, delta] += coefficient[mu, p] * self.v_ao[mu, nu, sigma, delta]

        for q in range(self.n_ao):
            for p, nu, sigma, delta in loop_indices:
                eri_mo_tmp2[p, q, sigma, delta] += coefficient[nu, q] * eri_mo_tmp1[p, nu, sigma, delta]

        for r in range(self.n_ao):
            for p, q, sigma, delta in loop_indices:
                eri_mo_tmp3[p, q, r, delta] += coefficient[sigma, r] * eri_mo_tmp2[p, q, sigma, delta]

        for s in range(self.n_ao):
            for p, q, r, delta in loop_indices:
                eri_mo_n5[p, q, r, s] += coefficient[delta, s] * eri_mo_tmp3[p, q, r, delta]

        return eri_mo_n5

    def hf_loop(self, n_ao, nuc_rep, n):
        # h_core_ao, v_ao, s_ao = self.get_integrals(n_ao)
        x_mat = calculate_s12_transformation_matrix(self.s_ao)
        p_mat = np.zeros((n_ao, n_ao), dtype=float)
        fock_mat = np.zeros((n_ao, n_ao), dtype=float)
        # coefficient = np.zeros((n_ao, n_ao), dtype=float)
        conv_criteria = 1e-11
        k = 0
        # energy = 0.0

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
                                    self.v_ao[mu, nu, sigma, delta] - 0.5 * self.v_ao[mu, delta, sigma, nu])
            # print(g_mat)
            fock_mat[:, :] = self.h_core_ao[:, :] + g_mat[:, :]
            # print(fock_mat)

            fock_p = np.dot(np.dot(x_mat.T, fock_mat), x_mat)
            eps, c_p = np.linalg.eigh(fock_p)
            # print(f'eps:{eps}')
            # dx = eps.argsort()
            # eps = eps[idx]
            # c_p = c_p[:, idx]
            coefficient = np.dot(x_mat, c_p)

            p_mat = np.zeros([n_ao, n_ao])

            for i in range(n_ao):
                for j in range(n_ao):
                    for a in range(int(n / 2)):
                        p_mat[i, j] = p_mat[i, j] + 2 * coefficient[i, a] * coefficient[j, a]

            q = np.dot(p_mat, self.h_core_ao + fock_mat)
            energy = np.trace(q) * 0.5

            # for i in range(n_ao):
            #    for j in range(n_ao):
            #            E += 0.5 * p_mat[j, i] * (h_core_ao[i,j] + fock_mat[i,j])
            print('Electronic energy =', energy)

            # print(f'new{p_mat}')
            # error = np.zeros((n_ao, n_ao))
            error = p_mat - p_mat_prev
            error_p = np.sqrt(np.sum(error ** 2)) / 4.0
            print(f'Convergence error:{error_p}')

            # check = np.trace(np.dot(p_mat, s_ao))
            # print(f'N_elec:{check}')

            if error_p < conv_criteria:
                energy_tot = energy + nuc_rep
                print(f'______Met Convergence Criteria______')
                print(f'Calculation converged with electronic energy:{energy}')
                print(f'Calculation converged with total energy:{energy_tot}')
                print(f'Density matrix:\n{p_mat}')
                print(f'Coefficient:\n{coefficient}')
                return energy_tot, self.v_ao, coefficient, fock_mat
