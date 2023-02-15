# Feb 08/2023
import numpy as np
from numpy import *
import re


def fock_transform(fock_matrix, c):
    # Transform fock matrix from AO basis to MO basis
    h_core_mo = np.dot(np.dot(c.T, fock_matrix), c)
    return h_core_mo


def repeat_loop(dim: int, nested: int) -> np.ndarray:
    return np.array(
        [np.tile(np.repeat(np.arange(dim), dim ** (nested - 1 - i)), dim ** i) for i in range(nested)]).T


def transform_v(v_ao, cmo):
    # Transform v_ao to v_mo
    nao = cmo.shape[0]
    nmo = cmo.shape[1]
    cmot = np.zeros(cmo.shape)
    cmot[:, :] = np.transpose(cmo)[:, :]

    v_muqrs = np.zeros((nao, nao, nao, nmo))
    for mu in range(nmo):
        for p in range(nao):
            for q in range(nao):
                for r in range(nao):
                    for s in range(nao):
                        v_muqrs[p, q, r, mu] = v_muqrs[p, q, r, mu] + v_ao[p, q, r, s] * cmo[s, mu]

    v_munurs = np.zeros((nao, nao, nao, nmo))
    for mu in range(nmo):
        for nu in range(nmo):
            for p in range(nao):
                for q in range(nao):
                    for r in range(nao):
                        v_munurs[nu, q, r, mu] = v_munurs[nu, q, r, mu] + v_muqrs[p, q, r, mu] * cmot[nu, p]

    v_munusigmas = np.zeros((nao, nao, nao, nmo))
    for mu in range(nmo):
        for nu in range(nmo):
            for sigma in range(nmo):
                for q in range(nao):
                    for r in range(nao):
                        v_munusigmas[nu, q, sigma, mu] = v_munusigmas[nu, q, sigma, mu] + v_munurs[nu, q, r, mu] * cmo[
                            r, sigma]

    v_mo = np.zeros((nao, nao, nao, nmo))
    for mu in range(nmo):
        for nu in range(nmo):
            for sigma in range(nmo):
                for delta in range(nmo):
                    for q in range(nao):
                        v_mo[nu, delta, sigma, mu] = v_mo[nu, delta, sigma, mu] + v_munusigmas[nu, q, sigma, mu] * \
                                                     cmot[
                                                         delta, q]
    return v_mo


def calculate_s12_transformation_matrix(s_mat):
    # Diagonalize Matrix
    # SL = Ls
    # X = S^(1/2) = L  s^(-1/2)  L^T
    # Create L and s from eigenvalue and eigenvector from matrix S
    s, l_vec = np.linalg.eigh(s_mat)
    inv_s_12 = (np.diag(s ** (-0.5)))
    x_mat = np.dot(l_vec, np.dot(inv_s_12, np.transpose(l_vec)))
    return x_mat


def find_nuc_energy(file_hf):
    # Read nuclear repulsion energy from file
    nuc_rep = ''
    f = open(file_hf, 'r')
    for line in f:
        if "Nuclear repulsion:" in line:
            toks = line.split()
            nuc_rep = float(toks[2])
    f.close()
    return nuc_rep


def substitute_string(string, rule):
    str_lst = list(string)
    rule1, rule2 = rule.replace(" ", "").split("->")
    idx_list = [[i.start() for i in re.finditer(c, string)] for c in rule1]
    for idx, pos_list in enumerate(idx_list):
        for pos in pos_list:
            str_lst[pos] = rule2[idx]
    return "".join(str_lst)


class scf_loop:
    def __init__(self, n_ao):
        # Initialize
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

    def mo_eri_einsum(self, c):
        eri_mo = np.einsum("ijkl, ip, jq, kr, ls -> pqrs", self.v_ao, c, c, c, c, optimize=True)
        return eri_mo

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
        conv_criteria = 1e-12
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
                return energy_tot, self.v_ao, coefficient, fock_mat, fock_p, x_mat, c_p
    
    def mp2_loop(self, v_mo, eps):
        e_mp2_corr = 0.0
        t = int(self.n_ao / 2)
        for a in range(0, t):
            for b in range(0, t):
                for r in range(t, self.n_ao):
                    for s in range(t, self.n_ao):
                        eps_sum = eps[a] + eps[b] - eps[r] - eps[s]
                        e_mp2_corr += (v_mo[a, b, r, s] * (2 * v_mo[r, s, a, b] - v_mo[r, s, b, a])) / eps_sum
        return e_mp2_corr

    def chem_to_phys(self, v_mo):
        loop_indices = repeat_loop(self.n_ao, 4)
        v_mo_new = np.zeros(v_mo.shape)
        for a, b, r, s in loop_indices:
            v_mo_new[a, b, r, s] = v_mo[a, r, b, s]
        return v_mo_new

    def mp3_loop(self, v_mo, eps):
        e_mp3 = 0.0
        num = int(self.n_ao / 2)
        for a in range(0, num):
            for b in range(0, num):
                for c in range(0, num):
                    for d in range(0, num):
                        for r in range(num, self.n_ao):
                            for s in range(num, self.n_ao):
                                # Diagram 2
                                eps_2 = (eps[a] + eps[d] - eps[r] - eps[s]) * (eps[c] + eps[b] - eps[r] - eps[s])
                                e_mp3 += (2 * v_mo[a, d, r, s] * v_mo[c, b, a, d] * v_mo[r, s, c, b]) / eps_2

                                # Diagram 7
                                eps_7 = (eps[a] + eps[c] - eps[r] - eps[s]) * (eps[d] + eps[b] - eps[r] - eps[s])
                                e_mp3 += (-1 * v_mo[a, c, r, s] * v_mo[d, b, a, c] * v_mo[s, r, d, b]) / eps_7

        for a in range(0, num):
            for b in range(0, num):
                for r in range(num, self.n_ao):
                    for s in range(num, self.n_ao):
                        for t in range(num, self.n_ao):
                            for u in range(num, self.n_ao):
                                # Diagram 1
                                eps_1 = (eps[a] + eps[b] - eps[r] - eps[u]) * (eps[a] + eps[b] - eps[t] - eps[s])
                                e_mp3 += (2 * v_mo[a, b, r, u] * v_mo[r, u, t, s] * v_mo[t, s, a, b]) / eps_1

                                # Diagram 8
                                eps_8 = (eps[a] + eps[b] - eps[t] - eps[r]) * (eps[a] + eps[b] - eps[u] - eps[s])
                                e_mp3 += (-1 * v_mo[a, b, r, t] * v_mo[t, r, u, s] * v_mo[u, s, a, b]) / eps_8

        for a in range(0, num):
            for b in range(0, num):
                for c in range(0, num):
                    for r in range(num, self.n_ao):
                        for s in range(num, self.n_ao):
                            for t in range(num, self.n_ao):
                                # Diagram 3
                                eps_3 = (eps[a] + eps[c] - eps[r] - eps[t]) * (eps[a] + eps[b] - eps[s] - eps[t])
                                e_mp3 += (-4 * v_mo[a, c, r, t] * v_mo[r, b, s, c] * v_mo[s, t, a, b]) / eps_3

                                # Diagram 4
                                eps_4 = (eps[b] + eps[c] - eps[r] - eps[t]) * (eps[a] + eps[c] - eps[s] - eps[t])
                                e_mp3 += (-4 * v_mo[b, c, r, t] * v_mo[r, a, s, b] * v_mo[s, t, a, c]) / eps_4

                                # Diagram 5
                                eps_5 = (eps[a] + eps[c] - eps[r] - eps[t]) * (eps[a] + eps[b] - eps[r] - eps[s])
                                e_mp3 += (8 * v_mo[a, c, r, t] * v_mo[b, t, s, c] * v_mo[r, s, a, b]) / eps_5

                                # Diagram 6
                                eps_6 = (eps[c] + eps[b] - eps[r] - eps[t]) * (eps[a] + eps[b] - eps[r] - eps[s])
                                e_mp3 += (2 * v_mo[c, b, r, t] * v_mo[a, t, s, c] * v_mo[r, s, a, b]) / eps_6

                                # Diagram 9
                                eps_9 = (eps[c] + eps[b] - eps[r] - eps[t]) * (eps[a] + eps[c] - eps[s] - eps[t])
                                e_mp3 += (2 * v_mo[b, c, r, t] * v_mo[a, r, b, s] * v_mo[t, s, a, c]) / eps_9

                                # Diagram 10
                                eps_10 = (eps[c] + eps[b] - eps[r] - eps[t]) * (eps[a] + eps[c] - eps[s] - eps[t])
                                e_mp3 += (2 * v_mo[c, b, r, t] * v_mo[r, a, s, b] * v_mo[s, t, a, c]) / eps_10

                                # Diagram 11
                                eps_11 = (eps[a] + eps[b] - eps[r] - eps[s]) * (eps[c] + eps[b] - eps[r] - eps[t])
                                e_mp3 += (-4 * v_mo[a, b, r, s] * v_mo[s, c, a, t] * v_mo[r, t, b, c]) / eps_11

                                # Diagram 12
                                eps_12 = (eps[b] + eps[c] - eps[t] - eps[r]) * (eps[a] + eps[b] - eps[r] - eps[s])
                                e_mp3 += (-4 * v_mo[b, c, r, t] * v_mo[a, t, s, c] * v_mo[r, s, a, b]) / eps_12
        return e_mp3

    def rmp3_loop(self, v_mo, eps):
        loop_indices = repeat_loop(self.n_ao, 4)
        g_mo = v_mo
        t_iajb = np.zeros(v_mo.shape)
        L_mo = 2 * g_mo - g_mo.swapaxes(-1, -3)
        for i, a, j, b in loop_indices:
            eps_sum = eps[i] + eps[j] - eps[a] - eps[b]
            if eps_sum > 1e-14:
                t_iajb[i, a, j, b] = g_mo[i, a, j, b] / eps_sum
        T_iajb = 4 * t_iajb - 2 * t_iajb.swapaxes(-1, -3)
        mp2_corr = (t_iajb * L_mo).sum()

        mp3_X_iajb = (
                + 0.5 * np.einsum("icjd, acbd -> iajb", t_iajb, g_mo)
                + 0.5 * np.einsum("kalb, kilj -> iajb", t_iajb, g_mo)
                + np.einsum("iakc, bjkc -> iajb", t_iajb, L_mo)
                - np.einsum("kajc, bcki -> iajb", t_iajb, g_mo)
                - np.einsum("kaic, bjkc -> iajb", t_iajb, g_mo)
        )
        mp3_corr = (T_iajb * mp3_X_iajb).sum()
        return -mp2_corr, mp3_corr

    def hf_greens_loop(self, omega, c, fock_ao, fock_mo, fock_sao, c_p):
        I_n = np.identity(2)
        A_ao_real = np.zeros(omega.shape)
        A_ao_imag = np.zeros(omega.shape)
        A_sao_real = np.zeros(omega.shape)
        A_sao_imag = np.zeros(omega.shape)
        A_mo_real = np.zeros(omega.shape)
        A_mo_imag = np.zeros(omega.shape)
        w_dim, n_dim = omega.shape
        s_mo = np.dot(np.dot(c.T, self.s_ao), c)
        s_sao = np.dot(np.dot(c_p.T, self.s_ao), c_p)
        for i in range(w_dim):
            for j in range(n_dim):
                G_ao = np.linalg.inv(np.dot(omega[i, j], self.s_ao) - fock_ao)
                G_sao = np.linalg.inv(np.dot(omega[i, j], I_n) - fock_sao)
                G_mo = np.linalg.inv(np.dot(omega[i, j], I_n) - fock_mo)
                A_ao = - (1 / np.pi) * np.trace(np.dot(G_ao, self.s_ao))
                A_sao = - (1 / np.pi) * np.trace(np.dot(G_sao, s_sao))
                A_mo = - (1 / np.pi) * np.trace(np.dot(G_mo, s_mo))
                A_ao_imag[i, j] += np.imag(A_ao)
                A_sao_imag[i, j] += np.imag(A_sao)
                A_mo_imag[i, j] += np.imag(A_mo)
                A_ao_real[i, j] += np.real(A_ao)
                A_sao_real[i, j] += np.real(A_sao)
                A_mo_real[i, j] += np.real(A_mo)
        return A_ao_imag, A_sao_imag, A_mo_imag, A_ao_real, A_sao_real, A_mo_real

    def gf2_loop(self, omega, c, c_p, x_mat, v_mo, fock_ao, fock_mo, fock_sao):
        I_n = np.identity(self.n_ao)
        eps = np.diag(fock_mo)
        loop_indices = repeat_loop(self.n_ao, 2)

        A_mo_real = np.zeros(omega.shape)
        A_mo_imag = np.zeros(omega.shape)

        # G_mo_new = np.zeros(omega.shape)
        A_ao_2_imag = np.zeros(omega.shape)

        # G_ao_new = np.zeros(omega.shape)
        A_sao_2_imag = np.zeros(omega.shape)

        A_mo_2_real = np.zeros(omega.shape)
        A_mo_2_imag = np.zeros(omega.shape)

        # G_sao_new = np.zeros(omega.shape)
        Sig_mo = np.zeros(omega.shape)
        Sig_ao = np.zeros(omega.shape)
        Sig_sao = np.zeros(omega.shape)

        w_dim, n_dim = omega.shape
        s_mo = np.dot(np.dot(c.T, self.s_ao), c)
        s_sao = np.dot(np.dot(c_p.T, self.s_ao), c_p)

        for k in range(w_dim):
            for p in range(n_dim):
                G_mo = np.linalg.inv(np.subtract(np.dot(omega[k, p], I_n), fock_mo))
                # G_ao = np.linalg.inv(np.dot(omega[k, p], self.s_ao) - fock_ao)
                # G_sao = np.linalg.inv(np.dot(omega[k, p], I_n) - fock_sao)

                A_mo = np.trace(np.dot(G_mo, self.s_ao))
                A_mo_imag[k, p] += - (1 / np.pi) * np.imag(np.trace(np.dot(G_mo, s_mo)))
                A_mo_real[k, p] += (-1 / np.pi) * np.real(A_mo)

                Sigma_mo = np.zeros((self.n_ao, self.n_ao), dtype=complex)
                # Sigma_ao = np.zeros((self.n_ao, self.n_ao), dtype=complex)
                # Sigma_sao = np.zeros((self.n_ao, self.n_ao), dtype=complex)
                t = int(self.n_ao / 2)

                for i, j in loop_indices:
                    for a in range(0, t):
                        for r in range(t, self.n_ao):
                            for s in range(t, self.n_ao):
                                eps_1 = omega[k, p] + eps[a] - eps[r] - eps[s]
                                Sigma_mo[i, j] += (v_mo[r, s, i, a] * (2 * v_mo[j, a, r, s] - v_mo[a, j, r, s])) / eps_1
                                # Sigma_ao[i, j] += (v_ao[r, s, i, a] * (
                                # 2 * v_ao[j, a, r, s] - v_ao[a, j, r, s])) / eps_1
                                # Sigma_sao[i, j] += (v_sao[r, s, i, a] * (
                                # 2 * v_sao[j, a, r, s] - v_sao[a, j, r, s])) / eps_1
                            for b in range(0, t):
                                eps_2 = omega[k, p] + eps[r] - eps[a] - eps[b]
                                Sigma_mo[i, j] += (v_mo[a, b, i, r] * (2 * v_mo[j, r, a, b] - v_mo[r, j, a, b])) / eps_2
                                # Sigma_ao[i, j] += (v_ao[a, b, i, r] * (
                                # 2 * v_ao[j, r, a, b] - v_ao[r, j, a, b])) / eps_2
                                # Sigma_sao[i, j] += (v_sao[a, b, i, r] * (
                                # 2 * v_sao[j, r, a, b] - v_sao[r, j, a, b])) / eps_2

                Sigma_ao = np.dot(np.dot(c, Sigma_mo), c.T)
                Sigma_sao = np.dot(np.dot(c_p, Sigma_mo), c_p.T)

                G_mo_new = np.linalg.inv(np.subtract(np.subtract(np.dot(omega[k, p], I_n), fock_mo), Sigma_mo))
                # G_fft = self.gf_fft(omega, np.real(G_mo))
                G_ao_new = np.linalg.inv(np.subtract(np.subtract(np.dot(omega[k, p], self.s_ao), fock_ao), Sigma_ao))
                G_sao_new = np.linalg.inv(np.subtract(np.subtract(np.dot(omega[k, p], I_n), fock_sao), Sigma_sao))

                t_mo = np.diag(np.imag(Sigma_mo))
                t_ao = np.diag(np.imag(Sigma_ao))
                t_sao = np.diag(np.imag(Sigma_sao))

                Sig_mo[k, p] = t_mo[1]
                Sig_ao[k, p] = t_ao[1]
                Sig_sao[k, p] = t_sao[1]

                # print(np.dot(np.dot(c.T, Sigma_ao), c) - Sigma_mo)
                A_mo_new = np.trace(np.dot(G_mo_new, s_mo))
                A_ao_new = np.trace(np.dot(G_ao_new, self.s_ao))
                A_sao_new = np.trace(np.dot(G_sao_new, s_sao))

                A_mo_2_real[k, p] += (-1 / np.pi) * np.real(A_mo_new)

                A_mo_2_imag[k, p] += (-1 / np.pi) * np.imag(A_mo_new)
                A_ao_2_imag[k, p] += (-1 / np.pi) * np.imag(A_ao_new)
                A_sao_2_imag[k, p] += (-1 / np.pi) * np.imag(A_sao_new)
        return Sig_mo, Sig_ao, Sig_sao, A_mo_imag, A_mo_2_imag, A_ao_2_imag, A_sao_2_imag

    def gf_fft(self, iw, g_iw):
        beta = 1 / 100
        n = lambda w: ((w * np.pi / beta) / 2 - 0.5), iw

        def fft(g_iw):
            N = len(n)
            if N == 1:
                return g_iw
            else:
                g_odd = fft(g_iw[::2])
                g_even = fft(g_iw[1::2])
                factor = np.exp(-1j * (2 * np.arrange(N) + 1) * np.pi / (beta * N))
                g_tau = np.cpncatenate([g_even+factor[:int(N/2)*g_odd, g_even+factor[int(N/2):]*g_odd]])
            return g_tau
        G_tau = fft(g_iw)
        return G_tau

    def gf_ft(self, tau, N, F_mo, beta):
        I_n = np.identity(self.n_ao)
        G = np.zeros((self.n_ao, self.n_ao))
        for n in range(0, 1000):
            iw = (2 * n + 1) * (np.pi / beta)
            G += np.real(np.linalg.inv(np.dot(1j * iw, I_n) - F_mo) * np.exp(-1j * iw * tau))
        for n in range(1000, N):
            iw = (2j * n + 1) * (np.pi / beta)
            G += np.real(1 / iw)
        return G[0,0]