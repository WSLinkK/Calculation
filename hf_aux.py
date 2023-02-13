import numpy as np
import scipy.linalg
import time

# n_ao: number of atomic orbital
# n_mo": number of molecular orbital


def get_integrals(n_ao):
    # Generate Molecular Integral
    file_v_int = "fort.3001"
    file_h_core_int = "fort.3005"
    file_eigenvalues = "fort.2024"
    file_eigenvectors = "fort.2999"
    file_overlap = "fort.3003"

    h_core_ao = get_t_int(file_h_core_int, n_ao)
    v_ao = get_v_int(file_v_int, n_ao)
    s_ao = get_overlap_matrix(file_overlap, n_ao)
    return h_core_ao, v_ao, s_ao


def get_t_int(file_t_int, n_ao):
    # Generate  Kinetic Integral for H_core
    f = open(file_t_int, 'r')
    t_int = np.zeros((n_ao, n_ao))
    for line in f:
        t_oks = line.split()
        i = int(t_oks[0])
        j = int(t_oks[1])
        t_int[i - 1, j - 1] = float(t_oks[2].replace("D", "E"))
        t_int[j - 1, i - 1] = t_int[i - 1, j - 1]
    f.close()
    return t_int


def get_v_int(file_integrals, n_ao):
    # Generate Two-Electron Integral
    v_int = np.zeros((n_ao, n_ao, n_ao, n_ao))
    f = open(file_integrals, 'r')
    for line in f:
        t_oks = line.split()
        i = int(t_oks[0])
        j = int(t_oks[1])
        k = int(t_oks[2])
        l = int(t_oks[3])
        v_int[i - 1, j - 1, k - 1, l - 1] = float(t_oks[4].replace("D", "E"))
    for mu in range(n_ao):
        for nu in range(mu + 1):
            for lam in range(n_ao):
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


def get_overlap_matrix(file_overlap, n_ao):
    # Generate Overlap Integral
    f = open(file_overlap, 'r')
    s_mat = np.zeros((n_ao, n_ao))
    for line in f:
        toks = line.split()
        i = int(toks[0])
        j = int(toks[1])
        s_mat[i - 1, j - 1] = float(toks[2].replace("D", "E"))
        s_mat[j - 1, i - 1] = s_mat[i - 1, j - 1]
    f.close()
    return s_mat


def calculate_s12_transformation_matrix(s_mat):
    # Diagonalize Matrix
    # SL = Ls
    # X = S^(1/2) = L  s^(-1/2)  L^T
    # Create L and s from eigenvalue and eigenvector from matrix S
    s, L = np.linalg.eigh(s_mat)
    inv_s_12 = (np.diag(s**(-0.5)))
    x_mat = np.dot(L, np.dot(inv_s_12, np.transpose(L)))
    return x_mat


def reject_linear_dependencies(s_eigval, s_eigvec):
    # Check for Linear dependency
    k = 0
    if s_eigval[0] <= 1.e-8:
        for eigval in s_eigval:
            if eigval[0] <= 1.e-15:
                k = k + 1
    return s_eigval[k:], s_eigvec[:, k:]


def calc_hf_energy(h_core_ao, g, p_new):
    energy = 0.5 * np.trace(np.dot(p_new, h_core_ao + (h_core_ao + g)))
    return energy


def find_nuc_energy(file_hf):
    f = open(file_hf, 'r')
    for line in f:
        if "Nuclear repulsion:" in line:
            toks = line.split()
            nuc_rep = float(toks[2])
    f.close()
    return nuc_rep


def calculate_unit_cell_correction(dmd, vint, dim):
    uc_two_el = np.zeros((dim, dim))
    g_c = np.zeros((dim, dim))
    g_x = np.zeros((dim, dim))

    for mu in range(dim):
        for nu in range(dim):
            for lam in range(dim):
                for sig in range(dim):
                    uc_two_el[mu, nu] = uc_two_el[mu, nu] + dmd[lam, sig] * (
                                vint[mu, nu, sig, lam] - 0.5 * vint[mu, lam, sig, nu])
                    g_c[mu, nu] = g_c[mu, nu] + dmd[lam, sig] * (vint[mu, nu, sig, lam])
                    g_x[mu, nu] = g_x[mu, nu] + dmd[lam, sig] * (-0.5 * vint[mu, lam, sig, nu])
    return uc_two_el, g_c, g_x


def calculate_density_matrix(noc, eigenvec):
    dm_ac = 2.0 * np.dot(eigenvec[:, 0:noc], np.transpose(eigenvec[:, 0:noc]))
    return dm_ac


def transform_v(v_ao, cmo):
    nao = cmo.shape[0]
    nmo = cmo.shape[1]
    cmot = np.zeros(cmo.shape)
    cmot[:, :] = np.transpose(cmo)[:, :]

    v_aux = np.zeros((nao, nao, nao, nmo))
    for mu in range(nmo):
        for i in range(nao):
            for j in range(nao):
                for k in range(nao):
                    for l in range(nao):
                        v_aux[i, j, k, mu] = v_aux[i, j, k, mu] + v_ao[i, j, k, l] * cmo[l, mu]

    v_ao_1 = np.zeros((nao, nao, nao, nmo))
    for mu in range(nmo):
        for nu in range(nmo):
            for i in range(nao):
                for j in range(nao):
                    for k in range(nao):
                        v_ao_1[nu, j, k, mu] = v_ao_1[nu, j, k, mu] + v_aux[i, j, k, mu] * cmot[nu, i]

    v_aux_2 = np.zeros((nao, nao, nao, nmo))
    for mu in range(nmo):
        for nu in range(nmo):
            for sigma in range(nmo):
                for j in range(nao):
                    for k in range(nao):
                        v_aux_2[nu, j, sigma, mu] = v_aux_2[nu, j, sigma, mu] + v_ao_1[nu, j, k, mu] * cmo[k, sigma]

    v_ao_2 = np.zeros((nao, nao, nao, nmo))
    for mu in range(nmo):
        for nu in range(nmo):
            for sigma in range(nmo):
                for delta in range(nmo):
                    for j in range(nao):
                        v_ao_2[nu, delta, sigma, mu] = v_ao_2[nu, delta, sigma, mu] + v_aux_2[nu, j, sigma, mu] * cmot[
                            delta, j]
    return v_ao_2


def mo_eri_transformation_n8(n_ao, v_ao, c):
    # O(N^8)
    v_mo = np.zeros(v_ao.shape, dtype=float)
    for p in range(n_ao):
        for q in range(n_ao):
            for r in range(n_ao):
                for s in range(n_ao):
                    for mu in range(n_ao):
                        for nu in range(n_ao):
                            for sigma in range(n_ao):
                                for delta in range(n_ao):
                                    v_mo[p, q, r, s] += c[mu, p] * c[nu, q] * v_ao[mu, nu, sigma, delta] * c[sigma, r] * c[delta, s]
    return v_mo


def repeat_loop(dim: int, nested: int) -> np.ndarray:
    return np.array([np.tile(np.repeat(np.arange(dim), dim**(nested - 1 - i)), dim**i) for i in range(nested)]).T


def mo_eri_transformation_n4(n_ao, v_ao, c):
    eri_mo = np.zeros(v_ao.shape, dtype=float)
    for p, q, r, s,mu, nu, sigma, delta in repeat_loop(n_ao, 8):
        eri_mo[p, q, r, s] += c[mu, p] * c[nu, q] * v_ao[mu, nu, sigma, delta] * c[sigma, r] * c[delta, s]
    return eri_mo


def mo_eri_transformation_n5(n_ao, v_ao, c):
    eri_mo_n5 = np.zeros(v_ao.shape, dtype=float)
    eri_mo_tmp1 = np.zeros(v_ao.shape, dtype=float)
    eri_mo_tmp2 = np.zeros(v_ao.shape, dtype=float)
    eri_mo_tmp3 = np.zeros(v_ao.shape, dtype=float)
    loop_indices = repeat_loop(n_ao, 4)

    for p in range(n_ao):
        for mu, nu, sigma, delta in loop_indices:
            eri_mo_tmp1[p, nu, sigma, delta] += c[mu, p] * v_ao[mu, nu, sigma, delta]

    for q in range(n_ao):
        for p, nu, sigma, delta  in loop_indices:
            eri_mo_tmp2[p, q, sigma, delta] += c[nu, q] * eri_mo_tmp1[p, nu, sigma, delta]

    for r in range (n_ao):
        for p, q, sigma, delta in loop_indices:
            eri_mo_tmp3[p, q, r, delta] += c[sigma, r] * eri_mo_tmp2[p, q, sigma, delta]

    for s in range (n_ao):
        for p, q, r, delta in loop_indices:
            eri_mo_n5[p, q, r, s] += c[delta, s] * eri_mo_tmp3[p, q, r, delta]

    return eri_mo_n5

def fock_transform(fock_matrix, c):
    h_core_mo = np.dot(np.dot(c.T, fock_matrix), c)
    return h_core_mo
