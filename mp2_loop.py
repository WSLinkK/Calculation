import rhf
import hf_aux
import numpy as np


def repeat_loop(dim: int, nested: int) -> np.ndarray:
    return np.array([np.tile(np.repeat(np.arange(dim), dim ** (nested - 1 - i)), dim ** i) for i in range(nested)]).T


name = 'H2'
R = 4.0
n_ao = 2
nuc_rep = hf_aux.find_nuc_energy(name + '_' + str(R) + '.out')
N = 2
rhf_scf = rhf.scf_loop(n_ao)
E_tot, v_ao, c, fock_mat = rhf_scf.hf_loop(n_ao, nuc_rep, N)

v_mo = hf_aux.mo_eri_transformation_n5(n_ao, v_ao, c)
diag_fock = rhf.fock_transform(fock_mat, c)
t = np.dot(np.dot(np.transpose(c), fock_mat), c)


def mp2_loop(n_ao, v_mo, diag_fock):
    eps = np.zeros(n_ao)
    for i in range(n_ao):
        eps[i] = diag_fock[i, i]
    e_mp2_tot = 0.0
    for a in range(n_ao):
        for r in range(n_ao):
            for b in range(n_ao):
                for s in range(n_ao):
                    eps_sum = eps[a] + eps[b] - eps[r] - eps[s]
                    if eps_sum > 1e-14:
                        e_mp2 = (v_mo[a, b, r, s] * (2* v_mo[r, s, a, b] - v_mo[r, s, b, a])) / eps_sum
                        e_mp2_tot += e_mp2
    return e_mp2_tot


mp2 = mp2_loop(n_ao, v_mo, diag_fock)
print(mp2)
