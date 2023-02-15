import os
import rhf
import sys
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc
from numpy import ones, copy, cos, tan, pi, linspace


def gaussxw(N):
    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3, 4 * N - 1, N) / (4 * N + 2)
    x = cos(pi * a + 1 / (8 * N * N * tan(a)))
    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = ones(N, float)
        p1 = copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = (N + 1) * (p0 - x * p1) / (1 - x * x)
        dx = p1 / dp
        x -= dx
        delta = max(abs(dx))
    # Calculate the weights
    w = 2 * (N + 1) * (N + 1) / (N * N * (1 - x * x) * dp * dp)
    return x, w


def gaussxwab(N, a, b):
    x, w = gaussxw(N)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

# --HF_for_H2_molecule

R = 1.4
n_ao = 2
name = 'H2'
E_HF = []
E_MP2 = []
E_MP3 = []

file_out = name + '_' + str(R) + '.out'
s = 'cp H2_data/' + name + '_' + str(R) + '.tar.gz . '
os.system(s)
s = 'cp H2_data/' + name + '_' + str(R) + '.out . '
os.system(s)
s = 'tar -xvzf ' + name + '_' + str(R) + '.tar.gz'
os.system(s)
# sys.stdout = open('H2_data_'+str(R)+'.txt', 'w')
print(f'_____________________{file_out}_____________________')
nuc_rep = rhf.find_nuc_energy(name + '_' + str(R) + '.out')
# N = input("Number of electrons? ")
N = 2
rhf_scf = rhf.scf_loop(n_ao)

print(f'_____________________Starting RHF Procedure_____________________')
E_hf, v_ao, c, fock_ao, fock_sao, x_mat, c_p = rhf_scf.hf_loop(n_ao, nuc_rep, N)
fock_mo = rhf.fock_transform(fock_ao, c)
print(f'fock_ao:\n{fock_ao}')
print(f'fock_sao:\n{fock_sao}')
print(f'fock_mo:\n{fock_mo}')
eps = np.diag(fock_mo)

v_ao_new = rhf_scf.chem_to_phys(v_ao)
v_mo = rhf_scf.mo_eri_transformation_n5(c)
v_sao = rhf_scf.mo_eri_transformation_n5(x_mat)
v_sao_new = rhf_scf.chem_to_phys(v_sao)
v_mo_new = rhf_scf.chem_to_phys(v_mo)
#print(v_sao)
mp2 = rhf_scf.mp2_loop(v_mo_new, eps)
print(f'E_mp2:\n{mp2}')
print(f"_____________________Starting HF-Green's Function Procedure_____________________")
w = np.arange(-10.0, 10.0, 0.001, dtype=float)
n = np.array([0.005, 0.00001, 0.2, 0.1]) #0.005, 0.00001, 1e-6,
omega = np.zeros((len(w), len(n)), dtype=complex)
for q in range(len(w)):
    for p in range(len(n)):
        omega[q, p] += w[q] + 1j * n[p]

#print(c_p)
#print(x_mat)
#print(np.dot(x_mat, c_p))
#Sig_mo, Sig_ao, Sig_sao, A_mo_imag, A_mo_2_imag, A_ao_2_imag, A_sao_2_imag = rhf_scf.gf2_loop(omega, c, c_p, x_mat, v_mo, fock_ao, fock_mo, fock_sao)


tau = np.linspace(0, 10, 1000)
G_tau = np.zeros(len(tau))
for i in range(0, len(tau)):
    G_tau[i] += rhf_scf.gf_ft(tau[i], 10000, fock_mo, 100.0)

'''
N = 100
beta = 100
x, w = gaussxwab(N, 0.0, 100.0)
G = lambda tau:G_tau[tau] * np.exp(-1j * iw * tau)
G_iw =[]

for i in range(N):
    iw = (2 * n + 1) * (np.pi / beta)
    G_iw.app = w[i] * G[x[i]]
'''
print(f"_____________________Creating Spectral Function Graph _____________________")



plt.figure()
plt.plot(tau, G_tau)
plt.savefig('ft.pdf', dpi=800)




