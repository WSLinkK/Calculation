import os
import rhf
import sys
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc
from pyscf import gto, scf, agf2

distance = [1.4]
n_ao = 2
name = 'H2'

mol = gto.M(atom='H 0 0.7 0; H 0 -0.7 0', basis='sto-3g', unit='au')
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.run()

gf2 = agf2.AGF2(mf)
gf2.conv_tol = 1e-7
gf2.run()

# Access the GreensFunction object and compute the spectrum


for r, R in enumerate(distance):
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
    v_mo = rhf_scf.mo_eri_transformation_n5(c)
    v_mo_new = rhf_scf.chem_to_phys(v_mo)

    mp2 = rhf_scf.mp2_loop(v_mo_new, eps)
    print(f'E_mp2:\n{mp2}')
    print(f"_____________________Starting HF-Green's Function Procedure_____________________")
    w = np.arange(-10.0, 10.0, 0.01, dtype=float)
    n = np.array([0.005, 0.00001, 0.02, 0.1]) #0.005, 0.00001, 1e-6,

    gf = gf2.gf
    eta = 0.02
    spectrum = gf.real_freq_spectrum(w, eta=eta)

    omega = np.zeros((len(w), len(n)), dtype=complex)
    for q in range(len(w)):
        for p in range(len(n)):
            omega[q, p] += w[q] + 1j * n[p]

    Sig_mo, Sig_ao, Sig_sao, A_mo_imag, A_mo_2_imag, A_ao_2_imag, A_sao_2_imag = rhf_scf.gf2_loop(omega, c, c_p, x_mat, v_mo_new, fock_ao, fock_mo, fock_sao)

    print(f"_____________________Creating Spectral Function Graph _____________________")

    plt.figure(1)
    plt.title(f'$\eta={n[2]}$')
    plt.plot(w, A_mo_imag[:, 2], '-k', label='GF_MO')
    plt.plot(w, A_mo_2_imag[:, 2], '--b', label='GF2_MO')
    plt.plot(w, -spectrum, ':r', label='pyscf_RAGF2')
    plt.xlabel('$\omega$')
    plt.ylabel('ImA$(\omega)$')
    plt.legend
    plt.xlim(-2, 2)
    plt.savefig('pyscf_vs_my.pdf', dpi=800)
