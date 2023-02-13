import os
import rhf
import sys
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc

# --HF_for_H2_molecule

distance = [
    0.7]  # 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.3, 4.6, 4.8, 4.9, 5.2, 5.5, 5.8, 6.0, 6.4, 6.8, 7.0, 10.0]
n_ao = 2
name = 'H2'
E_HF = []
E_MP2 = []
E_MP3 = []

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
    E_hf, v_ao, c, fock_ao, fock_sao = rhf_scf.hf_loop(n_ao, nuc_rep, N)
    fock_mo = rhf.fock_transform(fock_ao, c)
    print(f'fock_ao:\n{fock_ao}')
    print(f'fock_sao:\n{fock_sao}')
    print(f'fock_mo:\n{np.round(fock_mo, decimals=15)}')

    eps = np.diag(fock_mo)
    v_mo = rhf_scf.mo_eri_transformation_n5(c)
    v_mo_new = rhf_scf.chem_to_phys(v_mo)
    mp2 = rhf_scf.mp2_loop(v_mo_new, eps)
    print(f'E_mp2:\n{mp2}')
    print(f"_____________________Starting HF-Green's Function Procedure_____________________")
    w = np.arange(-10.0, 10.0, 0.01, dtype=float)
    n = np.array([0.0005, 0.00001, 1e-6, 0.1])
    omega = np.zeros((len(w), len(n)), dtype=complex)
    for q in range(len(w)):
        for p in range(len(n)):
            omega[q, p] += w[q] + 1j * n[p]
    print(omega[1,0])
    I_n = np.identity(2)
    s = np.dot(omega[1, 1], I_n)
    A_ao, A_sao, A_mo = rhf_scf.hf_greens_loop(omega, fock_ao, fock_mo, fock_sao, )
    print(A_mo[552])
    plt.figure()
    plt.plot(w, A_ao[:, 0], 'ok', label='$\eta = 0.0005$')  # x_axis, y_axis, color, name
    plt.plot(w, A_mo[:, 0], '.r', label='$\eta = 0.0005$')  # x_axis, y_axis, color, name
    #plt.plot(w, A_sao[:, 0], '-b', label='$\eta = 0.0005$')  # x_axis, y_axis, color, name
    plt.ylabel(r'$A(\omega)$')
    plt.xlabel(r'$\omega$')
    #plt.xlim(-5,1)
    plt.legend()
    #plt.show()
    plt.savefig('Spectral Function.pdf', dpi=800)
