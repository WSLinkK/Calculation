import os
import rhf
import sys
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc

# --HF_for_H2_molecule

distance = [1.4]
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
    Sig_mo, Sig_ao, Sig_sao, A_mo_imag, A_mo_2_imag, A_ao_2_imag, A_sao_2_imag = rhf_scf.gf2_loop(omega, c, c_p, x_mat, v_mo, fock_ao, fock_mo, fock_sao)

    print(f"_____________________Creating Spectral Function Graph _____________________")
    #A_hs_sao, A_ao, sigma = hf_jacob.hf_loop(n_ao, R, nuc_rep, N)
    #print(test)

    imag, axis_1 = plt.subplots(2, 2, figsize=(12, 10))
    sigma, axis_2 = plt.subplots(2, 2, figsize=(12, 10))
    t = 0

    for x in range(2):
        for y in range(2):
            #axis_1[x, y].plot(w, A_mo_imag[:, t], '-k', label='A_mo')
            axis_1[x, y].plot(w, A_mo_imag[:, t], '-k', label='A_mo')
            axis_1[x, y].plot(w, A_ao_2_imag[:, t], ':g', label='A_ao')
            axis_1[x, y].plot(w, A_mo_2_imag[:, t], ':r', label='A_sao')
            axis_1[x, y].set_title(f'$\eta={n[t]}$')
            axis_1[x, y].legend()

            axis_2[x, y].plot(w, Sig_mo[:, t], ':b', label='A_mo_gf2')
            axis_2[x, y].plot(w, Sig_ao[:, t], '-g', label='A_ao_gf2')
            axis_2[x, y].plot(w, Sig_sao[:, t], ':r', label='A_sao_gf2')
            axis_2[x, y].set_title(f'$\eta={n[t]}$')
            axis_2[x, y].legend()
            #axis_1[x, y].set_xlim(-1, 1)
            t += 1
    imag.supxlabel('$\omega$')
    imag.supylabel('ImA$(\omega)$')
    imag.suptitle(f'Distance = {R}')
    sigma.supxlabel('$\omega$')
    sigma.supylabel('ImA$(\Sigma)$')
    sigma.suptitle(f'Distance = {R}')
    imag.savefig(f'Spectral_Function_Figure/test{R}.pdf', dpi=800)  # f'Spectral_Function_Figure/Spectral_Function_GF_2_mo_r_Imaginary_Part_{R}.pdf', dpi=800
    sigma.savefig(f'Sigma_Function_Figure/btest{R}.pdf', dpi=800)
    #wabplt.show()

'''
    
    print(f"_____________________Creating Spectral Function Graph _____________________")
    imag, axis_1 = plt.subplots(2, 2, figsize=(12, 10))
    real, axis_2 = plt.subplots(2, 2, figsize=(12, 10))
    t = 0
    for x in range(2):
        for y in range(2):
            A = A_ao_imag[:, t]
            eig_1 = np.argmax(A)
            A_rm = np.delete(A, eig_1)
            w_rm = np.delete(w, eig_1)
            eig_2 = np.argmax(A_rm)

            axis_1[x, y].plot(w, A_ao_imag[:, t], '-k', label='A_ao')
            axis_1[x, y].plot(w, A_mo_imag[:, t], ':r', label='A_mo')
            axis_1[x, y].plot(w, A_sao_imag[:, t], '.b', label='A_sao', markersize='2')
            axis_1[x, y].set_title(f'$\eta={n[t]}$')
            axis_1[x, y].legend()
            axis_1[x, y].set_xlim(-3, 3)

            axis_2[x, y].plot(w, A_ao_real[:, t], '-k', label='A_ao')
            axis_2[x, y].plot(w, A_mo_real[:, t], ':r', label='A_mo')
            axis_2[x, y].plot(w, A_sao_real[:, t], '.b', label='A_sao', markersize='2')
            axis_2[x, y].set_title(f'$\eta={n[t]}$')
            axis_2[x, y].legend()
            # axis_2[x, y].set_xlim((w_rm[eig_2] - 12), (w[eig_1] + 12))
            t += 1
    imag.supxlabel('$\omega$')
    imag.supylabel('ImA$(\omega)$')
    real.supxlabel('$\omega$')
    real.supylabel('ReA$(\omega)$')
    imag.suptitle(f'Distance = {R}')
    real.suptitle(f'Distance = {R}')
    #imag.savefig(f'Spectral_Function_Figure/Spectral_Function_Imaginary_Part_{R}.pdf', dpi=800)
    #real.savefig(f'Spectral_Function_Figure/Spectral_Function_Real_Part_{R}.pdf', dpi=800)
    plt.show()
'''

'''           
    plt.figure('Imaginary_Part')
    plt.title(f'$\eta={n[0]}$')
    plt.plot(w, A_ao_imag[:, 0], '-k', label='A_ao')  # x_axis, y_axis, color, name
    plt.plot(w, A_mo_imag[:, 0], ':r', label='A_mo')  # x_axis, y_axis, color, name
    plt.plot(w, A_sao_imag[:, 0], '.b', label='A_sao')  # x_axis, y_axis, color, name
    plt.ylabel(r'$ImA(\omega)$')
    plt.xlabel(r'$\omega(k)$')
    plt.xlim((w_rm[eig_2] - 1), (w[eig_1] + 1))
    plt.legend()
    #plt.savefig(f'Spectral_Function_Imaginary_Part_eta={n[0]}.pdf', dpi=800)
    plt.figure('Real_Part')
    plt.title(f'$\eta={n[0]}$')
    plt.plot(w, A_ao_real[:, 0], '-k', label='A_ao')  # x_axis, y_axis, color, name
    plt.plot(w, A_mo_real[:, 0], ':r', label='A_mo')  # x_axis, y_axis, color, name
    plt.plot(w, A_sao_real[:, 0], '.b', label='A_sao')  # x_axis, y_axis, color, name
    plt.ylabel(r'$ReA(\omega)$')
    plt.xlabel(r'$\omega(k)$')
    plt.xlim((w_rm[eig_2] - 1), (w[eig_1] + 1))
    plt.legend()
    plt.show()
    # plt.savefig('Spectral_Function_Imaginary_Part_eta={n[0]}.pdf', dpi=800)
'''
'''
    plt.figure('Imaginary_Part')
    plt.title(f'$\eta={n[0]}$')
    plt.plot(w, A_mo_imag_t[:, 0], '-k', label='A_mo_mp2')  # x_axis, y_axis, color, name
    plt.plot(w, A_mo_imag[:, 0], ':r', label='A_mo')  # x_axis, y_axis, color, name
    # plt.plot(w, A_sao_imag[:, 0], '-b', label='A_sao')  # x_axis, y_axis, color, name
    plt.ylabel(r'$ImA(\omega)$')
    plt.xlabel(r'$\omega(k)$')
    plt.xlim(-1, 1)
    plt.ylim(0, 1500)
    plt.legend()
    plt.show()
'''
