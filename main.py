import os
import rhf
import sys
import numpy as np
#import matplotlib.pyplot as plt
import tracemalloc
# --HF_for_H2_molecule

distance = [1.4] # 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.3, 4.6, 4.8, 4.9, 5.2, 5.5, 5.8, 6.0, 6.4, 6.8, 7.0, 10.0]
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
    #sys.stdout = open('H2_data_'+str(R)+'.txt', 'w')
    print(f'_____________________{file_out}_____________________')
    nuc_rep = rhf.find_nuc_energy(name + '_' + str(R) + '.out')
    # N = input("Number of electrons? ")
    N = 2
    rhf_scf = rhf.scf_loop(n_ao)

    print(f'_____________________Starting RHF Procedure_____________________')
    E_hf, v_ao, c, fock_ao, fock_sao, x_mat, c_p = rhf_scf.hf_loop(n_ao, nuc_rep, N)

    print(f'_____________________Starting MP2 Procedure_____________________')
    E_HF += [E_hf]
    fock_mo = rhf.fock_transform(fock_ao, c)
    eps = np.diag(fock_mo)
    v_mo = rhf_scf.mo_eri_transformation_n5(c)
    v_mo_new = rhf_scf.chem_to_phys(v_mo)
    e_mp2 = rhf_scf.mp2_loop(v_mo_new, eps)
    E_MP2 += [E_hf + e_mp2]
    print(f'MP2_Correlation_Energy:{e_mp2}')
    print(f'MP2_Total_Energy:{E_hf + e_mp2}')

    print(f'_____________________Starting MP3 Procedure_____________________')
    e_mp3 = rhf_scf.mp3_loop(v_mo_new, eps)
    E_MP3 += [E_hf + e_mp2 + e_mp3]
    print(f'MP2+MP3_Correlation_Energy:{e_mp2 + e_mp3}')
    print(f'MP3_Total_Energy:{E_hf + e_mp2 + e_mp3}')
    mp2, mp3= rhf_scf.rmp3_loop(v_mo, eps)
    print(mp2)
    print(mp2 + mp3)
    # sys.stdout.close()

# E_HF = [x + 2 * 0.4666 for x in E_HF]
# E_MP2 = [x + 2 * 0.4666 for x in E_MP2]
# E_MP3 = [x + 2 * 0.4666 for x in E_MP3]

# print(f'_____________________Creating Energy Diagram_____________________')
#
# plt.figure()
# plt.plot(distance, E_HF, '.-k', label='RHF')  # x_axis, y_axis, color, name
# plt.plot(distance, E_MP2, '.-r', label='MP2')
# plt.plot(distance, E_MP3, '.-b', label='MP3')
# plt.xlabel(r'Distance (a.u.)')
# plt.ylabel(r'$Energy$ (a.u)')
# plt.hlines(y=0, xmin=0, xmax=10, color='k', linestyles=':') # add horizontal line
# plt.xlim(0, 8)
# plt.ylim(-0.3, 0.5)
# plt.legend()
# plt.savefig('H2_Energy_Diagram.pdf', dpi=800)
