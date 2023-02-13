import os
import rhf
import sys
import numpy as np
import matplotlib.pyplot as plt
# --HF_for_H2_molecule

distance = [0.7] #, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.3, 4.6, 4.8, 4.9, 5.2, 5.5, 5.8, 6.0, 6.4, 6.8, 7.0, 10.0]
n_ao = 2
name = 'H2'
E_HF = []
E_MP2 = []

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
    print(f'_____________________Starting SCF Procedure_____________________')
    E_hf, v_ao, c, fock_mat = rhf_scf.hf_loop(n_ao,  nuc_rep, N)
    print(f'_____________________Starting MP2 Procedure_____________________')
    E_HF += [E_hf]
    diag_fock = rhf.fock_transform(fock_mat, c)
    v_mo = rhf_scf.mo_eri_transformation_n5(c)
    print(f'n5:\n{v_mo}')
    v_mo_np = rhf_scf.mo_eri_einsum(c)
    print(f'np:\n{v_mo_np}')
    #e_mp2 = rhf_scf.mp2_loop(v_mo, diag_fock)
    #E_MP2 += [E_hf - e_mp2]
    #print(f'MP2_Correlation_Energy:{e_mp2}')
    #print(f'MP2_Total_Energy:{E_hf - e_mp2}')
    #sys.stdout.close()

#E_HF = [x + 2 * 0.4666 for x in E_HF]
#E_MP2 = [x + 2 * 0.4666 for x in E_MP2]

#print(f'_____________________Creating Energy Diagram_____________________')
#plt.figure()
#plt.plot(distance, E_HF, 'k', label='RHF')
#plt.plot(distance, E_MP2, 'r', label='MP2')
#plt.xlabel(r'Distance (A)')
#plt.ylabel(r'$E(H_2)-2E(H)$ (Ha)')
#plt.hlines(y=0, xmin=0, xmax=10, color='k', linestyles=':')
#plt.xlim(0, 10)
#plt.legend()
#plt.savefig('H2_Energy_Diagram.pdf', dpi=800)

