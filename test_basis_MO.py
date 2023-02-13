import hf_aux
import scipy.linalg
import numpy as np
import math
from numpy import linalg as LA
def hf_loop(nao,R,nuc_rep,N):
    energyold = 0

    #--point_1_and_2---specify--integrals

    h_core_ao,v_ao,s_ao=hf_aux.get_integrals(nao)
    print "h_core_ao"
    print h_core_ao
    print h_core_ao.shape
    for i in xrange(nao):
        for j in xrange(nao):
            for k in xrange(nao):
                for l in xrange(nao):
                    print i,j,k,l,v_ao[i,j,k,l]
    print "overlap"
    print s_ao
    # h_core_ao=H_core_{mu,nu}  kinteic energy integral + nuclear-electron attraction
    # v_ao=[ij|kl] two-electron integrals Coulomb repulsion
    #--point_3--diagonalize_overlap_matrix

    X=hf_aux.calculate_S12_transformation_matrix(s_ao)

    #---point_4--guess_for_density_matrix_P_old
    #---here_you_have_to_write_it
          #---point_7_calculate_transformed--F_trans=X_transpose*F*X
    X_three=np.zeros((nao,nao))
    for i in xrange(nao):
     for j in xrange(nao):
      X_three[i,j]=X[j,i]
    P_old=np.zeros((nao,nao))
    
    n_iter_max=40
    converged=False
    conv_thresh=1e-07 #threshold for convergence checking
    F=np.zeros((nao,nao))
    for iter in xrange(n_iter_max):
        if converged==False:
          #---point_5_prepare_G_matrix_from_equation_3.154_from_P_old_and_v_ao
            G=np.zeros((nao,nao)) 
            for i in xrange(nao):
                for j in xrange(nao):
                    for l in xrange(nao):
                        for k in xrange(nao):
                            G[i,j]=G[i,j]+P_old[l,k]*(v_ao[i,j,k,l]-0.5*v_ao[i,l,k,j])
            print G
          #---point_6_obtain-F=h_core+G
            F[:,:]=h_core_ao[:,:] + G[:,:]
    
        
          #---point_7_calculate_transformed--F_trans=X_transpose*F*X
            F_prime=np.dot(np.dot(X_three,F),X)
          #---here_you_have_to_write_it
          #---point_8_diagonalize--F_trans
            epsilon, C_p=LA.eigh(F_prime)
            print "Eigenvalues", iter
            print epsilon
            print C_p
          #-We will use a sepecial library function scipy.linalg.eigh 
          #-that performs diagonalization of a Hermitian matrix
          # epsilons are eigenvalues and C_p are eigenvectors
	#---point_9_calculate---C=X*C_p
            C=np.dot(X,C_p)       
          #---here_you_have_to_write_it
          #---point_10_form_a_new_density_matrix_(P_new)_from_3.145
            P_new=np.zeros((nao,nao))
            for i in xrange(nao):
                for j in xrange(nao):
                    for k in xrange(N/2):
                        P_new[i,j]=P_new[i,j]+2*C[i,k]*C[j,k]
          #---here_you_have_to_write_it
          #---calculate-energy--to--see--how--it--is--converging
            Q=np.dot(P_new,h_core_ao+F)
            Eo=0.0
            for i in xrange(nao):
                Eo=Eo+0.5*Q[i,i]
            energy=Eo
            print energy,"is the Hartree-Fock ground state energy."
            print " Energy total electronic",energy
            print "Energy ",energy+nuc_rep," in iteration ", iter 
          #--point_11_check_convergence
            u=0
            t=np.zeros((nao,nao))
            for i in xrange(nao):
                for j in xrange(nao):
                    t[i,j]=(P_new[i,j]-P_old[i,j])**(2.0)
                    u=u+t[i,j]
            tmp_conv=(u)**(0.5)/nao
            tmp_conv=abs(energy-energyold)
            if tmp_conv<=conv_thresh:
                converged=True
                print "-----Converged----------for R=",R, "on Iteration", iter
                print "----Eigenvalues----"
                for i in xrange(nao):
                    print i, epsilon[i]
                print "----Eigenvectors---"
                for i in xrange(nao):
                    print "eigenvector_number ",i
                    for j in xrange(nao):
                        print j, C[i,j]
                cmot=np.zeros(C.shape)
                cmot[:,:]=np.transpose(C)[:,:]
                H_core_MO=np.zeros((2,2))
                H_core_MO=np.dot(np.dot(C.T,h_core_ao),C)
               # for i in xrange(2):
                # for j in xrange(2):
                 # for u in xrange(2):
                  # for a in xrange(2):
                   # H_core_MO[u,a]=H_core_MO[u,a]+C[u,i]*h_core_ao[i,j]*C[j,a]
               # for j in xrange(2):
                # for u in xrange(2):
                 # for a in xrange(2):
                  # H_core_MO[u,a]=H_core_MO[u,a]+C[a,j]*H_core_MO[u,j]*C[j,a]
                print "-----In Molecular orbitals---H_core_MO=",H_core_MO
                print "-----This was h_core_ao=",h_core_ao
                def transform_v(v_ao,cmo):
                 nao=cmo.shape[0]
                 nmo=cmo.shape[1]
                 cmot=N.zeros(cmo.shape)
                 cmot[:,:]=N.transpose(cmo)[:,:]
                 v_aux=N.zeros((nao,nao,nao,nmo))
                 for mu in xrange(nmo):
                  for i in xrange(nao):
                   for j in xrange(nao):
                    for k in xrange(nao):
                     for l in xrange(nao):
                      v_aux[i,j,k,mu]= v_aux[i,j,k,mu]+v_ao[i,j,k,l]*cmo[l,mu]
                 v_ao[:,:,:,:]=0.0
                 for mu in xrange(nmo):
                  for nu in xrange(nmo):
                   for i in xrange(nao):
                    for j in xrange(nao):
                     for k in xrange(nao):
                      v_ao[nu,j,k,mu]= v_ao[nu,j,k,mu]+v_aux[i,j,k,mu]*cmot[nu,i]
                 v_aux[:,:,:,:]=0.0
                 for mu in xrange(nmo):
                  for nu in xrange(nmo):
                   for sigma in xrange(nmo):
                    for j in xrange(nao):
                     for k in xrange(nao):
                      v_aux[nu,j,sigma,mu]= v_aux[nu,j,sigma,mu]+v_ao[nu,j,k,mu]*cmo[k,sigma]
                 v_ao[:,:,:,:]=0.0
                  for mu in xrange(nmo):
                   for nu in xrange(nmo):
                    for sigma in xrange(nmo):
                     for delta in xrange(nmo):
                      for j in xrange(nao):
                       v_ao[nu,delta,sigma,mu]= v_ao[nu,delta,sigma,mu]+v_aux[nu,j,sigma,mu]*cmot[delta,j]
                 return v_ao
                print "This is in MO now----VMO=",VMO
                print "------This was v_ao Atomic orbtial matrinx---v_ao=",v_ao
                G_MO=np.zeros((2,2))#something is wrong with G_MO since it should be diagonol and is not
                P_MO=np.array([[2,0],[0,0]])
                for u in xrange(2):
                 for a in xrange(2):
                  for b in xrange(2):
                   for t in xrange(2):
                    G_MO[u,a]=G_MO[u,a]+P_MO[t,b]*(v_ao[u,a,b,t]-0.5*v_ao[u,t,b,a])
                F_MO=H_core_MO+G_MO
                print "---G_MO=",G_MO
                print "--Fock_MO=",F_MO
                print "F_prime=", F_prime 
                B=np.dot(np.dot(C.T,s_ao),C)
                print B
            else:
                P_old[:,:]=P_new[:,:]
                energyold = energy
                print "---------Diverged----------for R=",R, "on Iteration", iter
           # numb=0
           # for i in xrange(nao):
                #numb=numb+np.dot(P_old,s_ao)[i,i]
                #print "numb"
           # print numb
    #Now for MP2 Correction to the ground state.

    

    #MP2 calculation begins here.
            # import mp2
            # VMO = mp2.ao_to_mo_conv(v_ao,C)
            # print VMO
            # E_MP2 = mp2.calculate_MP2_corr(VMO,epsilon,N,nao)
            # print E_MP2,"is the 2nd-order correction term."
    
            # print "Thus the MP2 corrected energy is:"
            # E_cor=energy+nuc_rep + E_MP2

  #      print energy + E_MP2
    # print "Let's do some Green's Function stuff."

    # import selfenergy
    # import mp2
    # beta = 50
    # VMO = mp2.ao_to_mo_conv(v_ao,C)
    # output = open("green.txt","wr")
    # space = " "
    # output.write("omega")
    # output.write(space)
    # output.write("Im(G)")
    # output.write(space)
    # output.write("i*w*G(w)")
    # output.write("\n")
    
    # for m in range(5000):
    #     omega = selfenergy.calculate_omega(m,beta)
    #     sigma_2 = selfenergy.matsubara_self_energy(VMO,m,beta,epsilon,N)
    #     gree = selfenergy.greens_function(sigma_2,epsilon,omega)
    #     output.write(str(omega))
    #     output.write(space)
    #     output.write(str(numpy.imag(gree[0,0])))
    #     output.write(space)
    #     output.write(str(1j*omega*numpy.imag(gree[0,0])))
    #     output.write("\n")
    # output.close()
