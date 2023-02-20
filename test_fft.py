import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt


fock_mo= np.array([[-0.578227, 0.00000],[0.00000, 0.670075]])
print(fock_mo)

cut = 0.5
tau = np.arange(0, 120, cut)
beta = 1000
N = 10000
n = np.arange(0, N, 1)
I_n = np.identity(2)
eta = 0.02j
omega = lambda n:((2 * n + 1) * (np.pi / beta))
G = lambda w, t:(np.linalg.inv((w * I_n) - fock_mo) * np.exp(-w * t))
iw = omega(n) + eta
print(iw[0])
G_tau = np.zeros((2,2,len(tau)), complex)
ReG_tau = np.zeros(len(tau))
for t in range(len(tau)):
    test = np.zeros((2,2), complex)
    for i in range(beta):
        test += G(iw[i], tau[t])
    for i in range(beta, N):
        test += I_n/(iw[i])
    G_tau[:,:,t] += test/beta
    ReG_tau[t] += np.real(np.trace(G_tau[:,:,t]))
    print(f'tau={t * cut:.2f}, G(tau)={ReG_tau[t]}')

plt.figure()
plt.plot(tau, ReG_tau, 'k')
plt.savefig('G_tau.png', dpi=800)