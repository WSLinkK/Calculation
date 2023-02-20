import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt


fock_mo= np.array([[-0.578227, 0.00000],[0.00000, 0.670075]])
print(fock_mo)

cut = 0.2
tau = np.arange(0, 120, cut)
beta = 1000
N = 10000
n = np.arange(0, N, 0.5)
I_n = np.identity(2)
eta = 0.02j
omega = lambda n:((2 * n + 1) * (np.pi / beta))
G = lambda w, t:(np.linalg.inv((w * I_n) - fock_mo) * np.exp(-w * t))
G_iw = lambda w:(np.linalg.inv((w * I_n) - fock_mo))
w = omega(n)
w_neg = omega(-n)
iw = omega(n) + eta
iw_neg = omega(-n) + eta
A_w = np.zeros(len(iw))
A_w_neg = np.zeros(len(iw))
for i in range(len(iw)):
    A_w[i] += (-1/np.pi) * np.imag(np.trace(G_iw(iw[i])))
    A_w_neg[i] += (-1/np.pi) * np.imag(np.trace(G_iw(iw_neg[i])))

print(w[np.argmax(A_w)])
print(w_neg[np.argmax(A_w_neg)])


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
    #print(f'tau={t * cut:.2f}, G(tau)={ReG_tau[t]}')


plt.figure()
plt.plot(w, A_w, '-k', label='EA')
plt.plot(w_neg, A_w_neg, '-r', label='IP')
plt.xlim(-1, 1)
plt.savefig('A_tau.png', dpi=800)

plt.figure()
plt.plot(tau, ReG_tau, 'k')
plt.savefig('G_tau.png', dpi=800)