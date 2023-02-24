import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib.pyplot as plt


fock_mo= np.array([[-0.578227, 0.00000],[0.00000, 0.670075]])


# Def Functions
omega = lambda n:((2 * n + 1) * (np.pi / beta))
G = lambda w, t:(np.linalg.inv((w * I_n) - fock_mo) * np.exp(-w * t))
G_neg = lambda w, t:(np.linalg.inv((w * I_n) - fock_mo) * np.exp(w * t))
G_iw = lambda w:(np.linalg.inv((w * I_n) - fock_mo))

cut_tau = 1
cut_n = 0.5                         # narrow to get accurate mo_energy for G_HF
beta = 1000
N = 10000
eta = 0.02j
I_n = np.identity(2)
tau = np.arange(0, 120, cut_tau) 
n = np.arange(0, N, cut_n)
n_neg = np.arange(-N, 0, cut_n)                
boundry = int(1000/cut_n)           # Boundry Condition when tau = beta

w = omega(n)
w_neg = omega(n_neg)

iw = w + eta
iw_neg = w_neg + eta

w_t = np.append(w_neg, w)
print(w_t.shape)

A_w = np.zeros(len(iw))
A_w_neg = np.zeros(len(iw_neg))
G_w = np.zeros((len(iw), 2, 2), complex)
G_w_neg = np.zeros((len(iw_neg), 2, 2), complex)
# Getting Spectral Function
for i in range(len(iw)):
    G_w[i, :, :] += G_iw(iw[i])
    G_w_neg[i, :, :] += G_iw(iw_neg[i])
    A_w[i] += (-1/np.pi) * np.imag(np.trace(G_w[i, :, :]))
    A_w_neg[i] += (-1/np.pi) * np.imag(np.trace(G_w_neg[i, :, :]))

ReGW = np.real(np.trace(G_w, axis1=1, axis2=2))
plt.figure()
plt.plot(w, ReGW, 'k')
plt.savefig('ReG_w.png', format='png', dpi=800)

ImGW = np.imag(np.trace(G_w, axis1=1, axis2=2))
plt.figure()
plt.plot(w, ImGW, 'k', label='IP')
plt.savefig('ImG_w.png', format='png', dpi=800)

plt.figure()
plt.plot(w, A_w, '-k', label='EA')
plt.plot(w_neg, A_w_neg, '-r', label='IP')
#plt.plot(freq, A_w_fft, '.b', label='fft')
plt.xlim(-1, 1)
plt.savefig('A_tau.png', dpi=800)




# Time Domain 
G_tau = np.zeros((len(tau),2,2), complex)

ImG_tau = np.zeros(len(tau))
ReG_tau = np.zeros(len(tau))
fft_arr = len
for t in range(len(tau)):
    test = np.zeros((2,2), complex)
    for i in range(boundry):
        test += G(iw[i], tau[t])
        test += G_neg(iw_neg[i], tau[t])
    for i in range(boundry, N):
        test += I_n/(iw[i])
        test += I_n/(iw_neg[i])
    G_tau[t,:,:] += test/beta
    #print(G_tau[t,:,:])
    TrG = np.trace(G_tau[t,:,:])
    ImG_tau[t] += np.imag(TrG)
    ReG_tau[t] += np.real(TrG)
    print(f'tau={t * cut_tau:.2f}, G(tau)={TrG}')

G = np.zeros((N, 2, 2), complex)
for n in range(0, N):
    for t in range(tau):
        G[n, :, :] = G_tau[t,:,:] * np.exp(iw[n] * tau[t])

# Numpy fft test
G_fft = fftn(G, axes=(1, 2))
print(G_fft[0])
N = len(G_fft)
print(N)
n = np.arange(N)
freq = omega(n)
print(freq.shape)

A_w_fft = np.imag(np.trace(G_fft, axis1=1, axis2=2))
print(A_w_fft.shape)


plt.figure()
plt.plot(tau, A_w_fft, '-k', label='fft')
plt.savefig('fft.png', dpi=800)

print('Fock_mo:')
print(fock_mo)
print('mo_energy from G_HF:')
print(w[np.argmax(A_w)])
print(w_neg[np.argmax(A_w_neg)])






