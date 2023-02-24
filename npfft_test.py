import matplotlib.pyplot as plt
from numpy.fft import fft
import numpy as np



a = np.mgrid[:4, :2, :2][0]
print(a)
tra = np.trace(a, axis1=1, axis2=2)
print(tra)
print(a)
print(a.shape)
b = np.fft.fftn(a, axes=(1, 2))
print(b)
print(b.shape)