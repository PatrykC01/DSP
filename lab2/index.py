import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from scipy.signal.windows import hann, flattop

# Parametry dla wariantu 4
f1 = 600  # Hz
f2 = 600.25  # Hz
f3 = 599.75  # Hz
amplitude = 2
fs = 500  # Hz
N = 2000
k = np.arange(N)
x1 = amplitude * np.sin(2 * np.pi * f1 / fs * k)
x2 = amplitude * np.sin(2 * np.pi * f2 / fs * k)
x3 = amplitude * np.sin(2 * np.pi * f3 / fs * k)

w_rect = np.ones(N)
whann = hann(N, sym=False)
wflattop = flattop(N, sym=False)

X1wrect = fft(x1)
X2wrect = fft(x2)
X3wrect = fft(x3)

X1whann = fft(x1 * whann)
X2whann = fft(x2 * whann)
X3whann = fft(x3 * whann)

X1wflattop = fft(x1 * wflattop)
X2wflattop = fft(x2 * wflattop)
X3wflattop = fft(x3 * wflattop)

def fft2db(X):
    N = X.size
    Xtmp = 2 / N * X  
    Xtmp[0] *= 0.5  
    if N % 2 == 0:
        Xtmp[N//2] *= 0.5 
    return 20 * np.log10(np.abs(Xtmp))


df = fs / N
f = np.arange(N) * df


plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(f, fft2db(X1wrect), 'C0-', ms=3, label='f1=600Hz')
plt.plot(f, fft2db(X2wrect), 'C3-', ms=3, label='f2=600.25Hz')
plt.plot(f, fft2db(X3wrect), 'C2-', ms=3, label='f3=599.75Hz')
plt.xlim(75, 125)
plt.ylim(-60, 10)
plt.xticks(np.arange(75, 125, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.title('DFT spectra using FFT algorithm')

plt.subplot(3, 1, 2)
plt.plot(f, fft2db(X1whann), 'C0-', ms=3, label='f1=600Hz')
plt.plot(f, fft2db(X2whann), 'C3-', ms=3, label='f2=600.25Hz')
plt.plot(f, fft2db(X3whann), 'C2-', ms=3, label='f3=599.75Hz')
plt.xlim(75, 125)
plt.ylim(-60, 10)
plt.xticks(np.arange(75, 125, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.title('DFT spectra using FFT algorithm')

plt.subplot(3, 1, 3)
plt.plot(f, fft2db(X1wflattop), 'C0-', ms=3, label='f1=600Hz')
plt.plot(f, fft2db(X2wflattop), 'C3-', ms=3, label='f2=600.25Hz')
plt.plot(f, fft2db(X3wflattop), 'C2-', ms=3, label='f3=599.75Hz')
plt.xlim(75, 125)
plt.ylim(-60, 10)
plt.xticks(np.arange(75, 125, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.title('DFT spectra using FFT algorithm')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(f, fft2db(X1wrect), 'C0o-', ms=3, label='best case rect - f1=600Hz')
plt.plot(f, fft2db(X2wrect), 'C3o-', ms=3, label='worst case rect - f2=600.25Hz')
plt.xlim(75, 125)
plt.ylim(-60, 10)
plt.xticks(np.arange(75, 125, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.title('DFT spectra using FFT algorithm')

plt.subplot(3, 1, 2)
plt.plot(f, fft2db(X1whann), 'C0o-', ms=3, label='best case hann - f1=600Hz')
plt.plot(f, fft2db(X2whann), 'C3o-', ms=3, label='worst case hann - f2=600.25Hz')
plt.xlim(75, 125)
plt.ylim(-60, 10)
plt.xticks(np.arange(75, 125, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.title('DFT spectra using FFT algorithm')

plt.subplot(3, 1, 3)
plt.plot(f, fft2db(X1wflattop), 'C0o-', ms=3, label='best case flattop - f1=600Hz')
plt.plot(f, fft2db(X2wflattop), 'C3o-', ms=3, label='worst case flattop - f2=600.25Hz')
plt.xlim(75, 125)
plt.ylim(-60, 10)
plt.xticks(np.arange(75, 125, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.title('DFT spectra using FFT algorithm')
plt.tight_layout()
plt.show()

def winDTFTdB(w):
    N = w.size
    Nz = 100 * N 
    W = np.zeros(Nz)
    W[0:N] = w
    W = np.abs(fftshift(fft(W)))
    W /= np.max(W)  
    W = 20 * np.log10(W)
    Omega = 2 * np.pi / Nz * np.arange(Nz) - np.pi
    return Omega, W

plt.figure()
plt.plot([-np.pi, np.pi], [-3.01, -3.01], 'gray')   # mainlobe bandwidth
plt.plot([-np.pi, np.pi], [-13.3, -13.3], 'gray')    # rect max sidelobe
plt.plot([-np.pi, np.pi], [-31.5, -31.5], 'gray')     # hann max sidelobe
plt.plot([-np.pi, np.pi], [-93.6, -93.6], 'gray')     # flattop max sidelobe

Omega, W = winDTFTdB(w_rect)
plt.plot(Omega, W, label='rect')
Omega, W = winDTFTdB(whann)
plt.plot(Omega, W, label='hann')
Omega, W = winDTFTdB(wflattop)
plt.plot(Omega, W, label='flattop')
plt.xlim(-np.pi, np.pi)
plt.ylim(-120, 10)
plt.xlim(-np.pi/100, np.pi/100)  
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$|W(\Omega)| / dB$')
plt.legend()
plt.grid(True)
plt.title('Window DTFT spectra normalized to their mainlobe maximum')
plt.show()
