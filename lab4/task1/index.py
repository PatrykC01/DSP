import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth

# Parameters
f = 8 # Frequency (Hz)
t = np.linspace(0, 1, 1000, endpoint=False)  # Time vector

# Triangular wave (modification of sawtooth wave)
triangular_wave = sawtooth(2 * np.pi * f * t, width=0.5)

# Sampling parameters
f_sample = 12  # Sampling frequency (Hz)

# Sampling the signal
t_sample = np.arange(0, 1, 1 / f_sample)
samples = sawtooth(2 * np.pi * f * t_sample, width=0.5)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, triangular_wave, label="Original Triangular Wave")
plt.stem(t_sample, samples, linefmt='r-', markerfmt='ro', basefmt=" ", label="Sampled Signal")
plt.title("Triangular Wave Sampling")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

from scipy.signal import resample

# Reconstructing the signal
num_samples = 1000
reconstructed_signal = resample(samples, num_samples)

# Plotting the reconstruction
plt.figure(figsize=(10, 6))
plt.plot(t, triangular_wave, label='Original Triangular Wave')
plt.plot(t, reconstructed_signal, label='Reconstructed Signal', linestyle='--')
plt.title('Signal Reconstruction')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()
