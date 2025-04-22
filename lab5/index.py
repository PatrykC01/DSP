import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Parameters
N = 5000  # Total number of samples
fs = 1000  # Sampling frequency
segment_length = 1000  # Length of each segment
variance_values = [1, 5, 0.5, 2, 3]  # Piecewise variances

# Generate a non-stationary signal with piecewise variance
signal = np.concatenate([np.random.normal(0, np.sqrt(var), segment_length) for var in variance_values])

# Analyze local mean and variance
window_size = 500
step = 250
local_means = []
local_vars = []
positions = []

for start in range(0, len(signal) - window_size, step):
    window = signal[start:start + window_size]
    local_means.append(np.mean(window))
    local_vars.append(np.var(window))
    positions.append(start + window_size // 2)

# Plot local mean and variance
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(positions, local_means, marker='o')
plt.title("Local Mean (Sliding Window)")
plt.ylabel("Mean")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(positions, local_vars, marker='s', color='orange')
plt.title("Local Variance (Sliding Window)")
plt.xlabel("Sample Position")
plt.ylabel("Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

# Apply Welch's method in separate segments
segment_psds = []
frequencies = None

for i in range(len(variance_values)):
    start = i * segment_length
    end = start + segment_length
    f, Pxx = welch(signal[start:end], fs=fs, nperseg=256, noverlap=128, window='hann')
    segment_psds.append(Pxx)
    if frequencies is None:
        frequencies = f

# Plot PSDs for each segment
plt.figure(figsize=(10, 6))
for i, Pxx in enumerate(segment_psds):
    plt.semilogy(frequencies, Pxx, label=f"Segment {i+1}")
plt.title("PSD Comparison Across Segments (Welch's Method)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency [dB/Hz]")
plt.legend()
plt.grid(True)
plt.show()
