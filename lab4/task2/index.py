import numpy as np
import matplotlib.pyplot as plt

# Function for quantization
def quantize(signal, levels):
    min_val, max_val = min(signal), max(signal)
    step_size = (max_val - min_val) / levels
    quantized_signal = np.round((signal - min_val) / step_size) * step_size + min_val
    return quantized_signal

# Signal to quantize
original_signal = np.array([1.2, 2.3, 3.1, 4.5, 5.7])

# Quantization to 3 levels
quantized_signal = quantize(original_signal, levels=3)

# Calculate quantization error
quantization_error = original_signal - quantized_signal
print("Original Signal:", original_signal)
print("Quantized Signal:", quantized_signal)
print("Quantization Error:", quantization_error)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(original_signal, 'bo-', label="Original Signal")
plt.plot(quantized_signal, 'ro-', label="Quantized Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Quantization Effect")
plt.legend()
plt.grid()
plt.show()
