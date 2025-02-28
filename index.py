import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft

def calculate_K(N):
    """Calculates the K matrix for DFT/IDFT."""
    k = np.arange(N)
    mu = np.arange(N)
    K = np.outer(k, mu)
    return K

def calculate_W(N):
    """Calculates the W matrix for IDFT."""
    K = calculate_K(N)
    W = np.exp(+1j * 2 * np.pi / N * K)
    return W

def idft_matrix(X_mu, W, N):
    """Calculates IDFT using matrix notation."""
    x_k = 1/N * np.matmul(W, X_mu)
    return x_k

# Input signal xµ = [6, 2, 4, 3, 4, 5, 0, 0, 0]T
X_mu_variant = np.array([6, 2, 4, 3, 4, 5, 0, 0, 0]).reshape(-1, 1) # Reshape to column vector

N_values = [9, 16, 32] # Different values of N

for N in N_values:
    print(f"\n--- N = {N} ---")

    # Calculate matrices K and W
    K_matrix = calculate_K(N)
    W_matrix = calculate_W(N)

    print("\nMatrix K:")
    print(K_matrix)

    print("\nMatrix W:")
    print(W_matrix)

    # Extend X_mu to length N if N > 9, pad with zeros
    if N > 9:
        X_mu = np.concatenate((X_mu_variant, np.zeros((N - 9, 1))), axis=0)
    else:
        X_mu = X_mu_variant[:N] # Truncate if N < 9, though task implies N >= 9

    # Perform IDFT using matrix notation
    x_synthesized = idft_matrix(X_mu, W_matrix, N)
    k = np.arange(N)

    # Plot the synthesized signal
    plt.figure()
    plt.stem(k, np.real(x_synthesized), markerfmt='C0o', basefmt='C0:', linefmt='C0:', label='Real part')
    plt.stem(k, np.imag(x_synthesized), markerfmt='C1o', basefmt='C1:', linefmt='C1:', label='Imaginary part')
    plt.plot(k, np.real(x_synthesized), 'C0-', lw=0.5) 
    plt.plot(k, np.imag(x_synthesized), 'C1-', lw=0.5) 
    plt.xlabel("Sample k")
    plt.ylabel("Synthesized signal x[k]")
    plt.title(f"Synthesized discrete-time signal (N={N})")
    plt.legend()
    plt.grid(True)
    plt.show()