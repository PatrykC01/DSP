import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, lfilter

# =============================================================================
# Part A: Real-Time ECG Signal Processing (Simulation)
# =============================================================================

print("--- Part A: ECG Processing Simulation ---")

def synthetic_ecg(fs, duration, heart_rate=60):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False) 
    ecg = 0.6 * np.sin(2 * np.pi * heart_rate/60 * t) \
        + 0.2 * np.sin(2 * np.pi * 2 * heart_rate/60 * t) \
        + 0.1 * np.random.randn(len(t)) 
    return t, ecg

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def simulate_real_time_ecg_processing(ecg_signal, fs, block_size_ecg_sim): 

    total_samples = len(ecg_signal)
    num_blocks = total_samples // block_size_ecg_sim
    delay = block_size_ecg_sim / fs

    processed_signal = []
    t_axis = []

    print("Simulating real-time ECG filtering...")
    plt.figure(figsize=(10, 6)) 

    for i in range(num_blocks):
        block_start_idx = i * block_size_ecg_sim 
        block_end_idx = (i + 1) * block_size_ecg_sim 
        block = ecg_signal[block_start_idx:block_end_idx]
        
        filtered_block = bandpass_filter(block, fs)
        processed_signal.extend(filtered_block)
        
        current_t_block = np.arange(block_start_idx, block_end_idx) / fs
        t_axis.extend(current_t_block)

        plt.clf()
        plt.plot(t_axis, processed_signal, label="Filtered ECG", color='blue')
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"Real-Time ECG Simulation (Block {i+1}/{num_blocks})")
        plt.grid(True)
        plt.legend()
        plt.pause(0.01)  

        time.sleep(delay)  

    plt.show()
    return t_axis, processed_signal

fs_ecg = 300           # Sampling frequency (Hz)
duration_ecg = 8       # Signal duration (s)
block_size_ecg = 150   # Block size (samples)

t_original, ecg_original = synthetic_ecg(fs_ecg, duration_ecg) 
t_processed_ecg, ecg_processed = simulate_real_time_ecg_processing(ecg_original, fs_ecg, block_size_ecg_sim=block_size_ecg)

plt.figure(figsize=(12, 7))
plt.subplot(2,1,1)
plt.plot(t_original, ecg_original, label="Original ECG Signal")
plt.title("Original Synthetic ECG Signal (Variant 4)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)

plt.plot(t_processed_ecg, ecg_processed, label="Filtered ECG Signal", color='blue')
plt.title("Filtered ECG Signal after Block Processing (Variant 4)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# =============================================================================
# Part B: Kalman-Bucy Filtering
# =============================================================================

print("\n--- Part B: Kalman-Bucy Filter Simulation ---")

A_kb = -1.0
C_kb = 1.0
Q_kb = 0.8  
R_kb = 0.2  

dt_kb = 0.01 
T_kb = 10    
N_kb = int(T_kb / dt_kb)

P_val = 1.0     
x_true = 0.0  
x_hat = 0.0   

x_history = []
xhat_history = []
P_history = []
K_history = [] 

print(f"Kalman-Bucy Filter Parameters: A={A_kb}, C={C_kb}, Q={Q_kb}, R={R_kb}, P(0)={P_val}") 

for k in range(N_kb):
    w = np.random.normal(0, np.sqrt(Q_kb * dt_kb)) 
    x_true += dt_kb * (A_kb * x_true) + w 
    v = np.random.normal(0, np.sqrt(R_kb))       
    y = C_kb * x_true + v 

    K_gain = P_val * C_kb / R_kb  

    x_hat_dot = A_kb * x_hat + K_gain * (y - C_kb * x_hat) 
    x_hat += dt_kb * x_hat_dot
    
    P_dot = 2 * A_kb * P_val + Q_kb - (K_gain * C_kb * P_val) 
    P_val += dt_kb * P_dot

    x_history.append(x_true)
    xhat_history.append(x_hat)
    P_history.append(P_val)
    K_history.append(K_gain)


time_axis_kb = np.arange(0, T_kb, dt_kb)

plt.figure(figsize=(12, 8))

plt.subplot(3,1,1)
plt.plot(time_axis_kb, x_history, label="True State (x_true)")
plt.plot(time_axis_kb, xhat_history, label="Estimated State (x_hat)", linestyle="--")
plt.title("Kalman-Bucy Filter Simulation (Variant 4)")
plt.xlabel("Time [s]")
plt.ylabel("State x")
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(time_axis_kb, P_history, label="Error Covariance P(t)", color='green')
plt.title("Estimation Error Covariance P(t)")
plt.xlabel("Time [s]")
plt.ylabel("P(t)")
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(time_axis_kb, K_history, label="Kalman Gain K(t)", color='red')
plt.title("Kalman Gain K(t)")
plt.xlabel("Time [s]")
plt.ylabel("K(t)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
