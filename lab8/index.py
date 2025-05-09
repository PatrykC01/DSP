import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import neurokit2 as nk

ecg = nk.ecg_simulate(duration=20, sampling_rate=300)
fs = 300  

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

filtered_ecg = bandpass_filter(ecg, 0.7, 30, fs)

plt.plot(ecg, label='Raw ECG')
plt.plot(filtered_ecg, label='Filtered ECG')
plt.title("ECG Signal")
plt.legend()
plt.show()

signals, info = nk.ecg_process(filtered_ecg, sampling_rate=fs)
r_peaks = info["ECG_R_Peaks"]

plt.plot(filtered_ecg, label="Filtered ECG")
plt.plot(r_peaks, filtered_ecg[r_peaks], "ro", label="R-peaks")
plt.title("R-peak Detection")
plt.legend()
plt.show()

time = np.arange(0, len(ecg)) / fs
heart_rate = 60 / ((r_peaks[1:] - r_peaks[:-1]) / fs)
plt.plot(time[r_peaks[1:]], heart_rate)
plt.title("Heart Rate Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Heart Rate [bpm]")
plt.show()

print("Bandpass filtering (0.7–30 Hz) removes baseline wander and high-frequency noise, which significantly facilitates R-peak detection by the neurokit2 algorithm.")
