import numpy as np
import cv2
from matplotlib import pyplot as plt
import librosa
import librosa.display
from scipy.signal import chirp, square, sawtooth
from scipy.io.wavfile import write
from scipy.ndimage import gaussian_filter
import os

if not os.path.exists('generated_images'):
    os.makedirs('generated_images')
if not os.path.exists('generated_audio'):
    os.makedirs('generated_audio')
if not os.path.exists('results'):
    os.makedirs('results')

# 1. Signal Generation 

# Image: Gaussian-blurred dot in the center of a black image.
def generate_blurred_dot(size=256, sigma=10, filename='generated_images/blurred_dot.png'):
    """Generates a grayscale image with a Gaussian-blurred dot in the center."""
    img = np.zeros((size, size), dtype=np.float64) 
    img[size//2, size//2] = 1.0  
    blurred = gaussian_filter(img, sigma=sigma)
    
    blurred_norm = (blurred / np.max(blurred)) * 255 if np.max(blurred) > 0 else blurred
    cv2.imwrite(filename, blurred_norm.astype(np.uint8))
    print(f"Generated image saved as {filename}")
    return blurred_norm.astype(np.uint8)

# Audio: Low-frequency hum at 50 Hz for 5 seconds.
def generate_hum(freq=50, duration=5, fs=44100, amplitude=0.5, filename='generated_audio/hum_50hz.wav'):
    """Generates a low-frequency hum (sine wave) and saves it as a WAV file."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    y = amplitude * np.sin(2 * np.pi * freq * t)
   
    y_int16 = (y * 32767).astype(np.int16)
    write(filename, fs, y_int16)
    print(f"Generated audio saved as {filename}")
    return y, fs

image_filename = 'generated_images/blurred_dot.png'
audio_filename = 'generated_audio/hum_50hz.wav'
generated_image_v4 = generate_blurred_dot(size=256, sigma=15, filename=image_filename) 
generated_audio_y, generated_audio_sr = generate_hum(freq=50, duration=5, filename=audio_filename)

print("\n--- Starting Image Analysis ---")

# Task 1: Load Grayscale Image and Compute 2D DFT
print("Task 1: Loading generated image and computing 2D DFT...")
image_v4 = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
if image_v4 is None:
    raise ValueError(f"Failed to load generated image: {image_filename}")

f = np.fft.fft2(image_v4)
fshift = np.fft.fftshift(f) 
print("2D DFT computed.")

# Task 2: Visualize the Magnitude Spectrum
print("Task 2: Visualizing magnitude spectrum...")
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8) 

plt.figure(figsize=(8, 8))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (Variant 4 - Blurred Dot)')
plt.axis('off')
plt.colorbar()
plt.savefig('results/magnitude_spectrum_v4.png')
plt.show()
print("Magnitude spectrum saved as results/magnitude_spectrum_v4.png")

# Task 3: Apply Low-Pass and High-Pass Filters
print("Task 3: Applying Low-Pass and High-Pass filters...")
rows, cols = image_v4.shape
crow, ccol = rows // 2 , cols // 2

radius_lp = 30 
mask_lp = np.zeros((rows, cols), np.uint8)
cv2.circle(mask_lp, (ccol, crow), radius_lp, 1, thickness=-1)

mask_hp = 1 - mask_lp

fshift_lp = fshift * mask_lp
fshift_hp = fshift * mask_hp
print("Filters applied.")

# Task 4: Reconstruct Images from Filtered Spectra
print("Task 4: Reconstructing images from filtered spectra...")

f_ishift_lp = np.fft.ifftshift(fshift_lp)
img_back_lp = np.fft.ifft2(f_ishift_lp)
img_back_lp = np.abs(img_back_lp)

f_ishift_hp = np.fft.ifftshift(fshift_hp)
img_back_hp = np.fft.ifft2(f_ishift_hp)
img_back_hp = np.abs(img_back_hp)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image_v4, cmap='gray')
plt.title('Original Image (Blurred Dot)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_back_lp, cmap='gray')
plt.title(f'Low-Pass Filtered (r={radius_lp})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_back_hp, cmap='gray')
plt.title(f'High-Pass Filtered (r={radius_lp})')
plt.axis('off')

plt.tight_layout()
plt.savefig('results/filtered_images_v4.png')
plt.show()
print("Filtered images saved as results/filtered_images_v4.png")

print("\n--- Starting Audio Analysis ---")

# Task 5: Load Audio Signal and Compute Spectrogram
print("Task 5: Loading generated audio and computing spectrogram...")
y_v4, sr_v4 = generated_audio_y, generated_audio_sr 

D = librosa.stft(y_v4)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12, 5))
librosa.display.specshow(S_db, sr=sr_v4, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (Variant 4 - 50Hz Hum)')
plt.ylim(0, 1000) 
plt.tight_layout()
plt.savefig('results/spectrogram_v4.png')
plt.show()
print("Spectrogram saved as results/spectrogram_v4.png")

# Task 6: Compare Time-Domain and Frequency-Domain Representations
print("Task 6: Comparing Time and Frequency domain representations...")

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y_v4, sr=sr_v4)
plt.title('Time Domain Signal (50Hz Hum)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

Y_fft = np.fft.fft(y_v4)
freqs = np.fft.fftfreq(len(Y_fft), 1/sr_v4)

plt.subplot(2, 1, 2)
n_half = len(freqs) // 2
plt.plot(freqs[:n_half], np.abs(Y_fft[:n_half]))
plt.title('Frequency Domain Signal (Magnitude Spectrum)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 500) 

plt.tight_layout()
plt.savefig('results/time_freq_comparison_v4.png')
plt.show()
print("Time/Frequency comparison plot saved as results/time_freq_comparison_v4.png")

# Task 7: Apply Band-Pass Filter (to Image)
print("Task 7: Applying Band-Pass filter to the image...")
r1_bp = 10 
r2_bp = 40 

mask_bp = np.zeros((rows, cols), dtype=np.float32)
cv2.circle(mask_bp, (ccol, crow), r2_bp, 1.0, thickness=-1) 
cv2.circle(mask_bp, (ccol, crow), r1_bp, 0.0, thickness=-1) 

fshift_bp = fshift * mask_bp
f_ishift_bp = np.fft.ifftshift(fshift_bp)
img_bandpass = np.fft.ifft2(f_ishift_bp)
img_bandpass = np.abs(img_bandpass)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_v4, cmap='gray')
plt.title('Original Image (Blurred Dot)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_bandpass, cmap='gray')
plt.title(f'Band-Pass Filtered (r1={r1_bp}, r2={r2_bp})')
plt.axis('off')

plt.tight_layout()
plt.savefig('results/bandpass_filtered_image_v4.png')
plt.show()
print("Band-pass filtered image saved as results/bandpass_filtered_image_v4.png")

print("\n--- Analysis Complete ---")
