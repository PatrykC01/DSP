import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf

# Simulate ARMA(1,1) process
np.random.seed(42)
ar_params = [0.75]  # AR(1) coefficient
ma_params = [0.65]  # MA(1) coefficient
arma_process = ArmaProcess([1, -ar_params[0]], [1, ma_params[0]])
n_samples = 500
arma_signal = arma_process.generate_sample(n_samples)

# Fit AR(2) model
ar2_model = ARIMA(arma_signal, order=(2, 0, 0))
ar2_result = ar2_model.fit()

# Fit MA(2) model
ma2_model = ARIMA(arma_signal, order=(0, 0, 2))
ma2_result = ma2_model.fit()

# Fit ARMA(1,1) model
arma11_model = ARIMA(arma_signal, order=(1, 0, 1))
arma11_result = arma11_model.fit()

# Plot original signal and fitted values
plt.figure(figsize=(12, 6))
plt.plot(arma_signal, label="Original Signal", alpha=0.7)
plt.plot(ar2_result.fittedvalues, label="AR(2) Fit", linestyle="--")
plt.plot(ma2_result.fittedvalues, label="MA(2) Fit", linestyle=":")
plt.plot(arma11_result.fittedvalues, label="ARMA(1,1) Fit", linestyle="-.")
plt.legend()
plt.title("Model Fits Comparison")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Compare AIC and residual variance
print("Model Comparison:")
print(f"AR(2) AIC: {ar2_result.aic}, Residual Variance: {np.var(ar2_result.resid):.4f}")
print(f"MA(2) AIC: {ma2_result.aic}, Residual Variance: {np.var(ma2_result.resid):.4f}")
print(f"ARMA(1,1) AIC: {arma11_result.aic}, Residual Variance: {np.var(arma11_result.resid):.4f}")

# Plot residual autocorrelations
plt.figure(figsize=(12, 6))
plt.stem(acf(ar2_result.resid, nlags=20), linefmt="b-", markerfmt="bo", basefmt="r-", label="AR(2) Residual ACF")
plt.stem(acf(ma2_result.resid, nlags=20), linefmt="g-", markerfmt="go", basefmt="r-", label="MA(2) Residual ACF")
plt.stem(acf(arma11_result.resid, nlags=20), linefmt="m-", markerfmt="mo", basefmt="r-", label="ARMA(1,1) Residual ACF")
plt.title("Residual Autocorrelation Comparison")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.legend()
plt.grid(True)
plt.show()
