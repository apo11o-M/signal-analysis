import numpy as np
import matplotlib.pyplot as plt

d = np.genfromtxt("tx_iq.csv", delimiter=",", names=True)
i = d["i"]
q = d["q"]
x = i + 1j*q

# --- time domain snippet
Nsnip = min(len(x), 2000)
plt.figure()
plt.plot(i[:Nsnip], label="I")
plt.plot(q[:Nsnip], label="Q")
plt.title("Tone TX - Time domain (snippet)")
plt.xlabel("sample")
plt.grid(True)
plt.legend()

# --- FFT magnitude (uncalibrated bins)
Nfft = 16384
Nfft = min(Nfft, len(x))
w = np.hanning(Nfft)
X = np.fft.fftshift(np.fft.fft(x[:Nfft] * w))
mag = 20*np.log10(np.maximum(np.abs(X), 1e-12))

plt.figure()
plt.plot(mag)
plt.title("Tone TX - FFT magnitude (uncalibrated bins)")
plt.xlabel("FFT bin (shifted)")
plt.ylabel("dB")
plt.grid(True)

plt.show()
