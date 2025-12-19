import re
import numpy as np
import matplotlib.pyplot as plt


# -------- Parsing --------
FRAME_RE = re.compile(r"Frame ID:\s*(\d+)")
SIZE_RE  = re.compile(r"Buffer:\s*,\s*size:\s*(\d+)")
# Matches "(0.5,0)" "(0.499013,0.0313953)" including scientific notation
PAIR_RE  = re.compile(r"\(\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)")

def parse_log(path: str):
    """
    Returns:
      frames: list of dict {id: int, size: int, x: complex np.ndarray}
    """
    frames = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        m_id = FRAME_RE.search(lines[i])
        if not m_id:
            i += 1
            continue

        frame_id = int(m_id.group(1))

        # Find size line within the next few lines
        size = None
        j = i + 1
        while j < len(lines) and j < i + 8:
            m_size = SIZE_RE.search(lines[j])
            if m_size:
                size = int(m_size.group(1))
                break
            j += 1
        if size is None:
            # No buffer size found; skip this frame
            i += 1
            continue

        # Find the line (or lines) that contain the bracketed buffer
        # We'll collect until we see a closing ']'
        buf_text = ""
        k = j + 1
        while k < len(lines):
            buf_text += lines[k].strip() + " "
            if "]" in lines[k]:
                break
            k += 1

        pairs = PAIR_RE.findall(buf_text)
        x = np.array([float(a) + 1j * float(b) for (a, b) in pairs], dtype=np.complex64)

        # If log truncates with "...", x may be shorter than 'size'
        # We'll keep what we have but warn.
        if x.size != size:
            print(f"[warn] Frame {frame_id}: parsed {x.size} samples, header says {size}. "
                  f"(Likely truncated log with '...')")

        frames.append({"id": frame_id, "size": size, "x": x})
        i = k + 1

    return frames


# -------- STFT / Spectrogram --------
def stft(x: np.ndarray, n_fft: int, hop: int, window: str = "hann"):
    if x.size < n_fft:
        return np.empty((0, n_fft), dtype=np.complex64)

    if window == "hann":
        w = np.hanning(n_fft).astype(np.float32)
    else:
        w = np.ones(n_fft, dtype=np.float32)

    n_frames = 1 + (x.size - n_fft) // hop
    X = np.empty((n_frames, n_fft), dtype=np.complex64)

    for i in range(n_frames):
        start = i * hop
        seg = x[start:start + n_fft] * w
        X[i, :] = np.fft.fftshift(np.fft.fft(seg, n=n_fft))

    return X


def plot_spectrogram(frames, fs_hz: float, n_fft=1024, hop=256, eps=1e-12):
    # Concatenate frames into one long stream (simplest)
    x_all = np.concatenate([fr["x"] for fr in frames if fr["x"].size > 0])
    if x_all.size < n_fft:
        raise ValueError("Not enough samples to compute spectrogram. Increase log length or reduce n_fft.")

    X = stft(x_all, n_fft=n_fft, hop=hop)
    S = 20.0 * np.log10(np.abs(X) + eps)  # dB magnitude

    # Axes
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0/fs_hz))
    times = np.arange(S.shape[0]) * (hop / fs_hz)

    plt.figure()
    plt.imshow(
        S.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram (magnitude, dB)")
    plt.colorbar(label="dB")
    plt.tight_layout()


def plot_3d_waterfall(frames, fs_hz: float, n_fft=1024, hop=1024, eps=1e-12, max_slices=80):
    # 3D plot: each time-slice is one FFT magnitude curve
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x_all = np.concatenate([fr["x"] for fr in frames if fr["x"].size > 0])
    X = stft(x_all, n_fft=n_fft, hop=hop)
    if X.shape[0] == 0:
        raise ValueError("Not enough samples for 3D plot.")

    # optionally downsample time slices so it doesn't become unreadable
    idx = np.linspace(0, X.shape[0]-1, min(max_slices, X.shape[0])).astype(int)
    Xs = X[idx, :]

    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0/fs_hz))
    times = idx * (hop / fs_hz)

    Z = 20.0 * np.log10(np.abs(Xs) + eps)  # [time_slices, freq_bins]

    F, T = np.meshgrid(freqs, times)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(F, T, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s)")
    ax.set_zlabel("Magnitude (dB)")
    ax.set_title("3D Waterfall (STFT magnitude)")
    plt.tight_layout()


# -------- Main --------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Path to transmitter log file")
    ap.add_argument("--fs", type=float, required=True, help="Sample rate (Hz), e.g. 1e6")
    ap.add_argument("--fft", type=int, default=1024, help="FFT size")
    ap.add_argument("--hop", type=int, default=256, help="Hop size for spectrogram")
    ap.add_argument("--plot3d", action="store_true", help="Also generate 3D waterfall plot")
    args = ap.parse_args()

    frames = parse_log(args.logfile)
    if not frames:
        raise SystemExit("No frames parsed. Check log format / file path.")

    plot_spectrogram(frames, fs_hz=args.fs, n_fft=args.fft, hop=args.hop)
    if args.plot3d:
        # For 3D, use bigger hop so slices are distinct
        plot_3d_waterfall(frames, fs_hz=args.fs, n_fft=args.fft, hop=args.fft)

    plt.show()
