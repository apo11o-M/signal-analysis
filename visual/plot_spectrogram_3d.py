# python plot_spectrogram_3d.py --run ..\build\dumps\run_basic_tx_20251219_025903\ --stage tx --frame 0 --fs 1e7 --center --nfft 1024 --noverlap 768
# 
# plot_spectrogram_3d.py
#
# 3D spectrogram:
#   X = time (s)
#   Y = frequency (Hz)
#   Z = magnitude or power (linear or dB)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_frame(run_dir: Path, stage: str, frame_id: int | None):
    idx = pd.read_csv(run_dir / "index.csv")
    rows = idx[idx.stage == stage]

    if rows.empty:
        raise ValueError(f"No stage '{stage}' in index.csv")

    if frame_id is None:
        row = rows.iloc[0]
    else:
        row = rows[rows.frame_id == frame_id].iloc[0]

    path = run_dir / row.rel_path
    x = np.fromfile(path, dtype=np.complex64)
    return x, path


def compute_spectrogram(x, fs, nfft, noverlap, center):
    hop = nfft - noverlap
    if hop <= 0:
        raise ValueError("noverlap must be < nfft")

    win = np.hanning(nfft)
    win_energy = np.sum(win**2)

    nseg = 1 + (len(x) - nfft) // hop
    S = np.empty((nfft, nseg), dtype=np.float32)

    for k in range(nseg):
        seg = x[k * hop : k * hop + nfft] * win
        X = np.fft.fft(seg, nfft)
        P = np.abs(X)**2 / win_energy
        S[:, k] = P

    if center:
        S = np.fft.fftshift(S, axes=0)
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    else:
        freqs = np.fft.fftfreq(nfft, 1 / fs)

    times = np.arange(nseg) * hop / fs
    return times, freqs, S


def plot_3d_spectrogram(times, freqs, S, db, title):
    if db:
        S = 10 * np.log10(S + 1e-12)
        zlabel = "Power (dB)"
    else:
        zlabel = "Power"

    T, F = np.meshgrid(times, freqs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        T,
        F,
        S,
        cmap="viridis",
        linewidth=0,
        antialiased=False
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=12, label=zlabel)
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="3D spectrogram from DataWriter frame dumps")
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", default="tx", choices=["tx", "imp", "rx"])
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--fs", type=float, required=True)

    ap.add_argument("--nfft", type=int, default=1024)
    ap.add_argument("--noverlap", type=int, default=768)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--linear", action="store_true")

    ap.add_argument("--max_samples", type=int, default=None)

    args = ap.parse_args()

    run_dir = Path(args.run)
    x, path = load_frame(run_dir, args.stage, args.frame)

    if args.max_samples and x.size > args.max_samples:
        x = x[:args.max_samples]

    times, freqs, S = compute_spectrogram(
        x,
        fs=args.fs,
        nfft=args.nfft,
        noverlap=args.noverlap,
        center=args.center,
    )

    title = f"3D Spectrogram ({args.stage}) | {path.name}"
    plot_3d_spectrogram(times, freqs, S, db=not args.linear, title=title)


if __name__ == "__main__":
    main()
