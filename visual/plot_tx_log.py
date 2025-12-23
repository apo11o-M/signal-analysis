# python plot_tx_log.py --run ..\build\dumps\run_basic_tx_20251219_025903\ --stage tx --frame 0 --fs 1e7 --center
# 
# plot_spectrogram.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_frame(run_dir: Path, stage: str, frame_id: int):
    idx_path = run_dir / "frame_manifest.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"frame_manifest.csv not found in {run_dir}")

    idx = pd.read_csv(idx_path)

    # Normalize stage matching (your CSV uses "tx", "imp", "rx")
    rows = idx[idx["stage"] == stage].copy()
    if rows.empty:
        raise ValueError(f"No entries for stage='{stage}' in {idx_path}")

    if frame_id is None:
        # default: first available
        row = rows.iloc[0]
    else:
        rows2 = rows[rows["frame_id"] == frame_id]
        if rows2.empty:
            available = rows["frame_id"].head(10).tolist()
            raise ValueError(
                f"frame_id={frame_id} not found for stage='{stage}'. "
                f"Example available: {available} ..."
            )
        row = rows2.iloc[0]

    rel_path = Path(row["rel_path"])
    bin_path = run_dir / rel_path
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary file missing: {bin_path}")

    x = np.fromfile(bin_path, dtype=np.complex64)
    return x, bin_path


def plot_spectrogram(x: np.ndarray, fs: float, nfft: int, noverlap: int, title: str,
                     center: bool, db: bool, max_frames: int):
    # Optionally truncate for speed
    if max_frames is not None and x.size > max_frames:
        x = x[:max_frames]

    # Optional: center frequency at 0 Hz for complex baseband plots
    # For complex data, plt.specgram will show 0..fs unless we shift ourselves.
    # We'll compute via FFT bins manually so we can center cleanly.
    step = nfft - noverlap
    if step <= 0:
        raise ValueError("noverlap must be < nfft")

    nseg = 1 + max(0, (len(x) - nfft) // step)
    if nseg <= 0:
        raise ValueError("Signal too short for chosen nfft/noverlap")

    # Window
    win = np.hanning(nfft).astype(np.float32)
    win_norm = np.sum(win**2)

    # Compute STFT power
    S = np.empty((nfft, nseg), dtype=np.float32)
    for k in range(nseg):
        start = k * step
        seg = x[start:start+nfft]
        segw = seg * win
        X = np.fft.fft(segw, n=nfft)
        P = (np.abs(X)**2) / (win_norm if win_norm != 0 else 1.0)
        S[:, k] = P.astype(np.float32)

    if center:
        S = np.fft.fftshift(S, axes=0)
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0/fs))
    else:
        freqs = np.fft.fftfreq(nfft, d=1.0/fs)

    times = (np.arange(nseg) * step) / fs

    if db:
        # avoid log(0)
        S_plot = 10.0 * np.log10(S + 1e-12)
        zlabel = "Power (dB)"
    else:
        S_plot = S
        zlabel = "Power"

    # Plot
    plt.figure()
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    # imshow expects [y,x]
    plt.imshow(S_plot, aspect="auto", origin="lower", extent=extent)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label=zlabel)
    plt.tight_layout()


def main():
    ap = argparse.ArgumentParser(description="Plot spectrogram from DataWriter frame dumps (.c64).")
    ap.add_argument("--run", required=True, help="Run directory created by DataWriter (contains frame_manifest.csv)")
    ap.add_argument("--stage", default="tx", choices=["tx", "imp", "rx"], help="Pipeline stage")
    ap.add_argument("--frame", type=int, default=None, help="Frame id to load (default: first)")
    ap.add_argument("--fs", type=float, required=True, help="Sample rate in Hz")
    ap.add_argument("--nfft", type=int, default=2048, help="FFT size per segment")
    ap.add_argument("--noverlap", type=int, default=1536, help="Overlap samples (must be < nfft)")
    ap.add_argument("--center", action="store_true", help="Center frequency axis around 0 Hz")
    ap.add_argument("--linear", action="store_true", help="Use linear power instead of dB")
    ap.add_argument("--max_samples", type=int, default=None, help="Truncate to N samples for speed")

    args = ap.parse_args()

    run_dir = Path(args.run)
    x, path = load_frame(run_dir, args.stage, args.frame)

    title = f"Spectrogram: {args.stage}, file={path.name}, N={x.size}, fs={args.fs:g} Hz"
    plot_spectrogram(
        x=x,
        fs=args.fs,
        nfft=args.nfft,
        noverlap=args.noverlap,
        title=title,
        center=args.center,
        db=(not args.linear),
        max_frames=args.max_samples,
    )

    plt.show()


if __name__ == "__main__":
    main()
