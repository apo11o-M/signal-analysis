# python plot_tx_log.py --run ..\build\dumps\run_basic_tx_20251219_025903\ --stage tx --frame 0 --fs 1e7 --center
# 
# plot_spectrogram.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_rx_results(run_dir: Path):
    rx_path = run_dir / "rx" / "rx_results.csv"
    if not rx_path.exists():
        return None, rx_path
    df = pd.read_csv(rx_path)
    return df, rx_path


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


def load_all_frames(run_dir: Path, stage: str, frame_start: int = 0, frame_count: int | None = None):
    """
    Load multiple frames for a stage, concatenate into one long complex64 array.
    Returns: (x_concat, frame_ids, frame_len)
    """
    idx_path = run_dir / "frame_manifest.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"frame_manifest.csv not found in {run_dir}")

    idx = pd.read_csv(idx_path)
    rows = idx[idx["stage"] == stage].copy()
    if rows.empty:
        raise ValueError(f"No entries for stage='{stage}' in {idx_path}")

    # Sort by frame_id so time is correct
    rows = rows.sort_values("frame_id").reset_index(drop=True)

    # Apply range selection
    if frame_start is None:
        frame_start = 0
    if frame_start > 0:
        rows = rows[rows["frame_id"] >= frame_start].reset_index(drop=True)

    if frame_count is not None:
        rows = rows.head(frame_count).reset_index(drop=True)

    if rows.empty:
        raise ValueError("No frames selected after applying frame_start/frame_count")

    # Assume constant frame length (elem_count) for this stage.
    frame_len = int(rows.iloc[0]["elem_count"])
    if not (rows["elem_count"].astype(int) == frame_len).all():
        # If you later allow variable-length frames, you'll need a different time mapping.
        raise ValueError("Variable elem_count detected; this script assumes constant frame length per run/stage.")

    # Load and concatenate
    xs = []
    frame_ids = rows["frame_id"].astype(int).to_list()

    for _, row in rows.iterrows():
        rel_path = Path(row["rel_path"])
        bin_path = run_dir / rel_path
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary file missing: {bin_path}")

        x = np.fromfile(bin_path, dtype=np.complex64)
        # Sanity check
        if x.size != frame_len:
            raise ValueError(f"Unexpected frame length in {bin_path}: got {x.size}, expected {frame_len}")
        xs.append(x)

    x_concat = np.concatenate(xs) if len(xs) > 1 else xs[0]
    return x_concat, frame_ids, frame_len


def plot_spectrogram_ax(ax, x: np.ndarray, fs: float, nfft: int, noverlap: int, title: str,
                        center: bool, db: bool, max_samples: int):
    if max_samples is not None and x.size > max_samples:
        x = x[:max_samples]

    step = nfft - noverlap
    if step <= 0:
        raise ValueError("noverlap must be < nfft")

    nseg = 1 + max(0, (len(x) - nfft) // step)
    if nseg <= 0:
        raise ValueError("Signal too short for chosen nfft/noverlap")

    win = np.hanning(nfft).astype(np.float32)
    win_norm = np.sum(win**2)

    S = np.empty((nfft, nseg), dtype=np.float32)
    for k in range(nseg):
        start = k * step
        seg = x[start:start+nfft] * win
        X = np.fft.fft(seg, n=nfft)
        P = (np.abs(X)**2) / (win_norm if win_norm != 0 else 1.0)
        S[:, k] = P.astype(np.float32)

    if center:
        S = np.fft.fftshift(S, axes=0)
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0/fs))
    else:
        freqs = np.fft.fftfreq(nfft, d=1.0/fs)

    times = (np.arange(nseg) * step) / fs

    if db:
        S_plot = 10.0 * np.log10(S + 1e-12)
        zlabel = "Power (dB)"
    else:
        S_plot = S
        zlabel = "Power"

    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    im = ax.imshow(S_plot, aspect="auto", origin="lower", extent=extent)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    return im, zlabel, times


def plot_rx_ax(ax, rx_df: pd.DataFrame, fs: float, frame_len: int):
    # time axis per frame id
    t = (rx_df["frame_id"].to_numpy(dtype=float) * frame_len) / fs

    est_f = rx_df["est_freq_hz"].to_numpy(dtype=float)
    ax.plot(t, est_f)
    ax.set_ylabel("est_freq (Hz)")
    ax.set_xlabel("Time (s)")

    # Optional: snr on right axis
    if "snr_db" in rx_df.columns:
        ax2 = ax.twinx()
        ax2.plot(t, rx_df["snr_db"].to_numpy(dtype=float))
        ax2.set_ylabel("snr_db")

    # Optional: detected markers
    if "detected" in rx_df.columns:
        det = rx_df["detected"].to_numpy(dtype=int)
        # mark detections along the bottom
        det_t = t[det == 1]
        if det_t.size > 0:
            y0 = np.nanmin(est_f)
            ax.scatter(det_t, np.full_like(det_t, y0), marker="|")


def overlay_est_freq(ax_spec, rx_df: pd.DataFrame, fs: float, frame_len: int, center: bool):
    if rx_df is None or rx_df.empty:
        return

    # RX time axis (start of each frame)
    t = (rx_df["frame_id"].to_numpy(dtype=float) * frame_len) / fs
    f = rx_df["est_freq_hz"].to_numpy(dtype=float)

    # If your spectrogram is centered (fftshift), its y-axis is [-fs/2, fs/2].
    # If not centered, your y-axis is [0, fs) style. But your freq estimator is
    # signed baseband frequency (usually around +/-), so we may need mapping.
    #
    # For a baseband tone at +10 kHz, this is fine.
    # If you later allow negative freqs, and center=False, you can wrap to [0, fs).
    if not center:
        f = np.mod(f, fs)

    ax_spec.plot(t, f, linewidth=1.5)


def main():
    ap = argparse.ArgumentParser(description="Plot spectrogram from DataWriter frame dumps (.c64).")
    ap.add_argument("--run", required=True, help="Run directory created by DataWriter (contains frame_manifest.csv)")
    ap.add_argument("--stage", default="tx", choices=["tx", "imp", "rx"], help="Pipeline stage")
    # ap.add_argument("--frame", type=int, default=None, help="Frame id to load (default: first)")
    ap.add_argument("--frame_start", type=int, default=0, help="First frame_id to include (default: 0)")
    ap.add_argument("--frame_count", type=int, default=None, help="Number of frames to include (default: all)")
    ap.add_argument("--fs", type=float, required=True, help="Sample rate in Hz")
    ap.add_argument("--nfft", type=int, default=2048, help="FFT size per segment")
    ap.add_argument("--noverlap", type=int, default=1536, help="Overlap samples (must be < nfft)")
    ap.add_argument("--center", default=True, help="Center frequency axis around 0 Hz")
    ap.add_argument("--linear", default=False, help="Use linear power instead of dB")
    ap.add_argument("--max_samples", type=int, default=None, help="Truncate to N samples for speed")

    args = ap.parse_args()

    run_dir = Path(args.run)
    # x, path = load_frame(run_dir, args.stage, args.frame)
    x, frame_ids, frame_len = load_all_frames(
        run_dir=run_dir,
        stage=args.stage,
        frame_start=args.frame_start,
        frame_count=args.frame_count
    )
    rx_df, rx_path = load_rx_results(run_dir)

    # title = f"Spectrogram: {args.stage}, file={path.name}, N={x.size}, fs={args.fs:g} Hz"
    title = f"Spectrogram: {args.stage}, frames={len(frame_ids)}, frame_len={frame_len}, fs={args.fs:g} Hz"

    # if rx_df is None:
    #     # old behavior: just spectrogram
    #     plt.figure()
    #     plot_spectrogram(x, args.fs, args.nfft, args.noverlap, title, args.center, (not args.linear), args.max_samples)
    #     plt.show()
    #     return

    # Subplots: top spectrogram, bottom RX
    fig, (ax_spec, ax_rx) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 7)
    )

    im, zlabel, _ = plot_spectrogram_ax(
        ax=ax_spec,
        x=x,
        fs=args.fs,
        nfft=args.nfft,
        noverlap=args.noverlap,
        title=title,
        center=args.center,
        db=(not args.linear),
        max_samples=args.max_samples,
    )
    fig.colorbar(im, ax=ax_spec, label=zlabel)
    # overlay_est_freq(ax_spec, rx_df=rx_df, fs=args.fs, frame_len=frame_len, center=args.center)

    # Align RX time axis using frame_len = x.size (since you're plotting one frame dump)
    # plot_rx_ax(ax_rx, rx_df=rx_df, fs=args.fs, frame_len=int(x.size))
    plot_rx_ax(ax_rx, rx_df=rx_df, fs=args.fs, frame_len=frame_len)
    ax_rx.set_title(f"RX results: {rx_path.name}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
