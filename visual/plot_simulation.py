from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from frame_io import FrameSelection, load_frames_concat, load_rx_results


@dataclass(frozen=True)
class SpecConfig:
    fs: float
    nfft: int
    noverlap: int
    center: bool
    db: bool
    max_samples: Optional[int] = None


def _stft_power(
    x: np.ndarray,
    cfg: SpecConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT power (linear), returning (S, freqs, times).

    S shape: (nfft, nseg)
    """
    if cfg.max_samples is not None and x.size > cfg.max_samples:
        x = x[: cfg.max_samples]

    nfft = cfg.nfft
    noverlap = cfg.noverlap
    fs = cfg.fs

    step = nfft - noverlap
    if step <= 0:
        raise ValueError("noverlap must be < nfft")
    if x.size < nfft:
        raise ValueError("Signal too short for chosen nfft/noverlap")

    # number of segments
    nseg = 1 + (x.size - nfft) // step
    if nseg <= 0:
        raise ValueError("Signal too short for chosen nfft/noverlap")

    win = np.hanning(nfft).astype(np.float32)
    win_norm = float(np.sum(win * win)) or 1.0

    # Compute each segment FFT; keep it simple/clear.
    S = np.empty((nfft, nseg), dtype=np.float32)
    for k in range(nseg):
        start = k * step
        seg = x[start : start + nfft] * win
        X = np.fft.fft(seg, n=nfft)
        P = (np.abs(X) ** 2) / win_norm
        S[:, k] = P.astype(np.float32)

    if cfg.center:
        S = np.fft.fftshift(S, axes=0)
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    else:
        freqs = np.fft.fftfreq(nfft, d=1.0 / fs)

    times = (np.arange(nseg, dtype=np.float64) * step) / fs
    return S, freqs, times


def plot_spectrogram(ax, x: np.ndarray, cfg: SpecConfig, title: str):
    """
    Plot spectrogram on a given axis. Returns the image handle + colorbar label.
    """
    S, freqs, times = _stft_power(x, cfg)

    if cfg.db:
        S_plot = 10.0 * np.log10(S + 1e-12)
        zlabel = "Power (dB)"
    else:
        S_plot = S
        zlabel = "Power"

    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    print(f"Spectrogram extent: {extent}")
    im = ax.imshow(S_plot, aspect="auto", origin="lower", extent=extent)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return im, zlabel, times


def plot_rx(ax, rx_df: Optional[pd.DataFrame], fs: float, frame_len: int):
    """
    Plot RX estimated frequency and optional SNR/detection markers.
    """
    if rx_df is None or rx_df.empty:
        ax.text(0.5, 0.5, "No rx_results.csv found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    # time at start of each frame_id
    t = (rx_df["frame_id"].to_numpy(dtype=float) * frame_len) / fs

    if "est_freq_hz" in rx_df.columns:
        est_f = rx_df["est_freq_hz"].to_numpy(dtype=float)
        ax.plot(t, est_f, "m")
        ax.set_ylabel("est_freq (Hz) (magenta)")
    else:
        ax.text(0.5, 0.5, "rx_results.csv missing est_freq_hz", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Time (s)")

    # optional: snr on right axis
    if "snr_db" in rx_df.columns:
        ax2 = ax.twinx()
        ax2.plot(t, rx_df["snr_db"].to_numpy(dtype=float), 'g')
        ax2.set_ylabel("snr_db (green)")

    # optional: detected markers
    if "detected" in rx_df.columns and "est_freq_hz" in rx_df.columns:
        det = rx_df["detected"].to_numpy(dtype=int)
        det_t = t[det == 1]
        if det_t.size > 0:
            y0 = np.nanmin(est_f)
            ax.scatter(det_t, np.full_like(det_t, y0), marker="|")


def overlay_est_freq(ax_spec, rx_df: Optional[pd.DataFrame], fs: float, frame_len: int, center: bool):
    """
    Overlay RX estimated frequency line on top of the spectrogram.
    """
    if rx_df is None or rx_df.empty:
        return
    if "est_freq_hz" not in rx_df.columns:
        return

    t = (rx_df["frame_id"].to_numpy(dtype=float) * frame_len) / fs
    f = rx_df["est_freq_hz"].to_numpy(dtype=float)

    # If the spectrogram is not centered, map signed freq into [0, fs)
    # if not center:
    #     f = np.mod(f, fs)

    ax_spec.plot(t, f, 'w', linestyle='dotted', linewidth=1.5)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot spectrogram from DataWriter frame dumps (.c64).")

    ap.add_argument("--run", required=True, type=Path,
                    help="Run directory created by DataWriter (contains frame_manifest.csv)")
    ap.add_argument("--stage", default="tx", choices=["tx", "imp", "rx"],
                    help="Pipeline stage to plot")

    ap.add_argument("--frame_start", type=int, default=0,
                    help="First frame_id to include (default: 0)")
    ap.add_argument("--frame_count", type=int, default=None,
                    help="Number of frames to include (default: all)")

    ap.add_argument("--fs", type=float, default=4096, help="Sample rate in Hz")
    ap.add_argument("--nfft", type=int, default=2048, help="FFT size per segment")
    ap.add_argument("--noverlap", type=int, default=1536, help="Overlap samples (must be < nfft)")

    # Proper booleans (instead of default=True with a string)
    ap.add_argument("--center", dest="center", action="store_true", help="Center freq axis around 0 Hz")
    ap.add_argument("--no-center", dest="center", action="store_false", help="Do not center freq axis")
    ap.set_defaults(center=True)

    ap.add_argument("--db", dest="db", action="store_true", help="Plot power in dB (default)")
    ap.add_argument("--linear", dest="db", action="store_false", help="Plot linear power")
    ap.set_defaults(db=True)

    ap.add_argument("--max_samples", type=int, default=None, help="Truncate to N samples for speed")

    ap.add_argument("--overlay-rx", action="store_true",
                    help="Overlay RX est_freq on the spectrogram")

    return ap


def main():
    args = build_argparser().parse_args()

    sel = FrameSelection(stage=args.stage, frame_start=args.frame_start, frame_count=args.frame_count)
    x, frame_ids, frame_len = load_frames_concat(args.run, sel)
    rx_df, rx_path = load_rx_results(args.run)

    cfg = SpecConfig(
        fs=args.fs,
        nfft=args.nfft,
        noverlap=args.noverlap,
        center=args.center,
        db=args.db,
        max_samples=args.max_samples,
    )

    title = f"Spectrogram: {args.stage}, frames={len(frame_ids)}, frame_len={frame_len}, fs={args.fs:g} Hz"

    fig, (ax_spec, ax_rx) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 7),
    )

    im, zlabel, _ = plot_spectrogram(ax_spec, x, cfg, title)
    fig.colorbar(im, ax=ax_spec, label=zlabel)

    if args.overlay_rx:
        overlay_est_freq(ax_spec, rx_df, fs=args.fs, frame_len=frame_len, center=args.center)

    plot_rx(ax_rx, rx_df, fs=args.fs, frame_len=frame_len)
    ax_rx.set_title(f"RX results: {rx_path.name}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
