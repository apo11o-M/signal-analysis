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

# SINGLE TONE RECEIVER PLOTTING
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

# SINGLE TONE RECEIVER PLOTTING
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


# ==============================================================================
# ==============================================================================

def gen_ref_chirp(fs: float, L: int, f0: float, f1: float) -> np.ndarray:
    """
    Generate complex reference chirp of length L using phase accumulation.
    Matches the TX pattern: instantaneous freq linearly ramps from f0 to f1 over L samples.
    """
    if L <= 0:
        return np.zeros((0,), dtype=np.complex64)
    if L == 1:
        return np.ones((1,), dtype=np.complex64)

    ramp = (f1 - f0) / float(L - 1)  # Hz per sample
    phase = 0.0
    out = np.empty((L,), dtype=np.complex64)

    two_pi = 2.0 * np.pi
    for n in range(L):
        out[n] = np.cos(phase) + 1j * np.sin(phase)
        fn = f0 + ramp * n
        phase += two_pi * (fn / fs)
        phase = np.fmod(phase, two_pi)

    return out


def matched_filter_mag_vs_tau(
    x: np.ndarray,
    ref: np.ndarray,
    max_tau: int,
) -> Tuple[np.ndarray, int, float, float, float]:
    """
    Compute |corr(tau)| for tau=0..max_tau using constant correlation length.
    Returns:
      mags[tau], tau_hat, peak_mag, mean_mag, metric(=peak/mean)
    """
    x = np.asarray(x)
    ref = np.asarray(ref)
    N = int(x.size)
    L = int(ref.size)
    if L < 1:
        raise ValueError("Reference length must be >= 1")

    max_tau = int(min(max_tau, N - L))
    if max_tau < 0:
        raise ValueError("Signal too short for the chosen template length")

    mags = np.empty((max_tau + 1,), dtype=np.float64)

    # numpy.vdot(a,b) = conj(a) dot b
    for tau in range(max_tau + 1):
        seg = x[tau : tau + L]
        acc = np.vdot(ref, seg)  # sum conj(ref[n]) * seg[n]
        mags[tau] = np.abs(acc)

    tau_hat = int(np.argmax(mags))
    peak_mag = float(mags[tau_hat])
    mean_mag = float(np.mean(mags))
    metric = peak_mag / (mean_mag + 1e-12)
    return mags, tau_hat, peak_mag, mean_mag, metric


def plot_matched_filter_mag_vs_tau(
    ax: plt.Axes,
    x_frame: np.ndarray,
    fs: float,
    chirp_f0: float,
    chirp_f1: float,
    max_tau: int,
    title: str = "",
):
    """
    Plot matched filter magnitude vs tau for a single frame buffer x_frame.
    Uses a reference chirp length L = len(x_frame) - max_tau so that all taus use the same L.
    """
    x_frame = np.asarray(x_frame)
    N = x_frame.size
    max_tau = int(min(max_tau, N - 1))
    L = N - max_tau
    if L < 2:
        ax.text(0.5, 0.5, "Frame too short for chosen max_tau", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None

    ref = gen_ref_chirp(fs=fs, L=L, f0=chirp_f0, f1=chirp_f1)
    mags, tau_hat, peak_mag, mean_mag, metric = matched_filter_mag_vs_tau(x_frame, ref, max_tau=max_tau)

    ax.plot(np.arange(mags.size), mags)
    ax.axvline(tau_hat, linestyle="--")
    ax.set_xlabel("tau (samples)")
    ax.set_ylabel("|corr(tau)|")
    ax.set_title(title or f"Matched filter |corr(tau)|  tau_hat={tau_hat}, metric={metric:.2f}")
    ax.grid(True, alpha=0.2)

    # return info for the next stage (dechirp/FFT)
    return {
        "ref": ref,
        "L": L,
        "tau_hat": tau_hat,
        "peak_mag": peak_mag,
        "mean_mag": mean_mag,
        "metric": metric,
    }


def dechirp_and_fft(
    x_frame: np.ndarray,
    ref: np.ndarray,
    tau_hat: int,
    fs: float,
    nfft: int = 4096,
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align by tau_hat, dechirp by conj(ref), then FFT magnitude in dB.
    Returns (freqs_hz, mag_db).
    """
    x_frame = np.asarray(x_frame)
    ref = np.asarray(ref)

    L = ref.size
    seg = x_frame[tau_hat : tau_hat + L]
    z = seg * np.conj(ref)

    # FFT
    nfft = int(max(nfft, 1))
    Z = np.fft.fft(z, n=nfft)
    mag_db = 20.0 * np.log10(np.abs(Z) + 1e-12)
    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)

    if center:
        mag_db = np.fft.fftshift(mag_db)
        freqs = np.fft.fftshift(freqs)

    return freqs, mag_db


def plot_dechirped_fft_mag(
    ax: plt.Axes,
    x_frame: np.ndarray,
    fs: float,
    ref: np.ndarray,
    tau_hat: int,
    nfft: int = 4096,
    center: bool = True,
    title: str = "",
):
    freqs, mag_db = dechirp_and_fft(
        x_frame=x_frame,
        ref=ref,
        tau_hat=tau_hat,
        fs=fs,
        nfft=nfft,
        center=center,
    )
    ax.plot(freqs, mag_db)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title or "Dechirped FFT magnitude")
    ax.grid(True, alpha=0.2)


def plot_est_cfo_vs_frame(
    ax: plt.Axes,
    rx_df: Optional[pd.DataFrame],
    fs: float,
    frame_len: int,
    title: str = "",
):
    """
    Plot est_cfo_hz vs time (frame start times).
    If 'detected' exists, mark detected frames.
    """
    if rx_df is None or rx_df.empty:
        ax.text(0.5, 0.5, "No rx_results.csv found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    if "est_cfo_hz" not in rx_df.columns:
        ax.text(0.5, 0.5, "rx_results.csv missing est_cfo_hz", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    t = (rx_df["frame_id"].to_numpy(dtype=float) * frame_len) / fs
    cfo = rx_df["est_cfo_hz"].to_numpy(dtype=float)

    ax.plot(t, cfo)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimated CFO (Hz)")
    ax.set_title(title or "Estimated CFO vs time")
    ax.grid(True, alpha=0.2)

    if "detected" in rx_df.columns:
        det = rx_df["detected"].to_numpy(dtype=int)
        if np.any(det == 1):
            ax.scatter(t[det == 1], cfo[det == 1], s=18, marker="o")

# ==============================================================================
# ==============================================================================

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

    # SINGLE TONE RECEIVER PLOTTING
    # if args.overlay_rx:
    #     overlay_est_freq(ax_spec, rx_df, fs=args.fs, frame_len=frame_len, center=args.center)
    # plot_rx(ax_rx, rx_df, fs=args.fs, frame_len=frame_len)
    # ax_rx.set_title(f"RX results: {rx_path.name}")
    # plt.tight_layout()
    # plt.show()
    
    # CHIRP RECEIVER PLOTTING
    # Choose a single frame_id to inspect for MF + dechirp FFT.
    # Use the stage that matches what your receiver saw (usually "imp").
    mf_frame_id = args.frame_start  # or set explicitly
    mf_max_tau = 64                 # set >= your worst-case timing offset
    chirp_f0 = 1e3                  # set to your RX expected chirp start
    chirp_f1 = 5e3                  # set to your RX expected chirp end
    mf_nfft = 4096

    sel_one = FrameSelection(stage=args.stage, frame_start=mf_frame_id, frame_count=1)
    x1, frame_ids1, frame_len1 = load_frames_concat(args.run, sel_one)
    x1 = x1[:frame_len1]  # ensure exactly one frame

    fig2, (ax_mf, ax_fft, ax_cfo) = plt.subplots(3, 1, figsize=(10, 10))

    mf_info = plot_matched_filter_mag_vs_tau(
        ax=ax_mf,
        x_frame=x1,
        fs=args.fs,
        chirp_f0=chirp_f0,
        chirp_f1=chirp_f1,
        max_tau=mf_max_tau,
        title=f"MF magnitude vs tau (frame {mf_frame_id})",
    )

    if mf_info is not None:
        plot_dechirped_fft_mag(
            ax=ax_fft,
            x_frame=x1,
            fs=args.fs,
            ref=mf_info["ref"],
            tau_hat=mf_info["tau_hat"],
            nfft=mf_nfft,
            center=args.center,
            title="Dechirped FFT magnitude",
        )
    else:
        ax_fft.set_axis_off()
        ax_fft.text(0.5, 0.5, "MF failed; cannot dechirp", ha="center", va="center", transform=ax_fft.transAxes)

    plot_est_cfo_vs_frame(
        ax=ax_cfo,
        rx_df=rx_df,
        fs=args.fs,
        frame_len=frame_len,
        title="Estimated CFO vs frame",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
