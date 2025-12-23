from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


MANIFEST_NAME = "frame_manifest.csv"


@dataclass(frozen=True)
class FrameSelection:
    """Which frames to load from the manifest."""
    stage: str
    frame_start: int = 0
    frame_count: Optional[int] = None


def load_rx_results(run_dir: Path) -> tuple[Optional[pd.DataFrame], Path]:
    """
    Load receiver results CSV if present.

    Returns:
      (df_or_none, rx_path)
    """
    rx_path = run_dir / "rx" / "rx_results.csv"
    if not rx_path.exists():
        return None, rx_path
    return pd.read_csv(rx_path), rx_path


def load_manifest(run_dir: Path) -> tuple[pd.DataFrame, Path]:
    """
    Load the frame manifest CSV.
    """
    idx_path = run_dir / MANIFEST_NAME
    if not idx_path.exists():
        raise FileNotFoundError(f"{MANIFEST_NAME} not found in {run_dir}")
    return pd.read_csv(idx_path), idx_path


def select_manifest_rows(manifest: pd.DataFrame, sel: FrameSelection) -> pd.DataFrame:
    """
    Filter + sort manifest rows for a given stage and frame range.
    """
    if "stage" not in manifest.columns:
        raise ValueError("frame_manifest.csv missing required column: 'stage'")
    if "frame_id" not in manifest.columns:
        raise ValueError("frame_manifest.csv missing required column: 'frame_id'")
    if "rel_path" not in manifest.columns:
        raise ValueError("frame_manifest.csv missing required column: 'rel_path'")
    if "elem_count" not in manifest.columns:
        raise ValueError("frame_manifest.csv missing required column: 'elem_count'")

    rows = manifest[manifest["stage"] == sel.stage].copy()
    if rows.empty:
        raise ValueError(f"No entries for stage='{sel.stage}' in manifest")

    rows = rows.sort_values("frame_id").reset_index(drop=True)

    # Apply range selection
    if sel.frame_start > 0:
        rows = rows[rows["frame_id"] >= sel.frame_start].reset_index(drop=True)
    if sel.frame_count is not None:
        rows = rows.head(sel.frame_count).reset_index(drop=True)

    if rows.empty:
        raise ValueError("No frames selected after applying frame_start/frame_count")

    return rows


def validate_constant_frame_len(rows: pd.DataFrame) -> int:
    """
    Ensure elem_count is constant across selected rows.
    Returns frame_len if constant; raises otherwise.
    """
    frame_len = int(rows.iloc[0]["elem_count"])
    all_same = (rows["elem_count"].astype(int) == frame_len).all()
    if not all_same:
        raise ValueError(
            "Variable elem_count detected; plotting assumes constant frame length for time mapping."
        )
    return frame_len


def load_frames_concat(run_dir: Path, sel: FrameSelection) -> tuple[np.ndarray, list[int], int]:
    """
    Load selected frame binaries and concatenate into one complex64 array.

    Returns:
      (x_concat, frame_ids, frame_len)
    """
    manifest, _ = load_manifest(run_dir)
    rows = select_manifest_rows(manifest, sel)
    frame_len = validate_constant_frame_len(rows)

    xs: list[np.ndarray] = []
    frame_ids: list[int] = rows["frame_id"].astype(int).to_list()

    for _, row in rows.iterrows():
        bin_path = run_dir / Path(row["rel_path"])
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary file missing: {bin_path}")

        x = np.fromfile(bin_path, dtype=np.complex64)
        if x.size != frame_len:
            raise ValueError(f"Unexpected frame length in {bin_path}: got {x.size}, expected {frame_len}")
        xs.append(x)

    x_concat = np.concatenate(xs) if len(xs) > 1 else xs[0]
    return x_concat, frame_ids, frame_len
