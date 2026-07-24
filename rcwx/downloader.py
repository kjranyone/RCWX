"""HuggingFace model auto-download functionality."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Model repository and file mappings
# Use RVC's hubert_base.pt for better compatibility
HUBERT_REPO = "lj1995/VoiceConversionWebUI"
HUBERT_FILE = "hubert_base.pt"

RMVPE_REPO = "lj1995/VoiceConversionWebUI"
RMVPE_FILE = "rmvpe.pt"

# GTCRN streaming denoiser (MIT license, CPU ONNX, ~0.5MB).
# Official simplified streaming graph from the GTCRN repository.
GTCRN_URL = (
    "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/main/"
    "stream/onnx_models/gtcrn_simple.onnx"
)
GTCRN_FILE = "gtcrn_simple.onnx"


def get_hubert_path(models_dir: Path) -> Path:
    """Return the path to the HuBERT/ContentVec model."""
    return models_dir / "hubert" / HUBERT_FILE


def get_rmvpe_path(models_dir: Path) -> Path:
    """Return the path to the RMVPE model."""
    return models_dir / "rmvpe" / RMVPE_FILE


def get_gtcrn_path(models_dir: Path) -> Path:
    """Return the path to the GTCRN streaming denoiser ONNX model."""
    return models_dir / "gtcrn" / GTCRN_FILE


def download_gtcrn(models_dir: Path, force: bool = False) -> Path:
    """Download the GTCRN streaming ONNX model (~0.5MB) from GitHub.

    Args:
        models_dir: Directory to store models
        force: Force re-download even if exists

    Returns:
        Path to the downloaded model
    """
    target_path = get_gtcrn_path(models_dir)

    if target_path.exists() and not force:
        logger.info(f"GTCRN model already exists: {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading GTCRN from {GTCRN_URL}...")
    tmp_path = target_path.with_suffix(".onnx.part")
    urllib.request.urlretrieve(GTCRN_URL, tmp_path)
    tmp_path.replace(target_path)

    logger.info(f"GTCRN downloaded to: {target_path}")
    return target_path


def download_hubert(models_dir: Path, force: bool = False) -> Path:
    """
    Download the HuBERT/ContentVec model from HuggingFace.

    Args:
        models_dir: Directory to store models
        force: Force re-download even if exists

    Returns:
        Path to the downloaded model
    """
    target_path = get_hubert_path(models_dir)

    if target_path.exists() and not force:
        logger.info(f"HuBERT model already exists: {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading HuBERT from {HUBERT_REPO}...")

    downloaded_path = hf_hub_download(
        repo_id=HUBERT_REPO,
        filename=HUBERT_FILE,
        local_dir=target_path.parent,
        local_dir_use_symlinks=False,
    )

    logger.info(f"HuBERT downloaded to: {downloaded_path}")
    return Path(downloaded_path)


def download_rmvpe(models_dir: Path, force: bool = False) -> Path:
    """
    Download the RMVPE model from HuggingFace.

    Args:
        models_dir: Directory to store models
        force: Force re-download even if exists

    Returns:
        Path to the downloaded model
    """
    target_path = get_rmvpe_path(models_dir)

    if target_path.exists() and not force:
        logger.info(f"RMVPE model already exists: {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading RMVPE from {RMVPE_REPO}...")

    downloaded_path = hf_hub_download(
        repo_id=RMVPE_REPO,
        filename=RMVPE_FILE,
        local_dir=target_path.parent,
        local_dir_use_symlinks=False,
    )

    logger.info(f"RMVPE downloaded to: {downloaded_path}")
    return Path(downloaded_path)


def download_all(
    models_dir: Path,
    force: bool = False,
    callback: Optional[callable] = None,
) -> dict[str, Path]:
    """
    Download all required models.

    Args:
        models_dir: Directory to store models
        force: Force re-download even if exists
        callback: Optional progress callback(model_name, status)

    Returns:
        Dictionary of model names to paths
    """
    results = {}

    if callback:
        callback("hubert", "downloading")
    results["hubert"] = download_hubert(models_dir, force)
    if callback:
        callback("hubert", "done")

    if callback:
        callback("rmvpe", "downloading")
    results["rmvpe"] = download_rmvpe(models_dir, force)
    if callback:
        callback("rmvpe", "done")

    # Optional (denoise method "gtcrn") — non-fatal on network failure.
    try:
        if callback:
            callback("gtcrn", "downloading")
        results["gtcrn"] = download_gtcrn(models_dir, force)
        if callback:
            callback("gtcrn", "done")
    except Exception as e:
        logger.warning(f"GTCRN download failed (optional, non-fatal): {e}")
        if callback:
            callback("gtcrn", "failed")

    return results


def check_models(models_dir: Path) -> dict[str, bool]:
    """
    Check if required models exist.

    Args:
        models_dir: Directory containing models

    Returns:
        Dictionary of model names to existence status
    """
    return {
        "hubert": get_hubert_path(models_dir).exists(),
        "rmvpe": get_rmvpe_path(models_dir).exists(),
    }
