#!/usr/bin/env python3
"""
Batch-render generator-only latent de-identification sweeps.

The script intentionally contains only four methods:

    avg
        Full W+ interpolation toward the generator's W average.

    mid_avg
        W-average interpolation restricted to a fixed middle-layer baseline.

    rnd_offset
        A small deterministic random offset whose magnitude is calibrated from
        the generator's mapped W distribution. The offset is scaled linearly by
        1 - identity.

    image_matched_residual
        Samples mapped donor latents, renders them at the saved pose, selects a
        donor that balances image-space preservation with latent displacement,
        and applies only the donor's shared-W shift while preserving the source
        W+ layer residuals.

No external recognition, age, demographic, segmentation, landmark, or attribute
models are used. The image-matched method is therefore a heuristic: it can reduce
large changes in apparent age, color, texture, and outer appearance, but it cannot
guarantee preservation of identity-independent biometric characteristics.

Expected input layout:

    <input-dir>/processed.txt
    <input-dir>/<image-name>/generator.pkl
    <input-dir>/<image-name>/pose.npy
    <input-dir>/<image-name>/render_w_plus.npy
    <input-dir>/<image-name>/original.png

Each non-empty line in processed.txt is an image name without an extension.

Output layout:

    <output-dir>/<image-name>/original.png
    <output-dir>/<image-name>/<technique>/frame_000_identity_0.0000.png
    <output-dir>/<image-name>/<technique>/frame_001_identity_0.1000.png
    ...

Example:

    python deid_four_methods.py \
        --input-dir /home/container_output/asds_deid \
        --min-identity 0.0 \
        --max-identity 1.0 \
        --identity-step 0.1 \
        --mapping-samples 32 \
        --output-dir /home/container_output/asds_deid_renders
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import dnnlib
import utils.legacy as legacy


DEIDENTIFICATION_TECHNIQUES: Tuple[str, ...] = (
    "avg",
    "mid_avg",
    "rnd_offset",
    "image_matched_residual",
)


RND_OFFSET_FRACTION = 2.00
DONOR_RENDER_BATCH_SIZE = 1


@dataclass
class EditCache:
    """Deterministic edit data reused throughout one image's identity sweep."""

    w_avg: torch.Tensor
    random_offset_delta: torch.Tensor
    image_matched_residual_delta: torch.Tensor
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render four generator-only de-identification sweeps for all image "
            "directories listed in processed.txt."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing processed.txt and the image-named source "
            "directories."
        ),
    )
    parser.add_argument(
        "--min-identity",
        type=float,
        required=True,
        help="Minimum identity value to render. Must be in [0, 1].",
    )
    parser.add_argument(
        "--max-identity",
        type=float,
        required=True,
        help="Maximum identity value to render. Must be in [0, 1].",
    )
    parser.add_argument(
        "--identity-step",
        type=float,
        required=True,
        help=(
            "Increment between evaluated identity values. Both endpoints are "
            "included; the maximum is appended when the step does not land on it."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for rendered image folders and the batch report.",
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        choices=DEIDENTIFICATION_TECHNIQUES,
        default=list(DEIDENTIFICATION_TECHNIQUES),
        help="Techniques to render. Default: all four techniques.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, for example cuda, cuda:1, or cpu. Default: cuda.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("image", "image_raw", "image_depth"),
        default="image",
        help="Generator output tensor to save. Default: image.",
    )
    parser.add_argument(
        "--noise-mode",
        choices=("const", "random", "none"),
        default="const",
        help=(
            "Noise mode passed to generator.synthesis(). Use const for stable "
            "candidate comparison. Default: const."
        ),
    )
    parser.add_argument(
        "--sample-mult",
        type=float,
        default=3.0,
        help="Multiplier for depth sampling resolutions. Default: 3.0.",
    )
    parser.add_argument(
        "--nrr",
        type=int,
        default=None,
        help="Optional neural rendering resolution override.",
    )
    parser.add_argument(
        "--mapping-samples",
        type=int,
        default=64,
        help=(
            "Mapped donors rendered and compared by image_matched_residual. "
            "Larger values increase the chance of finding a compatible donor but "
            "increase runtime. Default: 32."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Base seed. A stable image-specific seed is derived from it so reruns "
            "produce the same random offset and donor bank. Default: 0."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete each existing output image directory before rendering it.",
    )
    return parser.parse_args()


def _mid_band(num_layers: int) -> Tuple[int, int]:
    """Return the fixed middle band used only by the mid_avg baseline."""

    start = max(1, int(0.2 * num_layers))
    end = min(num_layers, int(0.55 * num_layers))
    return start, end


def _mapping_module(generator: torch.nn.Module) -> torch.nn.Module:
    """Return the module that owns w_avg and num_ws."""

    backbone = getattr(generator, "backbone", None)
    backbone_mapping = getattr(backbone, "mapping", None)
    if backbone_mapping is not None:
        return backbone_mapping

    mapping = getattr(generator, "mapping", None)
    if isinstance(mapping, torch.nn.Module):
        return mapping

    raise AttributeError(
        "Generator has no mapping module at backbone.mapping or mapping."
    )


def _run_mapping(
    generator: torch.nn.Module,
    z: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Call the generator-level mapping API when EG3D exposes one."""

    mapping = getattr(generator, "mapping", None)
    if callable(mapping):
        return mapping(z, c)
    return _mapping_module(generator)(z, c)


def _w_avg_like(generator: torch.nn.Module, w: torch.Tensor) -> torch.Tensor:
    mapping = _mapping_module(generator)
    w_avg = mapping.w_avg.to(device=w.device, dtype=w.dtype)

    if w_avg.ndim == 1:
        w_avg = w_avg.reshape(1, 1, -1)
    elif w_avg.ndim == 2:
        w_avg = w_avg.unsqueeze(0)

    if w_avg.ndim != 3 or w_avg.shape[-1] != w.shape[-1]:
        raise ValueError(
            f"Unexpected w_avg shape {tuple(w_avg.shape)} for W+ shape "
            f"{tuple(w.shape)}."
        )

    if w_avg.shape[1] == 1 and w.shape[1] != 1:
        w_avg = w_avg.expand(w.shape[0], w.shape[1], w.shape[2])
    elif w_avg.shape[1] != w.shape[1]:
        raise ValueError(
            f"w_avg has {w_avg.shape[1]} style layers, but W+ has {w.shape[1]}."
        )

    if w_avg.shape[0] == 1 and w.shape[0] != 1:
        w_avg = w_avg.expand(w.shape[0], -1, -1)

    return w_avg


@torch.no_grad()
def sample_mapped_bank(
    generator: torch.nn.Module,
    c: torch.Tensor,
    w: torch.Tensor,
    count: int,
    random_generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample valid W+ donor candidates through the mapping network."""

    if count < 4:
        raise ValueError("--mapping-samples must be at least 4.")
    if w.ndim != 3 or w.shape[0] != 1:
        raise ValueError(
            f"Expected one source W+ with shape [1, num_ws, w_dim], got {tuple(w.shape)}."
        )
    if c.ndim != 2 or c.shape[0] != 1:
        raise ValueError(f"Expected pose shape [1, c_dim], got {tuple(c.shape)}.")

    z_dim = getattr(generator, "z_dim", None)
    if z_dim is None:
        raise AttributeError("Generator does not expose z_dim.")

    z = torch.randn(
        (count, z_dim),
        device=w.device,
        dtype=w.dtype,
        generator=random_generator,
    )
    mapped = _run_mapping(generator, z, c.repeat(count, 1))

    if mapped.ndim == 2:
        mapped = mapped.unsqueeze(1)
    if mapped.ndim != 3:
        raise ValueError(
            f"Mapping network returned shape {tuple(mapped.shape)}; expected "
            "[samples, num_ws, w_dim]."
        )
    if mapped.shape[-1] != w.shape[-1]:
        raise ValueError(
            f"Mapped donor W dimension {mapped.shape[-1]} does not match source "
            f"W dimension {w.shape[-1]}."
        )
    if mapped.shape[1] == 1 and w.shape[1] != 1:
        mapped = mapped.expand(mapped.shape[0], w.shape[1], w.shape[2])
    elif mapped.shape[1] != w.shape[1]:
        raise ValueError(
            f"Mapped donors have {mapped.shape[1]} layers, but source W+ has "
            f"{w.shape[1]}."
        )

    if not torch.isfinite(mapped).all():
        raise ValueError("Mapping network produced NaN or infinity in donor bank.")
    return mapped.to(dtype=w.dtype)


def _random_offset_delta(
    w: torch.Tensor,
    w_avg: torch.Tensor,
    donor_bank: torch.Tensor,
    random_generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build a small fixed random direction calibrated by mapped-latent spread.

    The direction is made orthogonal to the source displacement from w_avg only
    to avoid merely increasing or decreasing ordinary average truncation. This is
    still a random baseline and has no semantic interpretation.
    """

    noise = torch.randn(
        w.shape,
        device=w.device,
        dtype=w.dtype,
        generator=random_generator,
    )
    reference = w - w_avg
    denominator = reference.square().sum().clamp(min=1e-12)
    noise = noise - ((noise * reference).sum() / denominator) * reference

    noise_norm = noise.norm()
    if float(noise_norm.detach()) <= 1e-12:
        noise = torch.ones_like(w)
        noise_norm = noise.norm().clamp(min=1e-12)
    direction = noise / noise_norm

    bank_center = donor_bank.mean(dim=0, keepdim=True)
    mapped_radii = (donor_bank - bank_center).reshape(donor_bank.shape[0], -1).norm(
        dim=1
    )
    typical_radius = mapped_radii.median().clamp(min=1e-8)
    maximum_norm = RND_OFFSET_FRACTION * typical_radius
    delta = direction * maximum_norm

    return delta, {
        "fraction_of_mapped_radius": RND_OFFSET_FRACTION,
        "typical_mapped_radius": float(typical_radius.detach().cpu()),
        "maximum_offset_norm": float(maximum_norm.detach().cpu()),
        "actual_offset_norm": float(delta.norm().detach().cpu()),
    }


def _select_rgb_output(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Prefer EG3D's lower-resolution image_raw for donor comparison."""

    for key in ("image_raw", "image"):
        value = output.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            if value.shape[1] >= 3:
                return value[:, :3]
    raise KeyError(
        "Generator output contains no RGB 'image_raw' or 'image' tensor for "
        f"candidate comparison. Available keys: {sorted(output.keys())}"
    )


@torch.no_grad()
def _synthesize_rgb(
    generator: torch.nn.Module,
    ws: torch.Tensor,
    c: torch.Tensor,
    noise_mode: str,
) -> torch.Tensor:
    output = generator.synthesis(ws=ws, c=c, noise_mode=noise_mode)
    if not isinstance(output, dict):
        raise TypeError(
            f"generator.synthesis() returned {type(output).__name__}; expected dict."
        )
    return _select_rgb_output(output)


def _unit_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB generator tensor from [-1, 1] to finite [0, 1]."""

    image = ((image.float().clamp(-1, 1) + 1.0) * 0.5).clamp(0.0, 1.0)
    if not torch.isfinite(image).all():
        raise ValueError("Generated comparison image contains NaN or infinity.")
    return image


def _relative_crop(
    image: torch.Tensor,
    y0: float,
    y1: float,
    x0: float,
    x1: float,
) -> torch.Tensor:
    height, width = image.shape[-2:]
    top = max(0, min(height - 1, int(round(y0 * height))))
    bottom = max(top + 1, min(height, int(round(y1 * height))))
    left = max(0, min(width - 1, int(round(x0 * width))))
    right = max(left + 1, min(width, int(round(x1 * width))))
    return image[..., top:bottom, left:right]


def _rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    y = 0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b
    cr = 0.500000 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb, cr], dim=1)


def _image_descriptors(image: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Extract fixed image-space proxies from aligned face renders.

    These descriptors intentionally use no learned model:
      * center_color: central YCbCr mean and standard deviation
      * coarse_face: low-resolution central grayscale layout
      * edge_face: low-resolution central gradient structure
      * texture: central high-frequency residual energy
      * outer: low-resolution top/bottom/left/right appearance
    """

    image = _unit_rgb(image)
    center = _relative_crop(image, 0.16, 0.84, 0.22, 0.78)
    center_ycc = _rgb_to_ycbcr(center)

    color_mean = center_ycc.mean(dim=(-2, -1))
    color_std = center_ycc.std(dim=(-2, -1), unbiased=False)
    center_color = torch.cat([color_mean, color_std], dim=1)

    gray = center_ycc[:, 0:1]
    coarse_face = F.adaptive_avg_pool2d(gray, (12, 10)).flatten(1)

    dx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    edge = torch.sqrt(dx.square() + dy.square() + 1e-12)
    edge_face = F.adaptive_avg_pool2d(edge, (8, 8)).flatten(1)

    smooth = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
    texture_residual = (gray - smooth).abs()
    texture = torch.cat(
        [
            texture_residual.mean(dim=(-2, -1)),
            texture_residual.std(dim=(-2, -1), unbiased=False),
        ],
        dim=1,
    )

    top = _relative_crop(image, 0.00, 0.20, 0.00, 1.00)
    bottom = _relative_crop(image, 0.80, 1.00, 0.00, 1.00)
    left = _relative_crop(image, 0.20, 0.80, 0.00, 0.18)
    right = _relative_crop(image, 0.20, 0.80, 0.82, 1.00)
    outer = torch.cat(
        [
            F.adaptive_avg_pool2d(top, (4, 8)).flatten(1),
            F.adaptive_avg_pool2d(bottom, (4, 8)).flatten(1),
            F.adaptive_avg_pool2d(left, (8, 4)).flatten(1),
            F.adaptive_avg_pool2d(right, (8, 4)).flatten(1),
        ],
        dim=1,
    )

    return {
        "center_color": center_color,
        "coarse_face": coarse_face,
        "edge_face": edge_face,
        "texture": texture,
        "outer": outer,
    }


def _descriptor_error(
    candidate: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    return (candidate - source).square().mean(dim=1)


def _median_normalize(values: torch.Tensor) -> torch.Tensor:
    return values / values.median().clamp(min=1e-12)


@torch.no_grad()
def select_image_matched_donor(
    generator: torch.nn.Module,
    w: torch.Tensor,
    c: torch.Tensor,
    donor_bank: torch.Tensor,
    noise_mode: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Select a donor using only generator renders and fixed image statistics.

    Each preservation error is normalized by its donor-bank median. The donor's
    preservation cost is the worst normalized error across the five descriptors,
    so one severely changed property cannot be hidden by good scores elsewhere.
    Latent displacement is also median-normalized. The selected donor maximizes:

        normalized_latent_distance / (1 + worst_preservation_cost)

    This avoids hand-selected layer indices and cross-metric loss weights. It is
    nevertheless an unsupervised heuristic, not a demographic or age guarantee.
    """

    source_rgb = _synthesize_rgb(
        generator=generator,
        ws=w,
        c=c,
        noise_mode=noise_mode,
    )
    source_descriptors = _image_descriptors(source_rgb)

    candidate_descriptors: Dict[str, List[torch.Tensor]] = {
        key: [] for key in source_descriptors
    }

    batch_size = DONOR_RENDER_BATCH_SIZE
    for start in range(0, donor_bank.shape[0], batch_size):
        donor_batch = donor_bank[start : start + batch_size]
        pose_batch = c.repeat(donor_batch.shape[0], 1)
        donor_rgb = _synthesize_rgb(
            generator=generator,
            ws=donor_batch,
            c=pose_batch,
            noise_mode=noise_mode,
        )
        descriptors = _image_descriptors(donor_rgb)
        for key, value in descriptors.items():
            candidate_descriptors[key].append(value)

    candidate_descriptors = {
        key: torch.cat(parts, dim=0)
        for key, parts in candidate_descriptors.items()
    }

    raw_errors: Dict[str, torch.Tensor] = {}
    normalized_errors: List[torch.Tensor] = []
    for key, source_value in source_descriptors.items():
        error = _descriptor_error(candidate_descriptors[key], source_value)
        raw_errors[key] = error
        normalized_errors.append(_median_normalize(error))

    error_matrix = torch.stack(normalized_errors, dim=1)
    worst_preservation_cost = error_matrix.max(dim=1).values

    latent_distance = (donor_bank - w).reshape(donor_bank.shape[0], -1).norm(dim=1)
    normalized_distance = _median_normalize(latent_distance)
    score = normalized_distance / (1.0 + worst_preservation_cost)

    selected_index = int(torch.argmax(score).item())
    selected = donor_bank[selected_index : selected_index + 1]

    metric_metadata = {
        key: {
            "selected_error": float(values[selected_index].detach().cpu()),
            "bank_median_error": float(values.median().detach().cpu()),
            "selected_normalized_error": float(
                _median_normalize(values)[selected_index].detach().cpu()
            ),
        }
        for key, values in raw_errors.items()
    }

    metadata = {
        "selected_index": selected_index,
        "selection_output": "image_raw when available, otherwise image",
        "render_batch_size": DONOR_RENDER_BATCH_SIZE,
        "score": float(score[selected_index].detach().cpu()),
        "latent_distance": float(latent_distance[selected_index].detach().cpu()),
        "normalized_latent_distance": float(
            normalized_distance[selected_index].detach().cpu()
        ),
        "worst_preservation_cost": float(
            worst_preservation_cost[selected_index].detach().cpu()
        ),
        "metrics": metric_metadata,
    }
    return selected, metadata


def build_edit_cache(
    generator: torch.nn.Module,
    w: torch.Tensor,
    c: torch.Tensor,
    mapping_samples: int,
    noise_mode: str,
    random_generator: Optional[torch.Generator],
) -> EditCache:
    """Build all four method anchors once for a stable identity sweep."""

    w_avg = _w_avg_like(generator, w)
    donor_bank = sample_mapped_bank(
        generator=generator,
        c=c,
        w=w,
        count=mapping_samples,
        random_generator=random_generator,
    )

    random_offset_delta, random_metadata = _random_offset_delta(
        w=w,
        w_avg=w_avg,
        donor_bank=donor_bank,
        random_generator=random_generator,
    )

    selected_donor, donor_metadata = select_image_matched_donor(
        generator=generator,
        w=w,
        c=c,
        donor_bank=donor_bank,
        noise_mode=noise_mode,
    )

    # Preserve the source W+ residuals while shifting only the shared W component.
    source_shared = w.mean(dim=1, keepdim=True)
    donor_shared = selected_donor.mean(dim=1, keepdim=True)
    image_matched_residual_delta = (donor_shared - source_shared).expand_as(w)

    metadata = {
        "mapping_samples": mapping_samples,
        "random_offset": random_metadata,
        "image_matched_donor": donor_metadata,
        "image_matched_residual_norm": float(
            image_matched_residual_delta.norm().detach().cpu()
        ),
        "method_limitations": (
            "Image descriptors are fixed non-learned proxies. They do not measure "
            "identity, age, race, sex, or demographic attributes directly."
        ),
    }

    return EditCache(
        w_avg=w_avg,
        random_offset_delta=random_offset_delta,
        image_matched_residual_delta=image_matched_residual_delta,
        metadata=metadata,
    )


def edit_latent(
    generator: torch.nn.Module,
    w: torch.Tensor,
    c: torch.Tensor,
    identity: float,
    deid_mode: str,
    w_rand_cache: Optional[torch.Tensor] = None,
    w_std: Optional[float] = None,
    edit_cache: Optional[EditCache] = None,
) -> torch.Tensor:
    """
    Edit W+ with identity=1 returning the exact source latent.

    identity=0 means maximum configured edit strength, not guaranteed identity
    removal. ``w_rand_cache`` and ``w_std`` are retained only for compatibility
    with older call sites.
    """

    del w_std
    identity = float(identity)
    if not 0.0 <= identity <= 1.0:
        raise ValueError(f"identity must be in [0, 1], got {identity}.")
    if deid_mode not in DEIDENTIFICATION_TECHNIQUES:
        raise ValueError(f"Unknown deid_mode: {deid_mode}")
    if identity == 1.0:
        return w.clone()

    if edit_cache is None:
        edit_cache = build_edit_cache(
            generator=generator,
            w=w,
            c=c,
            mapping_samples=32,
            noise_mode="const",
            random_generator=None,
        )

    if w_rand_cache is not None:
        if w_rand_cache.shape != w.shape:
            raise ValueError(
                f"w_rand_cache shape {tuple(w_rand_cache.shape)} does not "
                f"match W+ shape {tuple(w.shape)}."
            )
        source_shared = w.mean(dim=1, keepdim=True)
        donor_shared = w_rand_cache.mean(dim=1, keepdim=True)
        edit_cache.image_matched_residual_delta = (
            donor_shared - source_shared
        ).expand_as(w)

    strength = 1.0 - identity
    edited = w.clone()

    if deid_mode == "avg":
        return w + strength * (edit_cache.w_avg - w)

    if deid_mode == "mid_avg":
        mid_start, mid_end = _mid_band(w.shape[1])
        edited[:, mid_start:mid_end] = w[:, mid_start:mid_end] + strength * (
            edit_cache.w_avg[:, mid_start:mid_end] - w[:, mid_start:mid_end]
        )
        return edited

    if deid_mode == "rnd_offset":
        return w + strength * edit_cache.random_offset_delta

    if deid_mode == "image_matched_residual":
        return w + strength * edit_cache.image_matched_residual_delta

    raise AssertionError(f"Unhandled deid_mode: {deid_mode}")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)


def load_generator(path: Path, device: torch.device) -> Tuple[torch.nn.Module, str]:
    """Load a generator through the project loader, with raw pickle fallback."""

    legacy_error = None
    try:
        with dnnlib.util.open_url(str(path)) as file:
            data = legacy.load_network_pkl(file)

        if not isinstance(data, dict) or "G_ema" not in data:
            raise KeyError("Loaded pickle does not contain a 'G_ema' entry.")

        generator = data["G_ema"]
        loader_name = "legacy.load_network_pkl"
    except Exception as error:
        legacy_error = f"{type(error).__name__}: {error}"
        with path.open("rb") as file:
            data = pickle.load(file)

        if isinstance(data, dict) and "G_ema" in data:
            generator = data["G_ema"]
        elif isinstance(data, torch.nn.Module):
            generator = data
        else:
            raise RuntimeError(
                "Raw pickle contained neither a PyTorch module nor a dictionary "
                "with a 'G_ema' entry."
            )
        loader_name = f"pickle.load fallback; legacy error: {legacy_error}"

    generator = generator.to(device).float()
    generator.eval().requires_grad_(False)
    return generator, loader_name


def configure_generator(
    generator: torch.nn.Module,
    sample_mult: float,
    nrr: Optional[int],
) -> Dict[str, Any]:
    if sample_mult <= 0:
        raise ValueError("--sample-mult must be greater than zero.")
    if nrr is not None and nrr <= 0:
        raise ValueError("--nrr must be greater than zero when provided.")

    details: Dict[str, Any] = {
        "rendering_kwargs_before": json_safe(
            getattr(generator, "rendering_kwargs", None)
        )
    }

    rendering_kwargs = getattr(generator, "rendering_kwargs", None)
    if isinstance(rendering_kwargs, dict):
        for key in ("depth_resolution", "depth_resolution_importance"):
            if key in rendering_kwargs:
                rendering_kwargs[key] = max(
                    1,
                    int(round(float(rendering_kwargs[key]) * sample_mult)),
                )

    if nrr is not None:
        generator.neural_rendering_resolution = nrr

    details["rendering_kwargs_after"] = json_safe(
        getattr(generator, "rendering_kwargs", None)
    )
    details["neural_rendering_resolution"] = json_safe(
        getattr(generator, "neural_rendering_resolution", None)
    )
    return details


def load_pose(path: Path, device: torch.device) -> torch.Tensor:
    pose = np.load(path)
    if pose.size != 25:
        raise ValueError(
            f"Pose must contain 25 values, but {path} has shape {pose.shape} "
            f"and size {pose.size}."
        )

    tensor = torch.from_numpy(pose.reshape(1, 25)).to(
        device=device,
        dtype=torch.float32,
    )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Pose contains NaN or infinity: {path}")
    return tensor


def load_latent(path: Path, device: torch.device) -> torch.Tensor:
    array = np.load(path)
    if array.ndim not in (1, 2, 3):
        raise ValueError(
            f"Latent must have 1, 2, or 3 dimensions, got {array.shape}: {path}"
        )

    latent = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    if not torch.isfinite(latent).all():
        raise ValueError(f"Latent contains NaN or infinity: {path}")
    return latent


def normalize_latent(
    latent: torch.Tensor,
    generator: torch.nn.Module,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Convert common W and W+ layouts to [batch, num_ws, w_dim]."""

    original_shape = list(latent.shape)
    actions: List[str] = []

    if latent.ndim == 1:
        latent = latent.unsqueeze(0).unsqueeze(0)
        actions.append("expanded [w_dim] to [1, 1, w_dim]")
    elif latent.ndim == 2:
        if latent.shape[0] == 1:
            latent = latent.unsqueeze(1)
            actions.append("expanded [1, w_dim] to [1, 1, w_dim]")
        else:
            latent = latent.unsqueeze(0)
            actions.append("expanded [num_ws, w_dim] to [1, num_ws, w_dim]")

    mapping = _mapping_module(generator)
    expected_num_ws = getattr(mapping, "num_ws", None)
    expected_w_dim = getattr(generator, "w_dim", latent.shape[-1])

    if latent.shape[-1] != expected_w_dim:
        raise ValueError(
            f"Latent W dimension is {latent.shape[-1]}, but generator expects "
            f"{expected_w_dim}."
        )

    if expected_num_ws is not None and latent.shape[1] != expected_num_ws:
        if latent.shape[1] == 1:
            latent = latent.repeat(1, expected_num_ws, 1)
            actions.append(
                f"repeated the single W vector across {expected_num_ws} layers"
            )
        else:
            raise ValueError(
                f"Latent has {latent.shape[1]} style vectors, but generator "
                f"expects {expected_num_ws}."
            )

    return latent, {
        "original_shape": original_shape,
        "normalized_shape": list(latent.shape),
        "actions": actions,
        "expected_num_ws": expected_num_ws,
        "expected_w_dim": expected_w_dim,
    }


def convert_output_to_image(tensor: torch.Tensor, image_mode: str) -> Image.Image:
    tensor = tensor.detach()

    if image_mode == "image_depth":
        tensor = tensor[0]
        if tensor.ndim == 3:
            tensor = tensor[0]

        minimum = tensor.min()
        maximum = tensor.max()
        denominator = maximum - minimum
        if float(denominator) > 0:
            tensor = (tensor - minimum) / denominator
        else:
            tensor = torch.zeros_like(tensor)

        array = (tensor * 255).to(torch.uint8).cpu().numpy()
        return Image.fromarray(array, mode="L")

    tensor = (tensor.clamp(-1, 1) + 1) * 0.5
    tensor = (tensor * 255).to(torch.uint8)
    array = tensor[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def render(
    generator: torch.nn.Module,
    latent: torch.Tensor,
    pose: torch.Tensor,
    image_mode: str,
    noise_mode: str,
) -> Image.Image:
    with torch.no_grad():
        output = generator.synthesis(
            ws=latent,
            c=pose,
            noise_mode=noise_mode,
        )

    if not isinstance(output, dict):
        raise TypeError(
            f"generator.synthesis() returned {type(output).__name__}; expected dict."
        )
    if image_mode not in output:
        raise KeyError(
            f"Generator output does not contain {image_mode!r}. Available keys: "
            f"{sorted(output.keys())}"
        )
    return convert_output_to_image(output[image_mode], image_mode)


def read_processed_names(processed_path: Path) -> List[str]:
    names: List[str] = []
    seen = set()

    for line_number, raw_line in enumerate(
        processed_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        name = raw_line.strip()
        if not name or name.startswith("#"):
            continue
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(
                f"Invalid directory name on line {line_number} of "
                f"{processed_path}: {name!r}"
            )
        if name not in seen:
            names.append(name)
            seen.add(name)

    if not names:
        raise ValueError(f"No image names found in {processed_path}.")
    return names


def build_identity_values(
    minimum: float,
    maximum: float,
    step: float,
) -> List[float]:
    for label, value in (("minimum", minimum), ("maximum", maximum)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Identity {label} must be finite and in [0, 1].")
    if minimum > maximum:
        raise ValueError("--min-identity cannot be greater than --max-identity.")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("--identity-step must be finite and greater than zero.")

    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-12):
        return [float(minimum)]

    count = int(math.floor((maximum - minimum) / step + 1e-12))
    values = [minimum + index * step for index in range(count + 1)]
    values = [min(maximum, max(minimum, value)) for value in values]

    if not math.isclose(values[-1], maximum, rel_tol=0.0, abs_tol=1e-9):
        values.append(maximum)
    else:
        values[-1] = maximum

    unique: List[float] = []
    for value in values:
        if not unique or not math.isclose(
            value,
            unique[-1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            unique.append(float(value))
    return unique


def stable_image_seed(base_seed: int, image_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{image_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def make_torch_generator(
    device: torch.device,
    seed: int,
) -> torch.Generator:
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(seed)
    return random_generator


def identity_filename(index: int, identity: float) -> str:
    return f"frame_{index:03d}_identity_{identity:.4f}.png"


def process_image(
    image_name: str,
    input_dir: Path,
    output_dir: Path,
    identities: Sequence[float],
    techniques: Sequence[str],
    device: torch.device,
    image_mode: str,
    noise_mode: str,
    sample_mult: float,
    nrr: Optional[int],
    mapping_samples: int,
    base_seed: int,
    overwrite: bool,
) -> Dict[str, Any]:
    source_dir = input_dir / image_name
    target_dir = output_dir / image_name

    paths = {
        "generator": source_dir / "generator.pkl",
        "pose": source_dir / "pose.npy",
        "latent": source_dir / "render_w_plus.npy",
        "original": source_dir / "original.png",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} file for {image_name}: {path}")

    if overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["original"], target_dir / "original.png")

    image_seed = stable_image_seed(base_seed, image_name)
    random_generator = make_torch_generator(device, image_seed)

    print(f"[INFO] {image_name}: loading generator", flush=True)
    generator, loader_name = load_generator(paths["generator"], device)
    generator_details = configure_generator(generator, sample_mult, nrr)

    pose = load_pose(paths["pose"], device)
    raw_latent = load_latent(paths["latent"], device)
    w, normalization = normalize_latent(raw_latent, generator)

    print(
        f"[INFO] {image_name}: evaluating {mapping_samples} mapped donor candidates",
        flush=True,
    )
    edit_cache = build_edit_cache(
        generator=generator,
        w=w,
        c=pose,
        mapping_samples=mapping_samples,
        noise_mode=noise_mode,
        random_generator=random_generator,
    )

    renders: Dict[str, Any] = {}
    for technique in techniques:
        technique_dir = target_dir / technique
        technique_dir.mkdir(parents=True, exist_ok=True)

        for stale_frame in technique_dir.glob("frame_*_identity_*.png"):
            stale_frame.unlink()

        technique_outputs: List[Dict[str, Any]] = []
        print(
            f"[INFO] {image_name}: rendering {technique} "
            f"({len(identities)} identities)",
            flush=True,
        )

        for frame_index, identity in enumerate(identities):
            edited_w = edit_latent(
                generator=generator,
                w=w,
                c=pose,
                identity=identity,
                deid_mode=technique,
                edit_cache=edit_cache,
            )

            if identity == 1.0 and not torch.equal(edited_w, w):
                raise AssertionError(
                    f"{technique} changed W+ at identity=1.0 for {image_name}."
                )
            if not torch.isfinite(edited_w).all():
                raise ValueError(
                    f"Edited W+ contains NaN or infinity: {image_name}/{technique} "
                    f"identity={identity}"
                )

            image = render(
                generator=generator,
                latent=edited_w,
                pose=pose,
                image_mode=image_mode,
                noise_mode=noise_mode,
            )
            filename = identity_filename(frame_index, identity)
            output_path = technique_dir / filename
            image.save(output_path)
            image.close()

            technique_outputs.append(
                {
                    "frame": frame_index,
                    "identity": identity,
                    "edit_strength": 1.0 - identity,
                    "path": str(output_path),
                }
            )

        renders[technique] = technique_outputs

    report = {
        "status": "success",
        "image_name": image_name,
        "source_dir": str(source_dir),
        "output_dir": str(target_dir),
        "seed": image_seed,
        "generator_loader": loader_name,
        "generator_class": (
            f"{generator.__class__.__module__}.{generator.__class__.__name__}"
        ),
        "generator_details": generator_details,
        "pose_shape": list(pose.shape),
        "latent_normalization": normalization,
        "identities": list(identities),
        "techniques": list(techniques),
        "edit_cache": edit_cache.metadata,
        "renders": renders,
    }

    report_path = target_dir / "render_manifest.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(report), file, indent=2, sort_keys=True)
        file.write("\n")

    del edit_cache, w, raw_latent, pose, generator
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[SUCCESS] {image_name}: completed", flush=True)
    return report


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    processed_path = input_dir / "processed.txt"

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir:
        raise ValueError(
            "--output-dir must differ from --input-dir because output image "
            "directories use the same names as the source directories."
        )
    if not processed_path.is_file():
        raise FileNotFoundError(f"processed.txt not found: {processed_path}")
    if args.mapping_samples < 4:
        raise ValueError("--mapping-samples must be at least 4.")
    if args.sample_mult <= 0:
        raise ValueError("--sample-mult must be greater than zero.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA-compatible GPU is visible."
        )

    identities = build_identity_values(
        minimum=args.min_identity,
        maximum=args.max_identity,
        step=args.identity_step,
    )
    image_names = read_processed_names(processed_path)
    techniques = list(dict.fromkeys(args.techniques))
    device = torch.device(args.device)

    if args.noise_mode == "random":
        print(
            "[WARNING] --noise-mode random makes donor-image comparison less "
            "stable. Use --noise-mode const for reproducible selection.",
            file=sys.stderr,
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    edit_configuration = {
        "mapping_samples": args.mapping_samples,
        "random_offset_fraction_of_mapped_radius": RND_OFFSET_FRACTION,
        "donor_render_batch_size": DONOR_RENDER_BATCH_SIZE,
        "selection_descriptors": [
            "central YCbCr mean/std",
            "coarse central grayscale",
            "central gradient structure",
            "central texture energy",
            "outer border appearance",
        ],
        "selection_score": (
            "median-normalized latent distance divided by one plus the worst "
            "median-normalized preservation error"
        ),
    }
    batch_report: Dict[str, Any] = {
        "status": "running",
        "input_dir": str(input_dir),
        "processed_path": str(processed_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "identities": identities,
        "techniques": techniques,
        "edit_configuration": edit_configuration,
        "images": {},
    }

    failures = 0
    for image_name in image_names:
        try:
            batch_report["images"][image_name] = process_image(
                image_name=image_name,
                input_dir=input_dir,
                output_dir=output_dir,
                identities=identities,
                techniques=techniques,
                device=device,
                image_mode=args.image_mode,
                noise_mode=args.noise_mode,
                sample_mult=args.sample_mult,
                nrr=args.nrr,
                mapping_samples=args.mapping_samples,
                base_seed=args.seed,
                overwrite=args.overwrite,
            )
        except Exception as error:
            failures += 1
            error_report = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            batch_report["images"][image_name] = error_report
            print(
                f"[FAILED] {image_name}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    batch_report["status"] = "success" if failures == 0 else "partial_failure"
    batch_report["image_count"] = len(image_names)
    batch_report["failure_count"] = failures

    report_path = output_dir / "batch_render_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(batch_report), file, indent=2, sort_keys=True)
        file.write("\n")

    if failures:
        print(
            f"[DONE] Completed with {failures} failure(s). Report: {report_path}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(f"[DONE] All images completed. Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
