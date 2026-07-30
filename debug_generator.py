#!/usr/bin/env python3
"""
Validate and render a generator saved by the resumable de-identification script.

The script:
1. Loads generator.pkl.
2. Loads pose.npy.
3. Loads render_w_plus.npy and, when available, w_plus.npy.
4. Validates tensor shapes, dtypes, and finite values.
5. Renders each latent with the saved generator.
6. Writes debug_report.json with generator and latent diagnostics.

Example:
    python debug_saved_generator.py \
        --image-dir /home/container_output/deid/6797_03F28 \
        --outdir /home/container_output/deid/6797_03F28/debug_render \
        --sample-mult 3
"""

import argparse
import json
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image

import dnnlib
import utils.legacy as legacy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug a saved EG3D/FaceDNeRF generator and latent files."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help=(
            "Processed image directory containing generator.pkl, pose.npy, "
            "render_w_plus.npy, and optionally w_plus.npy."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Debug output directory. Defaults to <image-dir>/debug_render.",
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=None,
        help="Generator path. Defaults to <image-dir>/generator.pkl.",
    )
    parser.add_argument(
        "--pose",
        type=Path,
        default=None,
        help="Pose path. Defaults to <image-dir>/pose.npy.",
    )
    parser.add_argument(
        "--render-latent",
        type=Path,
        default=None,
        help="Render latent path. Defaults to <image-dir>/render_w_plus.npy.",
    )
    parser.add_argument(
        "--original-latent",
        type=Path,
        default=None,
        help="Original W/W+ path. Defaults to <image-dir>/w_plus.npy when present.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. Default: cuda",
    )
    parser.add_argument(
        "--image-mode",
        choices=["image", "image_raw", "image_depth"],
        default="image",
    )
    parser.add_argument(
        "--sample-mult",
        type=float,
        default=3.0,
        help="Multiplier for depth_resolution settings.",
    )
    parser.add_argument(
        "--nrr",
        type=int,
        default=None,
        help="Optional neural rendering resolution override.",
    )
    parser.add_argument(
        "--trunc",
        type=float,
        default=1.0,
        help="Optional truncation applied to loaded latents.",
    )
    parser.add_argument(
        "--noise-mode",
        choices=["const", "random", "none"],
        default="const",
    )
    return parser.parse_args()


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
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return repr(value)


def load_generator(path: Path, device: torch.device) -> Tuple[torch.nn.Module, str]:
    """
    First try the project's normal legacy loader. If that fails, fall back to
    raw pickle loading because the resumable script writes {"G_ema": generator}.
    """
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
                "Raw pickle loaded successfully, but it contained neither a "
                "PyTorch module nor a dictionary with a 'G_ema' entry."
            )

        loader_name = f"pickle.load fallback; legacy error: {legacy_error}"

    generator = generator.to(device).float()
    generator.eval().requires_grad_(False)
    return generator, loader_name


def load_pose(path: Path, device: torch.device) -> torch.Tensor:
    pose = np.load(path)

    if pose.size != 25:
        raise ValueError(
            f"Pose must contain 25 values, but {path} has shape {pose.shape} "
            f"and size {pose.size}."
        )

    pose = pose.reshape(1, 25)
    tensor = torch.from_numpy(pose).to(device=device, dtype=torch.float32)

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
    """
    Convert common W and W+ layouts to [batch, num_ws, w_dim].
    """
    original_shape = list(latent.shape)
    actions = []

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

    mapping = generator.backbone.mapping
    expected_num_ws = getattr(mapping, "num_ws", None)
    expected_w_dim = getattr(generator, "w_dim", latent.shape[-1])

    if latent.shape[-1] != expected_w_dim:
        raise ValueError(
            f"Latent w dimension is {latent.shape[-1]}, but generator expects "
            f"{expected_w_dim}."
        )

    if expected_num_ws is not None and latent.shape[1] != expected_num_ws:
        if latent.shape[1] == 1:
            latent = latent.repeat(1, expected_num_ws, 1)
            actions.append(
                f"repeated the single W vector across {expected_num_ws} style layers"
            )
        else:
            raise ValueError(
                f"Latent has {latent.shape[1]} style vectors, but generator "
                f"expects {expected_num_ws}."
            )

    details = {
        "original_shape": original_shape,
        "normalized_shape": list(latent.shape),
        "actions": actions,
        "expected_num_ws": expected_num_ws,
        "expected_w_dim": expected_w_dim,
    }
    return latent, details


def apply_truncation(
    latent: torch.Tensor,
    generator: torch.nn.Module,
    truncation_psi: float,
) -> torch.Tensor:
    if truncation_psi == 1.0:
        return latent

    w_avg = generator.backbone.mapping.w_avg.to(
        device=latent.device,
        dtype=latent.dtype,
    )

    if w_avg.ndim == 1:
        w_avg = w_avg.reshape(1, 1, -1)
    elif w_avg.ndim == 2:
        w_avg = w_avg.unsqueeze(0)

    return w_avg + truncation_psi * (latent - w_avg)


def convert_output_to_image(
    tensor: torch.Tensor,
    image_mode: str,
) -> Image.Image:
    tensor = tensor.detach()

    if image_mode == "image_depth":
        # Depth output is usually one channel and is not necessarily in [-1, 1].
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
) -> Tuple[Image.Image, Dict[str, Any]]:
    with torch.no_grad():
        output = generator.synthesis(
            ws=latent,
            c=pose,
            noise_mode=noise_mode,
        )

    if image_mode not in output:
        raise KeyError(
            f"Generator output does not contain {image_mode!r}. "
            f"Available keys: {sorted(output.keys())}"
        )

    image_tensor = output[image_mode]
    image = convert_output_to_image(image_tensor, image_mode)

    output_info = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "minimum": float(value.min().detach().cpu()),
            "maximum": float(value.max().detach().cpu()),
            "finite": bool(torch.isfinite(value).all().detach().cpu()),
        }
        for key, value in output.items()
        if isinstance(value, torch.Tensor)
    }

    return image, output_info


def main() -> int:
    args = parse_args()

    image_dir = args.image_dir.expanduser().resolve()
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else image_dir / "debug_render"
    )
    generator_path = (
        args.generator.expanduser().resolve()
        if args.generator is not None
        else image_dir / "generator.pkl"
    )
    pose_path = (
        args.pose.expanduser().resolve()
        if args.pose is not None
        else image_dir / "pose.npy"
    )
    render_latent_path = (
        args.render_latent.expanduser().resolve()
        if args.render_latent is not None
        else image_dir / "render_w_plus.npy"
    )
    original_latent_path = (
        args.original_latent.expanduser().resolve()
        if args.original_latent is not None
        else image_dir / "w_plus.npy"
    )

    required_paths = [generator_path, pose_path, render_latent_path]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    if args.sample_mult <= 0:
        raise ValueError("--sample-mult must be greater than zero.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA-compatible GPU is visible."
        )

    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    report: Dict[str, Any] = {
        "status": "running",
        "image_dir": str(image_dir),
        "generator_path": str(generator_path),
        "pose_path": str(pose_path),
        "render_latent_path": str(render_latent_path),
        "original_latent_path": (
            str(original_latent_path) if original_latent_path.is_file() else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "arguments": vars(args),
        "renders": {},
    }

    try:
        print(f"[INFO] Loading generator: {generator_path}", flush=True)
        generator, loader_name = load_generator(generator_path, device)

        report["generator_loader"] = loader_name
        report["generator_class"] = (
            f"{generator.__class__.__module__}.{generator.__class__.__name__}"
        )
        report["generator_parameter_count"] = sum(
            parameter.numel() for parameter in generator.parameters()
        )
        report["generator_trainable_parameter_count"] = sum(
            parameter.numel()
            for parameter in generator.parameters()
            if parameter.requires_grad
        )
        report["generator_w_dim"] = getattr(generator, "w_dim", None)
        report["generator_num_ws"] = getattr(
            generator.backbone.mapping,
            "num_ws",
            None,
        )
        report["rendering_kwargs_before"] = json_safe(generator.rendering_kwargs)

        generator.rendering_kwargs["depth_resolution"] = int(
            generator.rendering_kwargs["depth_resolution"] * args.sample_mult
        )
        generator.rendering_kwargs["depth_resolution_importance"] = int(
            generator.rendering_kwargs["depth_resolution_importance"]
            * args.sample_mult
        )

        if args.nrr is not None:
            generator.neural_rendering_resolution = args.nrr

        report["rendering_kwargs_after"] = json_safe(generator.rendering_kwargs)
        report["neural_rendering_resolution"] = json_safe(
            getattr(generator, "neural_rendering_resolution", None)
        )

        pose = load_pose(pose_path, device)
        report["pose"] = {
            "shape": list(pose.shape),
            "dtype": str(pose.dtype),
            "minimum": float(pose.min().detach().cpu()),
            "maximum": float(pose.max().detach().cpu()),
            "finite": bool(torch.isfinite(pose).all().detach().cpu()),
        }

        loaded_latents: Dict[str, torch.Tensor] = {}

        latent_candidates = {
            "render_w_plus": render_latent_path,
        }
        if original_latent_path.is_file():
            latent_candidates["w_plus"] = original_latent_path

        for latent_name, latent_path in latent_candidates.items():
            print(f"[INFO] Loading latent: {latent_path}", flush=True)

            raw_latent = load_latent(latent_path, device)
            normalized_latent, normalization = normalize_latent(
                raw_latent,
                generator,
            )
            normalized_latent = apply_truncation(
                normalized_latent,
                generator,
                args.trunc,
            )

            loaded_latents[latent_name] = normalized_latent

            image, output_info = render(
                generator=generator,
                latent=normalized_latent,
                pose=pose,
                image_mode=args.image_mode,
                noise_mode=args.noise_mode,
            )

            output_path = outdir / f"{latent_name}_{args.image_mode}.png"
            image.save(output_path)
            image.close()

            report["renders"][latent_name] = {
                "source_path": str(latent_path),
                "output_path": str(output_path),
                "raw_shape": list(raw_latent.shape),
                "raw_dtype": str(raw_latent.dtype),
                "raw_minimum": float(raw_latent.min().detach().cpu()),
                "raw_maximum": float(raw_latent.max().detach().cpu()),
                "raw_mean": float(raw_latent.mean().detach().cpu()),
                "raw_standard_deviation": float(
                    raw_latent.std(unbiased=False).detach().cpu()
                ),
                "normalization": normalization,
                "normalized_minimum": float(
                    normalized_latent.min().detach().cpu()
                ),
                "normalized_maximum": float(
                    normalized_latent.max().detach().cpu()
                ),
                "normalized_mean": float(
                    normalized_latent.mean().detach().cpu()
                ),
                "normalized_standard_deviation": float(
                    normalized_latent.std(unbiased=False).detach().cpu()
                ),
                "generator_outputs": output_info,
            }

            print(f"[SUCCESS] Saved: {output_path}", flush=True)

        if "render_w_plus" in loaded_latents and "w_plus" in loaded_latents:
            render_latent = loaded_latents["render_w_plus"]
            original_latent = loaded_latents["w_plus"]

            if render_latent.shape == original_latent.shape:
                difference = render_latent - original_latent
                report["latent_comparison"] = {
                    "same_shape": True,
                    "allclose": bool(
                        torch.allclose(
                            render_latent,
                            original_latent,
                            rtol=1e-5,
                            atol=1e-6,
                        )
                    ),
                    "maximum_absolute_difference": float(
                        difference.abs().max().detach().cpu()
                    ),
                    "mean_absolute_difference": float(
                        difference.abs().mean().detach().cpu()
                    ),
                }
            else:
                report["latent_comparison"] = {
                    "same_shape": False,
                    "render_w_plus_shape": list(render_latent.shape),
                    "w_plus_shape": list(original_latent.shape),
                }

        report["status"] = "success"

    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()

        report_path = outdir / "debug_report.json"
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(json_safe(report), file, indent=2, sort_keys=True)
            file.write("\n")

        print(f"[FAILED] {type(error).__name__}: {error}", file=sys.stderr)
        print(f"[INFO] Partial report: {report_path}", file=sys.stderr)
        return 1

    report_path = outdir / "debug_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(report), file, indent=2, sort_keys=True)
        file.write("\n")

    print(f"[SUCCESS] Debug report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
