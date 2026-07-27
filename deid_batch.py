import argparse
import copy
import json
import logging
import os
import pickle
import platform
import shutil
import signal
import socket
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger("torch").setLevel(logging.ERROR)

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

import dnnlib
import utils.legacy as legacy
from criteria.id_loss import IDLoss
from editors import w_plus_editor_WD

import latent_vector_edit


DEFAULT_MODES = [
    "mid_mix",
    "mid_avg",
    "mid_interp",
    "mid_w_std_noise",
    "mid_orthogonal",
]


STOP_REQUESTED = False
STOP_SIGNAL = None
STATE = {}
STATE_PATH = None


class StopRequested(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run image de-identification with resumable image-level state."
    )

    parser.add_argument(
        "--network-path",
        required=True,
        help="Path to the base network .pkl file.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing <image_id>.png and <image_id>.npy.",
    )
    parser.add_argument(
        "--image-list",
        required=True,
        help="Text file containing one image ID per line.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where final per-image outputs will be saved.",
    )
    parser.add_argument(
        "--tmp-dir",
        default="./tmp_results",
        help="Temporary directory used by projection and PTI.",
    )

    parser.add_argument("--pre-iterations", type=int, default=300)
    parser.add_argument("--post-iterations", type=int, default=300)
    parser.add_argument("--w-avg-samples", type=int, default=600)

    parser.add_argument("--lambda-id", type=float, default=1.0)
    parser.add_argument("--lambda-origin", type=float, default=1.0)

    parser.add_argument("--pti-sample-mult", type=float, default=3.0)
    parser.add_argument("--render-sample-mult", type=float, default=3.0)

    parser.add_argument("--trunc-min", type=float, default=0.0)
    parser.add_argument("--trunc-max", type=float, default=1.0)
    parser.add_argument("--trunc-step", type=float, default=0.1)

    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
    )

    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Do not create GIF files.",
    )
    parser.add_argument("--gif-duration-ms", type=int, default=150)

    parser.add_argument(
        "--save-generator",
        action="store_true",
        help=(
            "Save generator.pkl, pose.npy, w_plus.npy, and "
            "render_w_plus.npy in each image output folder."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess images that already have an _SUCCESS marker.",
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable PyTorch anomaly detection. This is slower.",
    )

    return parser.parse_args()


def atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    temporary_path = (
        f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    )

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, path)


def to_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()

    return str(value)


def build_common_metadata(
    args,
    device,
    generator_base,
    base_depth_resolution,
    base_depth_resolution_importance,
    w_std,
    truncation_values,
    image_ids,
):
    network_stat = os.stat(args.network_path)

    gpu_devices = []
    for gpu_index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(gpu_index)
        gpu_devices.append(
            {
                "index": gpu_index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )

    return {
        "metadata_version": 1,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arguments": to_json_safe(vars(args)),
        "effective_processing_parameters": {
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "detect_anomaly": bool(args.detect_anomaly),
            "export_gif": not args.no_gif,
            "input_image_extension": ".png",
            "pose_extension": ".npy",
            "input_resize": [512, 512],
            "normalization_mean": [0.5, 0.5, 0.5],
            "normalization_std": [0.5, 0.5, 0.5],
            "synthesis_output_key": "image",
            "synthesis_noise_mode": "const",
            "projection_image_output_enabled": False,
            "projection_image_log_step": 20,
            "pti_image_output_enabled": False,
            "pti_image_log_step": 10,
            "base_depth_resolution": int(base_depth_resolution),
            "base_depth_resolution_importance": int(
                base_depth_resolution_importance
            ),
            "pti_depth_resolution": int(
                base_depth_resolution * args.pti_sample_mult
            ),
            "pti_depth_resolution_importance": int(
                base_depth_resolution_importance
                * args.pti_sample_mult
            ),
            "render_depth_resolution": int(
                base_depth_resolution * args.render_sample_mult
            ),
            "render_depth_resolution_importance": int(
                base_depth_resolution_importance
                * args.render_sample_mult
            ),
            "truncation_values": to_json_safe(truncation_values),
            "w_std": to_json_safe(w_std),
            "save_w_vectors_with_generator": bool(args.save_generator),
        },
        "generator": {
            "network_path": args.network_path,
            "network_size_bytes": network_stat.st_size,
            "network_modified_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
                time.localtime(network_stat.st_mtime),
            ),
            "rendering_kwargs": to_json_safe(
                generator_base.rendering_kwargs
            ),
            "parameters_require_grad_on_load": any(
                parameter.requires_grad
                for parameter in generator_base.parameters()
            ),
        },
        "dataset": {
            "dataset_dir": args.dataset_dir,
            "image_list": args.image_list,
            "image_count": len(image_ids),
            "image_ids": image_ids,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": getattr(Image, "__version__", None),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "hardware": {
            "hostname": socket.gethostname(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": gpu_devices,
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
        },
    }


def update_state(**values):
    if STATE_PATH is None:
        return

    STATE.update(values)
    STATE["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    atomic_write_json(STATE_PATH, STATE)


def handle_signal(signum, frame):
    global STOP_REQUESTED
    global STOP_SIGNAL

    STOP_REQUESTED = True
    STOP_SIGNAL = signum

    try:
        update_state(
            status="termination_requested",
            signal=signal.Signals(signum).name,
        )
    except Exception:
        pass


def check_stop():
    if STOP_REQUESTED:
        raise StopRequested(
            f"Termination requested by signal {STOP_SIGNAL}"
        )


def read_lines(path):
    with open(path, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        ]


def append_processed(path, image_id):
    with open(path, "a", encoding="utf-8") as file:
        file.write(image_id + "\n")
        file.flush()
        os.fsync(file.fileno())


def load_pose(path, device):
    pose = np.load(path)
    pose = np.reshape(pose, (1, 25))

    return torch.tensor(
        pose,
        device=device,
        dtype=torch.float32,
    )


def load_id_image(path, device):
    image = Image.open(path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
            transforms.Resize((512, 512)),
        ]
    )

    transformed_image = transform(image).to(device)

    return torch.squeeze(
        (transformed_image + 1) / 2
    ) * 255


def create_truncation_values(minimum, maximum, step):
    return np.arange(
        minimum,
        maximum + step / 2,
        step,
    )


def render_mode_frames(
    generator,
    latent,
    pose,
    deid_mode,
    mode_dir,
    w_rand_cache,
    w_std,
    truncation_values,
    export_gif,
    gif_duration_ms,
    image_mode="image",
):
    os.makedirs(mode_dir, exist_ok=True)

    frames = []

    for frame_index, truncation in enumerate(truncation_values):
        check_stop()

        edited_latent = latent.clone()

        edited_latent = latent_vector_edit.edit_latent(
            generator,
            edited_latent,
            pose,
            float(truncation),
            deid_mode,
            w_rand_cache=w_rand_cache,
            w_std=w_std,
        )

        with torch.no_grad():
            output = generator.synthesis(
                ws=edited_latent,
                c=pose,
                noise_mode="const",
            )
            image_tensor = output[image_mode]

        image_tensor = (
            image_tensor.clamp(-1, 1) + 1
        ) * 0.5

        image_tensor = (
            image_tensor * 255
        ).to(torch.uint8)

        image_array = (
            image_tensor[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        pil_image = Image.fromarray(image_array)

        frame_path = os.path.join(
            mode_dir,
            f"frame_{frame_index:04d}_trunc{truncation:.3f}.png",
        )

        pil_image.save(frame_path)

        if export_gif:
            frames.append(pil_image.copy())

        pil_image.close()

    if export_gif and frames:
        gif_path = os.path.join(
            mode_dir,
            f"{deid_mode}.gif",
        )

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=gif_duration_ms,
            loop=0,
        )

        for frame in frames:
            frame.close()

    return len(truncation_values)


def save_generator(generator, path):
    generator = generator.eval().requires_grad_(False).cpu()

    with open(path, "wb") as file:
        pickle.dump(
            {"G_ema": generator},
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return generator


def process_image(
    image_id,
    args,
    device,
    generator_base,
    base_depth_resolution,
    base_depth_resolution_importance,
    id_loss,
    w_std,
    truncation_values,
    in_progress_root,
):
    image_path = os.path.join(
        args.dataset_dir,
        f"{image_id}.png",
    )

    pose_path = os.path.join(
        args.dataset_dir,
        f"{image_id}.npy",
    )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if not os.path.isfile(pose_path):
        raise FileNotFoundError(
            f"Pose file not found: {pose_path}"
        )

    final_output_dir = os.path.join(
        args.output_dir,
        image_id,
    )

    temporary_output_dir = os.path.join(
        in_progress_root,
        image_id,
    )

    work_dir = os.path.join(
        args.tmp_dir,
        image_id,
    )

    shutil.rmtree(
        temporary_output_dir,
        ignore_errors=True,
    )
    shutil.rmtree(
        work_dir,
        ignore_errors=True,
    )

    os.makedirs(
        temporary_output_dir,
        exist_ok=False,
    )
    os.makedirs(
        work_dir,
        exist_ok=False,
    )

    shutil.copy2(
        image_path,
        os.path.join(
            temporary_output_dir,
            "original.png",
        ),
    )

    generator = None
    generator_final = None
    success = False
    start_time = time.time()

    try:
        update_state(
            current_image=image_id,
            stage="loading_inputs",
        )

        pose = load_pose(
            pose_path,
            device,
        )

        identity_image = load_id_image(
            image_path,
            device,
        )

        check_stop()

        generator = copy.deepcopy(generator_base)

        generator.rendering_kwargs[
            "depth_resolution"
        ] = int(
            base_depth_resolution
            * args.pti_sample_mult
        )

        generator.rendering_kwargs[
            "depth_resolution_importance"
        ] = int(
            base_depth_resolution_importance
            * args.pti_sample_mult
        )

        update_state(
            current_image=image_id,
            stage="w_plus_projection",
        )

        w_plus = w_plus_editor_WD.project(
            generator,
            pose,
            work_dir,
            identity_image,
            device=device,
            w_avg_samples=args.w_avg_samples,
            w_name=image_id,
            num_steps=args.pre_iterations,
            id_loss=id_loss,
            lamda_id=args.lambda_id,
            lamda_origin=args.lambda_origin,
            image_output_enabled=False,
            image_log_step=20,
        )

        check_stop()

        update_state(
            current_image=image_id,
            stage="pti",
        )

        generator_final = w_plus_editor_WD.project_pti(
            generator,
            pose,
            work_dir,
            identity_image,
            w_plus,
            device=device,
            w_avg_samples=args.w_avg_samples,
            w_name=image_id,
            num_steps_pti=args.post_iterations,
            id_loss=id_loss,
            lamda_id=args.lambda_id,
            lamda_origin=args.lambda_origin,
            image_output_enabled=False,
            image_log_step=10,
        )

        check_stop()

        generator_final.eval().requires_grad_(False)

        generator_final.rendering_kwargs[
            "depth_resolution"
        ] = int(
            base_depth_resolution
            * args.render_sample_mult
        )

        generator_final.rendering_kwargs[
            "depth_resolution_importance"
        ] = int(
            base_depth_resolution_importance
            * args.render_sample_mult
        )

        latent = w_plus

        if latent.ndim == 2:
            latent = latent.unsqueeze(0)

        if hasattr(
            generator_final.backbone.mapping,
            "num_ws",
        ):
            num_ws = (
                generator_final.backbone.mapping.num_ws
            )

            if (
                latent.ndim == 3
                and latent.shape[1] != num_ws
            ):
                if latent.shape[1] != 1:
                    raise ValueError(
                        "Unexpected number of W+ style vectors: "
                        f"{latent.shape[1]}, expected {num_ws}."
                    )

                latent = latent.repeat(
                    1,
                    num_ws,
                    1,
                )

        update_state(
            current_image=image_id,
            stage="precomputing_latents",
        )

        w_rand_cache = (
            latent_vector_edit.precompute_w_rand(
                generator_final,
                pose,
            )
        )

        for mode_index, mode in enumerate(
            args.modes,
            start=1,
        ):
            check_stop()

            update_state(
                current_image=image_id,
                stage="rendering",
                current_mode=mode,
                current_mode_index=mode_index,
                total_modes=len(args.modes),
            )

            mode_dir = os.path.join(
                temporary_output_dir,
                mode,
            )

            frame_count = render_mode_frames(
                generator=generator_final,
                latent=latent,
                pose=pose,
                deid_mode=mode,
                mode_dir=mode_dir,
                w_rand_cache=w_rand_cache,
                w_std=w_std,
                truncation_values=truncation_values,
                export_gif=not args.no_gif,
                gif_duration_ms=args.gif_duration_ms,
            )

            print(
                f"[{image_id}] Saved {frame_count} "
                f"frames for {mode}",
                flush=True,
            )

        check_stop()

        if args.save_generator:
            update_state(
                current_image=image_id,
                stage="saving_generator_and_latent",
            )

            shutil.copy2(
                pose_path,
                os.path.join(
                    temporary_output_dir,
                    "pose.npy",
                ),
            )

            # Exact W/W+ tensor returned by the projection stage.
            np.save(
                os.path.join(
                    temporary_output_dir,
                    "w_plus.npy",
                ),
                w_plus.detach().cpu().numpy(),
            )

            # W+ tensor after shape normalization/repetition. This is the
            # base latent that was passed to edit_latent() for rendering.
            np.save(
                os.path.join(
                    temporary_output_dir,
                    "render_w_plus.npy",
                ),
                latent.detach().cpu().numpy(),
            )

            generator_final = save_generator(
                generator_final,
                os.path.join(
                    temporary_output_dir,
                    "generator.pkl",
                ),
            )

            torch.cuda.empty_cache()

        elapsed_seconds = time.time() - start_time

        metadata = {
            "image_id": image_id,
            "source_image": image_path,
            "source_pose": pose_path,
            "completed_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            ),
            "elapsed_seconds": elapsed_seconds,
            "common_metadata_file": "../common_metadata.json",
            "generator_saved": bool(args.save_generator),
            "pose_saved": bool(args.save_generator),
            "w_plus_saved": bool(args.save_generator),
            "w_plus_shape": list(w_plus.shape),
            "render_w_plus_shape": list(latent.shape),
        }

        atomic_write_json(
            os.path.join(
                temporary_output_dir,
                "metadata.json",
            ),
            metadata,
        )

        with open(
            os.path.join(
                temporary_output_dir,
                "_SUCCESS",
            ),
            "w",
            encoding="utf-8",
        ) as file:
            file.write("completed\n")
            file.flush()
            os.fsync(file.fileno())

        update_state(
            current_image=image_id,
            stage="finalizing",
        )

        if os.path.exists(final_output_dir):
            shutil.rmtree(final_output_dir)

        os.replace(
            temporary_output_dir,
            final_output_dir,
        )

        success = True

        return elapsed_seconds

    finally:
        if generator is not None:
            del generator

        if generator_final is not None:
            del generator_final

        torch.cuda.empty_cache()

        if success:
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )


def main():
    global STATE_PATH
    global STATE

    args = parse_args()

    args.network_path = os.path.abspath(
        os.path.expanduser(args.network_path)
    )
    args.dataset_dir = os.path.abspath(
        os.path.expanduser(args.dataset_dir)
    )
    args.image_list = os.path.abspath(
        os.path.expanduser(args.image_list)
    )
    args.output_dir = os.path.abspath(
        os.path.expanduser(args.output_dir)
    )
    args.tmp_dir = os.path.abspath(
        os.path.expanduser(args.tmp_dir)
    )

    if not os.path.isfile(args.network_path):
        raise FileNotFoundError(
            f"Network not found: {args.network_path}"
        )

    if not os.path.isdir(args.dataset_dir):
        raise NotADirectoryError(
            f"Dataset directory not found: {args.dataset_dir}"
        )

    if not os.path.isfile(args.image_list):
        raise FileNotFoundError(
            f"Image list not found: {args.image_list}"
        )

    if args.trunc_step <= 0:
        raise ValueError(
            "--trunc-step must be greater than zero."
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )
    os.makedirs(
        args.tmp_dir,
        exist_ok=True,
    )

    in_progress_root = os.path.join(
        args.output_dir,
        ".in_progress",
    )

    os.makedirs(
        in_progress_root,
        exist_ok=True,
    )

    STATE_PATH = os.path.join(
        args.output_dir,
        "state.json",
    )

    processed_log_path = os.path.join(
        args.output_dir,
        "processed.txt",
    )

    image_ids = read_lines(args.image_list)

    slurm_job_id = os.environ.get(
        "SLURM_JOB_ID",
        "local",
    )

    STATE = {
        "status": "starting",
        "job_id": slurm_job_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "network_path": args.network_path,
        "dataset_dir": args.dataset_dir,
        "image_list": args.image_list,
        "output_dir": args.output_dir,
        "common_metadata": os.path.join(
            args.output_dir,
            "common_metadata.json",
        ),
        "total_images": len(image_ids),
        "started_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        ),
    }

    update_state()

    signal.signal(
        signal.SIGTERM,
        handle_signal,
    )
    signal.signal(
        signal.SIGINT,
        handle_signal,
    )

    torch.autograd.set_detect_anomaly(
        args.detect_anomaly
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA-compatible GPU detected."
        )

    device = torch.device("cuda")

    print("Detected GPUs:", flush=True)

    for gpu_index in range(
        torch.cuda.device_count()
    ):
        print(
            f"  GPU {gpu_index}: "
            f"{torch.cuda.get_device_name(gpu_index)}",
            flush=True,
        )

    completed_before_start = []

    for image_id in image_ids:
        success_marker = os.path.join(
            args.output_dir,
            image_id,
            "_SUCCESS",
        )

        if os.path.isfile(success_marker):
            completed_before_start.append(image_id)

    update_state(
        status="loading_network",
        completed_images=len(
            completed_before_start
        ),
    )

    print(
        f"[INFO] Loading base network: "
        f"{args.network_path}",
        flush=True,
    )

    with dnnlib.util.open_url(
        args.network_path
    ) as file:
        network_data = (
            legacy.load_network_pkl(file)
        )
        generator_base = (
            network_data["G_ema"].to(device)
        )

    base_depth_resolution = (
        generator_base.rendering_kwargs[
            "depth_resolution"
        ]
    )

    base_depth_resolution_importance = (
        generator_base.rendering_kwargs[
            "depth_resolution_importance"
        ]
    )

    w_std = latent_vector_edit.load_w_std()

    print(
        f"[INFO] w_std={float(w_std):.4f}",
        flush=True,
    )

    id_loss = IDLoss()

    truncation_values = (
        create_truncation_values(
            args.trunc_min,
            args.trunc_max,
            args.trunc_step,
        )
    )

    common_metadata = build_common_metadata(
        args=args,
        device=device,
        generator_base=generator_base,
        base_depth_resolution=base_depth_resolution,
        base_depth_resolution_importance=(
            base_depth_resolution_importance
        ),
        w_std=w_std,
        truncation_values=truncation_values,
        image_ids=image_ids,
    )

    common_metadata_path = os.path.join(
        args.output_dir,
        "common_metadata.json",
    )
    atomic_write_json(
        common_metadata_path,
        common_metadata,
    )

    print(
        f"[INFO] Saved common metadata: "
        f"{common_metadata_path}",
        flush=True,
    )

    completed_count = len(
        completed_before_start
    )

    update_state(
        status="running",
        completed_images=completed_count,
    )

    try:
        for image_index, image_id in enumerate(
            image_ids,
            start=1,
        ):
            check_stop()

            final_output_dir = os.path.join(
                args.output_dir,
                image_id,
            )

            success_marker = os.path.join(
                final_output_dir,
                "_SUCCESS",
            )

            if (
                os.path.isfile(success_marker)
                and not args.overwrite
            ):
                print(
                    f"[{image_id}] Already complete, "
                    "skipping.",
                    flush=True,
                )
                continue

            if (
                os.path.exists(final_output_dir)
                and not args.overwrite
                and not os.path.isfile(
                    success_marker
                )
            ):
                raise RuntimeError(
                    f"Output directory exists without "
                    f"an _SUCCESS marker: "
                    f"{final_output_dir}. "
                    f"Remove it or use --overwrite."
                )

            update_state(
                status="running",
                current_image=image_id,
                current_image_index=image_index,
                completed_images=completed_count,
                stage="starting_image",
                current_mode=None,
            )

            print(
                f"[{image_id}] Starting image "
                f"{image_index}/{len(image_ids)} "
                f"at {time.ctime()}",
                flush=True,
            )

            elapsed_seconds = process_image(
                image_id=image_id,
                args=args,
                device=device,
                generator_base=generator_base,
                base_depth_resolution=(
                    base_depth_resolution
                ),
                base_depth_resolution_importance=(
                    base_depth_resolution_importance
                ),
                id_loss=id_loss,
                w_std=w_std,
                truncation_values=(
                    truncation_values
                ),
                in_progress_root=(
                    in_progress_root
                ),
            )

            append_processed(
                processed_log_path,
                image_id,
            )

            completed_count += 1

            update_state(
                status="running",
                completed_images=completed_count,
                last_completed_image=image_id,
                current_image=None,
                current_mode=None,
                stage="between_images",
            )

            print(
                f"[{image_id}] Complete in "
                f"{elapsed_seconds:.2f} seconds",
                flush=True,
            )

        update_state(
            status="complete",
            completed_images=completed_count,
            current_image=None,
            current_mode=None,
            stage="complete",
            completed_at=time.strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            ),
        )

        print(
            f"All done - {time.ctime()}",
            flush=True,
        )

        return 0

    except StopRequested:
        update_state(
            status="interrupted",
            stage="interrupted",
            signal=(
                signal.Signals(
                    STOP_SIGNAL
                ).name
                if STOP_SIGNAL is not None
                else None
            ),
        )

        print(
            "Termination requested. Completed "
            "images are preserved. The current "
            "image will restart on the next run.",
            file=sys.stderr,
            flush=True,
        )

        if STOP_SIGNAL is not None:
            return 128 + STOP_SIGNAL

        return 1

    except Exception as error:
        update_state(
            status="failed",
            stage="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )

        raise


if __name__ == "__main__":
    raise SystemExit(main())