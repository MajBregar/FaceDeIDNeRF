import os
import shutil
import copy
import time
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
from PIL import Image
from torchvision.transforms import transforms

import dnnlib
import utils.legacy as legacy
from editors import w_plus_editor_WD
from criteria.id_loss import IDLoss

import latent_vector_edit


network_path = "./networks/ffhqrebalanced512-128.pkl"
dataset_dir = '/home/real_images/final_output'
output_dir = "/home/deid_processed"
tmp_results_dir = './tmp_results'

pre_iterations = 250
post_iterations = 150
pti_sample_mult = 3
gif_sample_mult = 3
trunc_min, trunc_max, trunc_step = 0.0, 1.0, 0.1
export_gif = True
gif_duration_ms = 150

modes = [
    # 'avg',
    # 'true_rnd',
    # 'rnd_avg_offset',
    # 'mapping_rnd',
    # 'mapping_interp',
    # 'w_noise',
    # 'layer_mix',
    # 'coarse_mix',
    # 'fine_mix',
    # 'pca_perturb',
    # 'orthogonal_noise',
    # 'style_shuffle',
    'mid_mix',
    'mid_avg',
    'mid_interp',
    'mid_w_std_noise',
    'mid_orthogonal'
]

input_dict = {
    "lamda_id": 1.0,
    "lamda_origin": 1.0,
}

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA-compatible GPU detected.")

device = torch.device('cuda')
print("Detected GPUs:")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


def load_pose(path):
    c = np.load(path)
    c = np.reshape(c, (1, 25))
    return torch.tensor(c, device=device, dtype=torch.float32)


def load_id_image(path):
    image = Image.open(path).convert('RGB')
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((512, 512))
    ])
    from_im = trans(image).to(device)
    return torch.squeeze((from_im + 1) / 2) * 255


def render_mode_frames(G, latent, c, deid_mode, mode_dir, w_rand_cache, w_std, image_mode='image'):
    os.makedirs(mode_dir, exist_ok=True)
    trunc_values = np.arange(trunc_min, trunc_max + trunc_step, trunc_step)
    frames = []

    for idx, trunc in enumerate(trunc_values):
        w = latent.clone()
        w = latent_vector_edit.edit_latent(
            G, w, c, trunc, deid_mode,
            w_rand_cache=w_rand_cache,
            w_std=w_std
        )

        with torch.no_grad():
            out = G.synthesis(ws=w, c=c, noise_mode='const')
            img = out[image_mode]

        img = (img.clamp(-1, 1) + 1) * 0.5
        img = (img * 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray(img)

        frame_path = os.path.join(mode_dir, f"frame_{idx:04d}_trunc{trunc:.3f}.png")
        pil_img.save(frame_path)
        frames.append(pil_img)

    if export_gif and frames:
        gif_path = os.path.join(mode_dir, f"{deid_mode}.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=gif_duration_ms,
            loop=0
        )

    return len(frames)


def read_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


os.makedirs(output_dir, exist_ok=True)
os.makedirs(tmp_results_dir, exist_ok=True)

images_list_path = os.path.join(output_dir, "images_to_process.txt")
processed_log_path = os.path.join(output_dir, "processed.txt")

if not os.path.exists(images_list_path):
    raise FileNotFoundError(
        f"{images_list_path} not found. Create it with one image_id per line "
        f"before running this script."
    )

image_ids = read_lines(images_list_path)
already_processed = set(read_lines(processed_log_path)) if os.path.exists(processed_log_path) else set()

work_root = os.path.join(tmp_results_dir, "_work")
frames_root = os.path.join(output_dir, "frames")

print(f"[INFO] Loading base network: {network_path}")
with dnnlib.util.open_url(network_path) as f:
    network_data = legacy.load_network_pkl(f)
    G_base = network_data['G_ema'].to(device)

base_depth_res = G_base.rendering_kwargs['depth_resolution']
base_depth_res_imp = G_base.rendering_kwargs['depth_resolution_importance']

w_std = latent_vector_edit.load_w_std()
print(f"[INFO] w_std={w_std:.4f}")

id_loss = IDLoss()

for image_id in image_ids:
    if image_id in already_processed:
        print(f"Skipping {image_id}, already in {processed_log_path}")
        continue

    print("Start:", time.ctime(), "Image:", image_id)

    image_tag = f"{image_id}"
    work_dir = os.path.join(work_root, image_tag)

    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    c = load_pose(os.path.join(dataset_dir, f"{image_id}.npy"))
    id_image = load_id_image(os.path.join(dataset_dir, f"{image_id}.png"))

    G = copy.deepcopy(G_base)
    G.rendering_kwargs['depth_resolution'] = int(base_depth_res * pti_sample_mult)
    G.rendering_kwargs['depth_resolution_importance'] = int(base_depth_res_imp * pti_sample_mult)

    w_plus = w_plus_editor_WD.project(
        G,
        c,
        work_dir,
        id_image,
        device=device,
        w_avg_samples=600,
        w_name=image_id,
        num_steps=pre_iterations,
        id_loss=id_loss,
        lamda_id=input_dict['lamda_id'],
        lamda_origin=input_dict['lamda_origin'],
        image_output_enabled=False,
        image_log_step=20
    )

    G_final = w_plus_editor_WD.project_pti(
        G,
        c,
        work_dir,
        id_image,
        w_plus,
        device=device,
        w_avg_samples=600,
        w_name=image_id,
        num_steps_pti=post_iterations,
        id_loss=id_loss,
        lamda_id=input_dict['lamda_id'],
        lamda_origin=input_dict['lamda_origin'],
        image_output_enabled=False,
        image_log_step=10
    )

    print("Fine-tuned generator ready -", time.ctime())

    G_final.eval().requires_grad_(False)
    G_final.rendering_kwargs['depth_resolution'] = int(base_depth_res * gif_sample_mult)
    G_final.rendering_kwargs['depth_resolution_importance'] = int(base_depth_res_imp * gif_sample_mult)

    latent = w_plus
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    if hasattr(G_final.backbone.mapping, "num_ws"):
        if latent.ndim == 3 and latent.shape[1] != G_final.backbone.mapping.num_ws:
            latent = latent.repeat(1, G_final.backbone.mapping.num_ws, 1)

    w_rand_cache = latent_vector_edit.precompute_w_rand(G_final, c)

    image_frames_dir = os.path.join(frames_root, image_tag)
    for mode in modes:
        mode_dir = os.path.join(image_frames_dir, mode)
        n_frames = render_mode_frames(G_final, latent, c, mode, mode_dir, w_rand_cache, w_std)
        print(f"  Saved {n_frames} frames for {mode} -> {mode_dir}", time.ctime())

    del G, G_final
    torch.cuda.empty_cache()
    shutil.rmtree(work_dir, ignore_errors=True)

    with open(processed_log_path, "a") as f:
        f.write(image_id + "\n")

print("All done -", time.ctime())