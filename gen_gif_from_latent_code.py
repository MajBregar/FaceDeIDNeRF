"""
Render multiple images from an EG3D / FaceDNeRF latent by sweeping truncation
and save the result as a GIF.
"""

import os
os.environ["PYTORCH_ROCM_F64"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_flush_denormal(True)

import click
import numpy as np
import dnnlib
import legacy
from PIL import Image

# -----------------------------------------------------------------------------

TRUNCATION_INTERVAL = (0.0, 1.0)
TRUNCATION_STEP = 0.01

# -----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', required=True)
@click.option('--latent', 'latent_path', required=True)
@click.option('--pose', 'pose_path', required=True)
@click.option('--outdir', required=True)
@click.option('--image-mode', type=click.Choice(['image', 'image_raw', 'image_depth']),
              default='image', show_default=True)
@click.option('--sample-mult', default=2.0, show_default=True)
@click.option('--nrr', default=None, type=int)
def main(
    network_pkl,
    latent_path,
    pose_path,
    outdir,
    image_mode,
    sample_mult,
    nrr
):

    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda')

    print(f"[INFO] Loading network: {network_pkl}")

    # -------------------------------------------------------------------------
    # Load generator
    # -------------------------------------------------------------------------
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).float()

    G.eval()

    # Improve NeRF sampling
    G.rendering_kwargs['depth_resolution'] = int(
        G.rendering_kwargs['depth_resolution'] * sample_mult
    )
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sample_mult
    )

    if nrr is not None:
        G.neural_rendering_resolution = nrr

    # -------------------------------------------------------------------------
    # Load latent
    # -------------------------------------------------------------------------
    latent = np.load(latent_path)
    latent = torch.tensor(latent, device=device, dtype=torch.float32)

    if latent.ndim == 2:
        latent = latent.unsqueeze(0)

    if hasattr(G.backbone.mapping, "num_ws"):
        if latent.ndim == 3 and latent.shape[1] != G.backbone.mapping.num_ws:
            latent = latent.repeat(1, G.backbone.mapping.num_ws, 1)

    print(f"[INFO] Latent shape: {latent.shape}")

    # -------------------------------------------------------------------------
    # Load camera
    # -------------------------------------------------------------------------
    c = np.load(pose_path)
    assert c.shape == (25,), f"Expected pose shape (25,), got {c.shape}"

    c = torch.tensor(c, device=device, dtype=torch.float32).unsqueeze(0)

    print(f"[INFO] Camera pose loaded: {c.shape}")

    # -------------------------------------------------------------------------
    # Generate frames
    # -------------------------------------------------------------------------
    trunc_values = np.arange(
        TRUNCATION_INTERVAL[0],
        TRUNCATION_INTERVAL[1] + TRUNCATION_STEP,
        TRUNCATION_STEP
    )

    frames = []

    truncation_mode = ['avg', 'true_rnd', 'rnd_avg_offset', 'mapping_rnd']
    select_trunc = 3

    for trunc in trunc_values:

        print(f"[INFO] Rendering truncation={trunc:.3f}")

        w = latent.clone()

        if trunc != 1.0:
            if truncation_mode[select_trunc] == 'avg':
                w_avg = G.backbone.mapping.w_avg
                w = w_avg + trunc * (w - w_avg)
            elif truncation_mode[select_trunc] == 'true_rnd':
                w_rand = torch.randn_like(w)
                w = w_rand + trunc * (w - w_rand)
            elif truncation_mode[select_trunc] == 'rnd_avg_offset':
                w_avg = G.backbone.mapping.w_avg
                noise = torch.randn_like(w_avg) * 0.3
                w_rand = w_avg + noise
                w = w_rand + trunc * (w - w_rand)
            elif truncation_mode[select_trunc] == 'mapping_rnd':
                z = torch.randn([1, G.z_dim], device=w.device)
                w_rand = G.mapping(z, c)
                w = w_rand + trunc * (w - w_rand)

        with torch.no_grad():
            out = G.synthesis(ws=w, c=c, noise_mode='const')
            img = out[image_mode]

        img = (img.clamp(-1, 1) + 1) * 0.5
        img = (img * 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()

        frames.append(Image.fromarray(img))
    frames.reverse()

    # -------------------------------------------------------------------------
    # Save GIF
    # -------------------------------------------------------------------------
    name = os.path.splitext(os.path.basename(latent_path))[0]
    gif_path = os.path.join(outdir, f"{name}_truncation.gif")

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )

    print(f"[SUCCESS] GIF saved → {gif_path}")

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()