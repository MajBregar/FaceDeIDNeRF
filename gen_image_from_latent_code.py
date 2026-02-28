"""
Render a single image from an edited EG3D / FaceDNeRF latent
using the ORIGINAL image camera pose.
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

@click.command()
@click.option('--network', 'network_pkl', required=True, help='EG3D network pickle or checkpoint')
@click.option('--latent', 'latent_path', required=True, help='Latent .npy file (W or W+)')
@click.option('--pose', 'pose_path', required=True, help='Camera pose .npy file (shape (25,))')
@click.option('--outdir', required=True, help='Output directory')
@click.option('--image-mode', type=click.Choice(['image', 'image_raw', 'image_depth']),
              default='image', show_default=True)
@click.option('--trunc', 'truncation_psi', default=1.0, show_default=True)
@click.option('--sample-mult', default=2.0, show_default=True)
@click.option('--nrr', default=None, type=int, help='Neural rendering resolution override')
def main(
    network_pkl,
    latent_path,
    pose_path,
    outdir,
    image_mode,
    truncation_psi,
    sample_mult,
    nrr
):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda')

    print(f"[INFO] Loading network: {network_pkl}")

    # -------------------------------------------------------------------------
    # Load generator
    # -------------------------------------------------------------------------
    if network_pkl.endswith(".pkl"):
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device).float()
    else:
        raise RuntimeError("Non-pkl checkpoints not supported in this minimal script")

    G.eval()

    # Increase NeRF sampling quality
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

    # Ensure batch dimension
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)

    # Expand W to W+ if needed
    if hasattr(G.backbone.mapping, "num_ws"):
        if latent.ndim == 3 and latent.shape[1] != G.backbone.mapping.num_ws:
            latent = latent.repeat(1, G.backbone.mapping.num_ws, 1)

    # Apply truncation (optional)
    if truncation_psi != 1.0:
        w_avg = G.backbone.mapping.w_avg
        latent = w_avg + truncation_psi * (latent - w_avg)


    print(f"[INFO] Latent shape: {latent.shape}")

    c = np.load(pose_path)
    assert c.shape == (25,), f"Expected pose shape (25,), got {c.shape}"

    c = torch.tensor(c, device=device, dtype=torch.float32).unsqueeze(0)

    print(f"[INFO] Camera pose loaded: {c.shape}")

    # -------------------------------------------------------------------------
    # Render single image
    # -------------------------------------------------------------------------
    with torch.no_grad():
        out = G.synthesis(
            ws=latent,
            c=c,
            noise_mode='const'
        )
        img = out[image_mode]

    # Convert from [-1, 1] → uint8
    img = (img.clamp(-1, 1) + 1) * 0.5
    img = (img * 255).to(torch.uint8)
    img = img[0].permute(1, 2, 0).cpu().numpy()

    # -------------------------------------------------------------------------
    # Save image
    # -------------------------------------------------------------------------
    name = os.path.splitext(os.path.basename(latent_path))[0]
    out_path = os.path.join(outdir, f"{name}.png")

    Image.fromarray(img).save(out_path)
    print(f"[SUCCESS] Saved image → {out_path}")

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
