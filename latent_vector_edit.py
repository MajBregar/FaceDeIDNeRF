import os
import numpy as np
import torch


def _mid_band(num_layers):
    start = max(1, int(0.2 * num_layers))
    end = min(num_layers, int(0.55 * num_layers))
    return start, end


def load_w_std():
    path = './w_std.npy'
    if os.path.exists(path):
        return float(np.load(path))
    return 1.0


def precompute_w_rand(G, c, k=16):
    z = torch.randn([k, G.z_dim], device=c.device)
    c_rep = c.repeat(k, 1)
    with torch.no_grad():
        w_rand = G.mapping(z, c_rep)
    return w_rand.mean(dim=0, keepdim=True)


def edit_latent(G, w, c, trunc, deid_mode, w_rand_cache=None, w_std=None):
    if w_std is None:
        w_std = load_w_std()

    def _w_rand():
        if w_rand_cache is not None:
            return w_rand_cache
        return precompute_w_rand(G, c)

    if deid_mode == 'avg':
        w_avg = G.backbone.mapping.w_avg
        return w_avg + trunc * (w - w_avg)

    if deid_mode == 'true_rnd':
        w_rand = torch.randn_like(w)
        return w_rand + trunc * (w - w_rand)

    if deid_mode == 'rnd_avg_offset':
        w_avg = G.backbone.mapping.w_avg
        noise = torch.randn_like(w_avg) * 0.1
        w_rand = w_avg + noise
        return w_rand + trunc * (w - w_rand)

    if deid_mode == 'mapping_rnd':
        w_rand = _w_rand()
        return w_rand + trunc * (w - w_rand)

    if deid_mode == 'mapping_interp':
        w_rand = _w_rand()
        return trunc * w + (1 - trunc) * w_rand

    if deid_mode == 'w_noise':
        noise = torch.randn_like(w) * (0.5 * trunc)   # arbitrary scale
        return w + noise

    if deid_mode == 'layer_mix':
        w_rand = _w_rand()
        mix_layer = int(trunc * w.shape[1])
        w[:, :mix_layer] = w_rand[:, :mix_layer]
        return w

    if deid_mode == 'coarse_mix':
        w_rand = _w_rand()
        w[:, :6] = w_rand[:, :6]
        return w

    if deid_mode == 'fine_mix':
        w_rand = _w_rand()
        w[:, 8:] = w_rand[:, 8:]
        return w

    if deid_mode == 'pca_perturb':
        direction = torch.randn_like(w)
        direction = direction / direction.norm()
        return w + trunc * direction

    if deid_mode == 'orthogonal_noise':
        noise = torch.randn_like(w)
        proj = (noise * w).sum() / (w * w).sum()
        noise = noise - proj * w
        return w + trunc * noise

    if deid_mode == 'style_shuffle':
        perm = torch.randperm(w.shape[1])
        return w[:, perm]


    if deid_mode == 'mid_mix':
        # Progressively swap middle layers with an averaged anonymous code.
        # Uses w_rand_cache so the mapping network isn't called per frame.
        mid_start, mid_end = _mid_band(w.shape[1])
        w_rand = _w_rand()
        mix_layer = mid_start + int(trunc * (mid_end - mid_start))
        w[:, mid_start:mix_layer] = w_rand[:, mid_start:mix_layer]
        return w

    if deid_mode == 'mid_avg':
        # Pull middle layers toward the dataset W average.
        # The training script's w_avg is computed with a frontal reference
        # camera, so it is not fully pose-conditioned. mid_interp is more
        # pose-aware (uses mapping network with the subject's actual c).
        w_avg = G.backbone.mapping.w_avg
        mid_start, mid_end = _mid_band(w.shape[1])
        w_band = w[:, mid_start:mid_end]
        w[:, mid_start:mid_end] = w_avg + trunc * (w_band - w_avg)
        return w

    if deid_mode == 'mid_interp':
        # Blend middle layers continuously toward a pose-conditioned anonymous
        # code. Smoother truncation curve than mid_mix (all layers blend
        # simultaneously rather than growing the swapped count discretely).
        mid_start, mid_end = _mid_band(w.shape[1])
        w_rand = _w_rand()
        w_band = w[:, mid_start:mid_end]
        w_rand_band = w_rand[:, mid_start:mid_end]
        w[:, mid_start:mid_end] = trunc * w_band + (1 - trunc) * w_rand_band
        return w

    if deid_mode == 'mid_w_std_noise':
        # Add W-manifold-calibrated Gaussian noise restricted to the middle
        # (identity-bearing) band. Scale = w_std * trunc means at trunc=1 the
        # perturbation is one W-space standard deviation, which is the
        # empirical boundary between plausible faces and artifacts based on how
        # the training script uses w_std to gate exploration noise.
        # Fine and coarse layers are untouched, preserving skin tone and pose.
        mid_start, mid_end = _mid_band(w.shape[1])
        noise = torch.randn_like(w[:, mid_start:mid_end]) * (w_std * trunc)
        w[:, mid_start:mid_end] = w[:, mid_start:mid_end] + noise
        return w

    if deid_mode == 'mid_orthogonal':
        # orthagonal noise - essentially no effect in practice
        mid_start, mid_end = _mid_band(w.shape[1])
        w_band = w[:, mid_start:mid_end]
        noise = torch.randn_like(w_band)
        proj = (noise * w_band).sum() / (w_band * w_band).sum().clamp(min=1e-8)
        noise = noise - proj * w_band
        noise = noise / noise.norm().clamp(min=1e-8)
        w[:, mid_start:mid_end] = w_band + (w_std * trunc) * noise
        return w

    raise ValueError(f"Unknown deid_mode: {deid_mode}")