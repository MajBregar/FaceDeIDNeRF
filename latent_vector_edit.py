import torch


def _mid_band(num_layers):
    start = max(1, int(0.2 * num_layers))
    end = min(num_layers, int(0.55 * num_layers))
    return start, end


def _avg_random_w(G, w, c, k=8):
    z = torch.randn([k, G.z_dim], device=w.device)
    c_rep = c.repeat(k, 1)
    w_rand = G.mapping(z, c_rep)
    return w_rand.mean(dim=0, keepdim=True)


def edit_latent(G, w, c, trunc, deid_mode):
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
        z = torch.randn([1, G.z_dim], device=w.device)
        w_rand = G.mapping(z, c)
        return w_rand + trunc * (w - w_rand)

    if deid_mode == 'mapping_interp':
        z = torch.randn([1, G.z_dim], device=w.device)
        w_rand = G.mapping(z, c)
        alpha = trunc
        return alpha * w + (1 - alpha) * w_rand

    if deid_mode == 'w_noise':
        noise = torch.randn_like(w) * (0.5 * trunc)
        return w + noise

    if deid_mode == 'layer_mix':
        z = torch.randn([1, G.z_dim], device=w.device)
        w_rand = G.mapping(z, c)
        mix_layer = int(trunc * w.shape[1])
        w[:, :mix_layer] = w_rand[:, :mix_layer]
        return w

    if deid_mode == 'coarse_mix':
        z = torch.randn([1, G.z_dim], device=w.device)
        w_rand = G.mapping(z, c)
        w[:, :6] = w_rand[:, :6]
        return w

    if deid_mode == 'fine_mix':
        z = torch.randn([1, G.z_dim], device=w.device)
        w_rand = G.mapping(z, c)
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

    if deid_mode == 'mid_mix':
        mid_start, mid_end = _mid_band(w.shape[1])
        w_rand = _avg_random_w(G, w, c)
        mix_layer = mid_start + int(trunc * (mid_end - mid_start))
        w[:, mid_start:mix_layer] = w_rand[:, mid_start:mix_layer]
        return w

    if deid_mode == 'mid_avg':
        w_avg = G.backbone.mapping.w_avg
        mid_start, mid_end = _mid_band(w.shape[1])
        w_band = w[:, mid_start:mid_end]
        w[:, mid_start:mid_end] = w_avg + trunc * (w_band - w_avg)
        return w

    if deid_mode == 'mid_interp':
        mid_start, mid_end = _mid_band(w.shape[1])
        w_rand = _avg_random_w(G, w, c)
        w_band = w[:, mid_start:mid_end]
        w_rand_band = w_rand[:, mid_start:mid_end]
        w[:, mid_start:mid_end] = trunc * w_band + (1 - trunc) * w_rand_band
        return w

    if deid_mode == 'style_shuffle':
        perm = torch.randperm(w.shape[1])
        return w[:, perm]

    raise ValueError(f"Unknown deid_mode: {deid_mode}")