import os
from itertools import product
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import torch

# Detect available GPUs
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA-compatible GPU detected.")
num_gpus = torch.cuda.device_count()
gpu_info = [(i, torch.cuda.get_device_name(i)) for i in range(num_gpus)]

print("Detected GPUs:")
for i, name in gpu_info:
    print(f"  GPU {i}: {name}")

gpu_ids_str = ",".join(str(i) for i, _ in gpu_info)

# Parameters

video_mult = 3
output_dir = "./output/"
image_dir = '/home/real_images/final_output'
save = '/home/deid_gifs'

image_ids = ["memzl2"]
input_dict = {
    "lamda_id": 1.0,
    "lamda_origin": 1.0,
}



modes = [
    # 'avg',               # full-vector blend toward dataset average; drags age/sex/race toward the dataset mean too
    # 'true_rnd',          # full-vector blend toward a random latent; randomizes demographics along with identity
    # 'rnd_avg_offset',    # same problem as avg, just offset by small noise around w_avg
    # 'mapping_rnd',       # full-vector blend toward a random mapped identity; no layer selectivity
    # 'mapping_interp',    # full-vector interpolation; same lack of selectivity as mapping_rnd
    # 'w_noise',           # isotropic noise across every layer; perturbs demographic-correlated layers equally
    # 'layer_mix',         # always starts swapping from layer 0; hits coarse (age/sex) early, fine (race) at high trunc
    # 'coarse_mix',        # only swaps coarse layers; mid layers stay intact, so identity barely changes
    # 'fine_mix',          # only swaps fine layers; mostly just shifts skin tone/color, identity stays recognizable
    # 'pca_perturb',       # full-vector perturbation along a random direction, not layer-restricted
    # 'orthogonal_noise',  # full-vector perturbation, same lack of layer selectivity
    # 'style_shuffle',     # permutes the layer axis itself; breaks facial coherence rather than de-identifying
    'mid_mix',
    'mid_avg',
    'mid_interp',
]
#mode = modes[2]

network_path = "./networks/ffhqrebalanced512-128.pkl"

for image_id, mode in product(image_ids, modes):


    output_dir_image = os.path.join(
        output_dir,
        f"{image_id}_"
        f"{input_dict['lamda_id']}_{input_dict['lamda_origin']}"
    )

    render_command = (
        f"PYTHONWARNINGS=\"ignore\" "
        f"python gen_gif_from_latent_code.py "
        f"--outdir '{save}' "
        f"--network '{output_dir_image}/checkpoints/fintuned_generator.pkl' "
        f"--latent '{output_dir_image}/checkpoints/{image_id}.npy' "
        f"--pose '{image_dir}/{image_id}.npy' "
        f"--sample-mult 3 "
        f"--deid_mode {mode}"
    )

    #print(render_command)
    os.system(render_command)