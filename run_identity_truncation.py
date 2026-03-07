import os

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

image_ids = ["memzl"]
input_dict = {
        "lamda_id": 1.0, #0.2, 
        "lamda_origin": 1.0, #0.2, 
        "lamda_illumination": 0.0
}


network_path = "./networks/ffhqrebalanced512-128.pkl"

for image_id in image_ids:

    truncation = 1.0
    save = '/home/output/test'

    output_dir_image = os.path.join(
        output_dir,
        f"{image_id}_"
        f"{input_dict['lamda_id']}_{input_dict['lamda_origin']}_"
        f"{input_dict['lamda_illumination']}"
    )

    render_command = (
        f"PYTHONWARNINGS=\"ignore\" "
        f"python gen_gif_from_latent_code.py "
        f"--outdir '{save}' "
        f"--network '{output_dir_image}/checkpoints/fintuned_generator.pkl' "
        f"--latent '{output_dir_image}/checkpoints/{image_id}.npy' "
        f"--pose '{image_dir}/{image_id}.npy' "
        f"--sample-mult 3"
    )

    #print(render_command)
    os.system(render_command)