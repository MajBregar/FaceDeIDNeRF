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
pre_iterations = 50
post_iterations = 50
dif_mult = 3 #higher numbers require more vram
video_mult = 3
output_dir = "./output/"
image_dir = '/home/real_images/final_output'

image_ids = ["juli"]
input_dict = {
        "lamda_id": 1.0, #0.2, 
        "lamda_origin": 1.0 #0.2
}


network_path = "./networks/ffhqrebalanced512-128.pkl"

for image_id in image_ids:
    print("Start:", time.ctime(), "Output Directory:", output_dir)

    command = (
        f"PYTHONWARNINGS=\"ignore\" "
        f"CUDA_VISIBLE_DEVICES={gpu_ids_str} python fine_tune_latent_space.py "
        f"--outdir='{output_dir}' "
        f"--network='{network_path}' "
        f"--sample_mult={dif_mult} "
        f"--image_path {image_dir}/{image_id}.png "
        f"--c_path {image_dir}/{image_id}.npy "
        f"--num_steps {pre_iterations} "
        f"--num_steps_pti {post_iterations} "
        f"--lamda_id {input_dict['lamda_id']} "
        f"--lamda_origin {input_dict['lamda_origin']} "
        f"--fine_tune_images_enabled {True} "
        f"--pre_image_log_steps {20} "
        f"--post_image_log_steps {10} "
    )
    #print(command)
    os.system(command)

    print("Saved fine tuned generator -", time.ctime())
    print("")
    print("")

    output_dir_image = os.path.join(
        output_dir,
        f"{image_id}_"
        f"{input_dict['lamda_id']}_{input_dict['lamda_origin']}"
    )

    render_command = (
        f"PYTHONWARNINGS=\"ignore\" "
        f"python gen_image_from_latent_code.py "
        f"--outdir '{output_dir_image}' "
        f"--network '{output_dir_image}/checkpoints/fintuned_generator.pkl' "
        f"--latent '{output_dir_image}/checkpoints/{image_id}.npy' "
        f"--pose '{image_dir}/{image_id}.npy' "
        f"--trunc 0.75 "
        f"--sample-mult 3"
    )

    # print(render_command)
    # os.system(render_command)
    # print("RENDERED IMAGE WITH ORIGINAL POSE:", time.ctime())    