
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#python run.py --outdir=projector_out --network=/disk1/haozhang/EG3D-diffusion_zh/eg3d/networks/ffhqrebalanced512-128.pkl --sample_mult=2  --image_path ./projector_test_data/00018.png --c_path ./projector_test_data/00018.npy

#python gen_videos_from_given_latent_code.py --outdir=out --trunc=0.7 --npy_path ./projector_out/00018_w_plus/00018_w_plus.npy   --network=/disk1/haozhang/EG3D-diffusion/eg3d/networks/ffhqrebalanced512-128.pkl --sample_mult=2

import sys
sys.path.append('DPR_model')
sys.path.append('DPR_utils')

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import re
from typing import List, Optional, Tuple, Union
import pickle
import click
import dnnlib
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import utils.legacy as legacy
from torchvision.transforms import transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
from editors import w_plus_editor_WD
from PIL import Image
from utils_SH import *      # type: ignore
from defineHourglass_512_gray_skip import *  # type: ignore
#from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from criteria.sd import StableDiffusion
# ----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.
    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

 
# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--image_path', help='image_path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--c_path', help='camera parameters path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--num_steps', 'num_steps', type=int, help='Multiplier for depth sampling in volume rendering', default=1000, show_default=True)
@click.option('--num_steps_pti', 'num_steps_pti', type=int, help='Multiplier for depth sampling in volume rendering', default=400, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--lamda_id', type=float, help='id loss wright', default=0.6, show_default=True)
@click.option('--lamda_origin', type=float, help='origin loss wright', default=0.6, show_default=True)
@click.option('--fine_tune_images_enabled', 'fine_tune_images_enabled', type=bool, help='Flag for enabling image generation during fine tuning', default=True, show_default=True)
@click.option('--pre_image_log_steps', 'pre_image_log_steps', type=int, help='After how many steps should output image', default=100, show_default=True)
@click.option('--post_image_log_steps', 'post_image_log_steps', type=int, help='After how many steps should output image', default=10, show_default=True)

def run(
        network_pkl: str,
        outdir: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        image_path:str,
        c_path:str,
        num_steps:int,
        num_steps_pti:int,
        lamda_id: float,
        lamda_origin: float,
        fine_tune_images_enabled: bool,
        pre_image_log_steps: int,
        post_image_log_steps: int
):

    os.makedirs(outdir, exist_ok=True)
    #print('Loading networks from "%s"...' % network_pkl)

    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        network_data = legacy.load_network_pkl(f)
        G = network_data['G_ema'].to(device)  # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    if nrr is not None: G.neural_rendering_resolution = nrr

    image = Image.open(image_path).convert('RGB')
    image_name = os.path.basename(image_path)[:-4]
    c = np.load(c_path)
    c = np.reshape(c,(1,25))

    c = torch.FloatTensor(c).cuda()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((512,512))
    ])
    from_im = trans(image).cuda()
    id_image = torch.squeeze((from_im.cuda() + 1) / 2) * 255

    # Identity and guidance models
    id_loss = IDLoss()

    # Output directory
    safe_name = (
        f"{image_name}_"
        f"{lamda_id}_{lamda_origin}"
    ).replace(' ', '_')

    outdir = os.path.join(outdir, safe_name)
    os.makedirs(outdir, exist_ok=True)

    w_plus = w_plus_editor_WD.project(
        G,
        c,
        outdir,
        id_image,
        device=torch.device('cuda'),
        w_avg_samples=600,
        w_name=image_name,
        num_steps=num_steps,
        id_loss=id_loss,
        lamda_id=lamda_id,
        lamda_origin=lamda_origin,
        image_output_enabled=fine_tune_images_enabled,
        image_log_step=pre_image_log_steps
    )

    G_final = w_plus_editor_WD.project_pti(
        G,
        c,
        outdir,
        id_image,
        w_plus,
        device=torch.device('cuda'),
        w_avg_samples=600,
        w_name=image_name,
        num_steps_pti=num_steps_pti,
        id_loss=id_loss,
        lamda_id=lamda_id,
        lamda_origin=lamda_origin,
        image_output_enabled=fine_tune_images_enabled,
        image_log_step=post_image_log_steps
    )

    outdir_ckeckpoints = os.path.join(outdir,"checkpoints")
    os.makedirs(outdir_ckeckpoints, exist_ok=True)
    np.save(f'{outdir_ckeckpoints}/{image_name}.npy', w_plus.cpu().detach())
    
    with open(f'{outdir_ckeckpoints}/fintuned_generator.pkl', 'wb') as f:
        network_data["G_ema"] = G_final.eval().requires_grad_(False).cpu()
        pickle.dump(network_data, f)
    

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------

