## WIP Deidentification Tool using NeRFs

## Installation

```bash
git submodule update --init --recursive

source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f env_A100P10.yml
conda activate deidnerf_py10

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9"

pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

bash download_deid_models.sh

sed -i "s/torch.pow(((image + 0.055) \/ 1.055), 2.4)/torch.pow(((image.abs() + 0.055) \/ 1.055), 2.4)/" \
$(python -c "import kornia, os; print(os.path.join(os.path.dirname(kornia.__file__), 'color/rgb.py'))")

# dataset preprocessing
cd ./dataset_preprocessing/Deep3DFaceRecon_pytorch
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/
cd ..

bash download_image_preprocessing_models.sh

# download the final model file and place it into the code directory
# 01_MorphableModel.mat
```


## Paper & Citation
Link to [**Paper**](https://arxiv.org/abs/2306.00783) 

If you find this work useful for your research, please cite our paper:

```bibtex
@article{zhang2024facednerf,
  title={FaceDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and Relighting with Diffusion Models},
  author={Zhang, Hao and DAI, Tianyuan and Xu, Yanbo and Tai, Yu-Wing and Tang, Chi-Keung},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
