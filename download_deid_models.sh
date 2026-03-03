
wget -P ./networks https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt

wget \
'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/eg3d/1/files?redirect=true&path=ffhqrebalanced512-128.pkl' \
-O ./networks/ffhqrebalanced512-128.pkl

curl -L \
'https://drive.usercontent.google.com/download?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn&export=download&confirm=t' \
-o ./networks/model_ir_se50.pth