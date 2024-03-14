ComfyUI DragAnything

## Install

```
pip install -r requirements.txt

cd custom_nodes/ComfyUI-DragAnything/pretrained_models

git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git
git-lfs clone https://huggingface.co/weijiawu/DragAnything
git-lfs clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
```

注意，如果不执行pip install -r requirements.txt进行安装，一定要执行pip install diffusers==0.19.3安装这个版本的diffusers

## Basic workflow

<img src="wf.png" raw=true>

https://github.com/chaojie/ComfyUI-DragAnything/blob/main/workflow.json

## With ComfyUI-Trajectory

```
cd custom_nodes
git clone https://github.com/chaojie/ComfyUI-Trajectory.git
```

<img src="wf_traj.png" raw=true>

https://github.com/chaojie/ComfyUI-DragAnything/blob/main/workflow_traj.json

## DragAnything

[DragAnything](https://github.com/showlab/DragAnything)
