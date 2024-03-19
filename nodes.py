import os
import sys
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
#sys.path.append(f'{comfy_path}/custom_nodes/ComfyUI-DragAnything')

import torch
import datetime
import numpy as np
from PIL import Image
from .pipeline.pipeline_svd_DragAnything import StableVideoDiffusionPipeline
from .models.DragAnything import DragAnythingSDVModel
from .models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import cv2
import re 
from scipy.ndimage import distance_transform_edt
import torchvision.transforms as T
import torch.nn.functional as F
from .utils.dift_util import DIFT_Demo, SDFeaturizer
from torchvision.transforms import PILToTensor
import json
import random

def save_gifs_side_by_side(batch_output, validation_control_images,output_folder,name = 'none', target_size=(512 , 512),duration=200):

    flattened_batch_output = batch_output
    def create_gif(image_list, gif_path, duration=100):
        pil_images = [validate_and_convert_image(img,target_size=target_size) for img in image_list]
        pil_images = [img for img in pil_images if img is not None]
        if pil_images:
            pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], loop=0, duration=duration)

    # Creating GIFs for each image list
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gif_paths = []
    
#     validation_control_images = validation_control_images*255 validation_images, 
    for idx, image_list in enumerate([validation_control_images, flattened_batch_output]):
        
#         if idx==0:
#             continue

        gif_path = os.path.join(output_folder, f"temp_{idx}_{timestamp}.gif")
        create_gif(image_list, gif_path)
        gif_paths.append(gif_path)

    # Function to combine GIFs side by side
    def combine_gifs_side_by_side(gif_paths, output_path):
        print(gif_paths)
        gifs = [Image.open(gif) for gif in gif_paths]

        # Assuming all gifs have the same frame count and duration
        frames = []
        for frame_idx in range(gifs[0].n_frames):
            combined_frame = None
            
                
            for gif in gifs:
                
                gif.seek(frame_idx)
                if combined_frame is None:
                    combined_frame = gif.copy()
                else:
                    combined_frame = get_concat_h(combined_frame, gif.copy())
            frames.append(combined_frame)
        print(gifs[0].info['duration'])
        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=duration)

    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    # Combine the GIFs into a single file
    combined_gif_path = os.path.join(output_folder, f"combined_frames_{name}_{timestamp}.gif")
    combine_gifs_side_by_side(gif_paths, combined_gif_path)

    # Clean up temporary GIFs
    for gif_path in gif_paths:
        os.remove(gif_path)

    return combined_gif_path

# Define functions
def validate_and_convert_image(image, target_size=(512 , 512)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

def create_image_grid(images, rows, cols, target_size=(512 , 512)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

def tensor_to_pil(tensor):
    """ Convert a PyTorch tensor to a PIL Image. """
    # Convert tensor to numpy array
    if len(tensor.shape) == 4:  # batch of images
        images = [Image.fromarray(img.numpy().transpose(1, 2, 0)) for img in tensor]
    else:  # single image
        images = Image.fromarray(tensor.numpy().transpose(1, 2, 0))
    return images

def save_combined_frames(batch_output, validation_images, validation_control_images, output_folder):
    # Flatten batch_output to a list of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Convert tensors in lists to PIL Images
    validation_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_images]
    validation_control_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_control_images]
    flattened_batch_output = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in batch_output]

    # Flatten lists if they contain sublists (for tensors converted to multiple images)
    validation_images = [img for sublist in validation_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    validation_control_images = [img for sublist in validation_control_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    flattened_batch_output = [img for sublist in flattened_batch_output for img in (sublist if isinstance(sublist, list) else [sublist])]

    # Combine frames into a list
    combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3
    rows = (num_images + cols - 1) // cols

    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols, target_size=(512, 512))
    if grid is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"combined_frames_{timestamp}.png"
        output_path = os.path.join(output_folder, filename)
        grid.save(output_path)
    else:
        print("Failed to create image grid")

def load_images_from_folder(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        matches = re.findall(r'\d+', filename)  # Find all sequences of digits in the filename
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'


    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images

def gen_gaussian_heatmap(imgSize=200):
    circle_img = np.zeros((imgSize, imgSize), np.float32)
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)
#         print(circle_mask)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    # 生成高斯图
    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

    # 如果要可视化对比正方形和最大内切圆高斯图的区别，注释下面这行即可
    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)*255).astype(np.uint8)
    # 将图像调整大小为 50x50
#         isotropicGrayscaleImage = cv2.resize(isotropicGrayscaleImage, (40, 40))
    return isotropicGrayscaleImage

def infer_model(model, image):
    transform = T.Compose([
        T.Resize((196, 196)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0).cuda()
#     cls_token = model.forward_features(image)
    cls_token = model(image, is_training=False)
    return cls_token

def find_largest_inner_rectangle_coordinates(mask_gray):

    refine_dist = cv2.distanceTransform(mask_gray.astype(np.uint8), cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(refine_dist)
    radius = int(maxVal)

    return maxLoc, radius

def get_ID(images_list,masks_list,dinov2):
        
    ID_images = []


    image = images_list
    mask = masks_list

#     try:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour) 

    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    image = image * mask
    
    image = image[y:y+h,x:x+w]
    
#     import random
#     cv2.imwrite("./{}.jpg".format(random.randint(1, 100)),image)
    
#     except:
#         pass
#         print("cv2.findContours error")

    image = Image.fromarray(image).convert('RGB')

    img_embedding = infer_model(dinov2, image)


    return img_embedding

def get_dift_ID(feature_map,mask):
        
#     feature_map = feature_map * 0
    
    new_feature = []
    non_zero_coordinates = np.column_stack(np.where(mask != 0))
    for coord in non_zero_coordinates:
#         feature_map[:, coord[0], coord[1]] = 1
        new_feature.append(feature_map[:, coord[0], coord[1]])
    
    stacked_tensor = torch.stack(new_feature, dim=0)
    # 在维度0上进行平均池化
    average_pooled_tensor = torch.mean(stacked_tensor, dim=0)

    return average_pooled_tensor


def extract_dift_feature(image, dift_model):
    if isinstance(image, Image.Image):
        image = image
    else:
        image = Image.open(image).convert('RGB')
           
    prompt = ''
    img_tensor = (PILToTensor()(image) / 255.0 - 0.5) * 2
    #print(f'{img_tensor}')
    dift_feature = dift_model.forward(img_tensor, prompt=prompt, up_ft_index=3,ensemble_size=8)
    return dift_feature

# cloud 
def get_condition(target_size=(512 , 512), original_size=(512 , 512), frame_number=20, first_frame=None, is_mask = False, side=20,model_id=None,mask_list=None,trajectory_list="[[]]"):
    images = []
    vis_images = []
    heatmap = gen_gaussian_heatmap()
    
    original_size = (original_size[1],original_size[0])
    size = (target_size[1],target_size[0])
    latent_size = (int(target_size[1]/8), int(target_size[0]/8))
    
    
    dift_model = SDFeaturizer(sd_id=model_id)
    #print(f'{first_frame}')
    keyframe_dift = extract_dift_feature(first_frame, dift_model=dift_model)
    
    ID_images=[]
    ids_list={}
    
    #with open(os.path.join(args["validation_image"],"demo.json"), 'r') as json_file:
    #    trajectory_json = json.load(json_file)
    trajectories=json.loads(trajectory_list)
    #mask_list = []
    trajectory_list = []
    radius_list = []
    
    ind=0
    for trajectory in trajectories:
        trajectory = [[int(i[0]/original_size[0]*size[0]),int(i[1]/original_size[1]*size[1])] for i in trajectory]
        trajectory_list.append(trajectory)
        
        #mask
        first_mask = mask_list[ind]
        
        mask_322 = cv2.resize(np.array(first_mask).astype(np.uint8),(int(target_size[1]), int(target_size[0])))
        _, radius = find_largest_inner_rectangle_coordinates(mask_322)
        radius_list.append(radius)   
        ind=ind+1 
    
    viss = 0
    if viss:
        mask_list_vis = [cv2.resize(i,(int(target_size[1]), int(target_size[0]))) for i in mask_list]
        
        vis_first_mask = show_mask(cv2.resize(np.array(first_frame).astype(np.uint8),(int(target_size[1]), int(target_size[0]))), mask_list_vis)
        vis_first_mask = cv2.cvtColor(vis_first_mask, cv2.COLOR_BGR2RGB)
        cv2.imwrite("test.jpg",vis_first_mask)
        assert False
        
        
    for idxx,point in enumerate(trajectory_list[0]):
        new_img = np.zeros(target_size, np.uint8)
        vis_img = new_img.copy()
        ids_embedding = torch.zeros((target_size[0], target_size[1], 320))
        
        if idxx>= frame_number:
            break
            
        for cc,(mask,trajectory,radius) in enumerate(zip(mask_list,trajectory_list,radius_list)):
            
            #print(f'cc{cc}ids_list{ids_list}')
            center_coordinate = trajectory[idxx]
            trajectory_ = trajectory[:idxx]
            side = min(radius,50)
#             side = radius
            
#             if cc>=1:
#                 continue
                
            # ID embedding
            if idxx == 0:
                # diffusion feature
                mask_32 = cv2.resize(mask.astype(np.uint8),latent_size)
                #print(f'mask_32{mask_32}')
                if len(np.column_stack(np.where(mask_32 != 0)))==0:
                    continue
                ids_list[cc] = get_dift_ID(keyframe_dift[0],mask_32)

                id_feature = ids_list[cc]
            else:
                id_feature = ids_list[cc]

            circle_img = np.zeros((target_size[0], target_size[1]), np.float32)
            circle_mask = cv2.circle(circle_img, (center_coordinate[0],center_coordinate[1]), side, 1, -1)
                      
    
            y1 = max(center_coordinate[1]-side,0)
            y2 = min(center_coordinate[1]+side,target_size[0]-1)
            x1 = max(center_coordinate[0]-side,0)
            x2 = min(center_coordinate[0]+side,target_size[1]-1)
            
            if x2-x1>3 and y2-y1>3:
                need_map = cv2.resize(heatmap, (x2-x1, y2-y1))
                new_img[y1:y2,x1:x2] = need_map.copy()
                
                if cc>=0:
                    vis_img[y1:y2,x1:x2] = need_map.copy()
                    if len(trajectory_) == 1:
                        vis_img[trajectory_[0][1],trajectory_[0][0]] = 255
                    else:
                        for itt in range(len(trajectory_)-1):
                            cv2.line(vis_img,(trajectory_[itt][0],trajectory_[itt][1]),(trajectory_[itt+1][0],trajectory_[itt+1][1]),(255,255,255),3)
                    


            # 获取非零像素的坐标
            non_zero_coordinates = np.column_stack(np.where(circle_mask != 0))
            for coord in non_zero_coordinates:
                ids_embedding[coord[0], coord[1]] = id_feature[0]
        
        ids_embedding = F.avg_pool1d(ids_embedding, kernel_size=2, stride=2)
        img = new_img

        # Ensure all images are in RGB format
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
        # Convert the numpy array to a PIL image
        pil_img = Image.fromarray(img)
        images.append(pil_img)
        vis_images.append(Image.fromarray(vis_img))
        ID_images.append(ids_embedding)
    return images,ID_images,vis_images



# Usage example
def convert_list_bgra_to_rgba(image_list):
    """
    Convert a list of PIL Image objects from BGRA to RGBA format.

    Parameters:
    image_list (list of PIL.Image.Image): A list of images in BGRA format.

    Returns:
    list of PIL.Image.Image: The list of images converted to RGBA format.
    """
    rgba_images = []
    for image in image_list:
        if image.mode == 'RGBA' or image.mode == 'BGRA':
            # Split the image into its components
            b, g, r, a = image.split()
            # Re-merge in RGBA order
            converted_image = Image.merge("RGBA", (r, g, b, a))
        else:
            # For non-alpha images, assume they are BGR and convert to RGB
            b, g, r = image.split()
            converted_image = Image.merge("RGB", (r, g, b))

        rgba_images.append(converted_image)

    return rgba_images

def show_mask(image, masks, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3)], axis=0)

        h, w = mask.shape[:2]

        color_a = np.concatenate([np.random.random(3)*255], axis=0)
        mask_image = mask.reshape(h, w, 1) * color_a.reshape(1, 1, -1)
        
    else:
        h, w = masks[0].shape[:2]
#         mask_image = mask1.reshape(h, w, 1) * np.array([30, 144, 255])
        mask_image = 0
        for idx,mask in enumerate(masks):
            if idx!=1 and idx!=0:
                continue
            color = np.concatenate([np.random.random(3)*255], axis=0)
            mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1) + mask_image

    return (np.array(image).copy()*0.4+mask_image*0.6).astype(np.uint8)

pretrained_weights_path=f'{comfy_path}/custom_nodes/ComfyUI-DragAnything/pretrained_models'
output_dir=f'{comfy_path}/custom_nodes/ComfyUI-DragAnything/saved_video'
pretrained_weights=os.listdir(pretrained_weights_path)

class DragAnythingLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svd_path": (pretrained_weights, {"default": "stable-video-diffusion-img2vid"}),
                "draganything_path": (pretrained_weights, {"default": "DragAnything"}),
            },
        }

    RETURN_TYPES = ("DragAnythingPipeline",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "run"
    CATEGORY = "DragAnything"

    def run(self,svd_path,draganything_path):
        svd_path=f'{pretrained_weights_path}/{svd_path}'
        draganything_path=f'{pretrained_weights_path}/{draganything_path}'
        controlnet = DragAnythingSDVModel.from_pretrained(draganything_path)
        unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(svd_path,subfolder="unet")
        pipeline = StableVideoDiffusionPipeline.from_pretrained(svd_path,controlnet=controlnet,unet=unet)
        pipeline.enable_model_cpu_offload()

        return (pipeline,)

class DragAnythingPipelineRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DragAnythingPipeline",),
                "sd_path": (pretrained_weights, {"default": "chilloutmix"}),
                "image": ("IMAGE",),
                "width": ("INT",{"default":576}),
                "height": ("INT",{"default":320}),
                "frame_number": ("INT",{"default":20}),
                "mask_list": ("IMAGE",),
                "trajectory_list": ("STRING", {"default": "[[]]"}),
                "num_inference_steps": ("INT",{"default":25}),
                "motion_bucket_id": ("INT",{"default":180}),
                "controlnet_cond_scale": ("FLOAT",{"default":1.0}),
                "decode_chunk_size": ("INT",{"default":8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "DragAnything"

    def run(self,pipeline,sd_path,image,width,height,frame_number,mask_list,trajectory_list,num_inference_steps,motion_bucket_id,controlnet_cond_scale,decode_chunk_size):
        pipeline.enable_model_cpu_offload()

        sd_path=f'{pretrained_weights_path}/{sd_path}'
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert('RGB')
        #image = np.array(image)
        # Convert RGB to BGR
        #image = image[:, :, ::-1].copy()
        masks=[]
        for mask in mask_list:
            mask_img=255.0 * mask.cpu().numpy()
            mask_img = Image.fromarray(np.clip(mask_img, 0, 255).astype(np.uint8)).convert('RGB')
            mask_img = np.array(mask_img)
            # Convert RGB to BGR
            #mask_img = mask_img[:, :, ::-1].copy()
            mask_img = cv2.cvtColor(np.array(mask_img).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            masks.append(mask_img)
            
        validation_image = image
        original_width, original_height = validation_image.size
        validation_image = validation_image.resize((width, height))
        validation_control_images,ids_embedding,vis_images = get_condition(target_size=(height , width),
                                                                        original_size=(original_height , original_width),
                                                                        frame_number = frame_number,first_frame = validation_image,
                                                                        side=100,model_id=sd_path,mask_list=masks,trajectory_list=trajectory_list)
        ids_embedding = torch.stack(ids_embedding, dim=0).permute(0, 3, 1, 2)

        val_save_dir = output_dir
        os.makedirs(val_save_dir, exist_ok=True)
        
        # Inference and saving loop
        video_frames = pipeline(validation_image, validation_control_images[:frame_number], decode_chunk_size=decode_chunk_size,num_frames=frame_number,num_inference_steps=num_inference_steps,motion_bucket_id=motion_bucket_id,controlnet_cond_scale=controlnet_cond_scale,height=height,width=width,ids_embedding=ids_embedding[:frame_number]).frames

        vis_images = [cv2.applyColorMap(np.array(img).astype(np.uint8), cv2.COLORMAP_JET) for img in vis_images]
        vis_images = [cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2RGB) for img in vis_images]
        
        vis_images = [Image.fromarray(img) for img in vis_images]
        
        video_frames = [img for sublist in video_frames for img in sublist]

        #save_gifs_side_by_side(video_frames, vis_images[:args["frame_number"]],val_save_dir,target_size=(width,height),duration=110)
        data = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in video_frames]
        return torch.cat(tuple(data), dim=0).unsqueeze(0)

class DragAnythingRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svd_path": (pretrained_weights, {"default": "stable-video-diffusion-img2vid"}),
                "draganything_path": (pretrained_weights, {"default": "DragAnything"}),
                "sd_path": (pretrained_weights, {"default": "chilloutmix"}),
                "image": ("IMAGE",),
                "width": ("INT",{"default":576}),
                "height": ("INT",{"default":320}),
                "frame_number": ("INT",{"default":20}),
                "mask_list": ("IMAGE",),
                "trajectory_list": ("STRING", {"default": "[[]]"}),
                "num_inference_steps": ("INT",{"default":25}),
                "motion_bucket_id": ("INT",{"default":180}),
                "controlnet_cond_scale": ("FLOAT",{"default":1.0}),
                "decode_chunk_size": ("INT",{"default":8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "DragAnything"

    def run(self,svd_path,draganything_path,sd_path,image,width,height,frame_number,mask_list,trajectory_list,num_inference_steps,motion_bucket_id,controlnet_cond_scale,decode_chunk_size):
        svd_path=f'{pretrained_weights_path}/{svd_path}'
        draganything_path=f'{pretrained_weights_path}/{draganything_path}'
        controlnet = DragAnythingSDVModel.from_pretrained(draganything_path)
        unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(svd_path,subfolder="unet")
        pipeline = StableVideoDiffusionPipeline.from_pretrained(svd_path,controlnet=controlnet,unet=unet)
        pipeline.enable_model_cpu_offload()

        sd_path=f'{pretrained_weights_path}/{sd_path}'
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert('RGB')
        #image = np.array(image)
        # Convert RGB to BGR
        #image = image[:, :, ::-1].copy()
        masks=[]
        for mask in mask_list:
            mask_img=255.0 * mask.cpu().numpy()
            mask_img = Image.fromarray(np.clip(mask_img, 0, 255).astype(np.uint8)).convert('RGB')
            mask_img = np.array(mask_img)
            # Convert RGB to BGR
            #mask_img = mask_img[:, :, ::-1].copy()
            mask_img = cv2.cvtColor(np.array(mask_img).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            masks.append(mask_img)
            
        validation_image = image
        original_width, original_height = validation_image.size
        validation_image = validation_image.resize((width, height))
        validation_control_images,ids_embedding,vis_images = get_condition(target_size=(height , width),
                                                                        original_size=(original_height , original_width),
                                                                        frame_number = frame_number,first_frame = validation_image,
                                                                        side=100,model_id=sd_path,mask_list=masks,trajectory_list=trajectory_list)
        ids_embedding = torch.stack(ids_embedding, dim=0).permute(0, 3, 1, 2)

        val_save_dir = output_dir
        os.makedirs(val_save_dir, exist_ok=True)
        
        # Inference and saving loop
        video_frames = pipeline(validation_image, validation_control_images[:frame_number], decode_chunk_size=decode_chunk_size,num_frames=frame_number,num_inference_steps=num_inference_steps,motion_bucket_id=motion_bucket_id,controlnet_cond_scale=controlnet_cond_scale,height=height,width=width,ids_embedding=ids_embedding[:frame_number]).frames

        vis_images = [cv2.applyColorMap(np.array(img).astype(np.uint8), cv2.COLORMAP_JET) for img in vis_images]
        vis_images = [cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2RGB) for img in vis_images]
        
        vis_images = [Image.fromarray(img) for img in vis_images]
        
        video_frames = [img for sublist in video_frames for img in sublist]

        #save_gifs_side_by_side(video_frames, vis_images[:args["frame_number"]],val_save_dir,target_size=(width,height),duration=110)
        data = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in video_frames]
        return torch.cat(tuple(data), dim=0).unsqueeze(0)

class DragAnythingRunRandom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svd_path": (pretrained_weights, {"default": "stable-video-diffusion-img2vid"}),
                "draganything_path": (pretrained_weights, {"default": "DragAnything"}),
                "sd_path": (pretrained_weights, {"default": "chilloutmix"}),
                "image": ("IMAGE",),
                "width": ("INT",{"default":576}),
                "height": ("INT",{"default":320}),
                "frame_number": ("INT",{"default":20}),
                "mask_list": ("IMAGE",),
                "num_inference_steps": ("INT",{"default":25}),
                "motion_bucket_id": ("INT",{"default":180}),
                "controlnet_cond_scale": ("FLOAT",{"default":1.0}),
                "decode_chunk_size": ("INT",{"default":8}),
                "move_speed": ("INT",{"default":5}),
                "move_border": ("INT",{"default":20}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "DragAnything"

    def run(self,svd_path,draganything_path,sd_path,image,width,height,frame_number,mask_list,num_inference_steps,motion_bucket_id,controlnet_cond_scale,decode_chunk_size,move_speed,move_border):
        trajectory_list="[]"
        trajectories=json.loads(trajectory_list)
        
        svd_path=f'{pretrained_weights_path}/{svd_path}'
        draganything_path=f'{pretrained_weights_path}/{draganything_path}'
        controlnet = controlnet = DragAnythingSDVModel.from_pretrained(draganything_path)
        unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(svd_path,subfolder="unet")
        pipeline = StableVideoDiffusionPipeline.from_pretrained(svd_path,controlnet=controlnet,unet=unet)
        pipeline.enable_model_cpu_offload()

        sd_path=f'{pretrained_weights_path}/{sd_path}'
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert('RGB')
        #image = np.array(image)
        # Convert RGB to BGR
        #image = image[:, :, ::-1].copy()
        masks=[]
        for mask in mask_list:
            mask_img=255.0 * mask.cpu().numpy()
            mask_img = Image.fromarray(np.clip(mask_img, 0, 255).astype(np.uint8)).convert('RGB')
            mask_img = np.array(mask_img)
            # Convert RGB to BGR
            #mask_img = mask_img[:, :, ::-1].copy()
            mask_img = cv2.cvtColor(np.array(mask_img).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            masks.append(mask_img)

            mask_trajectory=[]
            points=np.where(mask_img==1)
            x=np.mean(points[1])
            y=np.mean(points[0])
            mask_trajectory.append([x,y])
            
            for frame_ind in range(frame_number-1):
                x=x+random.randint(-move_speed,move_speed)
                y=y+random.randint(-move_speed,move_speed)
                if x<move_border:
                    x=move_border
                if y<move_border:
                    y=move_border
                if x>=image.size[0]-move_border:
                    x=image.size[0]-move_border
                if y>=image.size[1]-move_border:
                    y=image.size[1]-move_border
                mask_trajectory.append([x,y])
            trajectories.append(mask_trajectory)
        trajectory_list=json.dumps(trajectories)
        print(f'trajectory_list{trajectory_list}')

        validation_image = image
        original_width, original_height = validation_image.size
        validation_image = validation_image.resize((width, height))
        validation_control_images,ids_embedding,vis_images = get_condition(target_size=(height , width),
                                                                        original_size=(original_height , original_width),
                                                                        frame_number = frame_number,first_frame = validation_image,
                                                                        side=100,model_id=sd_path,mask_list=masks,trajectory_list=trajectory_list)
        ids_embedding = torch.stack(ids_embedding, dim=0).permute(0, 3, 1, 2)

        val_save_dir = output_dir
        os.makedirs(val_save_dir, exist_ok=True)
        
        # Inference and saving loop
        video_frames = pipeline(validation_image, validation_control_images[:frame_number], decode_chunk_size=decode_chunk_size,num_frames=frame_number,num_inference_steps=num_inference_steps,motion_bucket_id=motion_bucket_id,controlnet_cond_scale=controlnet_cond_scale,height=height,width=width,ids_embedding=ids_embedding[:frame_number]).frames

        vis_images = [cv2.applyColorMap(np.array(img).astype(np.uint8), cv2.COLORMAP_JET) for img in vis_images]
        vis_images = [cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2RGB) for img in vis_images]
        
        vis_images = [Image.fromarray(img) for img in vis_images]
        
        video_frames = [img for sublist in video_frames for img in sublist]

        #save_gifs_side_by_side(video_frames, vis_images[:args["frame_number"]],val_save_dir,target_size=(width,height),duration=110)
        data = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in video_frames]
        return torch.cat(tuple(data), dim=0).unsqueeze(0)

class DragAnythingPipelineRunRandom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DragAnythingPipeline",),
                "sd_path": (pretrained_weights, {"default": "chilloutmix"}),
                "image": ("IMAGE",),
                "width": ("INT",{"default":576}),
                "height": ("INT",{"default":320}),
                "frame_number": ("INT",{"default":20}),
                "mask_list": ("IMAGE",),
                "num_inference_steps": ("INT",{"default":25}),
                "motion_bucket_id": ("INT",{"default":180}),
                "controlnet_cond_scale": ("FLOAT",{"default":1.0}),
                "decode_chunk_size": ("INT",{"default":8}),
                "move_speed": ("INT",{"default":5}),
                "move_border": ("INT",{"default":20}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "DragAnything"

    def run(self,pipeline,sd_path,image,width,height,frame_number,mask_list,num_inference_steps,motion_bucket_id,controlnet_cond_scale,decode_chunk_size,move_speed,move_border):
        trajectory_list="[]"
        trajectories=json.loads(trajectory_list)

        sd_path=f'{pretrained_weights_path}/{sd_path}'
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert('RGB')
        #image = np.array(image)
        # Convert RGB to BGR
        #image = image[:, :, ::-1].copy()
        masks=[]
        for mask in mask_list:
            mask_img=255.0 * mask.cpu().numpy()
            mask_img = Image.fromarray(np.clip(mask_img, 0, 255).astype(np.uint8)).convert('RGB')
            mask_img = np.array(mask_img)
            # Convert RGB to BGR
            #mask_img = mask_img[:, :, ::-1].copy()
            mask_img = cv2.cvtColor(np.array(mask_img).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            masks.append(mask_img)

            mask_trajectory=[]
            points=np.where(mask_img==1)
            x=np.mean(points[1])
            y=np.mean(points[0])
            mask_trajectory.append([x,y])
            
            for frame_ind in range(frame_number-1):
                x=x+random.randint(-move_speed,move_speed)
                y=y+random.randint(-move_speed,move_speed)
                if x<move_border:
                    x=move_border
                if y<move_border:
                    y=move_border
                if x>=image.size[0]-move_border:
                    x=image.size[0]-move_border
                if y>=image.size[1]-move_border:
                    y=image.size[1]-move_border
                mask_trajectory.append([x,y])
            trajectories.append(mask_trajectory)
        trajectory_list=json.dumps(trajectories)
        print(f'trajectory_list{trajectory_list}')

        validation_image = image
        original_width, original_height = validation_image.size
        validation_image = validation_image.resize((width, height))
        validation_control_images,ids_embedding,vis_images = get_condition(target_size=(height , width),
                                                                        original_size=(original_height , original_width),
                                                                        frame_number = frame_number,first_frame = validation_image,
                                                                        side=100,model_id=sd_path,mask_list=masks,trajectory_list=trajectory_list)
        ids_embedding = torch.stack(ids_embedding, dim=0).permute(0, 3, 1, 2)

        val_save_dir = output_dir
        os.makedirs(val_save_dir, exist_ok=True)
        
        # Inference and saving loop
        video_frames = pipeline(validation_image, validation_control_images[:frame_number], decode_chunk_size=decode_chunk_size,num_frames=frame_number,num_inference_steps=num_inference_steps,motion_bucket_id=motion_bucket_id,controlnet_cond_scale=controlnet_cond_scale,height=height,width=width,ids_embedding=ids_embedding[:frame_number]).frames

        vis_images = [cv2.applyColorMap(np.array(img).astype(np.uint8), cv2.COLORMAP_JET) for img in vis_images]
        vis_images = [cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2RGB) for img in vis_images]
        
        vis_images = [Image.fromarray(img) for img in vis_images]
        
        video_frames = [img for sublist in video_frames for img in sublist]

        #save_gifs_side_by_side(video_frames, vis_images[:args["frame_number"]],val_save_dir,target_size=(width,height),duration=110)
        data = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in video_frames]
        return torch.cat(tuple(data), dim=0).unsqueeze(0)

class VHS_FILENAMES_STRING:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "filenames": ("VHS_FILENAMES",),
                    }
                }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "DragAnything"
    FUNCTION = "run"

    def run(self, filenames):
        return (filenames[1][-1],)

def get_allowed_dirs():
    dir = os.path.abspath(os.path.join(__file__, ".."))
    file = os.path.join(dir, "text_file_dirs.json")
    with open(file, "r") as f:
        return json.loads(f.read())


def get_valid_dirs():
    return get_allowed_dirs().keys()

def get_dir_from_name(name):
    dirs = get_allowed_dirs()
    if name not in dirs:
        raise KeyError(name + " dir not found")

    path = dirs[name]
    path = path.replace("$input", folder_paths.get_input_directory())
    path = path.replace("$output", folder_paths.get_output_directory())
    path = path.replace("$temp", folder_paths.get_temp_directory())
    return path


def is_child_dir(parent_path, child_path):
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)
    return os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])


def get_real_path(dir):
    dir = dir.replace("/**/", "/")
    dir = os.path.abspath(dir)
    dir = os.path.split(dir)[0]
    return dir

def get_file(root_dir, file):
    if file == "[none]" or not file or not file.strip():
        raise ValueError("No file")

    root_dir = get_dir_from_name(root_dir)
    root_dir = get_real_path(root_dir)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    full_path = os.path.join(root_dir, file)

    #if not is_child_dir(root_dir, full_path):
    #    raise ReferenceError()

    return full_path

class TextFileNode:
    RETURN_TYPES = ("STRING","BOOLEAN",)
    CATEGORY = "utils"

    def load_text(self, **kwargs):
        self.file = get_file(kwargs["root_dir"], kwargs["file"])
        if not os.path.exists(self.file):
            return ("",False,)
        with open(self.file, "r") as f:
            return (f.read(),True, )


class LoadText(TextFileNode):
    @classmethod
    def IS_CHANGED(self, **kwargs):
        return os.path.getmtime(self.file)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "root_dir": (list(get_valid_dirs()), {"default":"output"}),
                "file": ("STRING", {"default": "dragtest_1.txt"}),
            },
        }

    FUNCTION = "load_text"

class SaveText(TextFileNode):
    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "root_dir": (list(get_valid_dirs()), {"default":"output"}),
                "file": ("STRING", {"default": "dragtest_1.txt"}),
                "text": ("STRING", {"forceInput": True, "multiline": True})
            },
        }

    FUNCTION = "write_text"

    def write_text(self, **kwargs):
        self.file = get_file(kwargs["root_dir"], kwargs["file"])
        with open(self.file, "w") as f:
            f.write(kwargs["text"])

        return super().load_text(**kwargs)

NODE_CLASS_MAPPINGS = {
    "DragAnythingLoader":DragAnythingLoader,
    "DragAnythingRun":DragAnythingRun,
    "DragAnythingPipelineRun":DragAnythingPipelineRun,
    "DragAnythingRunRandom":DragAnythingRunRandom,
    "DragAnythingPipelineRunRandom":DragAnythingPipelineRunRandom,
    "VHS_FILENAMES_STRING":VHS_FILENAMES_STRING,
    "LoadText":LoadText,
    "SaveText":SaveText,
}

