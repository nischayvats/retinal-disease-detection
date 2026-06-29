import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, input_tensor, original_image):
    target_layer = model.resnet.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0]
    
    image_resized = cv2.resize(original_image, (512, 512))
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    cam_image = show_cam_on_image(image_normalized, grayscale_cam, use_rgb=True)
    
    return cam_image