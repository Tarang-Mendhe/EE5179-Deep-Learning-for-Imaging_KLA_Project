import os
import argparse
import torch
import numpy as np
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import logging

def load_image(image_path, device):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0).to(device)
    return img

def load_mask(mask_path, device):
    mask = Image.open(mask_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    mask = transform(mask).unsqueeze(0).to(device)
    return mask

def save_output_image(output, output_path):
    output_img = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output_img = (output_img * 255).astype(np.uint8)
    Image.fromarray(output_img).save(output_path)

def calculate_psnr_ssim(original, denoised, mask=None):
    original_np = original.squeeze().cpu().numpy().transpose(1, 2, 0)
    denoised_np = denoised.squeeze().cpu().numpy().transpose(1, 2, 0)

    if mask is not None:
        mask = mask.squeeze().cpu().numpy()
        original_np = original_np * mask
        denoised_np = denoised_np * mask

    psnr_value = psnr(torch.tensor(denoised_np), torch.tensor(original_np), data_range=1.0).item()
    ssim_value = ssim(original_np, denoised_np, channel_axis=-1, data_range=1.0)
    return psnr_value, ssim_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a deep learning model for image denoising with PSNR and SSIM evaluation in PyTorch.")
    parser.add_argument("input_dir", type=str, help="Directory of input test images")
    parser.add_argument("model_weights", type=str, help="Path to the model weights file")
    parser.add_argument("denoised_output_dir", type=str, help="Directory for saving denoised output images")
    parser.add_argument("--qualitative_dir", type=str, default=None, help="Directory for saving qualitative test images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (cuda or cpu)")

    args = parser.parse_args()
    main(args.input_dir, args.model_weights, args.denoised_output_dir, args.device, qualitative_dir=args.qualitative_dir)
