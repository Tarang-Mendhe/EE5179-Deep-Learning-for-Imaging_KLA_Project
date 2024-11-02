import os
import argparse
import torch
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio 
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import logging
import random
import matplotlib.pyplot as plt
from best_model_arch import UNet  # Replace with your actual model class

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

def get_unique_filename(output_dir, img_name):
    base_name, ext = os.path.splitext(img_name)
    output_path = os.path.join(output_dir, f"denoised_{img_name}")
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"denoised_{base_name}_{counter}{ext}")
        counter += 1
    return output_path

def calculate_psnr_ssim(original, denoised, mask=None):
    original = original.squeeze().cpu().numpy().transpose(1, 2, 0)
    denoised = denoised.squeeze().cpu().numpy().transpose(1, 2, 0)
    if mask is not None:
        mask = mask.squeeze().cpu().numpy()  
        mask = np.expand_dims(mask, axis=2)  
        original = original * mask
        denoised = denoised * mask
    psnr = PeakSignalNoiseRatio()
    psnr_value = psnr(torch.tensor(denoised), torch.tensor(original))
    ssim_value = ssim(original, denoised, data_range=1.0, channel_axis=2)
    return psnr_value, ssim_value

def main(input_dir, model_weights_path, denoised_output_dir, device, val_split='Val', qualitative_dir=None):
    logging.basicConfig(level=logging.INFO)
    try:
        n_classes = 3  
        model = UNet(n_class=n_classes)
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    os.makedirs(denoised_output_dir, exist_ok=True)
    output_dir = qualitative_dir if qualitative_dir else denoised_output_dir
    os.makedirs(output_dir, exist_ok=True)

    images_to_display = []
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category, val_split)
        if not os.path.isdir(category_path):
            logging.warning(f"Category path {category_path} does not exist. Skipping.")
            continue

        gt_clean_image_path = os.path.join(category_path, 'GT_clean_image')
        degraded_image_path = os.path.join(category_path, 'Degraded_image')
        defect_mask_path = os.path.join(category_path, 'Defect_mask')

        for subfolder in os.listdir(gt_clean_image_path):
            clean_subfolder = os.path.join(gt_clean_image_path, subfolder)
            degraded_subfolder = os.path.join(degraded_image_path, subfolder)
            mask_subfolder = os.path.join(defect_mask_path, subfolder)

            for img_name in os.listdir(clean_subfolder):
                clean_img = os.path.join(clean_subfolder, img_name)
                degraded_img = os.path.join(degraded_subfolder, img_name)
                img_base_name = os.path.splitext(img_name)[0]
                mask_img = os.path.join(mask_subfolder, f'{img_base_name}_mask.png')

                img = load_image(degraded_img, device)
                with torch.no_grad():
                    denoised = model(img).clamp(0, 1)

                output_path = get_unique_filename(output_dir, img_name)
                save_output_image(denoised, output_path)

                original_img = load_image(clean_img, device)
                mask = load_mask(mask_img, device)
                images_to_display.append((original_img, denoised, mask))
                
                if len(images_to_display) >= 5:
                    break
            if len(images_to_display) >= 5:
                break
        if len(images_to_display) >= 5:
            break

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    for i, (original_img, denoised_img, mask_img) in enumerate(images_to_display):
        original_img = original_img.squeeze().cpu().numpy().transpose(1, 2, 0)
        denoised_img = denoised_img.squeeze().cpu().numpy().transpose(1, 2, 0)
        mask_img = mask_img.squeeze().cpu().numpy()

        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(denoised_img)
        axes[i, 1].set_title("Denoised Image")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(mask_img, cmap='gray')
        axes[i, 2].set_title("Mask")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a deep learning model for image denoising with PSNR and SSIM evaluation in PyTorch.")
    parser.add_argument("input_dir", type=str, help="Directory of input test images")
    parser.add_argument("model_weights", type=str, help="Path to the model weights file")
    parser.add_argument("denoised_output_dir", type=str, help="Directory for saving denoised output images")
    parser.add_argument("--qualitative_dir", type=str, default=None, help="Directory for saving qualitative test images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (cuda or cpu)")

    args = parser.parse_args()
    main(args.input_dir, args.model_weights, args.denoised_output_dir, args.device, qualitative_dir=args.qualitative_dir)
