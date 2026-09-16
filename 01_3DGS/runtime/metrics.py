from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, selected_image_names=None):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if selected_image_names and fname not in selected_image_names:
            continue  # Skip if the file is not in the selected images list
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, selected_images=None):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"

                # Only select specified images if provided
                renders, gts, image_names = readImages(renders_dir, gt_dir, selected_image_names=selected_images)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    image_name = image_names[idx]
                    ssim_value = ssim(renders[idx], gts[idx]).item()
                    psnr_value = psnr(renders[idx], gts[idx]).item()
                    lpips_value = lpips(renders[idx], gts[idx], net_type='vgg').item()


                    if selected_images:
                        print(f"\nMetrics for image: {image_name}") 
                        # Print individual metrics for the image
                        print(f"  SSIM : {ssim_value:.7f}")
                        print(f"  PSNR : {psnr_value:.7f}")
                        print(f"  LPIPS: {lpips_value:.7f}")

                    # Append metrics to lists
                    ssims.append(ssim_value)
                    psnrs.append(psnr_value)
                    lpipss.append(lpips_value)

                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                     "PSNR": torch.tensor(psnrs).mean().item(),
                                                     "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--images_name', '-i', nargs="*", type=str, default=None, help="List of specific image name to evaluate")
    args = parser.parse_args()
    
    # If no images are selected, None will be passed to evaluate function
    evaluate(args.model_paths, selected_images=args.images_name)
