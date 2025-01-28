import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.utils import flow_to_image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from natsort import natsorted
import matplotlib.pyplot as plt

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()
    
    # use_dataset = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
    use_dataset = [ "softball"]
    use_cam_id = np.arange(31)
    with torch.no_grad():
        for dataset in use_dataset:
            for cam_id in use_cam_id:
                print("for", dataset, cam_id)
                for_flow_dir = f'../data/{dataset}/for_flow/{cam_id}'
                rev_flow_dir = f'../data/{dataset}/rev_flow/{cam_id}'
                os.makedirs(for_flow_dir, exist_ok=True)   
                os.makedirs(rev_flow_dir, exist_ok=True)   
                input_path = f"../data/{dataset}/ims/{cam_id}"
                images = glob.glob(os.path.join(input_path, '*.png')) + glob.glob(os.path.join(input_path, '*.jpg'))
                images = natsorted(images)
                image2 = None
                for i, _ in tqdm(enumerate(images)):
                    if i == (len(images) - 1):
                        continue
                    imfile1 = images[i + 0]
                    imfile2 = images[i + 1]
                    if image2 is None:
                        image1 = load_image(imfile1)
                    else:
                        image1 = image2
                    image2 = load_image(imfile2)
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)
                    # forward flow
                    _, for_flow_up = model(image1, image2, iters=20, test_mode=True)
                    for_flow_up = for_flow_up[0].cpu().numpy()
                    for_flow_up = for_flow_up.transpose(1,2,0)
                    # reverse flow
                    _, rev_flow_up = model(image2, image1, iters=20, test_mode=True)
                    rev_flow_up = rev_flow_up[0].cpu().numpy() 
                    rev_flow_up = rev_flow_up.transpose(1,2,0)
                    # H,W,2
                    np.save(f'{for_flow_dir}/{i}.npy', for_flow_up)      
                    np.save(f'{rev_flow_dir}/{i+1}.npy', rev_flow_up)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    demo(args)
