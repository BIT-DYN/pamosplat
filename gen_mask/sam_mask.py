import numpy as np
import cv2
import argparse
import torch
import glob
import clip
import sys
# your path to SAM
sys.path.append("/home/dyn/multimodal/Grounded-Segment-Anything/segment_anything")
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry
from PIL import Image
import matplotlib.pyplot as plt
import os
from natsort import natsorted
from tqdm import tqdm
import pickle
from tokenize_anything import model_registry
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack
from sentence_transformers import SentenceTransformer, util

# your path to SAM weight
sam_checkpoint = "/data/dyn/weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
# mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(   
    model=sam,
    points_per_side=60,
    pred_iou_thresh=0.92,
    stability_score_thresh=0.97,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=700,  )

def args_getter():
    parser = argparse.ArgumentParser(description='Sam embedded with CLIP')
    parser.add_argument(
        "--input_image",
        nargs="+",
        help="A list of space separated input rgb images; "
    )
    parser.add_argument('--output_dir',type=str,help='the results output directory')
    parser.add_argument('--down_sample',type=int,help='the results output directory')
    parser.add_argument('--use_frame',type=int,help='the results output directory')
    args = parser.parse_args()
    return args

def mask_getter(image:np.ndarray):
    print("start")
    global mask_generator
    masks = mask_generator.generate(image)
    print("ok")
    return masks


def min_rect_bbox(mask):
    nonzero_indices = np.nonzero(mask)
    if len(nonzero_indices[0]) == 0:
        return np.zeros((4, 2), dtype=np.intp)
    x, y, w, h = cv2.boundingRect(np.column_stack(nonzero_indices))
    rect = np.array([[y, x], [y, x + w], [y + h, x + w], [y + h, x]], dtype=np.intp)
    return rect


def resolve_overlaps(masks, height, width, min_pixel_threshold=50, max_pixel_threshold=50000):
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    label_map = np.zeros((height, width), dtype=np.int32) - 1  
    for idx, mask_data in enumerate(sorted_masks):
        mask = mask_data['segmentation']
        if np.sum(mask) < min_pixel_threshold or np.sum(mask) > max_pixel_threshold:
            continue
        label_map[mask] = idx 
    unique_mask_id = np.unique(label_map)
    all_mask = []
    all_bbox = []
    for id in unique_mask_id:
        if id == -1:
            continue
        mask_this = label_map==id
        bbox_this = min_rect_bbox(mask_this)
        all_mask.append(mask_this)
        all_bbox.append(bbox_this)
    return label_map, all_mask, all_bbox



def bbox_getter(bbox_original: list, height: int, width: int, out_rate = 1.3):
    # print(bbox_original)
    x_min, y_min = bbox_original[0]
    x_max, y_max = bbox_original[2]
    width_mask, height_mask = bbox_original[2] - bbox_original[0]
    width_mask_new = round(width_mask * out_rate)
    height_mask_new = round(height_mask * out_rate)
    width_inc = round((width_mask_new - width_mask) / 2)
    height_inc = round((height_mask_new - height_mask) / 2)
    left_inc = min(width_inc, x_min)
    top_inc = min(height_inc, y_min)
    right_inc = min(width_inc, width - x_max)
    bottom_inc = min(height_inc, height - y_max)
    x_min -= left_inc
    y_min -= top_inc
    x_max += right_inc
    y_max += bottom_inc
    return [x_min, y_min, x_max, y_max]


def main(args:argparse.Namespace) ->None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    model_type = "tap_vit_l"
    checkpoint = "/home/dyn/outdoor/tokenize-anything/weights/tap_vit_l_03f8ec.pkl"
    tap_model = model_registry[model_type](checkpoint=checkpoint)
    concept_weights = "/home/dyn/outdoor/tokenize-anything/weights/merged_2560.pkl"
    tap_model.concept_projector.reset_weights(concept_weights)
    tap_model.text_decoder.reset_cache(max_batch_size=1000)
    # your sbert path
    sbert_model = SentenceTransformer('/home/dyn/multimodal/SBERT/pretrained/model/all-MiniLM-L6-v2')
    
        
    # for dataset in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
    for dataset in ["softball"]:
        num = 31
        suffix = '*.jpg'
        max_pixel_threshold = 200000
        min_pixel_threshold = 10
        for n in range(num):  
            input_path = f"../data/{dataset}/ims/{n}"
            input_image = glob.glob(os.path.join(input_path, suffix))
            input_image = natsorted(input_image)
            output_dir = f"../data/{dataset}/mask/{n}"
        
            os.makedirs(output_dir, exist_ok=True)
            for idx, file in tqdm(enumerate(input_image)):
                # only need the init timestamp
                if idx > 0:
                    continue
                image_original = cv2.imread(file)
                image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
                height, width = image_original.shape[:2]
                masks = mask_getter(image_original)
                label_map, all_masks, all_bboxes = resolve_overlaps(masks, height, width, min_pixel_threshold=min_pixel_threshold, max_pixel_threshold=max_pixel_threshold)
                
                min_rects = all_bboxes
                img = image_original
                img_list, img_scales = im_rescale(img, scales=[1024], max_size=1024)
                input_size, original_size = img_list[0].shape, img.shape[:2]
                img_batch = im_vstack(img_list, fill_value=tap_model.pixel_mean_value, size=(1024, 1024))
                inputs = tap_model.get_inputs({"img": img_batch})
                inputs.update(tap_model.get_features(inputs))
                batch_points = np.zeros((len(min_rects), 2, 3), dtype=np.float32)
                for j in range(len(min_rects)):
                    batch_points[j, 0, 0] = min_rects[j][0, 0]
                    batch_points[j, 0, 1] = min_rects[j][0, 1]
                    batch_points[j, 0, 2] = 2
                    batch_points[j, 1, 0] = min_rects[j][2, 0]
                    batch_points[j, 1, 1] = min_rects[j][2, 1]
                    batch_points[j, 1, 2] = 3
                inputs["points"] = batch_points
                inputs["points"][:, :, :2] *= np.array(img_scales, dtype="float32")
                outputs = tap_model.get_outputs(inputs)
                iou_pred = outputs["iou_pred"].detach().cpu().numpy()
                point_score = batch_points[:, 0, 2].__eq__(2).__sub__(0.5)[:, None]
                rank_scores = iou_pred + point_score * ([1000] + [0] * (iou_pred.shape[1] - 1))
                mask_index = np.arange(rank_scores.shape[0]), rank_scores.argmax(1)
                sem_tokens = outputs["sem_tokens"][mask_index].unsqueeze_(1)
                captions = tap_model.generate_text(sem_tokens)
                caption_fts = sbert_model.encode(captions, convert_to_tensor=True, device="cuda")
                caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
                caption_fts = caption_fts.detach()
            
                mask_feature_all = []
                bbox_all = []
                for i, masks_data in enumerate(all_masks):
                    bbox = bbox_getter(all_bboxes[i], height, width, out_rate=1.2)
                    bbox_all.append(bbox)
                    crop_image = image_original[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    pil_crop_image = Image.fromarray(crop_image)
                    clip_crop_image = preprocess(pil_crop_image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        mask_feature = clip_model.encode_image(clip_crop_image)
                    mask_feature_all.append(mask_feature[0])
                # mask with vlm feature
                all_data = {
                    "instance_map": label_map,
                    "all_masks": all_masks,
                    "all_features": mask_feature_all,
                    "all_capfeat": caption_fts
                }
                
                out_filename = output_dir + "/" + str(idx) + ".pkl"
                with open(out_filename, 'wb') as f:
                    pickle.dump(all_data, f)   
                print("save to", out_filename)

if __name__ == '__main__':
    args = args_getter()
    main(args)



    
