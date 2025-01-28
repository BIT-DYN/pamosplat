import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
import open3d as o3d
import cv2
import math
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, l2_loss_v2, quat_mult,\
    o3d_knn, o3d_knn_group, params2rendervar, params2cpu, save_params, mask_2dto3d, mask_2dto3d_torch, \
    create_rotation_matrix, euler_to_quaternion,  compute_angles
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
import pickle
from scipy.spatial import cKDTree
import networkx as nx
import community
# from networkx.algorithms.community import greedy_modularity_communities, girvan_newman
from scipy.sparse import find
from collections import Counter
import clip
import matplotlib.pyplot as plt
import distinctipy
# from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import torch.nn.functional as F
from scipy.optimize import differential_evolution
import time
import random
import yaml
from scipy.spatial.transform import Rotation as R


def get_dataset(t, md, seq):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        camera_param = [w, h, k, w2c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"data/{seq}/ims/{fn}")))
        if im.shape[2] == 4:
            im = im[:, :, :3]  
            im[(im<0.1).all(axis=2)] = [0,0,0]
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        if fn.endswith('.jpg'):
            seg_filename = fn.replace('.jpg', '.png')
        else:
            seg_filename = fn
        seg = np.array(copy.deepcopy(Image.open(f"data/{seq}/seg/{seg_filename}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        camera_number = os.path.split(fn)[0]
        if t == 0:
            mask_path = f"data/{seq}/mask/{camera_number}/{t}.pkl"
            mask = pickle.load(open(mask_path, 'rb'))
            dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c, 'camera': camera_param, 'mask': mask})
        else:
            flow_path = f"data/{seq}/for_flow/{camera_number}/{t-1}.npy"  
            flow = np.load(flow_path)
            flow_path = f"data/{seq}/rev_flow/{camera_number}/{t}.npy"
            flow_reverse = np.load(flow_path)
            flow_length = np.linalg.norm(flow, axis=-1) 
            flow_reverse_length = np.linalg.norm(flow_reverse, axis=-1)  
            flow_weight = np.clip(flow_length + flow_reverse_length, 1, 6)
            flow_weight /= np.sum(flow_weight)  # normalize
            dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c, 'camera': camera_param, 'flow': flow, 'flow_weight': torch.tensor(flow_weight).cuda()})
    return dataset


def get_batch(todo_dataset, dataset, pre_dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
        if pre_dataset is not None:
            todo_pre_dataset = pre_dataset.copy()
    random_int = randint(0, len(todo_dataset) - 1)
    curr_data = todo_dataset.pop(random_int)
    pre_data = None
    if pre_dataset is not None:
        pre_data = todo_pre_dataset.pop(random_int)
    return curr_data, pre_data


def initialize_params(seq, md):
    init_pt_cld = np.load(f"data/{seq}/init_pt_cld.npz")["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],                                                                     # center
        'rgb_colors': init_pt_cld[:, 3:6],                                                                # RGB
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),                      # background
        'part_colors': np.tile([0, 0, 0], (seg.shape[0], 1)),                                       # part
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),                          # rot
        'logit_opacities': np.zeros((seg.shape[0], 1)),                                             # opacities
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),       # scales
        'cam_m': np.zeros((max_cams, 3)), 
        'cam_c': np.zeros((max_cams, 3)),  
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables


def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],   
        'rgb_colors': 0.0025,     
        'seg_colors': 0.0,  
        'part_colors': 0.0,    
        'unnorm_rotations': 0.001, 
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4, 
        'cam_c': 1e-4, 
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(configs, params, curr_data, pre_data, variables, part_index,  part_index_fg, is_initial_timestep, i, t, num_iter_per_timestep, seq):
    losses = {}
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    if is_initial_timestep:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    else:
        # flow-supervised
        losses['im'] =  0.2*l1_loss_v1(im, curr_data['im'], weight_map=curr_data['flow_weight']) + 0.6*l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))
        
    if is_initial_timestep:
        excess_radii_max = torch.relu(torch.max(torch.exp(params['log_scales']), dim=1).values - 0.03)
        losses['scale'] = torch.sum(excess_radii_max)  
        
    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]
        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None] 
        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        # local_rigidty
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])
        neighbor_pts = fg_pts[variables["neighbor_indices_part"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        # part_rigidty
        losses['iso_part'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist_part"], variables["neighbor_weight_part"])
        # some floor loss
        if seq in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:    
            losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()
        else:
            losses['floor'] = 0
        # bg loss
        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        if len(bg_pts) > 0:
            losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])
        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])
    loss_weights = {'im': 1.0, 'seg': 1.0, 'scale': 1.0, 'iso': 20.0, 'iso_part': 0.1, 'floor': 2.0, 'bg': 20.0, 'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def move_priori(configs, dataset, pre_dataset, params, part_colors, part_clipfeat, part_index, optimizer, variables, t, seq):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    pts_raw_save = pts.detach().clone()
    rot_raw_save = rot.detach().clone()
    is_fg = params['seg_colors'][:, 0] > 0.5
    fg_pts = pts[is_fg]
    fg_rots = rot[is_fg]
    # local rigidity
    neighbor_pts = fg_pts[variables["neighbor_indices_part"]]
    curr_offset = neighbor_pts - fg_pts[:, None]
    curr_offset = curr_offset.detach().float().contiguous()
    curr_offset = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)  
    offset_diff = torch.sqrt(((curr_offset - variables["neighbor_dist_part"]) ** 2) + 1e-20)  
    no_mask = variables["neighbor_weight_part"] < 0.02  
    diff_thre = 0.001   
    variables["neighbor_weight_part"] += 0.02 * torch.clip(1 - offset_diff / diff_thre, min=0) - 0.2 * torch.clip(torch.clip(offset_diff / diff_thre - 1, min=0), max=4)
    variables["neighbor_weight_part"] = torch.clip(variables["neighbor_weight_part"],min=0.0,max=1.0)
    
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["neighbor_dist_part"] = curr_offset
    variables["prev_col"] = params['rgb_colors'].detach()
    
    
    dataset_copy = dataset.copy()
    num_camera = len(dataset_copy)
    pre_dataset_copy = pre_dataset.copy()
    rgbrendervar = params2rendervar(params)
    rgbrendervar['colors_precomp'] = params['rgb_colors']
    grouprendervar = params2rendervar(params)
    grouprendervar['colors_precomp'] = params['part_colors']
    all_k = []
    all_w2c = []
    all_camere = []
    all_flow = []
    all_pre_rgb = []
    all_cur_rgb = []
    for cam in range(num_camera):
        pre_data = pre_dataset_copy[cam]
        all_pre_rgb.append(pre_data['im'])
        curr_data = dataset_copy[cam]
        w, h, k, w2c = curr_data['camera']
        all_cur_rgb.append(curr_data['im'])
        all_k.append(k)
        all_w2c.append(w2c)
        all_camere.append(curr_data['cam']) 
        all_flow.append(curr_data['flow'])
        
    # cur color diff
    part_diff_rgb = {} 
    for (camera, pre_rgb, cur_rgb) in zip(all_camere, all_pre_rgb, all_cur_rgb):
        pre_rgb = pre_rgb.permute(1, 2, 0).detach()
        cur_rgb = cur_rgb.permute(1, 2, 0).detach()
        group, _, _, = Renderer(raster_settings=camera)(**grouprendervar)
        group = group.permute(1, 2, 0)
        color_diff_dis = 0.05  # to part rendering
        for i in range(len(part_index)):
            color_diff = torch.abs(group - part_colors[i+1])
            mask_this = torch.all(color_diff < color_diff_dis, axis=2)
            this_mask_pre_rgb = pre_rgb[mask_this]
            this_mask_cur_rgb = cur_rgb[mask_this]
            diff = torch.abs(this_mask_pre_rgb - this_mask_cur_rgb)
            if i in part_diff_rgb:
                part_diff_rgb[i] = torch.cat((part_diff_rgb[i], diff), dim=0)
            else:
                part_diff_rgb[i] = diff
    part_score = []
    for i in range(len(part_index)):
        if i in part_diff_rgb:
            part_score.append(torch.mean(part_diff_rgb[i]).cpu().numpy())
        else:
            part_score.append(0)
    part_score = np.hstack(part_score)
    vel_ratio = 1
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    inertia_new_pts = pts.clone().detach()
    inertia_new_pts_copy = pts[is_fg].clone().cpu().detach().numpy()
    no_score = configs["no_score"]
    for idx, part_id in enumerate(part_index):
        if part_score[idx] < no_score:
            continue
        m_center_prev = torch.mean(variables["prev_pts"][part_id], axis=0)
        m_center = torch.mean(pts.clone().detach()[part_id], axis=0)
        m_offset = m_center - m_center_prev
        m_center_new = m_center + m_offset * vel_ratio                    
        inertia_new_pts[part_id] = inertia_new_pts[part_id] + m_offset
        all_torch_tensor_pre=(variables["prev_pts"][part_id]-m_center_prev).cpu().detach().numpy()
        all_torch_tensor=(pts.clone().detach()[part_id]-m_center).cpu().detach().numpy()
        H = np.dot(all_torch_tensor_pre.T, all_torch_tensor)
        U, S, Vt = np.linalg.svd(H)
        R_matrix = np.dot(U, Vt)
        rotation = R.from_matrix(R_matrix)
        rotated_cloud = torch.from_numpy(rotation.apply(all_torch_tensor)).to("cuda")
        inertia_new_pts[part_id] = torch.tensor(rotated_cloud + m_center_new, dtype=torch.float32).to("cuda")
    
    # gaussian_part_to_pixel
    all_part_cam_indices = []
    all_part_cam_pixel = []
    for cam_id, camera in enumerate(all_camere):
        part_cam_indices = {}
        part_cam_pixel  = {}
        group, _, depth, = Renderer(raster_settings=camera)(**grouprendervar)
        group = group.permute(1, 2, 0)
        all_masks = []
        for i in range(len(part_colors)):
            if i == 0: 
                continue
            color_diff = torch.abs(group - part_colors[i])
            mask_this = torch.all(color_diff < color_diff_dis, axis=2)
            all_masks.append(mask_this)
        k = all_k[cam_id]
        w2c = all_w2c[cam_id]
        means3D_np_raw  = params['means3D'].detach()
        mask_indices, pixel_coords_all_masks  = mask_2dto3d_torch(all_masks, depth, w2c, k, w, h, means3D_np_raw.detach(), threshold=0.05)
        
        use_pixel = 100
        for part_id, (mask_indice, pixel_coord) in enumerate(zip(mask_indices, pixel_coords_all_masks)):
            if len(mask_indice) > use_pixel:
                random_perm = torch.randperm(len(mask_indice))
                part_cam_indices[part_id] = mask_indice[random_perm[:use_pixel]]
                part_cam_pixel[part_id] = pixel_coord[random_perm[:use_pixel]]
            elif len(mask_indice) > use_pixel/4:
                part_cam_indices[part_id] = mask_indice
                part_cam_pixel[part_id] = pixel_coord
            else:
                continue
        all_part_cam_indices.append(part_cam_indices)
        all_part_cam_pixel.append(part_cam_pixel)
        
    # part motion prior
    for part_id,  index_this in enumerate(part_index):
        pc_this = means3D_np_raw[index_this] 
        centroid_point = torch.mean(pc_this, dim=0)  
        camera_k = []
        camera_w2c = []
        camera_indices = []
        camera_pixels = []
        gt_flows = []
        for cam_id, (part_cam_indices, part_cam_pixel) in enumerate(zip(all_part_cam_indices, all_part_cam_pixel)):
            if part_id in part_cam_indices:
                camera_k.append(torch.tensor(all_k[cam_id]).cuda())
                camera_w2c.append(torch.tensor(all_w2c[cam_id]).cuda())
                camera_indices.append(part_cam_indices[part_id])
                camera_pixels.append(part_cam_pixel[part_id])
                flow_torch = torch.tensor(all_flow[cam_id]).cuda().transpose(0,1)
                x_coords = part_cam_pixel[part_id][:, 0]  
                y_coords = part_cam_pixel[part_id][:, 1]  
                gt_flows.append(flow_torch[x_coords, y_coords])
        if len(camera_k) == 0:
            continue
        diffs = []
        for k, w2c, camera_indice, camera_pixel, gt_flow in zip(camera_k, camera_w2c, camera_indices, camera_pixels, gt_flows):
            new_pcs = means3D_np_raw[camera_indice]
            new_pcs_h = torch.cat([new_pcs, torch.ones(new_pcs.shape[0], 1).cuda()], dim=1)  # 齐次坐标
            cam_points_new_pcs = (w2c @ new_pcs_h.T).T
            img_points_h = (k @ cam_points_new_pcs[:, :3].T).T
            img_points_h = img_points_h[:, :2] / img_points_h[:, 2:]
            cal_flow = img_points_h - camera_pixel
            diffs.append(l1_loss_v2(cal_flow, gt_flow))
        diffs = torch.stack(diffs)
        raw_result = (torch.mean(diffs)).item()
        # no flow use interia
        if raw_result < 4:
            if part_score[part_id] >= no_score:
                means3D_np_raw  = params['means3D'].detach().clone()
                means3D_np_raw[index_this] = inertia_new_pts[index_this]
                new_params = {'means3D': means3D_np_raw}
                params = update_params_and_optimizer(new_params, params, optimizer)
            continue
        
        ratio = min(raw_result/20, 0.6)
        dis_the = 0.2 * ratio
        angle_the = np.pi/8 * ratio
        bounds = [(-dis_the, dis_the), (-dis_the, dis_the), (-dis_the, dis_the), 
                        (-angle_the, angle_the), (-angle_the, angle_the), (-angle_the, angle_the)]
        index_this = torch.tensor(index_this, device='cuda')
        
        result = differential_evolution(render_iou, bounds, args=(pc_this, index_this, means3D_np_raw.detach().clone(), centroid_point, 
                                                        camera_k, camera_w2c, camera_indices, camera_pixels, gt_flows),  strategy='best1bin', maxiter=int(25*ratio), popsize=2) 
        
        if result.fun < raw_result:
            best_params = result.x
            best_params = torch.tensor(best_params, device='cuda', dtype=torch.float32)
            tx, ty, tz, Rx, Ry, Rz = best_params
            rotation_matrix = create_rotation_matrix(Rx, Ry, Rz)
            pc_new  = (pc_this - centroid_point) @ rotation_matrix.T + centroid_point
            pc_new = pc_new + torch.tensor([tx, ty, tz], device='cuda')
            rot_raw  = params['unnorm_rotations'].detach().clone()
            rot_this = rot_raw[index_this]
            rot_raw[index_this] = quat_mult(rot_this, euler_to_quaternion(Rx, Ry, Rz))
            means3D_np_raw  = params['means3D'].detach().clone()
            means3D_np_raw[index_this] = pc_new
            new_params = {'means3D': means3D_np_raw, 'unnorm_rotations': rot_raw}
            params = update_params_and_optimizer(new_params, params, optimizer)
            
            
            part_diff_rgb = None
            for (camera, pre_rgb, cur_rgb) in zip(all_camere, all_pre_rgb, all_cur_rgb):
                pre_rgb = pre_rgb.permute(1, 2, 0).detach()
                cur_rgb = cur_rgb.permute(1, 2, 0).detach()
                rgbrendervar = params2rendervar(params)
                rgbrendervar['colors_precomp'] = params['rgb_colors']
                grouprendervar = params2rendervar(params)
                grouprendervar['colors_precomp'] = params['part_colors']
                group, _, _, = Renderer(raster_settings=camera)(**grouprendervar)
                group = group.permute(1, 2, 0)
                pre_render, _, _, = Renderer(raster_settings=camera)(**rgbrendervar)
                pre_render = pre_render.permute(1, 2, 0).detach()
                color_diff_dis = 0.05
                i = part_id
                color_diff = torch.abs(group - part_colors[i+1])
                mask_this = torch.all(color_diff < color_diff_dis, axis=2)
                this_mask_pre_rgb = pre_render[mask_this]
                this_mask_cur_rgb = cur_rgb[mask_this]
                diff = torch.abs(this_mask_pre_rgb - this_mask_cur_rgb)
                part_diff_rgb = torch.cat((part_diff_rgb, diff), dim=0) if part_diff_rgb is not None else diff
            color_diff = torch.mean(part_diff_rgb).cpu().numpy()
            # flow not good use interia
            if len(part_diff_rgb) > 0 and color_diff > no_score:
                means3D_np_raw  = params['means3D'].detach().clone()
                means3D_np_raw[index_this] = inertia_new_pts[index_this]
                new_params = {'means3D': means3D_np_raw}
                params = update_params_and_optimizer(new_params, params, optimizer)
            
    variables["prev_pts"] = pts_raw_save
    variables["prev_rot"] = rot_raw_save
    return params, variables       

                                                        
def render_iou(opt_params, pc_this, index_this, means3D_np_raw, centroid, camera_k, camera_w2c, camera_indices, camera_pixels, gt_flows):
    opt_params = torch.tensor(opt_params, device='cuda', dtype=torch.float32)
    tx, ty, tz, Rx, Ry, Rz = opt_params
    pc_new  = (pc_this - centroid) @ create_rotation_matrix(Rx, Ry, Rz).T + centroid
    pc_new = pc_new + opt_params[:3]
    means3D_np_raw = means3D_np_raw.detach()
    means3D_np_raw[index_this] = pc_new
    diffs = []
    for k, w2c, camera_indice, camera_pixel, gt_flow in zip(camera_k, camera_w2c, camera_indices, camera_pixels, gt_flows):
        new_pcs = means3D_np_raw[camera_indice]
        new_pcs_h = torch.cat([new_pcs, torch.ones(new_pcs.shape[0], 1).cuda()], dim=1)  # 齐次坐标
        cam_points_new_pcs = (w2c @ new_pcs_h.T).T
        img_points_h = (k @ cam_points_new_pcs[:, :3].T).T
        img_points_h = img_points_h[:, :2] / img_points_h[:, 2:]
        cal_flow = img_points_h - camera_pixel
        diffs.append(l1_loss_v2(cal_flow, gt_flow))
    diffs = torch.stack(diffs)
    result = (torch.mean(diffs)).item()
    return result
       


def initialize_post_first_timestep(all_label, params, variables, optimizer, seq, part_index, part_index_fg, num_knn=10, num_knn_part=10):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    index = np.arange(len(is_fg))[is_fg.detach().cpu().numpy()]
    neighbor_sq_dist, neighbor_indices, masks = o3d_knn_group(index, all_label, init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)    
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()
    variables["neighbor_angles"] = compute_angles(init_fg_pts.detach(), init_fg_pts.detach()[variables["neighbor_indices"]])
    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    centroid_dis = torch.zeros((len(params['means3D']))).cuda()
    for part_id in part_index:
        pc_this = params['means3D'][part_id]  
        centroid_point = torch.mean(pc_this, dim=0)  
        distances = torch.norm(pc_this - centroid_point, dim=1)
        centroid_dis[part_id] = distances
    variables["centroid_dis"] = centroid_dis.detach()
    neighbor_indices_part = torch.zeros((len(init_fg_pts), num_knn_part)).long().cuda() 
    for fg_part_id in part_index_fg:
        random_part_id = torch.empty((len(fg_part_id), num_knn_part), dtype=torch.long).cuda()
        for idx in range(len(fg_part_id)):
            indices = torch.randperm(len(fg_part_id))
            random_part_id[idx] = fg_part_id[indices[:num_knn_part]]
        neighbor_indices_part[fg_part_id] = random_part_id
    neighbor_weight_part = torch.ones((len(init_fg_pts), num_knn_part)).cuda()*0.2  
    variables["neighbor_indices_part"] = neighbor_indices_part.detach().long().contiguous()
    variables["neighbor_weight_part"] = neighbor_weight_part.detach().float().contiguous()
    neighbor_pts = init_fg_pts[variables["neighbor_indices_part"]]
    curr_offset = neighbor_pts - init_fg_pts[:, None]
    curr_offset = curr_offset.detach().float().contiguous()
    variables["neighbor_dist_part"] = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] == "rgb_colors":
            param_group['lr'] = 0.0005
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables
    
def gaussian_groupping(configs, seq, params, dataset, optimizer, variables, downsample=0.03, if_vis=False):
    
    num_camera = len(dataset)
    is_fg = (params['seg_colors'][:, 0] > 0.5).detach().cpu().numpy()                                                                                                                                                                                                                                                                                                                                                     
    means3D_np_raw  = params['means3D'].detach().cpu().numpy()                                                                                                                                                                                                                                                                                                                                               
    rgbcolors_np_raw  = params['rgb_colors'].detach().cpu().numpy()
    is_mask = is_fg
    means3D_np_fg_raw = means3D_np_raw[is_mask]
    rgbcolors_np_fg_raw =rgbcolors_np_raw[is_mask]
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(means3D_np_fg_raw)
    point_cloud.colors = o3d.utility.Vector3dVector(rgbcolors_np_fg_raw)
    downsampled_point_cloud = point_cloud.voxel_down_sample(downsample)
    means3D_np_fg = np.asarray(downsampled_point_cloud.points)
    num_gs = len(means3D_np_fg)
    
    gs_tree = cKDTree(means3D_np_fg)
    count_mat = np.zeros((num_gs, num_gs), dtype=int)
    vis_mat = np.zeros((num_gs, num_gs), dtype=int)
    gs_clipfeat = torch.zeros((num_gs, 512)).to("cuda")
    gs_featcount = torch.zeros((num_gs), dtype=int).to("cuda")
    gs_capfeat = torch.zeros((num_gs, 384)).to("cuda")
    dataset_copy = dataset.copy()
    save = False
    folder_path = f"mask_vis/{seq}"
    if not os.path.exists(folder_path):
        os.makedirs(f"mask_vis/{seq}", exist_ok=True)
        save = True
    bg_num = np.zeros((len(means3D_np_fg_raw), ), dtype=int) 
    for cam in range(num_camera):
        curr_data = dataset_copy.pop()
        camera = curr_data['cam']
        im, radius, depth, = Renderer(raster_settings=camera)(**params2rendervar(params))
        w, h, k, w2c = curr_data['camera']
        all_masks = curr_data['mask']['all_masks']
        all_features = curr_data['mask']['all_features']
        all_capfeat = curr_data['mask']['all_capfeat']
        label_map = curr_data['mask']['instance_map']
        if save:
            height, width = label_map.shape
            img = np.zeros((height, width, 3), dtype=np.uint8)  
            for idx, mask_data in enumerate(all_masks):
                color = (np.random.random(3) * 255).astype(np.uint8)  
                img[mask_data] = color
            cv2.imwrite(f"mask_vis/{seq}/{cam}.png", img)
            
        cur_seg = curr_data['seg'].permute(1, 2, 0)[...,0]
        bg_maks_index = []
        for mask_idx, mask_data in enumerate(all_masks):
            mask_data = torch.tensor(mask_data, device="cuda")
            if cur_seg[mask_data.bool()].float().mean() < 0.5:
                bg_maks_index.append(mask_idx)
        threshold = 0.10
        pcs, colors, mask_indices, _  = mask_2dto3d(all_masks, bg_maks_index, depth, w2c, k, w, h, means3D_np_fg, threshold=threshold)
        _, _, _, bg_gs_index  = mask_2dto3d(all_masks, bg_maks_index, depth, w2c, k, w, h, means3D_np_fg_raw, threshold=threshold)
        if bg_gs_index is not None:
            bg_num[bg_gs_index] += 1
        
        ok_mask_indices = []
        for maks_indice, feat, cap in zip(mask_indices, all_features, all_capfeat):
            if len(maks_indice) == 0:
                continue
            count_mat[maks_indice[:, None], maks_indice] += 1
            feat = feat/torch.norm(feat, dim=-1, keepdim=True)
            gs_clipfeat[maks_indice] += feat
            gs_featcount[maks_indice] += 1
            cap = cap/torch.norm(feat, dim=-1, keepdim=True)
            gs_capfeat[maks_indice] += cap
            ok_mask_indices.append(maks_indice)
        if len(ok_mask_indices) > 0:
            mask_indices = np.concatenate(ok_mask_indices)
            vis_mat[mask_indices[:, None], mask_indices] += 1
    
    np.fill_diagonal(count_mat, 0)
    rows_with_nonzero = np.where(np.any(count_mat != 0, axis=1))[0]
    means3D_np_fg = means3D_np_fg[rows_with_nonzero]
    mask_weights = np.divide(count_mat.astype('float'), vis_mat.astype('float')+0.00001)[np.ix_(rows_with_nonzero, rows_with_nonzero)]
    gs_clipfeat = gs_clipfeat[rows_with_nonzero]
    gs_featcount = gs_featcount[rows_with_nonzero]
    gs_capfeat = gs_capfeat[rows_with_nonzero]
    
    # use the
    mask_weights[mask_weights<configs["ratio_the"]] = 0
    rows_with_nonzero = np.where(np.any(mask_weights != 0, axis=1))[0]
    means3D_np_fg = means3D_np_fg[rows_with_nonzero]
    mask_weights = mask_weights[np.ix_(rows_with_nonzero, rows_with_nonzero)]
    gs_clipfeat = gs_clipfeat[rows_with_nonzero]
    gs_featcount = gs_featcount[rows_with_nonzero]
    gs_clipfeat /= gs_clipfeat.norm(p=2, dim=-1, keepdim=True)
    gs_capfeat = gs_capfeat[rows_with_nonzero]
    gs_capfeat /= gs_capfeat.norm(p=2, dim=-1, keepdim=True)
    

    G = nx.Graph()
    row, col, data = find(mask_weights)
    edges = [(r, c, w) for r, c, w in zip(row, col, data)]
    G.add_weighted_edges_from(edges)
    print("The final nodes number is", len(G.nodes()))
    nodes_np = np.array(sorted(G.nodes()))
    print("Waitting for Groupping over ......")
    partition = community.best_partition(G, resolution=float(configs["part_res"]), weight="weight")
    node_ids = np.vectorize(partition.get)(nodes_np)
    unique_ids, counts = np.unique(node_ids, return_counts=True)
    ok_communities = unique_ids[counts > num_gs*0.0001]
    unique_ids=unique_ids[ok_communities]
    
    colors = np.ones((len(G.nodes()), 3))
    part_point = []
    part_clipfeat = []
    part_capfeat = []
    part_label = [] 
    new_id = 0
    for idx, id in enumerate(unique_ids):
        colors[node_ids==id] = np.random.random(3)
        pc_this = means3D_np_fg[node_ids==id]
        clipfeat_this = gs_clipfeat[node_ids==id]
        capfeat_this = gs_capfeat[node_ids==id]
        pcd_this = o3d.geometry.PointCloud()
        pcd_this.points = o3d.utility.Vector3dVector(pc_this)
        pcd_clusters = pcd_this.cluster_dbscan(
            eps=configs["eps"],
            min_points=configs["min_points"],
        )
        pcd_clusters = np.array(pcd_clusters)
        counter = Counter(pcd_clusters)
        if counter and (-1 in counter):
            del counter[-1]
        if counter:
            most_common = counter.most_common(4)
            most_common_label, most_common_count = most_common[0]
            largest_mask = pcd_clusters == most_common_label
            clipfeat = clipfeat_this[largest_mask].mean(dim=0)
            clipfeat = clipfeat/torch.norm(clipfeat, dim=-1, keepdim=True)
            capfeat = capfeat_this[largest_mask].mean(dim=0)
            capfeat = capfeat/torch.norm(capfeat, dim=-1, keepdim=True)
            part_point.append(pc_this[largest_mask])
            part_clipfeat.append(clipfeat)
            part_capfeat.append(capfeat)
            part_label.append(np.tile(new_id+1, len(pc_this[largest_mask])))
            new_id += 1
            threshold = configs["num_thre"]
            for level in range(5):
                if len(most_common) > level+1:
                    next_common_label, next_common_count = most_common[level+1]
                    if next_common_count / most_common_count >= threshold:
                        next_largest_mask = pcd_clusters == next_common_label
                        next_clipfeat = clipfeat_this[next_largest_mask].mean(dim=0)
                        next_clipfeat = next_clipfeat / torch.norm(next_clipfeat, dim=-1, keepdim=True)
                        part_point.append(pc_this[next_largest_mask])
                        part_clipfeat.append(next_clipfeat)
                        next_capfeat = capfeat_this[next_largest_mask].mean(dim=0)
                        next_capfeat = next_capfeat / torch.norm(next_capfeat, dim=-1, keepdim=True)
                        part_capfeat.append(next_capfeat)
                        part_label.append(np.tile(new_id + 1, len(pc_this[next_largest_mask])))
                        new_id += 1 
    left_gs = np.vstack(part_point)
    left_label = np.concatenate(part_label)
    print("The gaussian part num is", new_id)
    # use random color
    random_colors = distinctipy.get_colors(new_id+1, pastel_factor=1)
    
    gs_tree = cKDTree(left_gs)  
    distances, indices = gs_tree.query(means3D_np_fg_raw)
    # filter some outliers
    out_mask_dis = distances>0.15 
    out_mask_bg = bg_num > num_camera/5
    out_mask = out_mask_dis | out_mask_bg
    out_all_mask = is_mask.copy()
    out_all_mask[out_all_mask] = out_mask
    out_num = np.count_nonzero(out_mask)
    old_seg_colors = params['seg_colors'].detach().cpu().numpy()
    new_seg_colors = old_seg_colors.copy()
    new_seg_colors[out_all_mask] = np.tile([0.3,0,0.7], (out_num,1)) 
    new_params = {'seg_colors': torch.tensor(new_seg_colors).cuda().float()}
    params = update_params_and_optimizer(new_params, params, optimizer)
    is_fg = (params['seg_colors'][:, 0] > 0.5).detach().cpu().numpy()   
    is_mask = is_fg
    all_label = np.zeros((len(params['means3D'])))
    all_label[is_mask] = left_label[indices][~out_mask] 
    all_label_fg = all_label[is_fg]
    part_index = []
    part_index_fg = []
    all_color = np.zeros((len(params['means3D']), 3))
    all_index = np.arange(len(params['means3D']))
    all_index_fg = np.arange(len(all_label_fg))
    for label in range((new_id+1)):
        if label == 0:
            all_color[all_label==label]=random_colors[label]
        else:
            part_index.append(all_index[all_label==label])
            part_index_fg.append(torch.tensor(all_index_fg[all_label_fg==label]).cuda())
            all_color[all_label==label]=random_colors[label]
    if if_vis:
        print("vis")
        gs_pcd = o3d.geometry.PointCloud()
        gs_pcd.points = o3d.utility.Vector3dVector(means3D_np_raw[is_mask])
        gs_pcd.colors = o3d.utility.Vector3dVector(all_color[is_mask])
        o3d.visualization.draw_geometries([gs_pcd])
    part_colors =  np.array(random_colors)
    part_colors = torch.tensor(part_colors, device="cuda")
    return part_index, part_index_fg, torch.stack(part_clipfeat), torch.stack(part_capfeat), all_label, all_color, part_colors, params, variables
                    
        
def cal_num_iter(dataset, params):
    rendervar = params2rendervar(params)
    all_color_diff = []
    for i in range(len(dataset)):
        curr_data = dataset[i]
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        curr_id = curr_data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        all_color_diff.append(l1_loss_v1(im, curr_data['im']))
    return torch.tensor(all_color_diff).mean().detach().cpu().numpy()

def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def train(seq, exp, configs, use_time =None):
    md = json.load(open(f"data/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    params, variables = initialize_params(seq, md)
    optimizer = initialize_optimizer(params, variables)
    output_params = []
    part_index = None
    part_index_fg = None
    for t in range(num_timesteps):
        if use_time is not None and t>use_time:
            continue
        dataset = get_dataset(t, md, seq)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        pre_dataset = None
        if not is_initial_timestep:
            pre_dataset = get_dataset(t-1, md, seq)
            params, variables = move_priori(configs, dataset, pre_dataset, params, part_colors, part_clipfeat, part_index, optimizer, variables, t, seq)
        # num_iter_per_timestep = configs["num_iter_initial"] if is_initial_timestep else configs["num_iter_post"]
        if is_initial_timestep:
            num_iter_per_timestep = configs["num_iter_initial"]
        else:
            loss = cal_num_iter(dataset, params)
            num_iter_per_timestep = np.max((np.min((configs["num_iter_post"], int(loss*100000))), int(configs["num_iter_post"]*0.75)))
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data, pre_data = get_batch(todo_dataset, dataset, pre_dataset)
            loss, variables = get_loss(configs, params, curr_data, pre_data, variables, part_index, part_index_fg, is_initial_timestep, i, t, num_iter_per_timestep, seq)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        progress_bar.close()
        if is_initial_timestep:
            with torch.no_grad():
                downsample = 0.03
                part_index, part_index_fg, part_clipfeat, part_capfeat, all_label, all_color, part_colors, params, variables \
                    = gaussian_groupping(configs, seq, params, dataset, optimizer, variables, downsample=downsample, if_vis=configs["vis_group"])
                new_params = {'part_colors': torch.tensor(all_color).cuda().float()}
                params = update_params_and_optimizer(new_params, params, optimizer)
                part_dir = {
                    'part_clipfeat': part_clipfeat,
                    'part_capfeat': part_capfeat,
                    'all_label': all_label,
                    'part_colors': part_colors
                }
            variables = initialize_post_first_timestep(all_label, params, variables, optimizer, seq, part_index, part_index_fg, num_knn=20, num_knn_part=10)
        output_params.append(params2cpu(params, is_initial_timestep, part_dir = part_dir))
        # save per N timestamps
        if t%10 == 0 and t > 0:
            save_params(output_params, seq, exp)
    save_params(output_params, seq, exp)


if __name__ == "__main__":
    exp_name = "PaMoSplat"
    # for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
    for sequence in ["softball"]:
        print("="*100)
        print("for sequence", sequence)
        with open('config/'+sequence+'.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        train(sequence, exp_name, configs)
        torch.cuda.empty_cache()