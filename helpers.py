import torch
import os
import open3d as o3d
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
import cv2
from scipy.optimize import differential_evolution
from scipy.optimize import minimize



def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam

def params2rendervar(params):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def l1_loss_v1(x, y, weight_map=None):
    l1 = torch.abs(x - y)
    if weight_map is not None:
        return (l1 * weight_map).sum() 
    else:
        return l1.mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()

def weighted_l2_loss_v1_part(x, y, w, part_index_fg):
    losses = torch.sqrt(((x - y) ** 2) * w + 1e-20)
    part_losses = []
    for group in part_index_fg:
        part_loss = losses[group].mean()  
        part_losses.append(part_loss)
    return torch.stack(part_losses).mean() 


def l2_loss_v1(x, y):
    return torch.sqrt(((x - y) ** 2) + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def l2_loss_v2(x, y):
    return torch.sqrt(((x - y) ** 2).sum(-1) + 1e-20).mean()



def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def o3d_knn_group(index, all_label, pts, num_knn):
    indices = []
    sq_dists = []
    masks = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for idx, p in zip(index, pcd.points):
        this_group = all_label[idx]
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        all_knn_idx = np.asarray(i[1:])
        all_knn_dis = np.asarray(d[1:])
        mask = all_label[index[all_knn_idx]] == this_group
        indices.append(all_knn_idx)
        sq_dists.append(all_knn_dis)
        masks.append(mask)
    return np.array(sq_dists), np.array(indices), np.array(masks)




def params2cpu(params, is_initial_timestep, part_dir=None):
    if is_initial_timestep:
        if part_dir is not None:
            part_dir.update(params)
            res = {k: (v.detach().cpu().contiguous().numpy() if not isinstance(v, np.ndarray) else v) for k, v in part_dir.items()}
        else:
            res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'part_colors','unnorm_rotations']}
    return res


def save_params(output_params, seq, exp):
    to_save = {}
    for k in output_params[0].keys():
        if k in output_params[1].keys():
            to_save[k] = np.stack([params[k] for params in output_params])
        else:
            to_save[k] = output_params[0][k]
    os.makedirs(f"output/{exp}/{seq}", exist_ok=True)
    np.savez(f"output/{exp}/{seq}/params", **to_save)


from scipy.spatial.transform import Rotation as R

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def quaternion_to_rotation_matrix(quaternion):
    return R.from_quat(quaternion).as_matrix()


def mask_2dto3d(all_masks, bg_maks_index, depth, w2c, k, w, h, world_points, threshold=0.1):
    k = torch.tensor(k).cuda().float()
    w2c = torch.tensor(w2c).cuda().float()
    world_points = torch.tensor(world_points).cuda().float()
    all_gs_indices = torch.arange(len(world_points))
    world_points_h = torch.cat([world_points, torch.ones(world_points.shape[0], 1).cuda().float()], dim=1)
    cam_points = (w2c @ world_points_h.T).T
    img_points = (k @ cam_points[:, :3].T).T
    img_points = img_points[:, :2] / img_points[:, 2:]
    valid_mask = (img_points[:, 0] >= 0) & (img_points[:, 0] < w) & (img_points[:, 1] >= 0) & (img_points[:, 1] < h)
    valid_img_points = img_points[valid_mask]
    valid_cam_points = cam_points[valid_mask]
    valid_world_points = world_points[valid_mask]
    valid_all_gs_indices = all_gs_indices[valid_mask]
    pixel_x = valid_img_points[:, 0].long()
    pixel_y = valid_img_points[:, 1].long()
    pixel_indices = pixel_y * w + pixel_x
    pixel_coords = valid_img_points.long()
    depth_values = depth[0, pixel_y, pixel_x]
    point_depths = valid_cam_points[:, 2]
    depth_diff = torch.abs(depth_values - point_depths)
    valid_depth_mask = depth_diff < threshold
    pixel_coords = pixel_coords[valid_depth_mask]
    valid_world_points = valid_world_points[valid_depth_mask]
    valid_all_gs_indices =valid_all_gs_indices[valid_depth_mask]
    pixel_indices = pixel_indices[valid_depth_mask]
    mask_points = []
    colors = []
    mask_indices = []
    bg_indices = []
    for mask_idx, mask in enumerate(all_masks):
        mask = mask.reshape(-1)
        mask_for_thismask = mask[pixel_indices.cpu().numpy()]
        mask_3d_points = valid_world_points.cpu().numpy()[mask_for_thismask]
        mask_gs_indices = valid_all_gs_indices.cpu().numpy()[mask_for_thismask]
        if mask_idx in bg_maks_index:
            bg_indices.append(mask_gs_indices)
            mask_indices.append([])
            continue
        mask_points.append(mask_3d_points)
        colors.append(np.tile(np.random.random(3), (len(mask_3d_points), 1)))
        mask_indices.append(mask_gs_indices)
    all_mask_points = np.vstack(mask_points)
    all_colors = np.vstack(colors)
    if len(bg_indices) > 0:
        bg_indices = np.concatenate(bg_indices)
    else:
        bg_indices = None
    return  all_mask_points, all_colors, mask_indices, bg_indices


def mask_2dto3d_torch(all_masks, depth, w2c, k, w, h, world_points, threshold=0.1):
    k = torch.tensor(k).cuda().float()
    w2c = torch.tensor(w2c).cuda().float()
    all_gs_indices = torch.arange(len(world_points)).cuda()
    world_points_h = torch.cat([world_points, torch.ones(world_points.shape[0], 1).cuda().float()], dim=1)
    cam_points = (w2c @ world_points_h.T).T
    img_points = (k @ cam_points[:, :3].T).T
    img_points = img_points[:, :2] / img_points[:, 2:]
    valid_mask = (img_points[:, 0] >= 0) & (img_points[:, 0] < w) & (img_points[:, 1] >= 0) & (img_points[:, 1] < h)
    valid_img_points = img_points[valid_mask]
    valid_cam_points = cam_points[valid_mask]
    valid_world_points = world_points[valid_mask]
    valid_all_gs_indices = all_gs_indices[valid_mask]
    pixel_x = valid_img_points[:, 0].long()
    pixel_y = valid_img_points[:, 1].long()
    pixel_indices = pixel_y * w + pixel_x
    pixel_coords = valid_img_points.long()
    depth_values = depth[0, pixel_y, pixel_x]
    point_depths = valid_cam_points[:, 2]
    depth_diff = torch.abs(depth_values - point_depths)
    valid_depth_mask = depth_diff < threshold
    pixel_coords = pixel_coords[valid_depth_mask]
    valid_world_points = valid_world_points[valid_depth_mask]
    valid_all_gs_indices = valid_all_gs_indices[valid_depth_mask]
    pixel_indices = pixel_indices[valid_depth_mask]
    mask_points = []
    colors = []
    mask_indices = []
    pixel_coords_all_masks = []
    bg_indices = []
    for mask_idx, mask in enumerate(all_masks):
        mask = mask.reshape(-1).cuda() 
        mask_for_thismask = mask[pixel_indices]
        mask_3d_points = valid_world_points[mask_for_thismask]
        mask_gs_indices = valid_all_gs_indices[mask_for_thismask]
        mask_pixel_coords = pixel_coords[mask_for_thismask]
        mask_points.append(mask_3d_points)
        mask_indices.append(mask_gs_indices)
        pixel_coords_all_masks.append(mask_pixel_coords)
    return mask_indices, pixel_coords_all_masks

    
    
def cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    # 计算每个张量的 L2 范数
    tensor1_norm = torch.norm(tensor1, p=2, dim=1, keepdim=True)
    tensor2_norm = torch.norm(tensor2, p=2, dim=1, keepdim=True)
    # 计算余弦相似度
    similarity = torch.mm(tensor1, tensor2.t()) / (tensor1_norm * tensor2_norm.t())
    return similarity



def create_rotation_matrix(Rx, Ry, Rz):
    cos = torch.cos
    sin = torch.sin
    R_x = torch.tensor([[1, 0, 0],
                        [0, cos(Rx), -sin(Rx)],
                        [0, sin(Rx), cos(Rx)]], device='cuda', dtype=torch.float32)
    R_y = torch.tensor([[cos(Ry), 0, sin(Ry)],
                        [0, 1, 0],
                        [-sin(Ry), 0, cos(Ry)]], device='cuda', dtype=torch.float32)
    R_z = torch.tensor([[cos(Rz), -sin(Rz), 0],
                        [sin(Rz), cos(Rz), 0],
                        [0, 0, 1]], device='cuda', dtype=torch.float32)
    return R_z @ R_y @ R_x



def euler_to_quaternion(Rx, Ry, Rz):
    Rx, Ry, Rz = map(lambda x: torch.tensor(x).cuda(), [Rx, Ry, Rz])
    cy = torch.cos(Rz * 0.5)
    sy = torch.sin(Rz * 0.5)
    cp = torch.cos(Ry * 0.5)
    sp = torch.sin(Ry * 0.5)
    cr = torch.cos(Rx * 0.5)
    sr = torch.sin(Rx * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def compute_angles(center_points, neighborhood_points):
    N = center_points.shape[0]  
    neighbors = neighborhood_points.shape[1]  # 20
    vectors = neighborhood_points - center_points.unsqueeze(1)  # (N, 20, 3)
    norm_vectors = vectors / vectors.norm(dim=-1, keepdim=True)  # (N, 20, 3)
    cos_sim_matrix = torch.einsum('nij,nmj->nim', norm_vectors, norm_vectors)  # (N, 20, 20)
    i, j = torch.triu_indices(neighbors, neighbors, offset=1)
    cos_angles = cos_sim_matrix[:, i, j]  # (N, 190)
    angles = torch.acos(cos_angles.clamp(-1 + 1e-6, 1 - 1e-6))  # Ensure values are in the range [-1, 1]
    return angles  # (N, 190)

