#%%
import os, sys

import numpy as np

from visualize import visualize_result, voxelize_numpy
from ahoi_utils import *
from train import NeuralNet
import pytorch3d.transforms as T
from trimesh import creation, transformations
import trimesh
from models_cvae import Encoder, Decoder, CVAE
from models_smpl_cvae import Encoder, Decoder, CVAESMPL
from visualize import *
import torch
from PIL import Image, ImageDraw
from io import BytesIO
from dataloaders import AhoiDatasetGlobal
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from models_voxel_pred import VoxelPredNet
from torch.utils.data import Subset


torch.autograd.set_detect_anomaly(True)

SAVE_IMAGE = True
N_STEP = 1
OUTPUT_DIR = '/home/jiangnan/AHOI/Supp/Visualize_opt'

inds = load_data('/home/jiangnan/AHOI/Supp/idx.npy')
# inds = [638737, 631562]
inds = [460722]


def evaluation(pcd1, pcd2):
    # pcd1_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l1').fit(pcd1)
    # min_2_to_1 = pcd1_nn.kneighbors(pcd2)[0]
    pcd2_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l1').fit(pcd2)
    min_1_to_2 = pcd2_nn.kneighbors(pcd1)[0]
    chamfer_dist = np.mean(min_1_to_2)

    return chamfer_dist

def evaluation_mesh(mesh1, mesh2):
    pcd_set1 = [mesh.vertices for mesh in mesh1]
    pcd_set2 = [mesh.vertices for mesh in mesh2]
    pcd_set1 = np.concatenate(pcd_set1)
    pcd_set2 = np.concatenate(pcd_set2)

    return evaluation(pcd_set1, pcd_set2)

def evaluation_cont_pen(human_pose, pcd_obj):
    smpl_output = smplx_model(body_pose=human_pose)
    joints = smpl_output.joints.detach().cpu().numpy().squeeze()
    vertices = smpl_output.vertices.detach().cpu()
    pelvis_transform = create_mat([0, 0, 0], joints[0], rot_type='rot_vec') \
                       @ create_mat([0, np.pi, 0], np.array([0, 0, 0]), rot_type='xyz')
    pcd_obj = trans_pcd_torch(pcd_obj, torch.tensor(pelvis_transform)).unsqueeze(0)
    # vertices = trans_pcd(vertices, np.linalg.inv(pelvis_transform))
    faces = torch.from_numpy(smplx_model.faces.astype(np.int64))
    inside = kaolin.ops.mesh.check_sign(vertices, faces, pcd_obj).squeeze()
    dist = torch.cdist(pcd_obj, vertices).squeeze()
    dist = torch.min(dist, dim=1)[0]

    if torch.count_nonzero(inside) == 0:
        pene = torch.tensor(0)
    else:
        pene = dist[inside].min()

    if torch.count_nonzero(inside) != 0:
        cont = torch.tensor(0)
    else:
        cont = dist[torch.logical_not(inside)].min()

    return pene, cont


def evaluation_rot_loc(rot_gt, loc_gt, mat_pt):
    loc_pt = mat_pt[:, :3, 3]
    rot_gt = torch.from_numpy(rot_gt)
    loc_gt = torch.from_numpy(loc_gt)
    inds = torch.logical_not(torch.isnan(rot_gt[:, 0]))
    loc_diff = torch.mean(torch.sqrt(torch.sum(torch.square(loc_pt[inds] - loc_gt[inds]))))

    return 0, loc_diff




    # pcd_obj = pcd_obj.cpu().numpy().squeeze()
    # hp = trimesh.points.PointCloud(vertices=vertices)
    # op = trimesh.points.PointCloud(vertices=pcd_obj)
    # scene = trimesh.Scene([hp, op])
    # scene.show()
    # pene = dist[]




# ind = 999
n_grid = 64
human_pose_all = load_data('/home/jiangnan/AHOI/temp_data/data_blocks_train/human_pose.npy')
human_betas_all = load_data('/home/jiangnan/AHOI/temp_data/data_blocks_train/human_betas.npy')
human_orient_all = load_data('/home/jiangnan/AHOI/temp_data/data_blocks_train/human_orient.npy')
# pare_human_pose_all = load_data('/home/jiangnan/AHOI/temp_data/data_blocks_val/pare_human_pose.npy')
# pare_human_betas_all = load_data('/home/jiangnan/AHOI/temp_data/data_blocks_val/pare_human_betas.npy')
img_name = load_data('/home/jiangnan/AHOI/temp_data/data_blocks_train/img_name.npy')

object_id_all = load_data('/home/jiangnan/AHOI/temp_data/data_blocks_train/object_id.npy')
object_rotation_all = load_data('/home/jiangnan/AHOI/temp_data/data_blocks_train/object_rotation.npy')
object_location_all = load_data('/home/jiangnan/AHOI/temp_data/data_blocks_train/object_location.npy')
object_meta_all = load_obj_meta(array=False)


input_size = 63
hidden_size = 500
output_size = 6
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load('/home/jiangnan/AHOI/checkpoint/global_model.ckpt'))
model.eval()


grid = gene_voxel_grid(N=n_grid, len=2, homo=False)
grid = torch.from_numpy(grid)

hidden_dim = 16
out_conv_channels = 512

learning_rate = 0.0001

model_cvae = CVAE(dim=n_grid, out_conv_channels=out_conv_channels, hidden_dim=hidden_dim).to(device)
# model_cvae.load_state_dict(torch.load('/home/jiangnan/AHOI/checkpoint/cvae_human_pena_1 (1).pth', map_location=device))
# model_cvae.load_state_dict(torch.load('/home/jiangnan/AHOI/checkpoint/cvae_humanpenal1.pth', map_location=device))
model_cvae.load_state_dict(torch.load('/home/jiangnan/AHOI/checkpoint/cvae_contrast (3).pth', map_location=device))
model_cvae.eval()

# model_cvae_smpl = CVAESMPL(dim=n_grid, out_conv_channels=out_conv_channels, hidden_dim=hidden_dim).to(device)
# # model_cvae.load_state_dict(torch.load('/home/jiangnan/AHOI/checkpoint/cvae_human_pena_1 (1).pth', map_location=device))
# model_cvae.load_state_dict(torch.load('/home/jiangnan/AHOI/checkpoint/cvae_humanpenal1.pth', map_location=device))
# # model_cvae_smpl.load_state_dict(torch.load('/home/jiangnan/AHOI/checkpoint/cvae_smpl.pth', map_location=device))
# model_cvae_smpl.eval()

# model_vox = VoxelPredNet(hidden_size=hidden_dim).to(device)
# model_vox.load_state_dict(torch.load('/home/jiangnan/AHOI/checkpoint/voxel_pred (copy).pth'))
# model_vox.eval()

smplx_model = smplx.create(MODEL_FOLDER, model_type='smplx',
                                   gender='male', ext='npz',
                                   num_betas=10,
                                   use_pca=False,
                                   create_global_orient=True,
                                   create_body_pose=True,
                                   create_betas=True,
                                   create_left_hand_pose=True,
                                   create_right_hand_pose=True,
                                   create_expression=True,
                                   create_jaw_pose=True,
                                   create_leye_pose=True,
                                   create_reye_pose=True,
                                   create_transl=True,
                                   batch_size=1)

#%%
train_dataset = AhoiDatasetGlobal(data_folder='/home/jiangnan/AHOI/temp_data/data_blocks_train', n_grid=n_grid, add_human=True)
train_dataset = Subset(train_dataset, indices=inds)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)


global_ind = 0

chamfer_dist_list = np.zeros((100, 2))
iou_list = np.zeros((100, 2))
pene_list = np.zeros((100, 2))
cont_list = np.zeros((100, 2))
stat = []

for i, (output, idx) in enumerate(train_loader):
    pelvis_transform = output['pelvis_transform'][0].detach().cpu().numpy()
    pelvis_transform_pare = output['pelvis_transform_pare'][0].detach().cpu().numpy()
    print(pelvis_transform)
    occ_human = output['occ_human']
    # occ_pare_human = output['occ_pare_human']
    occ_pare_human = occ_human
    occ_object = output['occ']
    # pare_human_pose = output['pare_human_pose']
    pare_human_pose = output['human_pose']
    pare_human_orient = output['human_orient']
    img = output['img']
    img_name = output['img_name']

    object_rotation = object_rotation_all[idx]
    object_location = object_location_all[idx]

    pcd_human = grid[occ_human.reshape(-1) > 0]
    pcd_pare_human = grid[occ_pare_human.reshape(-1) > 0]
    pcd_object = grid[occ_object.reshape(-1) > 0]


    object_id = output['object_id'][0].item()
    object_meta = object_meta_all[object_id]
    voxels = load_obj_voxel(object_id, n_grid=n_grid)
    voxels = torch.from_numpy(voxels).unsqueeze(1).to(torch.float32)



    # body_pose_torch = torch.from_numpy(pare_human_pose).to(device)
    with torch.no_grad():
        out_rot, out_loc = model(pare_human_pose)
        # out_loc -= out_loc
        out_rot_matrix = T.rotation_6d_to_matrix(out_rot)
        out_rot = T.matrix_to_euler_angles(out_rot_matrix, convention='XYZ')

        # occ_obj_pred = model_vox(img, occ_pare_human).squeeze().detach()

    out_rot = torch.nn.Parameter(out_rot, requires_grad=True)
    out_loc = torch.nn.Parameter(out_loc, requires_grad=True)
    rot_param = torch.nn.Parameter(torch.zeros(7), requires_grad=True)
    transl_param = torch.nn.Parameter(torch.zeros(7), requires_grad=True)

    optimizer_rot = torch.optim.SGD([out_rot], lr=learning_rate * 3., momentum=0.2)
    optimizer_loc = torch.optim.SGD([out_loc], lr=learning_rate, momentum=0.2)
    optimizer_rot_param = torch.optim.SGD([rot_param], lr=learning_rate * 3., momentum=0.2)
    optimizer_transl_param = torch.optim.SGD([transl_param], lr=learning_rate, momentum=0.2)


    images = []
    for step in range(N_STEP + 1):
        out_mat = create_mat_torch(out_rot, out_loc, rot_type='XYZ', affine=True)

        voxel_all = apply_part_transform(rot_param, transl_param, out_mat, object_meta, voxels)

        x = voxel_all[None, None, ...]
        x = x.to(device)
        # _, _, _, z = model_cvae(x, occ_pare_human)
        z, _ = model_cvae.encoder(x, occ_pare_human)

        #pred_loss = torch.sum(occ_obj_pred * voxel_all)

        with torch.no_grad():

            if step == 0 or step == N_STEP or SAVE_IMAGE:

                occ_obj = x.clone()
                occ_obj[occ_obj >= 0.5] = 1
                occ_obj[occ_obj < 0.5] = 0

                iou = torch.count_nonzero(torch.logical_and(occ_obj, occ_object)) / torch.count_nonzero(torch.logical_or(occ_obj, occ_object))
                print(f'IOU AT STEP {step}: {iou}')

                occ_obj = x.view(n_grid ** 3)
                occ_obj = occ_obj.to(torch.bool).detach().cpu().numpy()
                pcd_object_pt = trimesh.points.PointCloud(vertices=grid[occ_obj])



                # mesh_init = visualize_result(object_rotation=out_rot, object_location=out_loc, object_id=object_id, rigid=True, show=False)
                # mesh_gt = visualize_result(object_rotation=object_rotation, object_location=object_location, object_id=object_id, rigid=False, show=False)
                # mesh_chamfer = evaluation_mesh(mesh_init, mesh_gt)
                # print(f'Mesh Chamfer at step {step:02d}: ', mesh_chamfer)


            if SAVE_IMAGE:
                if step == N_STEP - 1:
                    # pcd_human = trans_pcd(pcd_human, pelvis_transform)
                    pcd_pare_human = trans_pcd(pcd_pare_human, pelvis_transform_pare)
                    pcd_object = trans_pcd(pcd_object, pelvis_transform)
                    pcd_human_trimesh = trimesh.points.PointCloud(vertices=pcd_human)
                    pcd_pare_human_trimesh = trimesh.points.PointCloud(vertices=pcd_pare_human)
                    pcd_object_trimesh = trimesh.points.PointCloud(vertices=pcd_object)

                    # out_mat = create_mat_torch(out_rot, out_loc, rot_type='XYZ', affine=False)
                    # part_mat = get_mat_from_urdf_param(rot_param, transl_param, out_mat, object_meta)
                    # part_mat = part_mat.detach().numpy()
                    # pelvis_mat = np.tile(pelvis_transform_pare[None, ...], (7, 1))
                    # # part_mat = np.matmul(pelvis_transform_pare, part_mat)
                    # part_rot = Rotation.from_matrix(part_mat[:, :3, :3]).as_euler('xyz')
                    # part_transl = part_mat[:, :3, 3]
                    # print(img_name)
                    #
                    # pare_human_pose = pare_human_pose.numpy()
                    # pare_human_orient = pare_human_orient.numpy()
                    #
                    # pare_human_orient_mat = Rotation.from_rotvec(pare_human_orient).as_matrix()
                    # adj_rot_mat = Rotation.from_euler('xyz', [0, np.pi / 4., 0]).as_matrix()
                    # pare_human_orient_mat = adj_rot_mat @ pare_human_orient_mat
                    # pare_human_orient = Rotation.from_matrix(pare_human_orient_mat).as_rotvec()
                    #
                    #
                    # visualize_result(object_rotation=object_rotation, object_location=object_location, object_id=object_id,
                    #                  rigid=False, show=False, human_pose=pare_human_pose, human_orient=pare_human_orient, human_transl=output['human_transl'].numpy(),
                    #                  save_name=os.path.join(OUTPUT_DIR, f'{idx.item():07d}_opt.jpg'))

                    out_rot_np = out_rot.detach().cpu().numpy()  # + [-0.4, 0, 0]
                    out_loc_np = out_loc.detach().cpu().numpy()  # + [-0.4, 0, 0]



                    # pcd_pt = visualize_result(object_location=out_loc_np, object_rotation=out_rot_np, object_id=object_id, rigid=True, show_axis=False, show=False)
                    # pcd_gt = visualize_result(object_location=object_location, object_rotation=object_rotation, object_id=object_id, rigid=False, show_axis=False, show=False)

                    pcd_pt = pcd_object_pt.apply_transform(pelvis_transform_pare)
                    pcd_gt = pcd_object_trimesh
                    pcd_gt.visual.vertex_colors = [240, 20, 20]

                    pcd_human_trimesh.export('/home/jiangnan/AHOI/Supp/Illustrate/human.obj')

                    # pcd_gt[0].visual.vertex_colors = [255, 10, 10]
                    # meshes = pcd_pt + pcd_gt + [pcd_human]
                    # meshes = [pcd_pt] + [pcd_human_trimesh] + [pcd_gt]
                    # meshes = [pcd_human_trimesh]
                    # visualize_result(human_pose=human_pose, human_orient=human_orient, object_location=object_location, object_rotation=object_rotation, object_id=object_id, voxel_n=n_grid, rigid=True, show_axis=False,
                    #                  save_name='/home/jiangnan/AHOI/temp_data/cmp_img.png', show=True, extra_meshes=meshes)
                #
                    # img_bytes = visualize_result(show_axis=False, show=True, return_img=True, extra_meshes=meshes)
                # img = Image.open(BytesIO(img_bytes))
                # images.append(img)

            # if step == 0 or step == N_STEP:
                # pcd_human_trimesh = trimesh.points.PointCloud(vertices=pcd_human)
                # scene = trimesh.Scene([pcd_human_trimesh, pcd_object_pt])
                # scene.show()



                # pcd_obj_torch = torch.from_numpy(pcd_object_pt.vertices).to(torch.float32)
                # pene, cont = evaluation_cont_pen(pare_human_pose, pcd_obj_torch)
                # dr, dl = evaluation_rot_loc(object_rotation, object_location, part_mat)

                # print(dr, dl)



                # chamfer_dist = evaluation(pcd_object_pt.vertices, pcd_object)
                # print(f'CHAMFER DIST IN STEP {step:02d}: {chamfer_dist}')
                # if step == 0:
                #     print(dr, dl)
                    #     chamfer_dist_list[global_ind, 0] = chamfer_dist
                    #     iou_list[global_ind, 0] = iou
                    # pene_list[global_ind, 0] = pene.item()
                    # cont_list[global_ind, 0] = cont.item()
                # else:
                #     print(dr, dl)
                    #     chamfer_dist_list[global_ind, 1] = chamfer_dist
                    #     iou_list[global_ind, 1] = iou
                    #     stat.append([object_id, img_name, chamfer_dist_list[global_ind, 0], chamfer_dist])
                    # pene_list[global_ind, 1] = pene.item()
                    # cont_list[global_ind, 1] = cont.item()


        # print(x.sum())
        loss = z @ z.T #- pred_loss * 0.001
        optimizer_rot.zero_grad()
        optimizer_loc.zero_grad()
        optimizer_rot_param.zero_grad()
        optimizer_transl_param.zero_grad()
        loss.backward()

        # print(out_mat)
        # print(rot_param)
        # print(transl_param)

        optimizer_rot.step()
        optimizer_loc.step()
        optimizer_rot_param.step()
        optimizer_transl_param.step()

    # if SAVE_IMAGE:
    #     for _ in range(10):
    #         images.append(img)
    #     images[0].save(f'/home/jiangnan/AHOI/temp_data/images/optimize/vis_{global_ind:02d}.gif', save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0 )

    global_ind += 1

    if global_ind >= 100:
        break

# chamfer_dist_list = chamfer_dist_list[chamfer_dist_list[:, 0] > 0]
# print(np.mean(chamfer_dist_list[:, 0] - chamfer_dist_list[:, 1]))
# print(np.mean(chamfer_dist_list[:, 0]), np.mean(chamfer_dist_list[:, 1]))
# print(np.mean(iou_list[:, 0] - iou_list[:, 1]))
# print(np.mean(iou_list[:, 0]), np.mean(iou_list[:, 1]))
print(np.mean(pene_list[:, 0] - pene_list[:, 1]))
print(np.mean(pene_list[:, 0]), np.mean(pene_list[:, 1]))
print(np.mean(cont_list[:, 0] - cont_list[:, 1]))
print(np.mean(cont_list[:, 0]), np.mean(cont_list[:, 1]))
# np.save(os.path.join(OUTPUT_DIR, 'chamfer.npy'), np.array(stat, dtype=object), allow_pickle=True)




