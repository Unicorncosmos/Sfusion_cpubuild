from cpulss import LSSViewTransformerBEVDepth   
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (640, 1600),
    'src_size': (900, 1600),
    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-51.2, 51.2, 0.4],                                                # 1
    'y': [-51.2, 51.2, 0.4],                                                # 2
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}
voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 128

multi_adj_frame_id_cfg = (1, 8, 8, 1)
num_adj = len(range(
    multi_adj_frame_id_cfg[0],
    multi_adj_frame_id_cfg[1]+multi_adj_frame_id_cfg[2]+1,
    multi_adj_frame_id_cfg[3]
))
out_size_factor = 4
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
radar_cfg = {
    'bbox_num': 100,
    'radar_fusion_type': "medium_fusion",  # in ['post_fusion', 'medium_fusion']
    'voxel_size': voxel_size,
    'out_size_factor': out_size_factor,
    'point_cloud_range': point_cloud_range,
    'grid_config': grid_config,
    'norm_bbox': True,  
    'pc_roi_method': 'pillars',
    'img_feats_bbox_dims': [1, 1, 0.5],
    'pillar_dims': [0.4, 0.4, 0.1],
    'pc_feat_name': ['pc_x', 'pc_y', 'pc_vx', 'pc_vy'],
    'hm_to_box_ratio': 1.0,
    'time_debug': False,
    'radar_head_task': [
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        ]
}

input_size=data_config['input_size']
in_channels=512
out_channels=128
depthnet_cfg=dict(use_dcn=False)
downsample=16


view_transformer = LSSViewTransformerBEVDepth(
    grid_config=grid_config,
    input_size=input_size,
    in_channels=512,
    out_channels=128,
    accelerate=False,  # Set to True if using acceleration
    depthnet_cfg=dict(use_dcn=False),  # Adjust as needed
    downsample = 16,  # Additional depthnet configuration if needed
) 

#----------------------------------CPU-LSS-TEST--------------------------------------------------------


import torch

# Define dummy input tensors
B = 1  # Batch size
N = 12  # Total number of frames
num_frame = 2  # Number of frames per group
H, W = 640, 1600  # Image height and width

# Create dummy input tensors
inputs = [
    torch.randn(B, N, 3, H, W),  # Image data
    torch.randn(B, N, 3, 3),    # Rotation matrices
    torch.randn(B, N, 3),       # Translation vectors
    torch.randn(B, N, 3, 3),    # Intrinsic matrices
    torch.randn(B, N, 3, 3),    # Post-rotation matrices
    torch.randn(B, N, 3),       # Post-translation vectors
    torch.randn(B, 3, 3)        # BDA matrices
]

# Define the data preprocessing function
def preprocess_data(inputs, num_frame):
    # Place the provided preprocessing code here
    B, N, _, H, W = inputs[0].shape
    N = N // num_frame
    imgs = inputs[0].view(B, N, num_frame, 3, H, W)
    imgs = torch.split(imgs, 1, 2)
    imgs = [t.squeeze(2) for t in imgs]
    rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
    extra = [
        rots.view(B, num_frame, N, 3, 3),
        trans.view(B, num_frame, N, 3),
        intrins.view(B, num_frame, N, 3, 3),
        post_rots.view(B, num_frame, N, 3, 3),
        post_trans.view(B, num_frame, N, 3)
    ]
    extra = [torch.split(t, 1, 1) for t in extra]
    extra = [[p.squeeze(1) for p in t] for t in extra]
    rots, trans, intrins, post_rots, post_trans = extra
    return imgs, rots, trans, intrins, post_rots, post_trans, bda

# Test the preprocessing function
imgs, rots, trans, intrins, post_rots, post_trans, bda = preprocess_data(inputs, num_frame)

print("---------------------------------------")
# Print the shapes of the processed tensors
print("Image frames shapes:", [img.shape for img in imgs])
print("Rotation matrices shapes:", [rot.shape for rot in rots])
print("Translation vectors shapes:", [tr.shape for tr in trans])
print("Intrinsic matrices shapes:", [intr.shape for intr in intrins])
print("Post-rotation matrices shapes:", [post_rot.shape for post_rot in post_rots])
print("Post-translation vectors shapes:", [post_tr.shape for post_tr in post_trans])
print("BDA matrices shape:", bda.shape)
print("---------------------------------------")


# Assuming 'model' is your LSSViewTransformerBEVDepth model
with torch.no_grad():
    rot = rots[0]  # Ensure 'rot' is on the correct device
    tran = trans[0]  # Ensure 'tran' is on the correct device
    intrin = intrins[0]  # Ensure 'intrin' is on the correct device
    post_rot = post_rots[0]  # Ensure 'post_rot' is on the correct device
    post_tran = post_trans[0]  # Ensure 'post_tran' is on the correct device
    output = view_transformer.get_mlp_input(
        rot, tran, intrin, post_rot, post_tran, bda
    )

print("---------------------------------------")
print("mlp_input:", output.shape)
print("---------------------------------------")