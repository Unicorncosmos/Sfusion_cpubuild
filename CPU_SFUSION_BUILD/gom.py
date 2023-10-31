from InternImage.internimage import InternImage
from LSSViewtransformer.cpulss import LSSViewTransformerBEVDepth

# data_config = {
#     'cams': [
#         'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
#         'CAM_BACK', 'CAM_BACK_RIGHT'
#     ],
#     'Ncams':
#     6,
#     'input_size': (640, 1600),
#     'src_size': (900, 1600),
#     # Augmentation
#     'resize': (-0.06, 0.11),
#     'rot': (-5.4, 5.4),
#     'flip': True,
#     'crop_h': (0.0, 0.0),
#     'resize_test': 0.00,
# }

# grid_config = {
#     'x': [-51.2, 51.2, 0.4],                                                # 1
#     'y': [-51.2, 51.2, 0.4],                                                # 2
#     'z': [-5, 3, 8],
#     'depth': [1.0, 60.0, 0.5],
# }
# voxel_size = [0.1, 0.1, 0.2]

# numC_Trans = 128

# multi_adj_frame_id_cfg = (1, 8, 8, 1)
# num_adj = len(range(
#     multi_adj_frame_id_cfg[0],
#     multi_adj_frame_id_cfg[1]+multi_adj_frame_id_cfg[2]+1,
#     multi_adj_frame_id_cfg[3]
# ))
# out_size_factor = 4
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# radar_cfg = {
#     'bbox_num': 100,
#     'radar_fusion_type': "medium_fusion",  # in ['post_fusion', 'medium_fusion']
#     'voxel_size': voxel_size,
#     'out_size_factor': out_size_factor,
#     'point_cloud_range': point_cloud_range,
#     'grid_config': grid_config,
#     'norm_bbox': True,  
#     'pc_roi_method': 'pillars',
#     'img_feats_bbox_dims': [1, 1, 0.5],
#     'pillar_dims': [0.4, 0.4, 0.1],
#     'pc_feat_name': ['pc_x', 'pc_y', 'pc_vx', 'pc_vy'],
#     'hm_to_box_ratio': 1.0,
#     'time_debug': False,
#     'radar_head_task': [
#             dict(num_class=1, class_names=['car']),
#             dict(num_class=2, class_names=['truck', 'construction_vehicle']),
#             dict(num_class=2, class_names=['bus', 'trailer']),
#             dict(num_class=1, class_names=['barrier']),
#             dict(num_class=2, class_names=['motorcycle', 'bicycle']),
#         ]
# }

# input_size=data_config['input_size']
# in_channels=512
# out_channels=128
# depthnet_cfg=dict(use_dcn=False)
# downsample=16
#-----------backbone_build----------------------------------------------

pretrained = r"C:\Users\GOKULNATH\Downloads\CPU_SFUSION_BUILD\backbone.pth" 
intern_image_model = InternImage(
        core_op='DCNv3_pytorch',
        channels=112,
        depths=[4, 4, 21, 4],
        groups=[7, 14, 28, 56],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=True,
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        init_cfg= pretrained
)
import torch
if pretrained:
  
  intern_image_model.init_weights()

input_data = torch.randn(6, 3, 640, 1600)

# Perform inference
with torch.no_grad():
    output_features = intern_image_model(input_data)
    # print(output_features)
    out0 = output_features[2].cpu().numpy()
    out1 = output_features[3].cpu().numpy()
    print("InternImage_Backbone_out")
    print("------------------------")
    print("backbone_out0",out0.shape)
    print("backbone_out1",out1.shape)



