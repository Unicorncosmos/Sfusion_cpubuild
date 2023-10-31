from InternImage.internimage import InternImage
from LSSViewtransformer.cpulss import LSSViewTransformerBEVDepth


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

if pretrained:
  intern_image_model.init_weights()


#-----------------------Transformer_build--------------------------------
view_transformer = LSSViewTransformerBEVDepth(
    grid_config=grid_config,
    input_size=input_size,
    in_channels=512,
    out_channels=128,
    accelerate=False,  # Set to True if using acceleration
    depthnet_cfg=dict(use_dcn=False),  # Adjust as needed
    downsample = 16,  # Additional depthnet configuration if needed
) 
from mmengine.config import Config
import torch
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet.utils import compat_cfg
import mmcv

def main():
    
    def prepare_inputs(inputs,num_frame):
        # split the inputs into each frame
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
    cfg = Config.fromfile(r"C:\Users\GOKULNATH\Downloads\config.py")

    cfg.model.pretrained = None

    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]
    

    # Build the dataset
    dataset = build_dataset(cfg.data.test)

    # Convert dict_keys to a list for creating the data loader
    data_loader_cfg = {
        **cfg.data.test_dataloader,
        'samples_per_gpu': 1,
        'workers_per_gpu': 0,  # Set to 0 to use a single worker (no multiprocessing)
        'dist': False,
        'shuffle': False,
    }

    data_loader = build_dataloader(dataset, **data_loader_cfg)
    prog_bar = mmcv.ProgressBar(len(dataset))   


    for i, data in enumerate(data_loader):
        with torch.no_grad():
            inputs = [t for t in data['img_inputs'][0]]
            imgs_meta = data['img_metas'][0].data[0]
            data['img_inputs'][0] = [t for t in data['img_inputs'][0]]
            data['radar_feat'] = data['radar_feat'][0].data
            data['img_metas'] = data['img_metas'][0].data
            # print("INPUTS:", inputs)
            imgs, rots, trans, intrins, post_rots, post_trans, bda = \
            prepare_inputs(inputs,2)
            # print("imgs, rots, trans, intrins, post_rots, post_trans, bda:",imgs, rots, trans, intrins, post_rots, post_trans, bda)
            bev_feat_list = []
            for img, rot, tran, intrin, post_rot, post_tran in zip(
                        imgs, rots, trans, intrins, post_rots, post_trans):

                        B, N, C, imH, imW = img.shape
                    
                        img = img.view(B * N, C, imH, imW)
                        backbone_out = intern_image_model(img)
                        mlp_input = view_transformer.get_mlp_input(
                        rots[0], trans[0], intrin, post_rot, post_tran, bda)
                        print("MLP_INPUT:", mlp_input)
                        print("BACKBONE_OUT[0]:", backbone_out[0].shape)
                        print("BACKBONE_OUT[1]:", backbone_out[1].shape)
                        break

if __name__ == '__main__':
    main()
