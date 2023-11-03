from CPU_SFUSION_BUILD.InternImage.internimage import InternImage
from CPU_SFUSION_BUILD.LSSViewtransformer.cpulss import LSSViewTransformerBEVDepth
from Radar_processing_pipelines import get_valid_radar_feat
from mmdet3d.core import bbox3d2result
import sys
import onnxruntime
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ]
common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))
# this function is taken from the mmdet3d library mmdet3d/models/detectors/bevdet.py
# this function is used to deserialize and serialized the output of the radar head
from bboxcoder import CenterPointBBoxCoder
from Centerheads import CenterPointBBoxCoder
pc_range=point_cloud_range[:2]
post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
max_num=500
score_threshold=0.1
voxel_size = [0.1, 0.1, 0.2]
out_size_factor=4,
voxel_size=voxel_size[:2],
code_size=9
bbox_coder = CenterPointBBoxCoder(pc_range,out_size_factor,voxel_size,max_num,code_size)
norm_bbox=True

heads = CenterPointBBoxCoder(tasks,bbox_coder,common_heads,norm_bbox)

def radar_head_result_serialize(self, outs):
    outs_ = []
    for out in outs:
        for key in ['sec_reg', 'sec_rot', 'sec_vel']:
            outs_.append(out[0][key])
    return outs_

def radar_head_result_deserialize(self, outs):
    outs_ = []
    keys = ['sec_reg', 'sec_rot', 'sec_vel']
    for head_id in range(len(outs) // 3):
        outs_head = [dict()]
        for kid, key in enumerate(keys):
            outs_head[0][key] = outs[head_id * 3 + kid]
        outs_.append(outs_head)
    return outs_

def pts_head_result_serialize(self, outs):
    outs_ = []
    for out in outs:
        for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
            outs_.append(out[0][key])
    return outs_

def pts_head_result_deserialize(self, outs):
    outs_ = []
    keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
    for head_id in range(len(outs) // 6):
        outs_head = [dict()]
        for kid, key in enumerate(keys):
            outs_head[0][key] = outs[head_id * 6 + kid]
        outs_.append(outs_head)
    return outs_


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
                        print("BACKBONE_OUT[0]:", backbone_out[2].shape)
                        print("BACKBONE_OUT[1]:", backbone_out[3].shape)
                        cfg.model.train_cfg = None
                        sess1_path =  'hvdet_stage1.onnx'
                        sess1_1_path = 'hvdet_stage1_1.onnx'
                        sess2_path = 'hvdet_stage2.onnx'
                        sess3_path = 'hvdet_stage3.onnx'
                        session1 = onnxruntime.InferenceSession(sess1_path, providers=['CPUExecutionProvider'])
                        session1_1 = onnxruntime.InferenceSession(sess1_1_path, providers=['CPUExecutionProvider'])
                        session2 = onnxruntime.InferenceSession(sess2_path, providers=['CPUExecutionProvider'])
                        session3 = onnxruntime.InferenceSession(sess3_path, providers=['CPUExecutionProvider'])
                            
                        sess1_out_img_feat, sess1_out_depth, sess1_out_tran_feat = session1.run(['img_feat', 'depth', 'tran_feat'], 
                                                                                        {
                                                                                            'backbone_out0':backbone_out[2].cpu().numpy(),
                                                                                            'backbone_out1':backbone_out[3].cpu().numpy(),
                                                                                            'mlp_input':mlp_input.cpu().numpy(),
                                                                                    #     # dynamic_axes={'backbone_in':[0], 'backbone_out':[0]})
                                                                                        })


                        sess1_out_img_feat = torch.tensor(sess1_out_img_feat).to(rot.device)
                        sess1_out_depth = torch.tensor(sess1_out_depth).to(rot.device)
                        sess1_out_tran_feat = torch.tensor(sess1_out_tran_feat).to(rot.device)
                        inputs = [sess1_out_img_feat, rot, tran, intrin, post_rot, post_tran, bda, mlp_input]
                        bev_feat, _ = view_transformer.view_transform(inputs, sess1_out_depth, sess1_out_tran_feat)
                
                        sess1_1_bev_feat = session1_1.run(['out_bev_feat'],
                                                        {'bev_feat': bev_feat.cpu().numpy()})
                        bev_feat_list.append(sess1_1_bev_feat[0])
                        multi_bev_feat = np.concatenate(bev_feat_list, axis=1)

                        output_names=['bev_feat'] + [f'output_{j}' for j in range(36)]
                        sess2_out = session2.run(output_names, 
                                            {
                                            'multi_bev_feat':multi_bev_feat,
                                            }) 
                        for i in range(len(sess2_out)):
                            sess2_out[i] = torch.tensor(sess2_out[i]).cuda()
                        bev_feat = sess2_out[0]
                        pts_outs = sess2_out[1:]


                        pts_out_dict = pts_head_result_deserialize(pts_outs)
                        radar_pc = data['radar_feat'][0]
        
                        radar_feat = get_valid_radar_feat(pts_out_dict, radar_pc, cfg.radar_cfg)
                        sec_feats = torch.cat([bev_feat, radar_feat], 1) 

                        output_names=[f'radar_out_{j}' for j in range(15)]

                        sess3_radar_out=session3.run(output_names, 
                                            {
                                            'sec_feat':sec_feats.cpu().numpy(),
                                            }) 
                        for i in range(len(sess3_radar_out)):
                            sess3_radar_out[i] = torch.tensor(sess3_radar_out[i]).to(pts_outs[0].device)
                        pts_outs = pts_head_result_deserialize(pts_outs)
                        sec_outs=radar_head_result_deserialize(sess3_radar_out)
    
                        for task_ind, task_out in enumerate(sec_outs):
                            ori_task_out = pts_outs[task_ind][0]
                            sec_task_out = task_out[0]
                            for k, v in ori_task_out.items():
                                sec_k = 'sec_' + k
                                if sec_k in sec_task_out.keys() and k != 'heatmap':
                                    pts_outs[task_ind][0][k] = sec_task_out[sec_k]
                        out_bbox_list = [dict() for _ in range(len(imgs_meta))]
                        bbox_list = heads.get_bboxes(
                                pts_outs, imgs_meta, rescale=False)
                        bbox_pts = [
                                bbox3d2result(bboxes, scores, labels)
                                for bboxes, scores, labels in bbox_list
                            ]

                        for result_dict, pts_bbox in zip(out_bbox_list, bbox_pts):
                                    result_dict['pts_bbox'] = pts_bbox
                        results.extend(out_bbox_list)
                        batch_size = len(out_bbox_list)
                        for _ in range(batch_size):
                                prog_bar.update()
                        mmcv.dump(results, args.out)                        

if __name__ == '__main__':
    main()

