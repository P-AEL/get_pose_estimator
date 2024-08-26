# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.apis.inferencers import MMPoseInferencer

filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    # standard bbox_thr=0.1, nms_thr=0.65
    rtmo=dict(bbox_thr=0.2, nms_thr=0.65, pose_based_nms=True),
)

def get_init_args(pose2d, device, scope = "mmpose", det_model = None, det_weights = None, det_cat_ids = 0, pose3d = None, pose3d_weights = None, show_progress = False):
    return {'pose2d': pose2d, 'pose2d_weights': None, 'scope': scope, 'device': device, 'det_model': det_model,
        'det_weights': det_weights, 'det_cat_ids': det_cat_ids, 'pose3d': pose3d, 'pose3d_weights': pose3d_weights,
        'show_progress': show_progress}

def get_call_args_rtmo(black_background = False, draw_bbox = False, draw_heatmap = False, bbox_thr = 0.2,
                        nms_thr = 0.65, pose_based_nms = True, kpt_thr = 0.8, tracking_thr = 0.8, use_oks_tracking = False,
                        disable_norm_pose_2d =  False, disable_rebase_keypoint = False, num_instances = 1, radius = 3,
                        thickness = 1, skeleton_style = "mmpose"):
    call_args = {"black_background": black_background, "draw_bbox": draw_bbox, "draw_heatmap": draw_heatmap, 
                 "bbox_thr": bbox_thr, "nms_thr": nms_thr, "pose_based_nms": pose_based_nms, "kpt_thr": kpt_thr, "tracking_thr": tracking_thr, 
                 "use_oks_tracking": use_oks_tracking, "disable_norm_pose_2d": disable_norm_pose_2d, "disable_rebase_keypoint": disable_rebase_keypoint,
                 "num_instances": num_instances, "radius": radius, "thickness": thickness, "skeleton_style": skeleton_style}
    return call_args

def get_rtmo_pose_estimator(pose2d, device):
    init_args = get_init_args(pose2d, device)
    return MMPoseInferencer(**init_args)

