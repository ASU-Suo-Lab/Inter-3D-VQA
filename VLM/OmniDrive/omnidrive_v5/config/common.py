import os.path as osp
import os

from omnidrive_v5.utils.paths import DEFAULT_DATASET_VERSION, resolve_dataset_version_paths


REPO_ROOT = "/home/suolab/LLM/VLM/OmniDrive"
VLM_ROOT = "/home/suolab/LLM/VLM"
LLM_ROOT = "/home/suolab/LLM"
_RESOLVED_PATHS = resolve_dataset_version_paths(
    os.getenv("OMNIDRIVE_DATASET_VERSION", DEFAULT_DATASET_VERSION),
    qa_json=os.getenv("OMNIDRIVE_QA_JSON"),
    evaluator=os.getenv("OMNIDRIVE_EVALUATOR"),
    prepared_dir=os.getenv("OMNIDRIVE_PREPARED_DIR"),
    work_dir=os.getenv("OMNIDRIVE_WORK_DIR"),
)
DATA_DIR = str(_RESOLVED_PATHS["prepared_dir"])
DEFAULT_WORK_DIR = str(_RESOLVED_PATHS["work_dir"])
DEFAULT_QA_JSON = str(_RESOLVED_PATHS["qa_json"])
DEFAULT_EVALUATOR = str(_RESOLVED_PATHS["evaluator"])
DEFAULT_PREDICTION_DIR = osp.join(DEFAULT_WORK_DIR, "predictions")

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
collect_keys = ["lidar2img", "intrinsics", "extrinsics", "timestamp"]
input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True)
ida_aug_conf = dict(
    resize_lim=(0.37, 0.45),
    final_dim=(320, 640),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=False,
)


def build_model(save_path, llm_path):
    return dict(
        type="Petr3D",
        save_path=save_path,
        use_grid_mask=True,
        frozen=True,
        use_lora=False,
        tokenizer=llm_path,
        lm_head=llm_path,
        reset_memory_on_new_scene=True,
        img_backbone=dict(
            type="EVAViT",
            img_size=640,
            patch_size=16,
            window_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4 * 2 / 3,
            window_block_indexes=(
                list(range(0, 2))
                + list(range(3, 5))
                + list(range(6, 8))
                + list(range(9, 11))
                + list(range(12, 14))
                + list(range(15, 17))
                + list(range(18, 20))
                + list(range(21, 23))
            ),
            qkv_bias=True,
            drop_path_rate=0.3,
            flash_attn=False,
            with_cp=True,
            frozen=False,
        ),
        map_head=dict(
            type="PETRHeadM",
            num_classes=1,
            in_channels=1024,
            out_dims=4096,
            memory_len=600,
            with_mask=True,
            topk_proposals=300,
            num_lane=1800,
            num_lanes_one2one=300,
            k_one2many=5,
            lambda_one2many=1.0,
            num_extra=256,
            n_control=11,
            pc_range=point_cloud_range,
            code_weights=[1.0, 1.0],
            with_ego_pos=False,
            use_ego_motion=False,
            transformer=dict(
                type="PETRTemporalTransformer",
                input_dimension=256,
                output_dimension=256,
                num_layers=6,
                embed_dims=256,
                num_heads=8,
                feedforward_dims=2048,
                dropout=0.1,
                with_cp=True,
                flash_attn=False,
            ),
            train_cfg=dict(
                assigner=dict(
                    type="LaneHungarianAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=1.5),
                    reg_cost=dict(type="LaneL1Cost", weight=0.02),
                    iou_cost=dict(type="IoUCost", weight=0.0),
                )
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.5),
            loss_bbox=dict(type="L1Loss", loss_weight=0.02),
            loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.0),
        ),
        pts_bbox_head=dict(
            type="StreamPETRHead",
            num_classes=10,
            in_channels=1024,
            out_dims=4096,
            num_query=600,
            with_mask=True,
            memory_len=600,
            topk_proposals=300,
            num_propagated=300,
            num_extra=256,
            n_control=11,
            match_with_velo=False,
            scalar=10,
            noise_scale=1.0,
            dn_weight=1.0,
            split=0.75,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            with_ego_pos=False,
            use_ego_motion=False,
            use_can_bus=False,
            transformer=dict(
                type="PETRTemporalTransformer",
                input_dimension=256,
                output_dimension=256,
                num_layers=6,
                embed_dims=256,
                num_heads=8,
                feedforward_dims=2048,
                dropout=0.1,
                with_cp=True,
                flash_attn=False,
            ),
            bbox_coder=dict(
                type="NMSFreeCoder",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                voxel_size=voxel_size,
                num_classes=10,
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0),
        ),
        train_cfg=dict(
            pts=dict(
                grid_size=[512, 512, 1],
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                out_size_factor=4,
                assigner=dict(
                    type="HungarianAssigner3D",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                    iou_cost=dict(type="IoUCost", weight=0.0),
                    pc_range=point_cloud_range,
                ),
            )
        ),
    )


def build_train_pipeline(llm_path, training):
    pipeline = [
        dict(type="LoadMultiViewImageFromFiles", to_float32=True),
        dict(type="ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, with_2d=False, training=training),
        dict(type="ResizeMultiview3D", img_scale=(640, 640), keep_ratio=False, multiscale_mode="value"),
        dict(type="NormalizeMultiviewImage", **img_norm_cfg),
        dict(type="PadMultiViewImage", size_divisor=32),
        dict(type="LoadIntersectionQASingleTrain", tokenizer=llm_path, max_length=512),
        dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys + ["prev_exists"]),
        dict(
            type="Collect3D",
            keys=["lane_pts", "input_ids", "vlm_labels", "gt_bboxes_3d", "gt_labels_3d", "img", "prev_exists"]
            + collect_keys,
            meta_keys=(
                "sample_idx",
                "scene_id",
                "scene_token",
                "timestamp",
                "question_ids",
                "qa_categories",
                "train_questions",
                "filename",
                "ori_shape",
                "img_shape",
                "pad_shape",
                "scale_factor",
                "flip",
                "img_norm_cfg",
                "gt_bboxes_3d",
                "gt_labels_3d",
            ),
        ),
    ]
    return pipeline


def build_test_pipeline(llm_path, qa_json):
    return [
        dict(type="LoadMultiViewImageFromFiles", to_float32=True),
        dict(type="ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, with_2d=False, training=False),
        dict(type="ResizeMultiview3D", img_scale=(640, 640), keep_ratio=False, multiscale_mode="value"),
        dict(type="NormalizeMultiviewImage", **img_norm_cfg),
        dict(type="PadMultiViewImage", size_divisor=32),
        dict(type="LoadIntersectionQATest", qa_json=qa_json, tokenizer=llm_path, max_length=512),
        dict(
            type="MultiScaleFlipAug3D",
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type="PETRFormatBundle3D",
                    collect_keys=collect_keys + ["prev_exists"],
                    class_names=class_names,
                    with_label=False,
                ),
                dict(
                    type="Collect3D",
                    keys=["input_ids", "img", "prev_exists"] + collect_keys,
                    meta_keys=(
                        "sample_idx",
                        "scene_id",
                        "timestamp",
                        "vlm_labels",
                        "question_ids",
                        "box_type_3d",
                        "box_mode_3d",
                        "filename",
                        "ori_shape",
                        "img_shape",
                        "pad_shape",
                        "scale_factor",
                        "flip",
                        "img_norm_cfg",
                    ),
                ),
            ],
        ),
    ]
