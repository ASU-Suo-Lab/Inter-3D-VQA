import os.path as osp

from omnidrive_v5.config.common import (
    DATA_DIR,
    DEFAULT_QA_JSON,
    DEFAULT_PREDICTION_DIR,
    DEFAULT_WORK_DIR,
    LLM_ROOT,
    REPO_ROOT,
    VLM_ROOT,
    build_model,
    build_test_pipeline,
    class_names,
    input_modality,
)


plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

val_batch_size_per_gpu = 1
workers_per_gpu = 4

work_dir = DEFAULT_WORK_DIR
data_root = REPO_ROOT
val_ann_file = osp.join(DATA_DIR, "infos_val.pkl")
qa_json = DEFAULT_QA_JSON
lane_file = osp.join(VLM_ROOT, "lanelet_align_gui_out", "aligned_centerlines.json")
llm_path = osp.join(REPO_ROOT, "2d_llm")
vision_pretrain = osp.join(REPO_ROOT, "vision_encoder", "eva02_petr_proj.pth")

test_pipeline = build_test_pipeline(llm_path, qa_json)

model = build_model(DEFAULT_PREDICTION_DIR, llm_path)
load_from = vision_pretrain

data = dict(
    samples_per_gpu=val_batch_size_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test=dict(
        samples_per_gpu=val_batch_size_per_gpu,
        type="SunLakesSequentialDatasetV5",
        ann_file=val_ann_file,
        qa_json=qa_json,
        lane_file=lane_file,
        data_root=data_root,
        seq_by="scene_id",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

evaluation = dict(interval=1, pipeline=test_pipeline)
checkpoint_config = None
dist_params = dict(backend="nccl")
log_level = "INFO"
opencv_num_threads = 0
mp_start_method = "fork"
