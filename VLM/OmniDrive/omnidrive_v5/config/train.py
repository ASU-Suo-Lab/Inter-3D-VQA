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
    build_train_pipeline,
    class_names,
    input_modality,
)


plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

num_gpus = 4
train_batch_size_per_gpu = 3
val_batch_size_per_gpu = 1
workers_per_gpu = 4
train_num_samples = 22104
val_num_samples = 306
val_loss_num_samples = 2285
global_batch_size = num_gpus * train_batch_size_per_gpu
num_iters_per_epoch = (train_num_samples + global_batch_size - 1) // global_batch_size
num_epochs = 3
max_iters = num_epochs * num_iters_per_epoch
warmup_iters = 100
logging_steps = 100
eval_steps = 400
save_steps = 400
metric_for_best_model = "eval_loss"
greater_is_better = False
max_keep_ckpts = 1
plot_loss = True

work_dir = DEFAULT_WORK_DIR
data_root = REPO_ROOT
train_ann_file = osp.join(DATA_DIR, "infos_train.pkl")
val_ann_file = osp.join(DATA_DIR, "infos_val.pkl")
qa_json = DEFAULT_QA_JSON
lane_file = osp.join(VLM_ROOT, "lanelet_align_gui_out", "aligned_centerlines.json")
llm_path = osp.join(REPO_ROOT, "2d_llm")
vision_pretrain = osp.join(REPO_ROOT, "vision_encoder", "eva02_petr_proj.pth")

train_pipeline = build_train_pipeline(llm_path, training=True)
val_loss_pipeline = build_train_pipeline(llm_path, training=False)
test_pipeline = build_test_pipeline(llm_path, qa_json)

model = build_model(DEFAULT_PREDICTION_DIR, llm_path)
load_from = vision_pretrain

data = dict(
    samples_per_gpu=train_batch_size_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type="SunLakesSequentialQATrainDatasetV5",
        ann_file=train_ann_file,
        qa_json=qa_json,
        lane_file=lane_file,
        data_root=data_root,
        seq_by="scene_id",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d="LiDAR",
    ),
    val_loss=dict(
        type="SunLakesSequentialQATrainDatasetV5",
        ann_file=val_ann_file,
        qa_json=qa_json,
        lane_file=lane_file,
        data_root=data_root,
        seq_by="scene_id",
        pipeline=val_loss_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d="LiDAR",
        samples_per_gpu=val_batch_size_per_gpu,
    ),
    val=dict(
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
    shuffler_sampler=dict(type="DistributedSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

optimizer = dict(
    constructor="LearningRateDecayOptimizerConstructor",
    type="AdamW",
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    paramwise_cfg=dict(
        decay_rate=0.9,
        head_decay_rate=4.0,
        lm_head_decay_rate=0.1,
        decay_type="vit_wise",
        num_layers=24,
    ),
)

optimizer_config = dict(type="Fp16OptimizerHook", loss_scale="dynamic", grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=warmup_iters,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type="IterBasedRunner", max_iters=max_iters)
find_unused_parameters = False
ddp_static_graph = True

validation_loss = dict(
    interval=eval_steps,
    checkpoint_subdir="checkpoints",
    save_best=True,
    save_last=True,
    metric_key=metric_for_best_model,
    rule="greater" if greater_is_better else "less",
    iters_per_epoch=num_iters_per_epoch,
    priority="LOW",
)

checkpoint_config = None

log_config = dict(
    interval=logging_steps,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)

custom_hooks = [
    dict(
        type="LossHistoryHook",
        logging_steps=logging_steps,
        log_subdir="logs",
        plot_subdir="plots",
        iters_per_epoch=num_iters_per_epoch,
        priority="LOWEST",
    ),
    dict(
        type="TrainingProgressBarHook",
        iters_per_epoch=num_iters_per_epoch,
        total_epochs=num_epochs,
        refresh_interval=1,
        priority="LOWEST",
    ),
]

evaluation = dict(interval=save_steps, pipeline=test_pipeline)

dist_params = dict(backend="nccl")
log_level = "INFO"
workflow = [("train", 1)]
opencv_num_threads = 0
mp_start_method = "fork"
resume_from = None
