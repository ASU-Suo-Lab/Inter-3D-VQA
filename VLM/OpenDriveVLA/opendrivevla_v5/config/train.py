from __future__ import annotations

from opendrivevla_v5.config.common import TRAINING_DEFAULTS


NUM_TRAIN_EPOCHS = TRAINING_DEFAULTS["num_train_epochs"]
MAX_STEPS = TRAINING_DEFAULTS["max_steps"]
PER_DEVICE_TRAIN_BATCH_SIZE = TRAINING_DEFAULTS["per_device_train_batch_size"]
GRADIENT_ACCUMULATION_STEPS = TRAINING_DEFAULTS["gradient_accumulation_steps"]
LEARNING_RATE = TRAINING_DEFAULTS["learning_rate"]
WEIGHT_DECAY = TRAINING_DEFAULTS["weight_decay"]
WARMUP_RATIO = TRAINING_DEFAULTS["warmup_ratio"]
LOGGING_STEPS = TRAINING_DEFAULTS["logging_steps"]
EVAL_STEPS = TRAINING_DEFAULTS["eval_steps"]
SAVE_STEPS = TRAINING_DEFAULTS["save_steps"]
SAVE_TOTAL_LIMIT = TRAINING_DEFAULTS["save_total_limit"]
SEED = TRAINING_DEFAULTS["seed"]
LORA_R = TRAINING_DEFAULTS["lora_r"]
LORA_ALPHA = TRAINING_DEFAULTS["lora_alpha"]
LORA_DROPOUT = TRAINING_DEFAULTS["lora_dropout"]

