from __future__ import annotations

import os
from pathlib import Path

from llava.train import train as base_train
from traffixqwen_v5.data.dataset import StrictIntersectionV5TrainDataset


def make_supervised_data_module(tokenizer, data_args, model):
    train_dataset = StrictIntersectionV5TrainDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        model=model,
    )
    eval_dataset = None
    train_path = Path(data_args.data_path).resolve()
    candidate_eval_path = Path(os.environ.get("TRAFFIXQWEN_V5_EVAL_DATA_PATH", str(train_path.with_name("val.json")))).resolve()
    if candidate_eval_path.is_file():
        eval_dataset = StrictIntersectionV5TrainDataset(
            tokenizer=tokenizer,
            data_path=str(candidate_eval_path),
            data_args=data_args,
            model=model,
        )

    debug_sample = train_dataset.build_supervision_debug(0)
    base_train.rank0_print(
        "[traffixqwen_v5] supervision sample "
        f"id={debug_sample['id']} supervised_tokens={debug_sample['supervised_token_count']}"
    )
    base_train.rank0_print(f"[traffixqwen_v5] supervision answer: {debug_sample['answer_text']}")
    base_train.rank0_print(f"[traffixqwen_v5] supervision decoded: {debug_sample['supervised_text']}")
    base_train.rank0_print(
        "[traffixqwen_v5] supervision lengths "
        f"input_tokens={debug_sample['input_length']} supervised_ratio={debug_sample['supervised_ratio']:.4f} "
        f"multimodal_cost_length={debug_sample['multimodal_cost_length']}"
    )
    length_values = train_dataset.lengths
    sorted_lengths = sorted(length_values)
    p50_index = len(sorted_lengths) // 2
    p95_index = min(len(sorted_lengths) - 1, int(len(sorted_lengths) * 0.95))
    base_train.rank0_print(
        "[traffixqwen_v5] train length stats "
        f"image_token_cost={train_dataset.image_token_cost} "
        f"min={sorted_lengths[0]} p50={sorted_lengths[p50_index]} "
        f"p95={sorted_lengths[p95_index]} max={sorted_lengths[-1]}"
    )

    if os.environ.get("TRAFFIXQWEN_V5_DEBUG_LOSS") == "1":
        debug_batch_size = max(1, int(os.environ.get("TRAFFIXQWEN_V5_DEBUG_BATCH_SIZE", "1")))
        debug_count = min(debug_batch_size, len(train_dataset))
        debug_rows = [train_dataset.build_supervision_debug(i) for i in range(debug_count)]
        input_lengths = [row["input_length"] for row in debug_rows]
        supervised_counts = [row["supervised_token_count"] for row in debug_rows]
        supervised_ratios = [row["supervised_ratio"] for row in debug_rows]
        multimodal_cost_lengths = [row["multimodal_cost_length"] for row in debug_rows]
        base_train.rank0_print(
            "[traffixqwen_v5] debug_loss batch "
            f"samples={debug_count} input_tokens_min={min(input_lengths)} input_tokens_max={max(input_lengths)} "
            f"supervised_tokens_min={min(supervised_counts)} supervised_tokens_max={max(supervised_counts)} "
            f"supervised_ratio_min={min(supervised_ratios):.4f} supervised_ratio_max={max(supervised_ratios):.4f} "
            f"cost_length_min={min(multimodal_cost_lengths)} cost_length_max={max(multimodal_cost_lengths)}"
        )

    data_collator = base_train.DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

base_train.make_supervised_data_module = make_supervised_data_module


def main() -> None:
    base_train.train()


if __name__ == "__main__":
    main()
