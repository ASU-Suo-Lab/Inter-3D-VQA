"""Intersection QA Generator V6.

The V6 template specification and natural-selector rules live in
`utils/create_QA_v6_templates.md`.
"""

from __future__ import annotations

import argparse
import os

from _qa_v6_runtime import IntersectionQAGeneratorV6Runtime


SAMPLE_ORDER = [
    "1_1_1_lane_first_object_type",
    "1_1_2_front_neighbor_type",
    "1_1_3_approach_vru_exists",
    "1_2_1_size_bucket",
    "1_3_1_environment",
    "1_3_2_vehicle_signal_state",
    "2_1_1_stopline_distance",
    "2_1_2_ped_to_far_edge",
    "2_2_4_longest_queue_lane",
    "3_1_1_current_motion_state",
    "3_2_1_waypoints",
    "3_3_1_safe_following",
    "3_4_3_primary_risk_subject",
    "4_1_1_overall_state",
    "4_2_2_notable_abnormal",
    "4_3_1_intersection_action",
    "4_3_4_object_action",
]


def main() -> None:
    runtime_cls = IntersectionQAGeneratorV6Runtime

    parser = argparse.ArgumentParser(description="Generate V6 intersection QA pairs.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on source frames.")
    parser.add_argument(
        "--keyframe-fps",
        type=float,
        default=runtime_cls.DEFAULT_KEYFRAME_FPS,
        help="Keyframe sampling rate per scene.",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=runtime_cls.DEFAULT_MAX_PER_TYPE,
        help="Maximum QA pairs to keep per subtemplate for each frame.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for frame-level QA generation. Use 1 to keep serial execution.",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pkl_path = os.path.join(root_dir, "sunlakes_infos_trainval.pkl")
    output_path = os.path.join(root_dir, runtime_cls.DEFAULT_OUTPUT_NAME)

    generator = runtime_cls(pkl_path)
    qa_pairs = generator.generate_dataset(
        output_path,
        max_frames=args.max_frames,
        keyframe_fps=args.keyframe_fps,
        max_per_type=args.max_per_type,
        num_workers=args.num_workers,
    )

    sample_by_template = {}
    for qa in qa_pairs:
        template_id = qa.get("subtemplate")
        if template_id not in sample_by_template:
            sample_by_template[template_id] = qa

    print("\n=== Sample QA Pairs ===")
    shown = 0
    for template_id in SAMPLE_ORDER:
        qa = sample_by_template.get(template_id)
        if qa is None:
            continue
        shown += 1
        print(f"\n{shown}. [{qa['chapter']}] {template_id}")
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")


if __name__ == "__main__":
    main()
