"""Intersection QA Generator V5.

The V5 template specification and geometry definitions live in
`utils/create_QA_v5_templates.md`.
"""

from __future__ import annotations

import argparse
import os

from _qa_v5_runtime import IntersectionQAGeneratorV5Runtime


SAMPLE_ORDER = [
    "1_1_1_fine_type",
    "1_1_4_relative_neighbor_type",
    "1_1_2_side_exists",
    "1_2_1_size_bucket",
    "2_1_1_stopline_distance",
    "2_2_2_ped_zone",
    "3_1_1_current_motion_state",
    "3_1_2_vehicle_maneuver",
    "3_2_2_future_region",
    "3_2_3_waypoints",
    "3_3_2_likely_long_queue_lane",
    "3_4_3_primary_risk_subject",
    "4_1_1_overall_state",
    "4_2_2_notable_abnormal",
    "4_3_1_intersection_action",
]


def main() -> None:
    runtime_cls = IntersectionQAGeneratorV5Runtime

    parser = argparse.ArgumentParser(description="Generate V5 intersection QA pairs.")
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
        default=16,
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
