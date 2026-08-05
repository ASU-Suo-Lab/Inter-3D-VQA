"""Intersection QA Generator V6.

The V6 template specification and natural-selector rules live in
`utils/create_QA_v6_templates.md`.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

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


BUCKET_FIELD_BY_TEMPLATE = {
    "1_1_1_lane_first_object_type": "object_type",
    "1_1_2_front_neighbor_type": "rel_dir",
    "1_1_3_approach_vru_exists": "exists",
    "1_1_4_approach_type_count": "object_type",
    "1_2_1_size_bucket": "size_bucket",
    "1_3_1_environment": "weather",
    "1_3_2_vehicle_signal_state": "signal_state",
    "2_1_1_stopline_distance": "object_type",
    "2_1_2_ped_to_far_edge": "crosswalk",
    "2_1_3_participant_distance": "pair_type",
    "2_1_4_nearest_vehicle": "direction",
    "2_2_1_ped_zone": "ped_zone",
    "2_2_2_lane_queue_count": "lane_function",
    "2_2_3_stopline_back_5m_count": "side",
    "2_2_4_longest_queue_lane": "lane_function",
    "2_2_5_crosswalk_blocking": "crosswalk_blocked",
    "3_1_1_current_motion_state": "motion_state",
    "3_1_2_vehicle_maneuver": "maneuver",
    "3_2_1_waypoints": "trajectory",
    "3_2_2_future_region": "future_region",
    "3_3_1_safe_following": "is_safe",
    "3_3_2_likely_long_queue_lane": "lane_function",
    "3_4_1_pair_conflict": "has_conflict",
    "3_4_2_nearest_conflict_participant": "conflict_partner_type",
    "3_4_3_primary_risk_subject": "risk_reason",
    "3_4_4_risk_pattern": "interaction_pattern",
    "4_1_1_overall_state": "overall_state",
    "4_1_2_approach_motion_status": "motion_label",
    "4_1_3_scene_summary": "summary_type",
    "4_1_4_heaviest_traffic_approach": "dominant_side",
    "4_2_1_speeding_risk": "has_speeding_risk",
    "4_2_2_notable_abnormal": "notable_abnormal",
    "4_3_1_intersection_action": "action_state",
    "4_3_2_approach_action": "action_state",
    "4_3_3_lane_action": "action_state",
    "4_3_4_object_action": "action_state",
}


def _normalize_bucket_value(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "None"
    if isinstance(value, dict):
        return "structured"
    if isinstance(value, list):
        return "structured"
    return str(value)


def _infer_bucket(template_id: str, qa: dict) -> str | None:
    structured = qa.get("structured_targets") or {}
    field = BUCKET_FIELD_BY_TEMPLATE.get(template_id)
    if field is not None and field in structured:
        value = structured[field]
        return _normalize_bucket_value(value)
    if field == "trajectory" and "trajectory" in structured:
        return "trajectory"
    if field == "signal_state" and qa.get("placeholder"):
        return "None"
    answer = qa.get("answer")
    if qa.get("placeholder") and answer is None:
        return "None"
    if isinstance(answer, str):
        lowered = answer.strip().lower()
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"
    return None


def _print_bucket_statistics(qa_pairs: list[dict]) -> None:
    bucket_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for qa in qa_pairs:
        template_id = qa.get("subtemplate")
        bucket = _infer_bucket(template_id, qa)
        if bucket is None:
            continue
        bucket_stats[template_id][bucket] += 1
    print("\n=== Template Bucket / State Counts ===")
    for template_id in sorted(bucket_stats):
        print(f"{template_id}:")
        for bucket, count in sorted(bucket_stats[template_id].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {bucket}: {count}")


def _print_template_count_comparison(pre_scene_counts: dict, post_scene_counts: dict, post_balance_counts: dict, post_ratio_counts: dict) -> None:
    print("\n=== Template Counts (Pre-TemporalSuppression -> Post-TemporalSuppression -> Post-Balance -> Post-Ratio) ===")
    all_template_ids = set(pre_scene_counts) | set(post_scene_counts) | set(post_balance_counts) | set(post_ratio_counts)
    for template_id in sorted(all_template_ids):
        print(
            f"{template_id}: "
            f"{pre_scene_counts.get(template_id, 0)} -> "
            f"{post_scene_counts.get(template_id, 0)} -> "
            f"{post_balance_counts.get(template_id, 0)} -> "
            f"{post_ratio_counts.get(template_id, 0)}"
        )


def _print_total_count_comparison(pre_scene_total: int, post_scene_total: int, post_balance_total: int, post_ratio_total: int) -> None:
    print("\n=== Total QA Counts (Pre-TemporalSuppression -> Post-TemporalSuppression -> Post-Balance -> Post-Ratio) ===")
    print(f"{pre_scene_total} -> {post_scene_total} -> {post_balance_total} -> {post_ratio_total}")


def _print_bucket_count_comparison(pre_scene_buckets: dict, post_scene_buckets: dict, post_balance_buckets: dict, post_ratio_buckets: dict) -> None:
    print("\n=== Template Bucket / State Counts (Pre-TemporalSuppression -> Post-TemporalSuppression -> Post-Balance -> Post-Ratio) ===")
    for template_id in sorted(set(pre_scene_buckets) | set(post_scene_buckets) | set(post_balance_buckets) | set(post_ratio_buckets)):
        print(f"{template_id}:")
        pre_scene_template = pre_scene_buckets.get(template_id, {})
        post_scene_template = post_scene_buckets.get(template_id, {})
        post_balance_template = post_balance_buckets.get(template_id, {})
        post_ratio_template = post_ratio_buckets.get(template_id, {})
        for bucket in sorted(set(pre_scene_template) | set(post_scene_template) | set(post_balance_template) | set(post_ratio_template)):
            print(
                f"  {bucket}: "
                f"{pre_scene_template.get(bucket, 0)} -> "
                f"{post_scene_template.get(bucket, 0)} -> "
                f"{post_balance_template.get(bucket, 0)} -> "
                f"{post_ratio_template.get(bucket, 0)}"
            )


def _format_share(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(count / total) * 100:.1f}%"


def _print_share_comparison(
    title: str,
    pre_scene_stats: dict,
    post_scene_stats: dict,
    post_balance_stats: dict,
    post_ratio_stats: dict,
    pre_scene_total: int,
    post_scene_total: int,
    post_balance_total: int,
    post_ratio_total: int,
) -> None:
    print(f"\n=== {title} Share (Pre-TemporalSuppression -> Post-TemporalSuppression -> Post-Balance -> Post-Ratio) ===")
    all_keys = set(pre_scene_stats) | set(post_scene_stats) | set(post_balance_stats) | set(post_ratio_stats)
    for key in sorted(all_keys):
        print(
            f"{key}: "
            f"{_format_share(pre_scene_stats.get(key, 0), pre_scene_total)} -> "
            f"{_format_share(post_scene_stats.get(key, 0), post_scene_total)} -> "
            f"{_format_share(post_balance_stats.get(key, 0), post_balance_total)} -> "
            f"{_format_share(post_ratio_stats.get(key, 0), post_ratio_total)}"
        )


def _print_section_share_comparison(
    pre_scene_section_stats: dict,
    post_scene_section_stats: dict,
    post_balance_section_stats: dict,
    post_ratio_section_stats: dict,
    pre_scene_chapter_stats: dict,
    post_scene_chapter_stats: dict,
    post_balance_chapter_stats: dict,
    post_ratio_chapter_stats: dict,
    section_to_chapter: dict[str, str],
) -> None:
    print("\n=== Section Share (within Chapter) (Pre-TemporalSuppression -> Post-TemporalSuppression -> Post-Balance -> Post-Ratio) ===")
    all_sections = set(pre_scene_section_stats) | set(post_scene_section_stats) | set(post_balance_section_stats) | set(post_ratio_section_stats)
    for section in sorted(all_sections):
        chapter = section_to_chapter.get(section)
        if chapter is None:
            continue
        print(
            f"{section}: "
            f"{_format_share(pre_scene_section_stats.get(section, 0), pre_scene_chapter_stats.get(chapter, 0))} -> "
            f"{_format_share(post_scene_section_stats.get(section, 0), post_scene_chapter_stats.get(chapter, 0))} -> "
            f"{_format_share(post_balance_section_stats.get(section, 0), post_balance_chapter_stats.get(chapter, 0))} -> "
            f"{_format_share(post_ratio_section_stats.get(section, 0), post_ratio_chapter_stats.get(chapter, 0))}"
        )


def _print_subtemplate_share_comparison(
    pre_scene_template_stats: dict,
    post_scene_template_stats: dict,
    post_balance_template_stats: dict,
    post_ratio_template_stats: dict,
    pre_scene_section_stats: dict,
    post_scene_section_stats: dict,
    post_balance_section_stats: dict,
    post_ratio_section_stats: dict,
    template_to_section: dict[str, str],
) -> None:
    print("\n=== Subtemplate Share (within Section) (Pre-TemporalSuppression -> Post-TemporalSuppression -> Post-Balance -> Post-Ratio) ===")
    all_templates = set(pre_scene_template_stats) | set(post_scene_template_stats) | set(post_balance_template_stats) | set(post_ratio_template_stats)
    for template_id in sorted(all_templates):
        section = template_to_section.get(template_id)
        if section is None:
            continue
        print(
            f"{template_id}: "
            f"{_format_share(pre_scene_template_stats.get(template_id, 0), pre_scene_section_stats.get(section, 0))} -> "
            f"{_format_share(post_scene_template_stats.get(template_id, 0), post_scene_section_stats.get(section, 0))} -> "
            f"{_format_share(post_balance_template_stats.get(template_id, 0), post_balance_section_stats.get(section, 0))} -> "
            f"{_format_share(post_ratio_template_stats.get(template_id, 0), post_ratio_section_stats.get(section, 0))}"
        )


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

    with open(output_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    metadata = payload.get("metadata", {})
    template_registry = metadata.get("template_registry", {})
    template_to_section = {
        template_id: spec.get("section")
        for template_id, spec in template_registry.items()
        if spec.get("section")
    }
    section_to_chapter = {
        spec.get("section"): spec.get("chapter")
        for spec in template_registry.values()
        if spec.get("section") and spec.get("chapter")
    }

    pre_scene_template_counts = metadata.get("pre_scene_filter_template_statistics", metadata.get("pre_balance_template_statistics", {}))
    post_scene_template_counts = metadata.get("post_scene_filter_template_statistics", metadata.get("pre_balance_template_statistics", {}))
    pre_scene_chapter_counts = metadata.get("pre_scene_filter_chapter_statistics", metadata.get("pre_balance_chapter_statistics", {}))
    post_scene_chapter_counts = metadata.get("post_scene_filter_chapter_statistics", metadata.get("pre_balance_chapter_statistics", {}))
    pre_scene_section_counts = metadata.get("pre_scene_filter_section_statistics", metadata.get("pre_balance_section_statistics", {}))
    post_scene_section_counts = metadata.get("post_scene_filter_section_statistics", metadata.get("pre_balance_section_statistics", {}))
    pre_template_counts = metadata.get("pre_balance_template_statistics", {})
    pre_chapter_counts = metadata.get("pre_balance_chapter_statistics", {})
    pre_section_counts = metadata.get("pre_balance_section_statistics", {})
    post_balance_template_counts = metadata.get("post_balance_template_statistics", {})
    post_balance_chapter_counts = metadata.get("post_balance_chapter_statistics", {})
    post_balance_section_counts = metadata.get("post_balance_section_statistics", {})
    post_template_counts = metadata.get("post_ratio_template_statistics", metadata.get("template_statistics", {}))
    post_chapter_counts = metadata.get("post_ratio_chapter_statistics", metadata.get("chapter_statistics", {}))
    post_section_counts = metadata.get("post_ratio_section_statistics", metadata.get("section_statistics", {}))
    pre_scene_total_qas = int(metadata.get("pre_scene_filter_total_qas", metadata.get("pre_balance_total_qas", 0)))
    post_scene_total_qas = int(metadata.get("post_scene_filter_total_qas", metadata.get("pre_balance_total_qas", 0)))
    pre_total_qas = int(metadata.get("pre_balance_total_qas", 0))
    post_balance_total_qas = int(metadata.get("post_balance_total_qas", 0))
    post_ratio_total_qas = int(metadata.get("post_ratio_total_qas", metadata.get("total_qas", 0)))
    pre_scene_bucket_counts = metadata.get("pre_scene_filter_template_bucket_statistics", metadata.get("pre_balance_template_bucket_statistics", {}))
    post_scene_bucket_counts = metadata.get("post_scene_filter_template_bucket_statistics", metadata.get("pre_balance_template_bucket_statistics", {}))
    pre_bucket_counts = metadata.get("pre_balance_template_bucket_statistics", {})
    post_balance_bucket_counts = metadata.get("post_balance_template_bucket_statistics", {})
    post_bucket_counts = metadata.get("post_ratio_template_bucket_statistics", metadata.get("template_bucket_statistics", {}))

    if pre_scene_template_counts and post_scene_template_counts and post_balance_template_counts and post_template_counts:
        _print_total_count_comparison(pre_scene_total_qas, post_scene_total_qas, post_balance_total_qas, post_ratio_total_qas)
        _print_template_count_comparison(pre_scene_template_counts, post_scene_template_counts, post_balance_template_counts, post_template_counts)
        _print_share_comparison(
            "Chapter",
            pre_scene_chapter_counts,
            post_scene_chapter_counts,
            post_balance_chapter_counts,
            post_chapter_counts,
            pre_scene_total_qas,
            post_scene_total_qas,
            post_balance_total_qas,
            post_ratio_total_qas,
        )
        _print_section_share_comparison(
            pre_scene_section_counts,
            post_scene_section_counts,
            post_balance_section_counts,
            post_section_counts,
            pre_scene_chapter_counts,
            post_scene_chapter_counts,
            post_balance_chapter_counts,
            post_chapter_counts,
            section_to_chapter,
        )
        _print_subtemplate_share_comparison(
            pre_scene_template_counts,
            post_scene_template_counts,
            post_balance_template_counts,
            post_template_counts,
            pre_scene_section_counts,
            post_scene_section_counts,
            post_balance_section_counts,
            post_section_counts,
            template_to_section,
        )
    else:
        template_counts = {}
        for qa in qa_pairs:
            template_id = qa.get("subtemplate")
            template_counts[template_id] = template_counts.get(template_id, 0) + 1
        print("\n=== Template Counts ===")
        for template_id in sorted(template_counts):
            print(f"{template_id}: {template_counts[template_id]}")

    if pre_scene_bucket_counts and post_scene_bucket_counts and post_balance_bucket_counts and post_bucket_counts:
        _print_bucket_count_comparison(pre_scene_bucket_counts, post_scene_bucket_counts, post_balance_bucket_counts, post_bucket_counts)
    else:
        _print_bucket_statistics(qa_pairs)

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
