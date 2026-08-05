import re


LIDAR_TEMPLATE_FAMILY_NAMES = (
    "unknown",
    "object_identity",
    "count_queue",
    "geometry_distance",
    "spatial_scene",
    "route_waypoint",
    "risk_interaction",
    "motion_state",
)

LIDAR_TEMPLATE_FAMILY_TO_ID = {name: idx for idx, name in enumerate(LIDAR_TEMPLATE_FAMILY_NAMES)}

# Per-family initial (agent_scale, scene_scale, output_scale) for the trainable LiDAR router.
# The values are deliberately mild: they bias known LiDAR-friendly numeric families
# toward object memory without hard disabling any branch.
LIDAR_TEMPLATE_ROUTE_INITIAL_SCALES = (
    (1.00, 1.00, 1.00),  # unknown
    (1.05, 0.95, 1.05),  # object_identity
    (1.30, 0.95, 1.15),  # count_queue
    (1.25, 1.00, 1.15),  # geometry_distance
    (1.10, 1.15, 1.05),  # spatial_scene
    (1.25, 1.05, 0.80),  # route_waypoint
    (1.10, 1.20, 1.00),  # risk_interaction
    (1.00, 1.00, 0.70),  # motion_state
)

_SUBTEMPLATE_PATTERN = re.compile(r"Subtemplate:\s*([A-Za-z0-9_]+)")


def extract_subtemplate_from_prompt(prompt: list[dict[str, str]]) -> str | None:
    for message in reversed(prompt):
        if message.get("role") != "user":
            continue
        match = _SUBTEMPLATE_PATTERN.search(message.get("content", ""))
        if match is not None:
            return match.group(1)
    return None


def lidar_template_family_id_from_subtemplate(subtemplate: str | None) -> int:
    value = (subtemplate or "").lower()
    if not value:
        return LIDAR_TEMPLATE_FAMILY_TO_ID["unknown"]

    if "speed" in value or "accel" in value or "motion_state" in value or "maneuver" in value:
        return LIDAR_TEMPLATE_FAMILY_TO_ID["motion_state"]
    if "risk" in value or "conflict" in value or "abnormal" in value or "safe_following" in value:
        return LIDAR_TEMPLATE_FAMILY_TO_ID["risk_interaction"]
    if "count" in value or "queue" in value:
        return LIDAR_TEMPLATE_FAMILY_TO_ID["count_queue"]
    if "distance" in value or "nearest" in value or "far_edge" in value or "stopline" in value:
        return LIDAR_TEMPLATE_FAMILY_TO_ID["geometry_distance"]
    if "waypoint" in value or "future_region" in value or "action" in value:
        return LIDAR_TEMPLATE_FAMILY_TO_ID["route_waypoint"]
    if (
        "lane" in value
        or "zone" in value
        or "crosswalk" in value
        or "blocking" in value
        or "overall_state" in value
        or "scene_summary" in value
        or "flow_imbalance" in value
        or "signal_state" in value
        or "weather" in value
    ):
        return LIDAR_TEMPLATE_FAMILY_TO_ID["spatial_scene"]
    if value.startswith("1_"):
        return LIDAR_TEMPLATE_FAMILY_TO_ID["object_identity"]

    return LIDAR_TEMPLATE_FAMILY_TO_ID["unknown"]


def lidar_template_family_name_from_id(family_id: int | None) -> str:
    if family_id is None or family_id < 0 or family_id >= len(LIDAR_TEMPLATE_FAMILY_NAMES):
        return "unknown"
    return LIDAR_TEMPLATE_FAMILY_NAMES[family_id]


def lidar_template_family_name_from_subtemplate(subtemplate: str | None) -> str:
    return lidar_template_family_name_from_id(lidar_template_family_id_from_subtemplate(subtemplate))
