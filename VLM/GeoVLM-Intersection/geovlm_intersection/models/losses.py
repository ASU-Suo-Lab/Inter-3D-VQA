from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from geovlm_intersection.data.targets import SPEEDING_RISK_SPEED_THRESHOLD_MPS, SUBTEMPLATE_TO_INDEX

POSITION_3D_LOSS_WEIGHT = 1.0
IMAGE_REF_LOSS_WEIGHT = 0.25
SPEEDING_RISK_POSITIVE_WEIGHT = 113.0 / 257.0


@dataclass(frozen=True)
class GeoVLMLossOutput:
    total_loss: torch.Tensor
    total_loss_sum: torch.Tensor
    active_count: int
    components: dict[str, torch.Tensor]


def compute_structured_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    component_weights: dict[str, float] | None = None,
) -> GeoVLMLossOutput:
    components: dict[str, torch.Tensor] = {}
    loss_weights = {
        "position_3d": POSITION_3D_LOSS_WEIGHT,
        "image_ref": IMAGE_REF_LOSS_WEIGHT,
    }
    if component_weights:
        loss_weights.update(component_weights)

    if "object_type_index" in targets:
        components["object_type"] = F.cross_entropy(outputs["object_type_logits"], targets["object_type_index"])
    if "object_selection_index" in targets:
        components["object_selection"] = F.cross_entropy(
            outputs["object_selection_logits"],
            targets["object_selection_index"],
        )
    if "side_index" in targets:
        components["side"] = F.cross_entropy(outputs["side_logits"], targets["side_index"])
    if "motion_state_index" in targets:
        components["motion_state"] = F.cross_entropy(outputs["motion_state_logits"], targets["motion_state_index"])
    if "risk_reason_index" in targets:
        components["risk_reason"] = F.cross_entropy(outputs["risk_reason_logits"], targets["risk_reason_index"])
    if "intersection_action_index" in targets:
        components["intersection_action"] = F.cross_entropy(
            outputs["intersection_action_logits"],
            targets["intersection_action_index"],
        )
    if "side_action_index" in targets:
        components["side_action"] = F.cross_entropy(outputs["side_action_logits"], targets["side_action_index"])
    if "lane_action_index" in targets:
        components["lane_action"] = F.cross_entropy(outputs["lane_action_logits"], targets["lane_action_index"])
    if "object_action_index" in targets:
        components["object_action"] = F.cross_entropy(outputs["object_action_logits"], targets["object_action_index"])
    if "lane_function_index" in targets:
        components["lane_function"] = F.cross_entropy(outputs["lane_function_logits"], targets["lane_function_index"])
    if "camera_index" in targets:
        components["camera"] = F.cross_entropy(outputs["camera_logits"], targets["camera_index"])
    if "binary_answer" in targets:
        binary_targets = targets["binary_answer"]
        binary_logits = outputs["binary_answer_logit"]
        binary_weights = torch.ones_like(binary_targets)
        if "subtemplate_index" in targets:
            subtemplate_ids = targets["subtemplate_index"].reshape(-1)
            speeding_mask = subtemplate_ids == SUBTEMPLATE_TO_INDEX["4_2_1_speeding_risk"]
            positive_mask = speeding_mask & (binary_targets > 0.5)
            binary_weights = torch.where(
                positive_mask,
                torch.full_like(binary_weights, SPEEDING_RISK_POSITIVE_WEIGHT),
                binary_weights,
            )
        components["binary"] = F.binary_cross_entropy_with_logits(
            binary_logits,
            binary_targets,
            weight=binary_weights,
        )
        if "subtemplate_index" in targets and "speed_value" in targets:
            subtemplate_ids = targets["subtemplate_index"].reshape(-1)
            speeding_mask = subtemplate_ids == SUBTEMPLATE_TO_INDEX["4_2_1_speeding_risk"]
            positive_mask = speeding_mask & (binary_targets > 0.5)
            if positive_mask.any():
                positive_prob = torch.sigmoid(binary_logits[positive_mask])
                speed_value = outputs["speed_value"][positive_mask]
                threshold_gap = torch.relu(
                    speed_value.new_tensor(SPEEDING_RISK_SPEED_THRESHOLD_MPS) - speed_value
                )
                components["binary_speed_consistency"] = (positive_prob * threshold_gap).mean()
    if "count_value" in targets:
        components["count"] = F.smooth_l1_loss(outputs["count_value"], targets["count_value"])
    if "distance_value" in targets:
        components["distance"] = F.smooth_l1_loss(outputs["distance_value"], targets["distance_value"])
    if "speed_value" in targets:
        components["speed"] = F.smooth_l1_loss(outputs["speed_value"], targets["speed_value"])
    if "acceleration_value" in targets:
        components["acceleration"] = F.smooth_l1_loss(outputs["acceleration_value"], targets["acceleration_value"])
    if "position_3d" in targets:
        components["position_3d"] = F.smooth_l1_loss(outputs["position_3d"], targets["position_3d"])
    if "image_ref" in targets:
        components["image_ref"] = F.smooth_l1_loss(outputs["image_ref"], targets["image_ref"])

    if not components:
        raise ValueError("No structured loss components were active for the current sample.")

    total_loss = torch.stack(
        [value * loss_weights.get(name, 1.0) for name, value in components.items()]
    ).sum()
    return GeoVLMLossOutput(
        total_loss=total_loss,
        total_loss_sum=total_loss,
        active_count=1,
        components=components,
    )
