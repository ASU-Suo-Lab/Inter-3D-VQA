from __future__ import annotations

from dataclasses import dataclass

from geovlm_intersection.data.targets import (
    CAMERA_VOCAB,
    GeoVLMSupervision,
    INTERSECTION_ACTION_VOCAB,
    LANE_ACTION_VOCAB,
    LANE_FUNCTION_VOCAB,
    MOTION_STATE_VOCAB,
    OBJECT_ACTION_VOCAB,
    OBJECT_TYPE_VOCAB,
    RISK_REASON_VOCAB,
    SIDE_ACTION_VOCAB,
    SIDE_VOCAB,
    SPEEDING_RISK_SPEED_THRESHOLD_MPS,
    denormalize_image_ref,
    denormalize_position_3d,
)
from geovlm_intersection.models.semantic_decoder import (
    normalize_answer_text,
    uses_decoder_final_prediction,
    uses_hybrid_structured_render,
)


ACTION_STATE_TO_SENTENCE = {
    "CONFLICT_SUPPRESSION": "Traffic should reduce aggressive movement and prioritize conflict avoidance.",
    "FLOW_CALMING": "Traffic should move more calmly and with reduced speed.",
    "FLOW_STABLE": "Traffic should continue in a coordinated and orderly way.",
    "LANE_CLEARANCE_MAINTENANCE": "Traffic should maintain clearance and avoid tight local conflicts.",
    "LANE_GENERAL_ORDER": "Traffic should proceed in an orderly manner.",
    "LANE_PREPARE_TO_STOP": "Traffic should prepare to stop and avoid pressing forward.",
    "LANE_QUEUE_PRESERVATION": "Traffic should preserve queue order and avoid disruption.",
    "LANE_SPEED_REDUCTION": "Traffic should reduce speed and proceed more conservatively.",
    "QUEUE_MANAGEMENT": "Traffic should proceed in a more orderly and tightly managed way.",
    "SIDE_CLEARANCE_PROTECTION": "Traffic should keep safer local spacing.",
    "SIDE_CROSSING_AWARENESS": "Traffic should yield clearly to crossing activity.",
    "SIDE_GENERAL_CAUTION": "Traffic should proceed cautiously and remain orderly.",
    "SIDE_QUEUE_STABILIZATION": "Traffic should stabilize queue movement and avoid unnecessary disruption.",
    "SIDE_SPEED_MODERATION": "Traffic should moderate speed and maintain safer spacing.",
}

OBJECT_ACTION_SUFFIX = {
    "OBJECT_PREPARE_TO_STOP": "prepare to stop",
    "OBJECT_PROCEED_CAUTIOUSLY": "proceed cautiously",
    "OBJECT_SLOW_DOWN": "slow down and keep safer local spacing",
    "OBJECT_YIELD_NOW": "yield now",
}

ACTION_STATE_ALLOWED_BY_SUBTEMPLATE = {
    "4_3_1_intersection_action": tuple(INTERSECTION_ACTION_VOCAB),
    "4_3_2_side_action": tuple(SIDE_ACTION_VOCAB),
    "4_3_3_lane_action": tuple(LANE_ACTION_VOCAB),
    "4_3_4_object_action": tuple(OBJECT_ACTION_VOCAB),
}

SEMANTIC_CANONICAL_SENTENCE_MAP = {
    "4_3_1_intersection_action": tuple(
        ACTION_STATE_TO_SENTENCE[action_state] for action_state in INTERSECTION_ACTION_VOCAB
    ),
    "4_3_2_side_action": tuple(
        ACTION_STATE_TO_SENTENCE[action_state] for action_state in SIDE_ACTION_VOCAB
    ),
    "4_3_3_lane_action": tuple(
        ACTION_STATE_TO_SENTENCE[action_state] for action_state in LANE_ACTION_VOCAB
    ),
}


@dataclass(frozen=True)
class DecodedStructuredPrediction:
    subtemplate: str
    object_type: str | None = None
    object_type_raw: str | None = None
    side: str | None = None
    side_raw: str | None = None
    motion_state: str | None = None
    motion_state_raw: str | None = None
    risk_reason: str | None = None
    risk_reason_raw: str | None = None
    action_state: str | None = None
    action_state_raw_top1: str | None = None
    action_state_allowed_set: tuple[str, ...] = ()
    action_state_family_mismatch: bool = False
    lane_function: str | None = None
    lane_function_raw: str | None = None
    binary_answer: bool | None = None
    binary_answer_raw: bool | None = None
    count_value: float | None = None
    count_value_raw: float | None = None
    distance_value: float | None = None
    distance_value_raw: float | None = None
    speed_value: float | None = None
    speed_value_raw: float | None = None
    acceleration_value: float | None = None
    acceleration_value_raw: float | None = None
    position_3d: tuple[float, float] | None = None
    position_3d_raw: tuple[float, float] | None = None
    camera_name: str | None = None
    camera_name_raw: str | None = None
    image_ref: tuple[float, float] | None = None
    image_ref_raw: tuple[float, float] | None = None
    selected_object_index: int | None = None
    selected_object_score: float | None = None
    secondary_object_index: int | None = None
    secondary_object_score: float | None = None


@dataclass(frozen=True)
class FinalPredictionResult:
    prediction: str
    prediction_source: str
    decoded_payload: dict[str, object] | None
    decoder_raw_output: str | None
    structured_overrides: dict[str, object] | None
    decoder_error: str | None
    structured_error: str | None


def _decode_vocab(logits, vocab: list[str]) -> str:
    return vocab[int(logits.argmax(dim=-1).item())]


def _decode_scalar(value) -> float:
    return float(value.reshape(-1)[0].detach().cpu().item())


def _decode_int(value) -> int:
    return int(value.reshape(-1)[0].detach().cpu().item())


def _decode_vector(value) -> tuple[float, float]:
    vector = value.reshape(-1).detach().cpu().tolist()
    return float(vector[0]), float(vector[1])


def _decode_action_state(subtemplate: str, outputs: dict[str, object]) -> tuple[str | None, str | None, tuple[str, ...], bool]:
    if subtemplate == "4_3_1_intersection_action" and "intersection_action_logits" in outputs:
        label = _decode_vocab(outputs["intersection_action_logits"], INTERSECTION_ACTION_VOCAB)
        return label, label, tuple(INTERSECTION_ACTION_VOCAB), False
    if subtemplate == "4_3_2_side_action" and "side_action_logits" in outputs:
        label = _decode_vocab(outputs["side_action_logits"], SIDE_ACTION_VOCAB)
        return label, label, tuple(SIDE_ACTION_VOCAB), False
    if subtemplate == "4_3_3_lane_action" and "lane_action_logits" in outputs:
        label = _decode_vocab(outputs["lane_action_logits"], LANE_ACTION_VOCAB)
        return label, label, tuple(LANE_ACTION_VOCAB), False
    if subtemplate == "4_3_4_object_action" and "object_action_logits" in outputs:
        label = _decode_vocab(outputs["object_action_logits"], OBJECT_ACTION_VOCAB)
        return label, label, tuple(OBJECT_ACTION_VOCAB), False
    return None, None, ACTION_STATE_ALLOWED_BY_SUBTEMPLATE.get(subtemplate, ()), False


def _constrain_motion_state(motion_state: str | None, speed_value: float | None) -> str | None:
    if motion_state is None or speed_value is None:
        return motion_state
    if speed_value < 0.2:
        return "stopped"
    if speed_value < 0.5 and motion_state not in {"stopped", "creeping"}:
        return "creeping"
    return motion_state


def _constrain_speeding_binary(binary_answer: bool | None, speed_value: float | None) -> bool | None:
    if binary_answer is not True:
        return binary_answer
    if speed_value is None:
        return binary_answer
    if speed_value < SPEEDING_RISK_SPEED_THRESHOLD_MPS:
        return False
    return True


def _location_phrase(side: str | None) -> str:
    if side == "center":
        return "in the center area"
    if side in {"north", "south", "east", "west"}:
        return f"on the {side} approach"
    return "in the scene"


def _location_phrase_full(side: str | None) -> str:
    if side == "center":
        return "in the center area of the intersection"
    if side in {"north", "south", "east", "west"}:
        return f"on the {side} approach of the intersection"
    return "in the intersection"


def _blob(position_3d: tuple[float, float] | None, camera_name: str | None, image_ref: tuple[float, float] | None) -> str:
    if position_3d is None or camera_name is None or image_ref is None:
        raise ValueError("location blob requires position_3d, camera_name, and image_ref")
    return f"<{position_3d[0]:.1f},{position_3d[1]:.1f},{camera_name},{image_ref[0]:.1f},{image_ref[1]:.1f}>"


def _indexed_vocab_value(index: int | None, vocab: list[str]) -> str | None:
    if index is None:
        return None
    if index < 0 or index >= len(vocab):
        return None
    return str(vocab[index])


def _action_state_from_supervision(supervision: GeoVLMSupervision) -> str | None:
    if supervision.intersection_action_index is not None:
        return _indexed_vocab_value(supervision.intersection_action_index, INTERSECTION_ACTION_VOCAB)
    if supervision.side_action_index is not None:
        return _indexed_vocab_value(supervision.side_action_index, SIDE_ACTION_VOCAB)
    if supervision.lane_action_index is not None:
        return _indexed_vocab_value(supervision.lane_action_index, LANE_ACTION_VOCAB)
    if supervision.object_action_index is not None:
        return _indexed_vocab_value(supervision.object_action_index, OBJECT_ACTION_VOCAB)
    return None


def render_canonical_answer_from_supervision(
    supervision: GeoVLMSupervision,
    *,
    fallback_answer: str = "",
) -> str:
    decoded = DecodedStructuredPrediction(
        subtemplate=supervision.subtemplate,
        object_type=_indexed_vocab_value(supervision.object_type_index, OBJECT_TYPE_VOCAB),
        side=_indexed_vocab_value(supervision.side_index, SIDE_VOCAB),
        motion_state=_indexed_vocab_value(supervision.motion_state_index, MOTION_STATE_VOCAB),
        risk_reason=_indexed_vocab_value(supervision.risk_reason_index, RISK_REASON_VOCAB),
        action_state=_action_state_from_supervision(supervision),
        lane_function=_indexed_vocab_value(supervision.lane_function_index, LANE_FUNCTION_VOCAB),
        binary_answer=(bool(round(supervision.binary_answer)) if supervision.binary_answer is not None else None),
        count_value=supervision.count_value,
        distance_value=supervision.distance_value,
        speed_value=supervision.speed_value,
        acceleration_value=supervision.acceleration_value,
        position_3d=supervision.position_3d,
        camera_name=_indexed_vocab_value(supervision.camera_index, CAMERA_VOCAB),
        image_ref=supervision.image_ref,
    )
    try:
        return render_decoded_answer(decoded)
    except Exception:
        return str(fallback_answer).strip()


def _token_overlap_score(text: str, candidate: str) -> tuple[float, float]:
    text_tokens = set(normalize_answer_text(text).split())
    candidate_tokens = set(normalize_answer_text(candidate).split())
    if not text_tokens or not candidate_tokens:
        return (0.0, 0.0)
    intersection = len(text_tokens & candidate_tokens)
    union = len(text_tokens | candidate_tokens)
    return (intersection / max(1, union), intersection / max(1, len(candidate_tokens)))


def _best_matching_sentence(text: str, candidates: tuple[str, ...]) -> str:
    return max(candidates, key=lambda candidate: _token_overlap_score(text, candidate))


def _extract_object_type(text: str, decoded_payload: dict[str, object] | None) -> str | None:
    lowered = str(text).lower()
    for object_type in sorted(OBJECT_TYPE_VOCAB, key=len, reverse=True):
        if object_type in lowered:
            return object_type
    if isinstance(decoded_payload, dict):
        value = decoded_payload.get("object_type")
        if isinstance(value, str) and value in OBJECT_TYPE_VOCAB:
            return value
    return None


def _extract_side(text: str, decoded_payload: dict[str, object] | None) -> str | None:
    lowered = str(text).lower()
    if "center area" in lowered or "center" in lowered:
        return "center"
    for side in ("north", "south", "east", "west"):
        if f"{side} approach" in lowered or f"on the {side}" in lowered:
            return side
    if isinstance(decoded_payload, dict):
        value = decoded_payload.get("side")
        if isinstance(value, str) and value in SIDE_VOCAB:
            return value
    return None


def _extract_object_action(text: str, decoded_payload: dict[str, object] | None) -> str | None:
    lowered = str(text).lower()
    for action_state, suffix in OBJECT_ACTION_SUFFIX.items():
        if suffix in lowered:
            return action_state
    if "yield" in lowered:
        return "OBJECT_YIELD_NOW"
    if "prepare to stop" in lowered or "stop" in lowered:
        return "OBJECT_PREPARE_TO_STOP"
    if "slow down" in lowered:
        return "OBJECT_SLOW_DOWN"
    if "cautious" in lowered:
        return "OBJECT_PROCEED_CAUTIOUSLY"
    if isinstance(decoded_payload, dict):
        value = decoded_payload.get("action_state")
        if isinstance(value, str) and value in OBJECT_ACTION_VOCAB:
            return value
    return None


def canonicalize_decoder_prediction(
    *,
    subtemplate: str,
    prediction: str,
    decoded_payload: dict[str, object] | None = None,
) -> str:
    raw_prediction = str(prediction).strip()
    if not raw_prediction:
        return ""
    if subtemplate in SEMANTIC_CANONICAL_SENTENCE_MAP:
        return _best_matching_sentence(raw_prediction, SEMANTIC_CANONICAL_SENTENCE_MAP[subtemplate])
    if subtemplate == "1_1_1_fine_type":
        object_type = _extract_object_type(raw_prediction, decoded_payload)
        side = _extract_side(raw_prediction, decoded_payload)
        if object_type is not None and side is not None:
            return f"It is a {object_type} located {_location_phrase_full(side)}."
    if subtemplate == "4_3_4_object_action":
        object_type = _extract_object_type(raw_prediction, decoded_payload)
        side = _extract_side(raw_prediction, decoded_payload)
        action_state = _extract_object_action(raw_prediction, decoded_payload)
        suffix = OBJECT_ACTION_SUFFIX.get(action_state) if action_state is not None else None
        if object_type is not None and side is not None and suffix is not None:
            return f"The {object_type} {_location_phrase(side)} should {suffix}."
    return raw_prediction


def decode_prediction_outputs(subtemplate: str, outputs: dict[str, object]) -> DecodedStructuredPrediction:
    object_type_raw = _decode_vocab(outputs["object_type_logits"], OBJECT_TYPE_VOCAB) if "object_type_logits" in outputs else None
    side_raw = _decode_vocab(outputs["side_logits"], SIDE_VOCAB) if "side_logits" in outputs else None
    motion_state_raw = _decode_vocab(outputs["motion_state_logits"], MOTION_STATE_VOCAB) if "motion_state_logits" in outputs else None
    risk_reason_raw = _decode_vocab(outputs["risk_reason_logits"], RISK_REASON_VOCAB) if "risk_reason_logits" in outputs else None
    lane_function_raw = _decode_vocab(outputs["lane_function_logits"], LANE_FUNCTION_VOCAB) if "lane_function_logits" in outputs else None
    camera_name_raw = _decode_vocab(outputs["camera_logits"], CAMERA_VOCAB) if "camera_logits" in outputs else None
    binary_answer_raw = (_decode_scalar(outputs["binary_answer_logit"]) > 0.0) if "binary_answer_logit" in outputs else None
    count_value_raw = _decode_scalar(outputs["count_value"]) if "count_value" in outputs else None
    distance_value_raw = _decode_scalar(outputs["distance_value"]) if "distance_value" in outputs else None
    speed_value_raw = _decode_scalar(outputs["speed_value"]) if "speed_value" in outputs else None
    acceleration_value_raw = _decode_scalar(outputs["acceleration_value"]) if "acceleration_value" in outputs else None
    position_3d_raw = _decode_vector(outputs["position_3d"]) if "position_3d" in outputs else None
    image_ref_raw = _decode_vector(outputs["image_ref"]) if "image_ref" in outputs else None
    selected_object_index = _decode_int(outputs["selected_object_index"]) if "selected_object_index" in outputs else None
    selected_object_score = _decode_scalar(outputs["selected_object_score"]) if "selected_object_score" in outputs else None
    secondary_object_index = _decode_int(outputs["secondary_object_index"]) if "secondary_object_index" in outputs else None
    secondary_object_score = _decode_scalar(outputs["secondary_object_score"]) if "secondary_object_score" in outputs else None

    motion_state = motion_state_raw
    binary_answer = binary_answer_raw
    if subtemplate == "3_1_1_current_motion_state":
        motion_state = _constrain_motion_state(motion_state_raw, speed_value_raw)
    if subtemplate == "4_2_1_speeding_risk":
        binary_answer = _constrain_speeding_binary(binary_answer_raw, speed_value_raw)

    action_state, action_state_raw_top1, action_state_allowed_set, action_state_family_mismatch = _decode_action_state(
        subtemplate, outputs
    )

    return DecodedStructuredPrediction(
        subtemplate=subtemplate,
        object_type=object_type_raw,
        object_type_raw=object_type_raw,
        side=side_raw,
        side_raw=side_raw,
        motion_state=motion_state,
        motion_state_raw=motion_state_raw,
        risk_reason=risk_reason_raw,
        risk_reason_raw=risk_reason_raw,
        action_state=action_state,
        action_state_raw_top1=action_state_raw_top1,
        action_state_allowed_set=action_state_allowed_set,
        action_state_family_mismatch=action_state_family_mismatch,
        lane_function=lane_function_raw,
        lane_function_raw=lane_function_raw,
        binary_answer=binary_answer,
        binary_answer_raw=binary_answer_raw,
        count_value=count_value_raw,
        count_value_raw=count_value_raw,
        distance_value=distance_value_raw,
        distance_value_raw=distance_value_raw,
        speed_value=speed_value_raw,
        speed_value_raw=speed_value_raw,
        acceleration_value=acceleration_value_raw,
        acceleration_value_raw=acceleration_value_raw,
        position_3d=denormalize_position_3d(position_3d_raw) if position_3d_raw is not None else None,
        position_3d_raw=position_3d_raw,
        camera_name=camera_name_raw,
        camera_name_raw=camera_name_raw,
        image_ref=denormalize_image_ref(image_ref_raw) if image_ref_raw is not None else None,
        image_ref_raw=image_ref_raw,
        selected_object_index=selected_object_index,
        selected_object_score=selected_object_score,
        secondary_object_index=secondary_object_index,
        secondary_object_score=secondary_object_score,
    )


def render_decoded_answer(decoded: DecodedStructuredPrediction) -> str:
    subtemplate = decoded.subtemplate
    if subtemplate == "1_1_1_fine_type":
        return f"It is a {decoded.object_type} located {_location_phrase_full(decoded.side)}."

    if subtemplate == "1_1_2_side_exists":
        return "Yes." if decoded.binary_answer else "No."

    if subtemplate == "1_1_3_side_count":
        count = int(round(decoded.count_value or 0.0))
        label = decoded.object_type or "object"
        plural = label if count == 1 else (
            "buses" if label == "bus" else
            "construction vehicles" if label == "construction_vehicle" else
            "golf carts" if label == "golf cart" else
            f"{label}s"
        )
        return f"The {decoded.side} approach currently has {count} {plural}."

    if subtemplate == "1_1_4_relative_neighbor_type":
        return f"It is a {decoded.object_type} at {_blob(decoded.position_3d, decoded.camera_name, decoded.image_ref)}."

    if subtemplate == "2_1_2_ped_to_far_edge":
        return f"The pedestrian on the {decoded.side} crosswalk is {decoded.distance_value:.1f} m from the exit area."

    if subtemplate == "2_1_4_nearest_vehicle_to_ped":
        return f"It is a {decoded.object_type} on the {decoded.side} approach, {decoded.distance_value:.1f} m away."

    if subtemplate == "3_1_1_current_motion_state":
        location = _location_phrase(decoded.side)
        if decoded.acceleration_value is not None and abs(decoded.acceleration_value) >= 0.05:
            accel_state = "accelerating" if decoded.acceleration_value >= 0.0 else "decelerating"
            return (
                f"The {decoded.object_type} {location} is {decoded.motion_state} at {decoded.speed_value:.1f} m/s "
                f"and {accel_state} at {decoded.acceleration_value:.1f} m/s^2."
            )
        return f"The {decoded.object_type} {location} is {decoded.motion_state} at {decoded.speed_value:.1f} m/s."

    if subtemplate == "3_4_2_nearest_conflict_participant":
        return (
            f"The most probable participant is the {decoded.object_type} at "
            f"{_blob(decoded.position_3d, decoded.camera_name, decoded.image_ref)}."
        )

    if subtemplate == "3_4_3_primary_risk_subject":
        return (
            f"The primary risk subject is the {decoded.object_type} at "
            f"{_blob(decoded.position_3d, decoded.camera_name, decoded.image_ref)} because of {decoded.risk_reason}."
        )

    if subtemplate == "4_2_1_speeding_risk":
        if not decoded.binary_answer:
            return "No."
        return (
            f"Yes. A {decoded.object_type} {_location_phrase_full(decoded.side)} is still moving at about "
            f"{decoded.speed_value:.1f} m/s."
        )

    if subtemplate in {"4_3_1_intersection_action", "4_3_2_side_action", "4_3_3_lane_action"}:
        if decoded.action_state not in ACTION_STATE_TO_SENTENCE:
            raise KeyError(f"Unsupported action_state renderer for {subtemplate}: {decoded.action_state}")
        return ACTION_STATE_TO_SENTENCE[decoded.action_state]

    if subtemplate == "4_3_4_object_action":
        suffix = OBJECT_ACTION_SUFFIX.get(decoded.action_state)
        if suffix is None:
            raise KeyError(f"Unsupported object action_state renderer: {decoded.action_state}")
        return f"The {decoded.object_type} {_location_phrase(decoded.side)} should {suffix}."

    raise KeyError(f"Unsupported render subtemplate: {subtemplate}")


def resolve_final_prediction(
    *,
    subtemplate: str,
    outputs: dict[str, object],
    decoder_raw_output: str | None,
) -> FinalPredictionResult:
    decoded_payload: dict[str, object] | None = None
    structured_overrides: dict[str, object] | None = None
    decoder_error: str | None = None
    structured_error: str | None = None
    prediction = ""
    prediction_source = "structured"

    if uses_decoder_final_prediction(subtemplate):
        decoded = None
        if str(decoder_raw_output or "").strip():
            prediction = str(decoder_raw_output).strip()
            prediction_source = "decoder"
        else:
            decoder_error = f"Decoder returned an empty answer for semantic template {subtemplate}."
        try:
            decoded = decode_prediction_outputs(subtemplate, outputs)
            decoded_payload = decoded.__dict__
        except Exception as exc:
            structured_error = f"{type(exc).__name__}: {exc}"
        if prediction:
            prediction = canonicalize_decoder_prediction(
                subtemplate=subtemplate,
                prediction=prediction,
                decoded_payload=decoded_payload,
            )
        return FinalPredictionResult(
            prediction=prediction,
            prediction_source=prediction_source,
            decoded_payload=decoded_payload,
            decoder_raw_output=decoder_raw_output,
            structured_overrides=None,
            decoder_error=decoder_error,
            structured_error=structured_error,
        )

    try:
        decoded = decode_prediction_outputs(subtemplate, outputs)
        decoded_payload = decoded.__dict__
        prediction = render_decoded_answer(decoded)
        prediction_source = "structured"
        if uses_hybrid_structured_render(subtemplate):
            structured_overrides = decoded_payload
    except Exception as exc:
        structured_error = f"{type(exc).__name__}: {exc}"

    if uses_hybrid_structured_render(subtemplate) and not str(decoder_raw_output or "").strip():
        decoder_error = f"Decoder returned an empty answer for hybrid template {subtemplate}."

    return FinalPredictionResult(
        prediction=prediction,
        prediction_source=prediction_source,
        decoded_payload=decoded_payload,
        decoder_raw_output=decoder_raw_output,
        structured_overrides=structured_overrides,
        decoder_error=decoder_error,
        structured_error=structured_error,
    )


def compute_final_text_match(answer: str, prediction: str) -> float:
    return float(normalize_answer_text(answer) == normalize_answer_text(prediction))
