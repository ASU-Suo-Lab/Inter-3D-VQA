from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from geovlm_intersection.data.targets import OBJECT_CENTRIC_SUBTEMPLATES, POSITION_3D_NORM_X, POSITION_3D_NORM_Y, SUBTEMPLATE_TO_INDEX


@dataclass
class GeoVLMConfig:
    num_cameras: int = 4
    image_token_dim: int = 1024
    bev_token_dim: int = 512
    object_token_dim: int = 512
    question_token_dim: int = 1024
    image_token_budget_per_camera: int = 256
    bev_token_budget: int = 1024
    object_token_budget: int = 128
    question_token_budget: int = 256
    fusion_dim: int = 1024
    scene_memory_slots: int = 192
    fusion_layers: int = 6
    fusion_heads: int = 8
    llm_condition_tokens: int = 32
    subtemplate_classes: int = len(SUBTEMPLATE_TO_INDEX)
    object_type_classes: int = 10
    side_classes: int = 5
    motion_state_classes: int = 8
    risk_reason_classes: int = 5
    intersection_action_classes: int = 4
    side_action_classes: int = 5
    lane_action_classes: int = 5
    object_action_classes: int = 4
    lane_function_classes: int = 3


@dataclass
class GeoVLMEncodedInputs:
    image_tokens: torch.Tensor
    bev_tokens: torch.Tensor
    object_tokens: torch.Tensor
    raw_object_tokens: torch.Tensor
    question_tokens: torch.Tensor
    subtemplate_ids: torch.Tensor


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SceneMemoryBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.image_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.object_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.bev_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.question_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = FeedForward(dim, dim * 4)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(6))

    def forward(
        self,
        scene_memory: torch.Tensor,
        image_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        bev_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        residual = scene_memory
        scene_memory = self.self_attn(scene_memory, scene_memory, scene_memory, need_weights=False)[0]
        scene_memory = self.norms[0](scene_memory + residual)

        residual = scene_memory
        scene_memory = self.image_attn(scene_memory, image_tokens, image_tokens, need_weights=False)[0]
        scene_memory = self.norms[1](scene_memory + residual)

        residual = scene_memory
        scene_memory = self.object_attn(scene_memory, object_tokens, object_tokens, need_weights=False)[0]
        scene_memory = self.norms[2](scene_memory + residual)

        residual = scene_memory
        scene_memory = self.bev_attn(scene_memory, bev_tokens, bev_tokens, need_weights=False)[0]
        scene_memory = self.norms[3](scene_memory + residual)

        residual = scene_memory
        scene_memory = self.question_attn(scene_memory, question_tokens, question_tokens, need_weights=False)[0]
        scene_memory = self.norms[4](scene_memory + residual)

        residual = scene_memory
        scene_memory = self.ffn(scene_memory)
        scene_memory = self.norms[5](scene_memory + residual)
        return scene_memory


class StructuredHeads(nn.Module):
    def __init__(self, config: GeoVLMConfig) -> None:
        super().__init__()
        dim = config.fusion_dim
        self.object_query = nn.Linear(dim, dim)
        self.object_type = nn.Linear(dim, config.object_type_classes)
        self.side_global = nn.Linear(dim, config.side_classes)
        self.side_object = nn.Linear(dim, config.side_classes)
        self.motion_state = nn.Linear(dim, config.motion_state_classes)
        self.risk_reason = nn.Linear(dim, config.risk_reason_classes)
        self.intersection_action = nn.Linear(dim, config.intersection_action_classes)
        self.side_action = nn.Linear(dim, config.side_action_classes)
        self.lane_action = nn.Linear(dim, config.lane_action_classes)
        self.object_action = nn.Linear(dim, config.object_action_classes)
        self.lane_function = nn.Linear(dim, config.lane_function_classes)
        self.binary_answer_global = nn.Linear(dim, 1)
        self.binary_answer_object = nn.Linear(dim, 1)
        self.count = nn.Linear(dim, 1)
        self.distance_global = nn.Linear(dim, 1)
        self.distance_object = nn.Linear(dim, 1)
        self.speed = nn.Linear(dim, 1)
        self.acceleration = nn.Linear(dim, 1)
        self.position_3d_delta = nn.Linear(dim, 2)
        self.camera_logits = nn.Linear(dim, config.num_cameras)
        self.image_ref = nn.Linear(dim, 2)

    @staticmethod
    def _apply_mask(base: torch.Tensor, replacement: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return base
        view_shape = [mask.shape[0]] + [1] * (base.ndim - 1)
        return torch.where(mask.view(*view_shape), replacement, base)

    def forward(
        self,
        *,
        global_context: torch.Tensor,
        object_context: torch.Tensor,
        object_tokens: torch.Tensor,
        raw_object_tokens: torch.Tensor,
        object_route_mask: torch.Tensor,
        binary_object_mask: torch.Tensor,
        distance_object_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        object_query = self.object_query(global_context).unsqueeze(1)
        object_selection_logits = torch.matmul(object_query, object_tokens.transpose(1, 2)).squeeze(1)
        object_anchor_xy = raw_object_tokens[..., :2]
        object_selection_weights = torch.softmax(object_selection_logits, dim=-1)
        selected_anchor_xy = torch.bmm(object_selection_weights.unsqueeze(1), object_anchor_xy).squeeze(1)
        anchor_xy_norm = selected_anchor_xy / selected_anchor_xy.new_tensor([POSITION_3D_NORM_X, POSITION_3D_NORM_Y])

        side_logits = self._apply_mask(
            self.side_global(global_context),
            self.side_object(object_context),
            object_route_mask,
        )
        binary_answer_logit = self._apply_mask(
            self.binary_answer_global(global_context).squeeze(-1),
            self.binary_answer_object(object_context).squeeze(-1),
            binary_object_mask,
        )
        distance_value = self._apply_mask(
            self.distance_global(global_context).squeeze(-1),
            self.distance_object(object_context).squeeze(-1),
            distance_object_mask,
        )
        position_delta = 0.25 * torch.tanh(self.position_3d_delta(object_context))
        return {
            "object_selection_logits": object_selection_logits,
            "object_selection_weights": object_selection_weights,
            "selected_object_anchor_xy": selected_anchor_xy,
            "object_type_logits": self.object_type(object_context),
            "side_logits": side_logits,
            "motion_state_logits": self.motion_state(object_context),
            "risk_reason_logits": self.risk_reason(object_context),
            "intersection_action_logits": self.intersection_action(global_context),
            "side_action_logits": self.side_action(global_context),
            "lane_action_logits": self.lane_action(global_context),
            "object_action_logits": self.object_action(object_context),
            "lane_function_logits": self.lane_function(global_context),
            "binary_answer_logit": binary_answer_logit,
            "count_value": self.count(global_context).squeeze(-1),
            "distance_value": distance_value,
            "speed_value": self.speed(object_context).squeeze(-1),
            "acceleration_value": self.acceleration(object_context).squeeze(-1),
            "position_3d": anchor_xy_norm + position_delta,
            "camera_logits": self.camera_logits(object_context),
            "image_ref": torch.sigmoid(self.image_ref(object_context)),
        }


class GeoVLMModel(nn.Module):
    """
    Fusion core for a new intersection VLM.

    The current implementation intentionally assumes external backbones already
    converted raw images and global point clouds into token tensors. This keeps
    the architecture decision-complete while avoiding premature binding to a
    specific image or sparse point-cloud backbone.
    """

    def __init__(self, config: GeoVLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeoVLMConfig()
        dim = self.config.fusion_dim
        self.camera_embedding = nn.Embedding(self.config.num_cameras, dim)
        self.image_proj = nn.Linear(self.config.image_token_dim, dim)
        self.bev_proj = nn.Linear(self.config.bev_token_dim, dim)
        self.object_proj = nn.Linear(self.config.object_token_dim, dim)
        self.question_proj = nn.Linear(self.config.question_token_dim, dim)
        self.subtemplate_embedding = nn.Embedding(self.config.subtemplate_classes, dim)
        self.scene_memory = nn.Parameter(torch.randn(self.config.scene_memory_slots, dim) * 0.02)
        self.fusion_blocks = nn.ModuleList(
            SceneMemoryBlock(dim=dim, heads=self.config.fusion_heads) for _ in range(self.config.fusion_layers)
        )
        self.scene_pool = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.global_route = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.object_route = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.structured_heads = StructuredHeads(self.config)
        self.semantic_prefix_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.config.question_token_dim),
        )

    @staticmethod
    def _mask_from_ids(subtemplate_ids: torch.Tensor, names: set[str]) -> torch.Tensor:
        allowed = torch.zeros(len(SUBTEMPLATE_TO_INDEX), dtype=torch.bool, device=subtemplate_ids.device)
        for name in names:
            allowed[SUBTEMPLATE_TO_INDEX[name]] = True
        return allowed[subtemplate_ids]

    @staticmethod
    def _compress_sequence_tokens(tokens: torch.Tensor, token_budget: int) -> torch.Tensor:
        if token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got: {token_budget}")
        if tokens.ndim != 3:
            raise ValueError(f"Expected rank-3 tokens [B, T, D], got shape={tuple(tokens.shape)}")
        if tokens.shape[1] <= token_budget:
            return tokens
        pooled = F.adaptive_avg_pool1d(tokens.transpose(1, 2), token_budget)
        return pooled.transpose(1, 2)

    def _compress_image_tokens(self, image_tokens: torch.Tensor) -> torch.Tensor:
        if image_tokens.ndim != 4:
            raise ValueError(f"image_tokens must be [B, C, T, D], got shape={tuple(image_tokens.shape)}")
        batch, cameras, _, hidden = image_tokens.shape
        flattened = image_tokens.reshape(batch * cameras, image_tokens.shape[2], hidden)
        compressed = self._compress_sequence_tokens(flattened, self.config.image_token_budget_per_camera)
        return compressed.reshape(batch, cameras, compressed.shape[1], hidden)

    def _project_image_tokens(self, image_tokens: torch.Tensor) -> torch.Tensor:
        batch, cameras, tokens, _ = image_tokens.shape
        projected = self.image_proj(image_tokens)
        camera_ids = torch.arange(cameras, device=image_tokens.device)
        camera_emb = self.camera_embedding(camera_ids).view(1, cameras, 1, -1)
        projected = projected + camera_emb
        return projected.view(batch, cameras * tokens, -1)

    def build_semantic_prefix_tokens(
        self,
        *,
        scene_memory: torch.Tensor,
        global_context: torch.Tensor,
        selected_object_context: torch.Tensor,
    ) -> torch.Tensor:
        semantic_source = torch.cat(
            [
                global_context.unsqueeze(1),
                selected_object_context.unsqueeze(1),
                scene_memory[:, : self.config.llm_condition_tokens],
            ],
            dim=1,
        )
        return self.semantic_prefix_proj(semantic_source)

    def encode_inputs(
        self,
        *,
        image_tokens: torch.Tensor,
        bev_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        raw_object_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        subtemplate_ids: torch.Tensor,
    ) -> GeoVLMEncodedInputs:
        if image_tokens.ndim != 4:
            raise ValueError(f"image_tokens must be [B, 4, T, D], got shape={tuple(image_tokens.shape)}")
        if image_tokens.shape[1] != self.config.num_cameras:
            raise ValueError(
                f"image_tokens camera dimension must equal {self.config.num_cameras}, got {image_tokens.shape[1]}"
            )
        if bev_tokens.ndim != 3 or object_tokens.ndim != 3 or raw_object_tokens.ndim != 3 or question_tokens.ndim != 3:
            raise ValueError("bev_tokens, object_tokens, raw_object_tokens, and question_tokens must all be rank-3 tensors.")
        if raw_object_tokens.shape[:2] != object_tokens.shape[:2]:
            raise ValueError(
                f"raw_object_tokens shape must match object_tokens on [B, T], got {tuple(raw_object_tokens.shape)} vs {tuple(object_tokens.shape)}"
            )

        image_tokens_comp = self._compress_image_tokens(image_tokens)
        bev_tokens_comp = self._compress_sequence_tokens(bev_tokens, self.config.bev_token_budget)
        object_tokens_comp = self._compress_sequence_tokens(object_tokens, self.config.object_token_budget)
        question_tokens_comp = self._compress_sequence_tokens(question_tokens, self.config.question_token_budget)
        raw_object_tokens_comp = self._compress_sequence_tokens(raw_object_tokens, self.config.object_token_budget)

        return GeoVLMEncodedInputs(
            image_tokens=self._project_image_tokens(image_tokens_comp),
            bev_tokens=self.bev_proj(bev_tokens_comp),
            object_tokens=self.object_proj(object_tokens_comp),
            raw_object_tokens=raw_object_tokens_comp,
            question_tokens=self.question_proj(question_tokens_comp),
            subtemplate_ids=subtemplate_ids.long(),
        )

    def forward_encoded(self, encoded_inputs: GeoVLMEncodedInputs) -> dict[str, torch.Tensor]:
        batch = encoded_inputs.image_tokens.shape[0]
        scene_memory = self.scene_memory.unsqueeze(0).expand(batch, -1, -1)
        for block in self.fusion_blocks:
            scene_memory = block(
                scene_memory=scene_memory,
                image_tokens=encoded_inputs.image_tokens,
                object_tokens=encoded_inputs.object_tokens,
                bev_tokens=encoded_inputs.bev_tokens,
                question_tokens=encoded_inputs.question_tokens,
            )

        pooled_scene = self.scene_pool(scene_memory.mean(dim=1))
        subtemplate_context = self.subtemplate_embedding(encoded_inputs.subtemplate_ids)
        global_context = self.global_route(torch.cat([pooled_scene, subtemplate_context], dim=-1))
        object_selection_query = self.structured_heads.object_query(global_context).unsqueeze(1)
        object_selection_logits = torch.matmul(object_selection_query, encoded_inputs.object_tokens.transpose(1, 2)).squeeze(1)
        object_selection_weights = torch.softmax(object_selection_logits, dim=-1)
        selected_object_context = torch.bmm(object_selection_weights.unsqueeze(1), encoded_inputs.object_tokens).squeeze(1)
        object_context = self.object_route(torch.cat([pooled_scene, selected_object_context, subtemplate_context], dim=-1))
        object_route_mask = self._mask_from_ids(encoded_inputs.subtemplate_ids, OBJECT_CENTRIC_SUBTEMPLATES)
        binary_object_mask = self._mask_from_ids(encoded_inputs.subtemplate_ids, {"4_2_1_speeding_risk"})
        distance_object_mask = self._mask_from_ids(encoded_inputs.subtemplate_ids, {"2_1_4_nearest_vehicle_to_ped"})
        structured = self.structured_heads(
            global_context=global_context,
            object_context=object_context,
            object_tokens=encoded_inputs.object_tokens,
            raw_object_tokens=encoded_inputs.raw_object_tokens,
            object_route_mask=object_route_mask,
            binary_object_mask=binary_object_mask,
            distance_object_mask=distance_object_mask,
        )
        semantic_prefix_tokens = self.build_semantic_prefix_tokens(
            scene_memory=scene_memory,
            global_context=global_context,
            selected_object_context=selected_object_context,
        )
        return {
            "scene_memory": scene_memory,
            "scene_pooled": pooled_scene,
            "global_context": global_context,
            "selected_object_context": selected_object_context,
            "semantic_prefix_tokens": semantic_prefix_tokens,
            "llm_conditioning_tokens": semantic_prefix_tokens,
            **structured,
        }

    def forward(
        self,
        *,
        image_tokens: torch.Tensor,
        bev_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        raw_object_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        subtemplate_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded_inputs = self.encode_inputs(
            image_tokens=image_tokens,
            bev_tokens=bev_tokens,
            object_tokens=object_tokens,
            raw_object_tokens=raw_object_tokens,
            question_tokens=question_tokens,
            subtemplate_ids=subtemplate_ids,
        )
        return self.forward_encoded(encoded_inputs)


def shape_smoke_forward(config: GeoVLMConfig | None = None) -> dict[str, tuple[int, ...]]:
    cfg = config or GeoVLMConfig()
    model = GeoVLMModel(cfg)
    batch = 2
    outputs = model(
        image_tokens=torch.randn(batch, cfg.num_cameras, 256, cfg.image_token_dim),
        bev_tokens=torch.randn(batch, 400, cfg.bev_token_dim),
        object_tokens=torch.randn(batch, 128, cfg.object_token_dim),
        raw_object_tokens=torch.randn(batch, 128, 11),
        question_tokens=torch.randn(batch, 64, cfg.question_token_dim),
        subtemplate_ids=torch.zeros(batch, dtype=torch.long),
    )
    return {key: tuple(value.shape) for key, value in outputs.items()}
