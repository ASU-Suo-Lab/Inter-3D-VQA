import torch
import torch.nn as nn


class TokenProjectionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class ObjectPrefixEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.object_fc1 = nn.Linear(input_dim, hidden_dim)
        self.object_act = nn.GELU()
        self.object_fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.object_fc1(object_features)
        hidden_states = self.object_act(hidden_states)
        hidden_states = self.object_fc2(hidden_states)
        return hidden_states


class ObjectGeometryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.object_geometry_fc1 = nn.Linear(input_dim, hidden_dim)
        self.object_geometry_act = nn.GELU()
        self.object_geometry_fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, object_geometry_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.object_geometry_fc1(object_geometry_features)
        hidden_states = self.object_geometry_act(hidden_states)
        hidden_states = self.object_geometry_fc2(hidden_states)
        return hidden_states


class ObjectNumericEncoder(nn.Module):
    def __init__(self, input_dim: int, label_vocab_size: int, label_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(label_vocab_size, label_emb_dim, padding_idx=0)
        self.numeric_fc1 = nn.Linear(input_dim + label_emb_dim, hidden_dim)
        self.numeric_act = nn.GELU()
        self.numeric_fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, object_numeric_features: torch.Tensor, object_numeric_label_ids: torch.Tensor) -> torch.Tensor:
        label_ids = object_numeric_label_ids.clamp(min=0, max=self.label_embedding.num_embeddings - 1)
        label_embeds = self.label_embedding(label_ids)
        hidden_states = torch.cat([object_numeric_features, label_embeds], dim=-1)
        hidden_states = self.numeric_fc1(hidden_states)
        hidden_states = self.numeric_act(hidden_states)
        hidden_states = self.numeric_fc2(hidden_states)
        return hidden_states


class SceneEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scene_fc1 = nn.Linear(input_dim, hidden_dim)
        self.scene_act = nn.GELU()
        self.scene_fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.scene_fc1(scene_features)
        hidden_states = self.scene_act(hidden_states)
        hidden_states = self.scene_fc2(hidden_states)
        return hidden_states


class LidarNumericHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class LidarTemplateRouter(nn.Module):
    def __init__(self, initial_scales: tuple[tuple[float, ...], ...], output_dim: int = 2) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"router output_dim must be positive, got: {output_dim}")
        self.output_dim = output_dim
        self.route_logits = nn.Embedding(len(initial_scales), output_dim)
        self.reset_parameters(initial_scales)

    def _prepare_scale_tensor(self, scales: tuple[tuple[float, ...], ...]) -> torch.Tensor:
        scales = torch.tensor(
            scales,
            dtype=self.route_logits.weight.dtype,
            device=self.route_logits.weight.device,
        )
        if scales.ndim != 2:
            raise ValueError("route scales must be a 2D tuple of route scale rows.")
        if scales.size(1) < self.output_dim:
            padding = torch.ones(
                (scales.size(0), self.output_dim - scales.size(1)),
                dtype=scales.dtype,
                device=scales.device,
            )
            scales = torch.cat([scales, padding], dim=1)
        elif scales.size(1) > self.output_dim:
            scales = scales[:, : self.output_dim]
        scales = scales.clamp(min=1.0e-3, max=2.0 - 1.0e-3)
        return scales

    def reset_parameters(self, initial_scales: tuple[tuple[float, ...], ...]) -> None:
        scales = self._prepare_scale_tensor(initial_scales)
        logits = torch.log(scales / (2.0 - scales))
        with torch.no_grad():
            self.route_logits.weight.copy_(logits)

    def forward(self, template_family_ids: torch.Tensor) -> torch.Tensor:
        family_ids = template_family_ids.to(device=self.route_logits.weight.device, dtype=torch.long).view(-1)
        family_ids = family_ids.clamp(min=0, max=self.route_logits.num_embeddings - 1)
        return 2.0 * torch.sigmoid(self.route_logits(family_ids))


class ObjectQueryConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, query_length: int, num_heads: int = 8) -> None:
        super().__init__()
        self.object_queries = nn.Parameter(torch.zeros(query_length, output_dim))
        nn.init.normal_(self.object_queries, mean=0.0, std=0.02)
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.query_norm = nn.LayerNorm(output_dim)
        self.memory_norm = nn.LayerNorm(output_dim)
        self.query_attn = nn.MultiheadAttention(output_dim, num_heads=num_heads, batch_first=True)
        self.query_self_norm = nn.LayerNorm(output_dim)
        self.query_self_attn = nn.MultiheadAttention(output_dim, num_heads=num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
        )
        self.slot_confidence_head = nn.Linear(output_dim, 1)

    def forward(
        self,
        object_hidden_states: torch.Tensor,
        object_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = object_hidden_states.size(0)
        memory_states = self.memory_norm(self.input_proj(object_hidden_states))
        query_states = self.query_norm(self.object_queries).unsqueeze(0).expand(batch_size, -1, -1)

        if object_mask is None:
            key_padding_mask = None
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=object_hidden_states.device)
        else:
            object_mask = object_mask.to(device=object_hidden_states.device, dtype=torch.bool)
            valid_mask = object_mask.any(dim=1)
            key_padding_mask = ~object_mask
            if not bool(valid_mask.all()):
                memory_states = memory_states.clone()
                key_padding_mask = key_padding_mask.clone()
                memory_states[~valid_mask] = 0.0
                key_padding_mask[~valid_mask] = False

        query_context, query_weights = self.query_attn(
            query_states,
            memory_states,
            memory_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        connected_states = query_states + query_context
        query_self_context, _ = self.query_self_attn(
            self.query_self_norm(connected_states),
            self.query_self_norm(connected_states),
            self.query_self_norm(connected_states),
            need_weights=False,
        )
        connected_states = connected_states + query_self_context
        connected_states = connected_states + self.output_mlp(self.output_norm(connected_states))

        if object_mask is None:
            valid_object_mask = torch.ones(
                batch_size,
                object_hidden_states.size(1),
                dtype=torch.bool,
                device=object_hidden_states.device,
            )
        else:
            valid_object_mask = object_mask.to(device=object_hidden_states.device, dtype=torch.bool)

        masked_query_weights = query_weights.masked_fill(~valid_object_mask.unsqueeze(1), float("-inf"))
        slot_assignment_indices = masked_query_weights.argmax(dim=-1)
        slot_logits = self.slot_confidence_head(connected_states).squeeze(-1)
        if not bool(valid_mask.all()):
            slot_logits = slot_logits.clone()
            slot_logits[~valid_mask] = -20.0
        slot_probs = torch.sigmoid(slot_logits)
        memory_mask = slot_probs >= 0.5
        memory_mask = memory_mask & valid_mask.unsqueeze(1)
        connected_states = connected_states * slot_probs.unsqueeze(-1).to(dtype=connected_states.dtype)
        connected_states = connected_states * valid_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=connected_states.dtype)
        return connected_states, memory_mask, slot_logits, slot_assignment_indices


class SceneLatentConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, latent_length: int, num_heads: int = 8) -> None:
        super().__init__()
        self.scene_latents = nn.Parameter(torch.zeros(latent_length, output_dim))
        nn.init.normal_(self.scene_latents, mean=0.0, std=0.02)
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.latent_norm = nn.LayerNorm(output_dim)
        self.memory_norm = nn.LayerNorm(output_dim)
        self.cross_attn = nn.MultiheadAttention(output_dim, num_heads=num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
        )

    def forward(
        self,
        scene_hidden_states: torch.Tensor,
        scene_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = scene_hidden_states.size(0)
        memory_states = self.memory_norm(self.input_proj(scene_hidden_states))
        latent_states = self.latent_norm(self.scene_latents).unsqueeze(0).expand(batch_size, -1, -1)

        if scene_mask is None:
            key_padding_mask = None
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=scene_hidden_states.device)
        else:
            scene_mask = scene_mask.to(device=scene_hidden_states.device, dtype=torch.bool)
            valid_mask = scene_mask.any(dim=1)
            key_padding_mask = ~scene_mask
            if not bool(valid_mask.all()):
                memory_states = memory_states.clone()
                key_padding_mask = key_padding_mask.clone()
                memory_states[~valid_mask] = 0.0
                key_padding_mask[~valid_mask] = False

        connected_states, _ = self.cross_attn(
            latent_states,
            memory_states,
            memory_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        connected_states = connected_states + self.output_mlp(self.output_norm(connected_states))
        memory_mask = valid_mask.unsqueeze(1).expand(batch_size, latent_states.size(1))
        connected_states = connected_states * memory_mask.unsqueeze(-1).to(dtype=connected_states.dtype)
        return connected_states, memory_mask


class LidarDecoderAdapterBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, gate_bias: float = -6.0) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.agent_norm = nn.LayerNorm(hidden_dim)
        self.scene_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.agent_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scene_gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.agent_gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for gate_mlp in (self.scene_gate_mlp, self.agent_gate_mlp, self.output_gate_mlp):
            last_linear = gate_mlp[-1]
            nn.init.normal_(last_linear.weight, mean=0.0, std=1.0e-3)
            nn.init.constant_(last_linear.bias, gate_bias)
        self._last_adapter_metrics = {
            "adapter_scene_gate_mean": 0.0,
            "adapter_agent_gate_mean": 0.0,
            "adapter_output_gate_mean": 0.0,
            "adapter_scene_delta_norm": 0.0,
            "adapter_agent_delta_norm": 0.0,
            "adapter_scene_route_scale_mean": 0.0,
            "adapter_agent_route_scale_mean": 0.0,
            "adapter_output_route_scale_mean": 0.0,
        }
        self._last_adapter_sample_metrics = None
        self._last_gate_usage_regularizer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        agent_memory: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
        scene_memory: torch.Tensor | None = None,
        scene_mask: torch.Tensor | None = None,
        agent_route_scale: torch.Tensor | None = None,
        scene_route_scale: torch.Tensor | None = None,
        output_route_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.size(1) == 0:
            return hidden_states

        norm_hidden_states = self.hidden_norm(hidden_states)
        scene_delta = hidden_states.new_zeros(hidden_states.shape)
        agent_delta = hidden_states.new_zeros(hidden_states.shape)
        scene_gate_mean = hidden_states.new_zeros(())
        agent_gate_mean = hidden_states.new_zeros(())
        output_gate_mean = hidden_states.new_zeros(())
        scene_delta_norm = hidden_states.new_zeros(())
        agent_delta_norm = hidden_states.new_zeros(())
        scene_gate_per_sample = hidden_states.new_zeros((hidden_states.size(0),))
        agent_gate_per_sample = hidden_states.new_zeros((hidden_states.size(0),))
        output_gate_per_sample = hidden_states.new_zeros((hidden_states.size(0),))
        scene_delta_norm_per_sample = hidden_states.new_zeros((hidden_states.size(0),))
        agent_delta_norm_per_sample = hidden_states.new_zeros((hidden_states.size(0),))
        scene_route_scale_per_sample = hidden_states.new_zeros((hidden_states.size(0),))
        agent_route_scale_per_sample = hidden_states.new_zeros((hidden_states.size(0),))
        output_route_scale_per_sample = hidden_states.new_zeros((hidden_states.size(0),))

        if scene_memory is not None and scene_mask is not None:
            scene_mask = scene_mask.to(device=hidden_states.device, dtype=torch.bool)
            norm_scene_memory = self.scene_norm(scene_memory)
            scene_key_padding_mask = ~scene_mask
            valid_scene_mask = scene_mask.any(dim=1)
            if not bool(valid_scene_mask.all()):
                norm_scene_memory = norm_scene_memory.clone()
                scene_key_padding_mask = scene_key_padding_mask.clone()
                norm_scene_memory[~valid_scene_mask] = 0.0
                scene_key_padding_mask[~valid_scene_mask] = False
            scene_delta, _ = self.scene_attn(
                norm_hidden_states,
                norm_scene_memory,
                norm_scene_memory,
                key_padding_mask=scene_key_padding_mask,
                need_weights=False,
            )
            scene_gate = torch.sigmoid(self.scene_gate_mlp(hidden_states))
            scene_route_scale_per_sample = (
                scene_route_scale.to(device=hidden_states.device, dtype=hidden_states.dtype).view(-1)
                if scene_route_scale is not None
                else hidden_states.new_ones((hidden_states.size(0),))
            )
            scene_delta = scene_gate * scene_delta * scene_route_scale_per_sample.view(-1, 1, 1)
            scene_gate_mean = scene_gate.mean()
            scene_delta_norm = scene_delta.norm(dim=-1).mean()
            scene_gate_per_sample = scene_gate.mean(dim=(1, 2))
            scene_delta_norm_per_sample = scene_delta.norm(dim=-1).mean(dim=1)

        if agent_memory is not None and agent_mask is not None:
            agent_mask = agent_mask.to(device=hidden_states.device, dtype=torch.bool)
            norm_agent_memory = self.agent_norm(agent_memory)
            agent_key_padding_mask = ~agent_mask
            valid_agent_mask = agent_mask.any(dim=1)
            if not bool(valid_agent_mask.all()):
                norm_agent_memory = norm_agent_memory.clone()
                agent_key_padding_mask = agent_key_padding_mask.clone()
                norm_agent_memory[~valid_agent_mask] = 0.0
                agent_key_padding_mask[~valid_agent_mask] = False
            agent_delta, _ = self.agent_attn(
                norm_hidden_states,
                norm_agent_memory,
                norm_agent_memory,
                key_padding_mask=agent_key_padding_mask,
                need_weights=False,
            )
            agent_gate = torch.sigmoid(self.agent_gate_mlp(hidden_states))
            agent_route_scale_per_sample = (
                agent_route_scale.to(device=hidden_states.device, dtype=hidden_states.dtype).view(-1)
                if agent_route_scale is not None
                else hidden_states.new_ones((hidden_states.size(0),))
            )
            agent_delta = agent_gate * agent_delta * agent_route_scale_per_sample.view(-1, 1, 1)
            agent_gate_mean = agent_gate.mean()
            agent_delta_norm = agent_delta.norm(dim=-1).mean()
            agent_gate_per_sample = agent_gate.mean(dim=(1, 2))
            agent_delta_norm_per_sample = agent_delta.norm(dim=-1).mean(dim=1)

        if scene_memory is None and agent_memory is None:
            self._last_adapter_metrics = {
                "adapter_scene_gate_mean": 0.0,
                "adapter_agent_gate_mean": 0.0,
                "adapter_output_gate_mean": 0.0,
                "adapter_scene_delta_norm": 0.0,
                "adapter_agent_delta_norm": 0.0,
                "adapter_scene_route_scale_mean": 0.0,
                "adapter_agent_route_scale_mean": 0.0,
                "adapter_output_route_scale_mean": 0.0,
            }
            self._last_adapter_sample_metrics = {
                "adapter_scene_gate_mean": scene_gate_per_sample.detach(),
                "adapter_agent_gate_mean": agent_gate_per_sample.detach(),
                "adapter_output_gate_mean": output_gate_per_sample.detach(),
                "adapter_scene_delta_norm": scene_delta_norm_per_sample.detach(),
                "adapter_agent_delta_norm": agent_delta_norm_per_sample.detach(),
                "adapter_scene_route_scale_mean": scene_route_scale_per_sample.detach(),
                "adapter_agent_route_scale_mean": agent_route_scale_per_sample.detach(),
                "adapter_output_route_scale_mean": output_route_scale_per_sample.detach(),
            }
            self._last_gate_usage_regularizer = hidden_states.new_zeros(())
            return hidden_states

        fused_delta = self.fusion_mlp(torch.cat([scene_delta, agent_delta], dim=-1))
        output_gate = torch.sigmoid(self.output_gate_mlp(hidden_states))
        output_route_scale_per_sample = (
            output_route_scale.to(device=hidden_states.device, dtype=hidden_states.dtype).view(-1)
            if output_route_scale is not None
            else hidden_states.new_ones((hidden_states.size(0),))
        )
        output_gate_mean = output_gate.mean()
        output_gate_per_sample = output_gate.mean(dim=(1, 2))
        usage_terms = []
        if scene_memory is not None:
            usage_terms.append(torch.relu(hidden_states.new_tensor(0.01) - scene_gate_mean))
        if agent_memory is not None:
            usage_terms.append(torch.relu(hidden_states.new_tensor(0.01) - agent_gate_mean))
        usage_terms.append(torch.relu(hidden_states.new_tensor(0.01) - output_gate_mean))
        self._last_gate_usage_regularizer = (
            torch.stack(usage_terms).mean() if usage_terms else hidden_states.new_zeros(())
        )
        self._last_adapter_metrics = {
            "adapter_scene_gate_mean": float(scene_gate_mean.detach().cpu()),
            "adapter_agent_gate_mean": float(agent_gate_mean.detach().cpu()),
            "adapter_output_gate_mean": float(output_gate_mean.detach().cpu()),
            "adapter_scene_delta_norm": float(scene_delta_norm.detach().cpu()),
            "adapter_agent_delta_norm": float(agent_delta_norm.detach().cpu()),
            "adapter_scene_route_scale_mean": float(scene_route_scale_per_sample.mean().detach().cpu()),
            "adapter_agent_route_scale_mean": float(agent_route_scale_per_sample.mean().detach().cpu()),
            "adapter_output_route_scale_mean": float(output_route_scale_per_sample.mean().detach().cpu()),
        }
        self._last_adapter_sample_metrics = {
            "adapter_scene_gate_mean": scene_gate_per_sample.detach(),
            "adapter_agent_gate_mean": agent_gate_per_sample.detach(),
            "adapter_output_gate_mean": output_gate_per_sample.detach(),
            "adapter_scene_delta_norm": scene_delta_norm_per_sample.detach(),
            "adapter_agent_delta_norm": agent_delta_norm_per_sample.detach(),
            "adapter_scene_route_scale_mean": scene_route_scale_per_sample.detach(),
            "adapter_agent_route_scale_mean": agent_route_scale_per_sample.detach(),
            "adapter_output_route_scale_mean": output_route_scale_per_sample.detach(),
        }
        return hidden_states + output_gate * fused_delta * output_route_scale_per_sample.view(-1, 1, 1)
