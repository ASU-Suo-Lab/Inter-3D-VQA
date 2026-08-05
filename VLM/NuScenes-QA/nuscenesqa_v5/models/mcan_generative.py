from __future__ import annotations

from dataclasses import dataclass, asdict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nuscenesqa_v5.data.dataset import PAD_ID


@dataclass
class MCANGenerativeConfig:
    object_feature_dim: int
    bbox_feature_dim: int
    hidden_size: int
    char_embed_size: int
    bbox_embed_size: int
    layers: int
    multi_head: int
    ff_size: int
    dropout: float
    flat_mlp_size: int
    flat_glimpses: int
    vocab_size: int

    def to_dict(self) -> dict:
        return asdict(self)


class FC(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout_r: float = 0.0, use_relu: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU(inplace=True) if use_relu else None
        self.dropout = nn.Dropout(dropout_r) if dropout_r > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_size: int, mid_size: int, out_size: int, dropout_r: float = 0.0, use_relu: bool = True) -> None:
        super().__init__()
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MHAtt(nn.Module):
    def __init__(self, config: MCANGenerativeConfig) -> None:
        super().__init__()
        self.config = config
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_merge = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        batch_size = query.size(0)
        head_dim = self.config.hidden_size // self.config.multi_head
        value = self.linear_v(value).view(batch_size, -1, self.config.multi_head, head_dim).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.config.multi_head, head_dim).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, self.config.multi_head, head_dim).transpose(1, 2)
        attended = self.att(value, key, query, mask)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.config.hidden_size)
        return self.linear_merge(attended)

    def att(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self, config: MCANGenerativeConfig) -> None:
        super().__init__()
        self.mlp = MLP(config.hidden_size, config.ff_size, config.hidden_size, dropout_r=config.dropout, use_relu=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SA(nn.Module):
    def __init__(self, config: MCANGenerativeConfig) -> None:
        super().__init__()
        self.mhatt = MHAtt(config)
        self.ffn = FFN(config)
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = LayerNorm(config.hidden_size)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = LayerNorm(config.hidden_size)

    def forward(self, y: torch.Tensor, y_mask: torch.Tensor | None) -> torch.Tensor:
        y = self.norm1(y + self.dropout1(self.mhatt(y, y, y, y_mask)))
        y = self.norm2(y + self.dropout2(self.ffn(y)))
        return y


class SGA(nn.Module):
    def __init__(self, config: MCANGenerativeConfig) -> None:
        super().__init__()
        self.mhatt1 = MHAtt(config)
        self.mhatt2 = MHAtt(config)
        self.ffn = FFN(config)
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = LayerNorm(config.hidden_size)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = LayerNorm(config.hidden_size)
        self.dropout3 = nn.Dropout(config.dropout)
        self.norm3 = LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_mask: torch.Tensor | None, y_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class MCA_ED(nn.Module):
    def __init__(self, config: MCANGenerativeConfig) -> None:
        super().__init__()
        self.enc_list = nn.ModuleList([SA(config) for _ in range(config.layers)])
        self.dec_list = nn.ModuleList([SGA(config) for _ in range(config.layers)])

    def forward(self, y: torch.Tensor, x: torch.Tensor, y_mask: torch.Tensor | None, x_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        for enc in self.enc_list:
            y = enc(y, y_mask)
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)
        return y, x


class AttFlat(nn.Module):
    def __init__(self, config: MCANGenerativeConfig) -> None:
        super().__init__()
        self.config = config
        self.mlp = MLP(config.hidden_size, config.flat_mlp_size, config.flat_glimpses, dropout_r=config.dropout, use_relu=True)
        self.linear_merge = nn.Linear(config.hidden_size * config.flat_glimpses, config.hidden_size)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None) -> torch.Tensor:
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)
        att_list = [torch.sum(att[:, :, i : i + 1] * x, dim=1) for i in range(self.config.flat_glimpses)]
        x_atted = torch.cat(att_list, dim=1)
        return self.linear_merge(x_atted)


def make_mask_from_zeros(feature: torch.Tensor) -> torch.Tensor:
    return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)


def make_mask_from_pad(ids: torch.Tensor) -> torch.Tensor:
    return ids.eq(PAD_ID).unsqueeze(1).unsqueeze(2)


class MCANGenerativeQA(nn.Module):
    def __init__(self, config: MCANGenerativeConfig) -> None:
        super().__init__()
        self.config = config
        question_hidden = config.hidden_size // 2
        self.char_embedding = nn.Embedding(config.vocab_size, config.char_embed_size, padding_idx=PAD_ID)
        self.question_encoder = nn.LSTM(
            input_size=config.char_embed_size,
            hidden_size=question_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bbox_linear = nn.Linear(config.bbox_feature_dim, config.bbox_embed_size)
        self.object_linear = nn.Linear(config.object_feature_dim + config.bbox_embed_size, config.hidden_size)
        self.backbone = MCA_ED(config)
        self.attflat_img = AttFlat(config)
        self.attflat_lang = AttFlat(config)
        self.proj_norm = LayerNorm(config.hidden_size)
        self.decoder_init = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder = nn.GRU(config.char_embed_size, config.hidden_size, batch_first=True)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size)

    def encode_context(
        self,
        object_features: torch.Tensor,
        bbox_features: torch.Tensor,
        question_ids: torch.Tensor,
    ) -> torch.Tensor:
        question_mask = make_mask_from_pad(question_ids)
        question_emb = self.char_embedding(question_ids)
        question_feat, _ = self.question_encoder(question_emb)
        object_mask = make_mask_from_zeros(object_features)
        bbox_emb = self.bbox_linear(bbox_features.to(torch.float32))
        object_feat = self.object_linear(torch.cat([object_features.to(torch.float32), bbox_emb], dim=-1))
        question_feat, object_feat = self.backbone(question_feat, object_feat, question_mask, object_mask)
        lang_flat = self.attflat_lang(question_feat, question_mask)
        obj_flat = self.attflat_img(object_feat, object_mask)
        return self.proj_norm(lang_flat + obj_flat)

    def forward(
        self,
        object_features: torch.Tensor,
        bbox_features: torch.Tensor,
        question_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_target_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context = self.encode_context(object_features, bbox_features, question_ids)
        hidden = torch.tanh(self.decoder_init(context)).unsqueeze(0)
        decoder_emb = self.char_embedding(decoder_input_ids)
        decoder_outputs, _ = self.decoder(decoder_emb, hidden)
        logits = self.output_proj(decoder_outputs)
        output = {"logits": logits}
        if decoder_target_ids is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                decoder_target_ids.reshape(-1),
                ignore_index=PAD_ID,
            )
            output["loss"] = loss
        return output

    @torch.no_grad()
    def generate(
        self,
        object_features: torch.Tensor,
        bbox_features: torch.Tensor,
        question_ids: torch.Tensor,
        decoder_prefix_ids: torch.Tensor | None,
        decoder_prefix_mask: torch.Tensor | None,
        max_new_tokens: int,
        bos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        context = self.encode_context(object_features, bbox_features, question_ids)
        hidden0 = torch.tanh(self.decoder_init(context)).unsqueeze(0)
        batch_size = question_ids.shape[0]
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=question_ids.device)
        hidden = hidden0
        if decoder_prefix_ids is not None and decoder_prefix_mask is not None:
            prefix_ids = decoder_prefix_ids.to(question_ids.device)
            prefix_mask = decoder_prefix_mask.to(question_ids.device)
            warm_tokens = torch.cat([generated, prefix_ids], dim=1)
            warm_lengths = prefix_mask.to(torch.long).sum(dim=1) + 1
            packed = pack_padded_sequence(
                self.char_embedding(warm_tokens),
                lengths=warm_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, hidden = self.decoder(packed, hidden0)
            warm_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
            last_indices = (warm_lengths - 1).to(question_ids.device)
            batch_indices = torch.arange(batch_size, device=question_ids.device)
            logits = self.output_proj(warm_outputs[batch_indices, last_indices])
            next_token = torch.argmax(logits, dim=-1)
            generated = torch.cat([generated, prefix_ids], dim=1)
        else:
            decoder_emb = self.char_embedding(generated[:, -1:])
            decoder_outputs, hidden = self.decoder(decoder_emb, hidden)
            logits = self.output_proj(decoder_outputs[:, -1])
            next_token = torch.argmax(logits, dim=-1)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=question_ids.device)
        for _ in range(max_new_tokens):
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos_id)
            if torch.all(finished):
                break
            decoder_emb = self.char_embedding(next_token.unsqueeze(1))
            decoder_outputs, hidden = self.decoder(decoder_emb, hidden)
            logits = self.output_proj(decoder_outputs[:, -1])
            next_token = torch.argmax(logits, dim=-1)
        return generated[:, 1:]
