import torch
from torch import nn

from .config import BertConfig


def _gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 2.0**0.5))


def _activation(name):
    if name == "gelu":
        return _gelu
    if name == "relu":
        return torch.relu
    raise ValueError("Only 'gelu' and 'relu' activations are supported")


class BertEmbeddings(nn.Module):
    """Word, position, and segment embeddings, matching BERT's first block."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (batch_size, seq_len), dtype=torch.long, device=input_ids.device
            )

        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.layer_norm(embeddings))


class TPDCIMMath:
    """Behavioral TP-DCIM matmul helpers.

    The helpers keep ordinary PyTorch tensors for training compatibility, but
    the loops and modes follow the paper's hardware view:
    - VIHA for QK^T uses stored K rows without an explicit transpose buffer.
    - A-stationary AV keeps attention blocks stationary and streams V blocks.
    - TCS optionally removes low-activity bit-serial query planes before QK^T.
    """

    def __init__(self, config):
        self.config = config
        self.stats = {}

    def reset(self):
        self.stats = {}

    def _add_stat(self, name, value):
        if self.config.dcim_collect_stats:
            self.stats[name] = self.stats.get(name, 0) + int(value)

    def threshold_compute_skip(self, query):
        thresholds = self.config.dcim_tcs_thresholds
        if thresholds is None:
            return query
        if len(thresholds) != 8:
            raise ValueError("dcim_tcs_thresholds must contain 8 MSB-to-LSB values")

        # TCS is an INT8 bit-serial approximation: low-popcount bit planes are
        # deactivated before entering the DCIM macro.
        scale = query.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        q_int = torch.clamp((query.detach().abs() / scale * 127).round(), 0, 127).to(
            torch.int16
        )
        sign = torch.sign(query)
        kept = torch.zeros_like(q_int)
        skipped_planes = 0

        for bit in range(7, -1, -1):
            plane = (q_int >> bit) & 1
            bit_sum = plane.sum(dim=-1, keepdim=True)
            threshold = thresholds[7 - bit]
            active = bit_sum > threshold
            kept = kept + plane * active.to(torch.int16) * (1 << bit)
            skipped_planes += int((~active).sum().item())

        self._add_stat("tcs_skipped_bit_planes", skipped_planes)
        skipped = sign * kept.to(query.dtype) * scale / 127.0
        return query + (skipped - query).detach()

    def qk(self, query, key):
        if self.config.dcim_qk_mode == "hiva":
            scores = torch.matmul(query, key.transpose(-1, -2))
            self._add_stat("qk_hiva_blocks", 1)
            return scores

        # VIHA models the transposable adder-tree path: K stays row-major in the
        # macro, while accumulation happens horizontally. Slicing avoids creating
        # a software transpose buffer for the whole K matrix.
        parts = []
        block = self.config.dcim_block_size
        for start in range(0, key.size(-2), block):
            k_block = key[..., start : start + block, :]
            parts.append(torch.einsum("bhqd,bhkd->bhqk", query, k_block))
            self._add_stat("qk_viha_blocks", 1)
        return torch.cat(parts, dim=-1)

    def av(self, attention_probs, value):
        if self.config.dcim_av_mode == "v_stationary":
            self._add_stat("av_v_stationary_blocks", 1)
            return torch.matmul(attention_probs, value)

        # A-stationary keeps an attention tile in the QKG core and streams V
        # through the VG core, reducing the V-stationary underutilization noted
        # in the paper.
        output = torch.zeros(
            *attention_probs.shape[:-1],
            value.size(-1),
            dtype=value.dtype,
            device=value.device,
        )
        block = self.config.dcim_block_size
        for start in range(0, value.size(-2), block):
            a_block = attention_probs[..., start : start + block]
            v_block = value[..., start : start + block, :]
            output = output + torch.matmul(a_block, v_block)
            self._add_stat("av_a_stationary_blocks", 1)
        return output


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dcim = TPDCIMMath(config)

    def _transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self.dcim.reset()
        query = self._transpose_for_scores(self.query(hidden_states))
        key = self._transpose_for_scores(self.key(hidden_states))
        value = self._transpose_for_scores(self.value(hidden_states))
        query = self.dcim.threshold_compute_skip(query)

        attention_scores = self.dcim.qk(query, key)
        attention_scores = attention_scores / (self.attention_head_size**0.5)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = self.dcim.av(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), self.all_head_size)

        if output_attentions:
            return context, attention_probs, self.dcim.stats
        return context, None, self.dcim.stats


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.layer_norm(hidden_states + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_output, attn_probs, stats = self.self(
            hidden_states, attention_mask, output_attentions
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output, attn_probs, stats


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = _activation(config.hidden_act)

    def forward(self, hidden_states):
        return self.act(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.layer_norm(hidden_states + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attention_output, attn_probs, stats = self.attention(
            hidden_states, attention_mask, output_attentions
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attn_probs, stats


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False,
    ):
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        dcim_stats = []

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, attn_probs, stats = layer_module(
                hidden_states, attention_mask, output_attentions
            )
            if output_attentions:
                all_attentions.append(attn_probs)
            dcim_stats.append(stats)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states, all_attentions, dcim_stats


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token = hidden_states[:, 0]
        return self.activation(self.dense(first_token))


class BertModel(nn.Module):
    """Minimal BERT encoder with TP-DCIM-aware self-attention."""

    def __init__(self, config=None, add_pooling_layer=True):
        super().__init__()
        self.config = config if config is not None else BertConfig()
        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config) if add_pooling_layer else None
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _extend_attention_mask(self, attention_mask, input_shape, device):
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        mask = attention_mask[:, None, None, :].to(
            dtype=self.embeddings.word_embeddings.weight.dtype
        )
        return (1.0 - mask) * -10000.0

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_hidden_states=False,
        output_attentions=False,
    ):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        extended_attention_mask = self._extend_attention_mask(
            attention_mask, input_ids.shape, input_ids.device
        )
        sequence_output, hidden_states, attentions, dcim_stats = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
            "hidden_states": hidden_states,
            "attentions": attentions,
            "dcim_stats": dcim_stats,
        }
