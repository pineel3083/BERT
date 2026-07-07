# tpdcim_bert_patch.py

import types
import torch
import torch.nn.functional as F

try:
    from transformers.cache_utils import EncoderDecoderCache
except Exception:
    EncoderDecoderCache = None


TCS_THRESHOLD_ORDER = "lsb_to_msb"
TCS_ACTIVE_RULE = "bsum > threshold"


def _zero_tcs_stats(enable_tcs=False, tcs_thresholds=None):
    return {
        "enable_tcs": enable_tcs,
        "tcs_thresholds": tcs_thresholds,
        "tcs_threshold_order": TCS_THRESHOLD_ORDER,
        "tcs_active_rule": TCS_ACTIVE_RULE,
        "bit_ops_total": 0,
        "bit_ops_skipped_by_tcs": 0,
        "tcs_rows_total": 0,
        "tcs_rows_skipped": 0,
        "tcs_row_skip_ratio": 0.0,
    }


def prepare_tpdcim_attention_mask(attention_mask, query):
    """
    Convert HF BERT masks into additive attention-score masks.

    Supported inputs:
        [B, N]          : 1=keep, 0=mask
        [B, 1, 1, N]    : already expanded
        [B, 1, N, N]    : already expanded

    Output values:
        keep position: 0
        masked position: -10000
    """
    if attention_mask is None:
        return None

    mask = attention_mask.to(device=query.device)

    # Already additive: keep=0, masked=-10000 or another large negative value.
    if mask.dtype.is_floating_point:
        if mask.numel() > 0 and mask.min().item() < 0:
            return mask.to(dtype=query.dtype)

    mask = mask.to(dtype=query.dtype)
    additive_mask = (1.0 - mask) * -10000.0

    if additive_mask.dim() == 2:
        additive_mask = additive_mask[:, None, None, :]
    elif additive_mask.dim() == 3:
        additive_mask = additive_mask[:, None, :, :]
    elif additive_mask.dim() == 4:
        pass
    else:
        raise ValueError(
            f"Unsupported attention_mask shape: {attention_mask.shape}"
        )

    return additive_mask


def make_sparse_block_mask(
    num_blocks,
    local_window=1,
    global_blocks=(0,),
    num_random_blocks=0,
    device=None,
):
    """
    BigBird-like block-level sparse attention mask.

    mask[qi, kj] = True  -> compute Q block qi against K block kj
    mask[qi, kj] = False -> skip that QK tile in the hardware-stat model
    """
    mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device=device)

    for qi in range(num_blocks):
        k0 = max(0, qi - local_window)
        k1 = min(num_blocks, qi + local_window + 1)
        mask[qi, k0:k1] = True

    for g in global_blocks:
        if 0 <= g < num_blocks:
            mask[g, :] = True
            mask[:, g] = True

    # Deterministic pseudo-random blocks keep evaluation reproducible.
    for qi in range(num_blocks):
        for r in range(num_random_blocks):
            kj = (qi * 1103515245 + 12345 + r * 97) % num_blocks
            mask[qi, kj] = True

    return mask


def qkt_tiled_fp32(
    query,
    key,
    scaling,
    tile_n=256,
    enable_sparse=False,
    local_window=1,
    global_blocks=(0,),
    num_random_blocks=0,
):
    """
    FP32 QK^T tile simulation.

    query/key shape: [B, H, N, D]
    dense mode computes all block tiles; sparse mode computes only block_mask=True
    tiles and leaves skipped tiles at -inf before softmax.
    """
    B, H, N, D = query.shape
    num_blocks = (N + tile_n - 1) // tile_n

    if enable_sparse:
        block_mask = make_sparse_block_mask(
            num_blocks=num_blocks,
            local_window=local_window,
            global_blocks=global_blocks,
            num_random_blocks=num_random_blocks,
            device=query.device,
        )
        scores = query.new_full((B, H, N, N), float("-inf"))
    else:
        block_mask = torch.ones(num_blocks, num_blocks, dtype=torch.bool, device=query.device)
        scores = query.new_empty(B, H, N, N)

    computed_tiles = 0
    skipped_tiles = 0

    for qi in range(num_blocks):
        q0 = qi * tile_n
        q1 = min(q0 + tile_n, N)
        q_tile = query[:, :, q0:q1, :]

        for kj in range(num_blocks):
            if not bool(block_mask[qi, kj]):
                skipped_tiles += 1
                continue

            k0 = kj * tile_n
            k1 = min(k0 + tile_n, N)
            k_tile = key[:, :, k0:k1, :]

            scores[:, :, q0:q1, k0:k1] = torch.matmul(
                q_tile,
                k_tile.transpose(2, 3),
            )
            computed_tiles += 1

    scores = scores * scaling
    total_tiles = num_blocks * num_blocks

    stats = {
        "N": N,
        "D": D,
        "tile_n": tile_n,
        "num_blocks": num_blocks,
        "qk_mode": "fp32",
        "quantization": "none",
        "enable_sparse": enable_sparse,
        "computed_qk_tiles": computed_tiles,
        "skipped_qk_tiles": skipped_tiles,
        "total_qk_tiles": total_tiles,
        "sparse_density": computed_tiles / total_tiles,
    }
    stats.update(_zero_tcs_stats())

    return scores, stats


def quantize_symmetric_int8(x, eps=1e-8):
    """
    Symmetric INT8 quantization.

    x_fp32 ~= x_int8 * scale, with x_int8 in [-127, 127]. This simulation uses
    one scale per tile to keep INT8 and bit-serial paths directly comparable.
    """
    max_abs = x.detach().abs().amax()
    scale = torch.clamp(max_abs / 127.0, min=eps)

    x_int8 = torch.clamp(
        torch.round(x / scale),
        -127,
        127,
    ).to(torch.int8)

    return x_int8, scale


def qkt_tiled_int8(
    query,
    key,
    scaling,
    tile_n=256,
    enable_sparse=False,
    local_window=1,
    global_blocks=(0,),
    num_random_blocks=0,
):
    """INT8 simulated QK^T tiling without bit-serial decomposition."""
    B, H, N, D = query.shape
    num_blocks = (N + tile_n - 1) // tile_n

    if enable_sparse:
        block_mask = make_sparse_block_mask(
            num_blocks=num_blocks,
            local_window=local_window,
            global_blocks=global_blocks,
            num_random_blocks=num_random_blocks,
            device=query.device,
        )
        scores = query.new_full((B, H, N, N), float("-inf"))
    else:
        block_mask = torch.ones(
            num_blocks,
            num_blocks,
            dtype=torch.bool,
            device=query.device,
        )
        scores = query.new_empty(B, H, N, N)

    computed_tiles = 0
    skipped_tiles = 0

    for qi in range(num_blocks):
        q0 = qi * tile_n
        q1 = min(q0 + tile_n, N)
        q_tile = query[:, :, q0:q1, :]
        q_int8, q_scale = quantize_symmetric_int8(q_tile)

        for kj in range(num_blocks):
            if not bool(block_mask[qi, kj]):
                skipped_tiles += 1
                continue

            k0 = kj * tile_n
            k1 = min(k0 + tile_n, N)
            k_tile = key[:, :, k0:k1, :]
            k_int8, k_scale = quantize_symmetric_int8(k_tile)

            # PyTorch may not support the desired INT8 matmul on every device, so
            # this is a functional simulation over quantized integer values.
            score_int_like = torch.matmul(
                q_int8.to(torch.float32),
                k_int8.to(torch.float32).transpose(2, 3),
            )
            score_fp = score_int_like.to(query.dtype) * (q_scale * k_scale)
            scores[:, :, q0:q1, k0:k1] = score_fp
            computed_tiles += 1

    scores = scores * scaling
    total_tiles = num_blocks * num_blocks

    stats = {
        "N": N,
        "D": D,
        "tile_n": tile_n,
        "num_blocks": num_blocks,
        "qk_mode": "int8",
        "quantization": "per_tile_symmetric_int8",
        "enable_sparse": enable_sparse,
        "computed_qk_tiles": computed_tiles,
        "skipped_qk_tiles": skipped_tiles,
        "total_qk_tiles": total_tiles,
        "sparse_density": computed_tiles / total_tiles,
    }
    stats.update(_zero_tcs_stats())

    return scores, stats


def qkt_tiled_bitserial(
    query,
    key,
    scaling,
    tile_n=256,
    enable_sparse=False,
    local_window=1,
    global_blocks=(0,),
    num_random_blocks=0,
    enable_tcs=False,
    tcs_thresholds=None,
):
    """
    Reconstruct INT8 QK^T using bit-serial Q input planes.

    TCS convention is intentionally unchanged from the current experiments:
    thresholds are interpreted LSB->MSB, and rows are active when
    bsum > threshold. Rows with bsum <= threshold are zeroed in the hardware
    activity model before the bit-plane matmul.
    """
    B, H, N, D = query.shape
    num_blocks = (N + tile_n - 1) // tile_n

    if tcs_thresholds is None:
        tcs_thresholds = [0] * 8
    if len(tcs_thresholds) != 8:
        raise ValueError("tcs_thresholds must contain 8 values in LSB-to-MSB order")

    if enable_sparse:
        block_mask = make_sparse_block_mask(
            num_blocks=num_blocks,
            local_window=local_window,
            global_blocks=global_blocks,
            num_random_blocks=num_random_blocks,
            device=query.device,
        )
        scores = query.new_full((B, H, N, N), float("-inf"))
    else:
        block_mask = torch.ones(
            num_blocks,
            num_blocks,
            dtype=torch.bool,
            device=query.device,
        )
        scores = query.new_empty(B, H, N, N)

    computed_tiles = 0
    skipped_tiles = 0
    bit_ops_total = 0
    bit_ops_skipped_by_tcs = 0
    tcs_rows_total = 0
    tcs_rows_skipped = 0

    for qi in range(num_blocks):
        q0 = qi * tile_n
        q1 = min(q0 + tile_n, N)
        q_tile = query[:, :, q0:q1, :]
        q_int8, q_scale = quantize_symmetric_int8(q_tile)
        q_uint8 = torch.bitwise_and(q_int8.to(torch.int16), 255)

        for kj in range(num_blocks):
            if not bool(block_mask[qi, kj]):
                skipped_tiles += 1
                continue

            k0 = kj * tile_n
            k1 = min(k0 + tile_n, N)
            k_tile = key[:, :, k0:k1, :]
            k_int8, k_scale = quantize_symmetric_int8(k_tile)
            k_fp = k_int8.to(torch.float32)

            acc = torch.zeros(
                (B, H, q1 - q0, k1 - k0),
                device=query.device,
                dtype=torch.float32,
            )

            for bit in range(8):
                bit_ops_total += 1

                q_bit = torch.bitwise_and(
                    torch.bitwise_right_shift(q_uint8, bit),
                    1,
                ).to(torch.float32)

                # Hardware proxy: sum the 64 bit-serial inputs per row. This is
                # activity/energy accounting, not a CUDA speedup claim.
                bsum = q_bit.sum(dim=-1)

                if enable_tcs:
                    threshold = tcs_thresholds[bit]
                    active_row = bsum > threshold

                    tcs_rows_total += active_row.numel()
                    tcs_rows_skipped += active_row.numel() - active_row.sum().item()

                    if not bool(active_row.any()):
                        bit_ops_skipped_by_tcs += 1
                        continue

                    q_bit = q_bit * active_row.unsqueeze(-1).to(q_bit.dtype)

                bit_weight = float(1 << bit) if bit < 7 else -128.0
                partial = torch.matmul(
                    q_bit,
                    k_fp.transpose(2, 3),
                )
                acc = acc + bit_weight * partial

            score_fp = acc.to(query.dtype) * (q_scale * k_scale)
            scores[:, :, q0:q1, k0:k1] = score_fp
            computed_tiles += 1

    scores = scores * scaling
    total_tiles = num_blocks * num_blocks

    stats = {
        "N": N,
        "D": D,
        "tile_n": tile_n,
        "num_blocks": num_blocks,
        "qk_mode": "bitserial",
        "quantization": "per_tile_symmetric_int8",
        "enable_sparse": enable_sparse,
        "computed_qk_tiles": computed_tiles,
        "skipped_qk_tiles": skipped_tiles,
        "total_qk_tiles": total_tiles,
        "sparse_density": computed_tiles / total_tiles,
        "enable_tcs": enable_tcs,
        "tcs_thresholds": tcs_thresholds,
        "tcs_threshold_order": TCS_THRESHOLD_ORDER,
        "tcs_active_rule": TCS_ACTIVE_RULE,
        "bit_ops_total": bit_ops_total,
        "bit_ops_skipped_by_tcs": bit_ops_skipped_by_tcs,
        "tcs_rows_total": tcs_rows_total,
        "tcs_rows_skipped": tcs_rows_skipped,
        "tcs_row_skip_ratio": (
            0.0 if tcs_rows_total == 0 else tcs_rows_skipped / tcs_rows_total
        ),
    }

    return scores, stats


def tpdcim_eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    """
    HF eager attention equivalent with only QK^T replaced by TP-DCIM stats paths.
    AV remains functionally unchanged: attn_weights @ value.
    """
    use_qk_tiling = getattr(module, "tpdcim_enable_qk_tiling", False)

    if use_qk_tiling:
        tile_n = getattr(module, "tpdcim_tile_n", 256)
        qk_mode = getattr(module, "tpdcim_qk_mode", "fp32")

        common_kwargs = dict(
            query=query,
            key=key,
            scaling=scaling,
            tile_n=tile_n,
            enable_sparse=getattr(module, "tpdcim_enable_sparse", False),
            local_window=getattr(module, "tpdcim_local_window", 1),
            global_blocks=getattr(module, "tpdcim_global_blocks", (0,)),
            num_random_blocks=getattr(module, "tpdcim_num_random_blocks", 0),
        )

        if qk_mode == "fp32":
            attn_weights, stats = qkt_tiled_fp32(**common_kwargs)
        elif qk_mode == "int8":
            attn_weights, stats = qkt_tiled_int8(**common_kwargs)
        elif qk_mode == "bitserial":
            attn_weights, stats = qkt_tiled_bitserial(
                **common_kwargs,
                enable_tcs=getattr(module, "tpdcim_enable_tcs", False),
                tcs_thresholds=getattr(module, "tpdcim_tcs_thresholds", None),
            )
        else:
            raise ValueError(f"Unsupported tpdcim_qk_mode: {qk_mode}")
        module.tpdcim_last_stats = stats
    else:
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        module.tpdcim_last_stats = None

    attention_mask = prepare_tpdcim_attention_mask(attention_mask, query)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(
        attn_weights,
        p=dropout,
        training=module.training,
    )

    head_mask = kwargs.get("head_mask", None)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    # AV is intentionally left exact/unchanged for now. A-stationary AV should be
    # added later as a stats-only utilization model unless functional tiling is needed.
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def patched_bert_self_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    past_key_values=None,
    **kwargs,
):
    """Instance-level BertSelfAttention.forward patch."""
    if past_key_values is None and "past_key_value" in kwargs:
        past_key_values = kwargs.pop("past_key_value")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.attention_head_size)

    query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
    key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
    value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

    if past_key_values is not None:
        current_past_key_values = past_key_values
        if EncoderDecoderCache is not None and isinstance(past_key_values, EncoderDecoderCache):
            current_past_key_values = past_key_values.self_attention_cache

        key_layer, value_layer = current_past_key_values.update(
            key_layer,
            value_layer,
            self.layer_idx,
        )

    attn_output, attn_weights = tpdcim_eager_attention_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        dropout=0.0 if not self.training else self.dropout.p,
        scaling=self.scaling,
        head_mask=head_mask,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return attn_output, attn_weights


def enable_qk_tiling(
    model,
    tile_n=256,
    enable_sparse=False,
    local_window=1,
    global_blocks=(0,),
    num_random_blocks=0,
    qk_mode="fp32",
    enable_tcs=False,
    tcs_thresholds=None,
):
    """Patch all BertSelfAttention modules with TP-DCIM QK simulation settings."""
    if qk_mode not in ("fp32", "int8", "bitserial"):
        raise ValueError("qk_mode must be one of: fp32, int8, bitserial")
    if enable_tcs and tcs_thresholds is not None and len(tcs_thresholds) != 8:
        raise ValueError("tcs_thresholds must contain 8 values in LSB-to-MSB order")

    if hasattr(model, "bert"):
        layers = model.bert.encoder.layer
    else:
        layers = model.encoder.layer

    for layer in layers:
        attn = layer.attention.self

        attn.tpdcim_enable_qk_tiling = True
        attn.tpdcim_tile_n = tile_n
        attn.tpdcim_enable_sparse = enable_sparse
        attn.tpdcim_local_window = local_window
        attn.tpdcim_global_blocks = global_blocks
        attn.tpdcim_num_random_blocks = num_random_blocks
        attn.tpdcim_qk_mode = qk_mode
        attn.tpdcim_enable_tcs = enable_tcs
        attn.tpdcim_tcs_thresholds = tcs_thresholds
        attn.tpdcim_tcs_threshold_order = TCS_THRESHOLD_ORDER
        attn.tpdcim_tcs_active_rule = TCS_ACTIVE_RULE
        attn.tpdcim_last_stats = None
        attn.forward = types.MethodType(patched_bert_self_attention_forward, attn)


def disable_qk_tiling(model):
    """Disable QK tiling flags without attempting to restore original bound methods."""
    if hasattr(model, "bert"):
        layers = model.bert.encoder.layer
    else:
        layers = model.encoder.layer

    for layer in layers:
        attn = layer.attention.self
        attn.tpdcim_enable_qk_tiling = False
        attn.tpdcim_last_stats = None
