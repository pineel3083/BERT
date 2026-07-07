# tpdcim_bert_patch.py

import types
import torch
import torch.nn.functional as F

try:
    from transformers.cache_utils import EncoderDecoderCache
except Exception:
    EncoderDecoderCache = None


def prepare_tpdcim_attention_mask(attention_mask, query):
    """
    HF BERT에서 넘어오는 attention_mask를
    attention score에 더할 수 있는 additive mask로 변환한다.

    입력 가능 형태:
        [B, N]          : 1=keep, 0=mask
        [B, 1, 1, N]    : 이미 확장된 mask
        [B, 1, N, N]    : 이미 확장된 mask

    출력:
        [B, 1, 1, N] 또는 [B, 1, N, N]
        keep 위치: 0
        mask 위치: -10000
    """
    if attention_mask is None:
        return None

    mask = attention_mask.to(device=query.device)

    # 이미 additive mask인 경우
    # 보통 keep=0, masked=-10000 또는 매우 작은 음수
    if mask.dtype.is_floating_point:
        if mask.numel() > 0 and mask.min().item() < 0:
            return mask.to(dtype=query.dtype)

    # 여기부터는 1/0 mask라고 가정
    # 1 = keep, 0 = mask
    mask = mask.to(dtype=query.dtype)

    additive_mask = (1.0 - mask) * -10000.0

    if additive_mask.dim() == 2:
        # [B, N] -> [B, 1, 1, N]
        additive_mask = additive_mask[:, None, None, :]

    elif additive_mask.dim() == 3:
        # [B, N, N] -> [B, 1, N, N]
        additive_mask = additive_mask[:, None, :, :]

    elif additive_mask.dim() == 4:
        # 이미 broadcast 가능한 형태
        pass

    else:
        raise ValueError(
            f"Unsupported attention_mask shape: {attention_mask.shape}"
        )

    return additive_mask

# def qkt_tiled_fp32(query, key, scaling, tile_n=256):
#     """
#     query: [B, H, N, D]
#     key:   [B, H, N, D]

#     return:
#         scores: [B, H, N, N]

#     원래:
#         scores = query @ key.transpose(-1, -2)

#     변경:
#         Q, K를 token tile 단위로 잘라서
#         scores[:, :, q0:q1, k0:k1] = Q_tile @ K_tile.T
#     """
#     B, H, N, D = query.shape

#     scores = query.new_empty(B, H, N, N)

#     computed_tiles = 0

#     for q0 in range(0, N, tile_n):
#         q1 = min(q0 + tile_n, N)
#         q_tile = query[:, :, q0:q1, :]  # [B, H, Tq, D]

#         for k0 in range(0, N, tile_n):
#             k1 = min(k0 + tile_n, N)
#             k_tile = key[:, :, k0:k1, :]  # [B, H, Tk, D]

#             scores[:, :, q0:q1, k0:k1] = torch.matmul(
#                 q_tile,
#                 k_tile.transpose(2, 3),
#             )

#             computed_tiles += 1

#     scores = scores * scaling

#     stats = {
#         "N": N,
#         "D": D,
#         "tile_n": tile_n,
#         "computed_qk_tiles": computed_tiles,
#     }

#     return scores, stats

def make_sparse_block_mask(
    num_blocks,
    local_window=1,
    global_blocks=(0,),
    num_random_blocks=0,
    device=None,
):
    """
    block-level sparse attention mask.

    mask[qi, kj] = True  -> Q block qi가 K block kj를 계산
    mask[qi, kj] = False -> 계산 skip
    """
    mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device=device)

    # 1. local window
    # 예: local_window=1이면 i-1, i, i+1 block 계산
    for qi in range(num_blocks):
        k0 = max(0, qi - local_window)
        k1 = min(num_blocks, qi + local_window + 1)
        mask[qi, k0:k1] = True

    # 2. global block
    # 예: block 0은 모든 block과 연결
    for g in global_blocks:
        if 0 <= g < num_blocks:
            mask[g, :] = True
            mask[:, g] = True

    # 3. deterministic random block
    # 매번 랜덤이면 결과가 바뀌니까 고정 패턴 사용
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
    query: [B, H, N, D]
    key:   [B, H, N, D]

    dense mode:
        모든 Q/K block 계산

    sparse mode:
        block_mask=True인 block만 계산
        나머지 score는 -inf 유지
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

        # sparse에서 계산하지 않는 곳은 softmax 후 0이 되어야 하므로 -inf
        scores = query.new_full((B, H, N, N), float("-inf"))
    else:
        block_mask = torch.ones(num_blocks, num_blocks, dtype=torch.bool, device=query.device)
        scores = query.new_empty(B, H, N, N)

    computed_tiles = 0
    skipped_tiles = 0

    for qi in range(num_blocks):
        q0 = qi * tile_n
        q1 = min(q0 + tile_n, N)

        q_tile = query[:, :, q0:q1, :]  # [B, H, Tq, D]

        for kj in range(num_blocks):
            if not bool(block_mask[qi, kj]):
                skipped_tiles += 1
                continue

            k0 = kj * tile_n
            k1 = min(k0 + tile_n, N)

            k_tile = key[:, :, k0:k1, :]  # [B, H, Tk, D]

            scores[:, :, q0:q1, k0:k1] = torch.matmul(
                q_tile,
                k_tile.transpose(2, 3),
            )

            computed_tiles += 1

    scores = scores * scaling

    stats = {
        "N": N,
        "D": D,
        "tile_n": tile_n,
        "num_blocks": num_blocks,
        "enable_sparse": enable_sparse,
        "computed_qk_tiles": computed_tiles,
        "skipped_qk_tiles": skipped_tiles,
        "total_qk_tiles": num_blocks * num_blocks,
        "sparse_density": computed_tiles / (num_blocks * num_blocks),
    }

    return scores, stats

def quantize_symmetric_int8(x, eps=1e-8):
    """
    symmetric INT8 quantization.

    x_fp32 ≈ x_int8 * scale
    x_int8 range: -127 ~ 127

    여기서는 per-tile scale 하나를 사용한다.
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
    """
    INT8 simulated QK tiling.

    query: [B, H, N, D]
    key:   [B, H, N, D]

    각 Q/K tile을 INT8로 quantize한 뒤:
        Q_int8 @ K_int8.T
    를 수행하고 다시 float score로 dequantize한다.

    아직 bit-serial/TCS는 아님.
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

        q_tile = query[:, :, q0:q1, :]  # [B, H, Tq, D]
        q_int8, q_scale = quantize_symmetric_int8(q_tile)

        for kj in range(num_blocks):
            if not bool(block_mask[qi, kj]):
                skipped_tiles += 1
                continue

            k0 = kj * tile_n
            k1 = min(k0 + tile_n, N)

            k_tile = key[:, :, k0:k1, :]  # [B, H, Tk, D]
            k_int8, k_scale = quantize_symmetric_int8(k_tile)

            # PyTorch CUDA에서 int8 matmul 지원이 제한적일 수 있어서
            # functional simulation은 float matmul로 수행.
            # 값 자체는 INT8로 quantized된 값이다.
            score_int_like = torch.matmul(
                q_int8.to(torch.float32),
                k_int8.to(torch.float32).transpose(2, 3),
            )

            score_fp = score_int_like.to(query.dtype) * (q_scale * k_scale)

            scores[:, :, q0:q1, k0:k1] = score_fp

            computed_tiles += 1

    scores = scores * scaling

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
        "total_qk_tiles": num_blocks * num_blocks,
        "sparse_density": computed_tiles / (num_blocks * num_blocks),
    }

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
    B, H, N, D = query.shape
    num_blocks = (N + tile_n - 1) // tile_n

    if tcs_thresholds is None:
        tcs_thresholds = [0] * 8

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

                # TCS: row마다 bit input의 1 개수 계산
                # q_bit shape: [B, H, Tq, D]
                # bsum shape : [B, H, Tq]
                bsum = q_bit.sum(dim=-1)

                if enable_tcs:
                    threshold = tcs_thresholds[bit]

                    # bsum > threshold 인 row만 계산 유지
                    active_row = bsum > threshold

                    tcs_rows_total += active_row.numel()
                    tcs_rows_skipped += active_row.numel() - active_row.sum().item()

                    # 전부 inactive면 이 bit-plane matmul 자체 skip
                    if not bool(active_row.any()):
                        bit_ops_skipped_by_tcs += 1
                        continue

                    # inactive row는 q_bit를 0으로 만들어 결과 기여 제거
                    q_bit = q_bit * active_row.unsqueeze(-1).to(q_bit.dtype)

                if bit < 7:
                    bit_weight = float(1 << bit)
                else:
                    bit_weight = -128.0

                partial = torch.matmul(
                    q_bit,
                    k_fp.transpose(2, 3),
                )

                acc = acc + bit_weight * partial

            score_fp = acc.to(query.dtype) * (q_scale * k_scale)
            scores[:, :, q0:q1, k0:k1] = score_fp

            computed_tiles += 1

    scores = scores * scaling

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
        "total_qk_tiles": num_blocks * num_blocks,
        "sparse_density": computed_tiles / (num_blocks * num_blocks),
        "bit_ops_total": bit_ops_total,
        "enable_tcs": enable_tcs,
        "tcs_thresholds": tcs_thresholds,
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
    기존 eager_attention_forward를 흉내 내되,
    QK^T 부분만 tiled version으로 바꾼 함수.

    query/key/value:
        [B, H, N, D]
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
            stats["qk_mode"] = "fp32"
        elif qk_mode == "int8":
            attn_weights, stats = qkt_tiled_int8(**common_kwargs)
            stats["qk_mode"] = "int8"
        elif qk_mode == "bitserial":
            attn_weights, stats = qkt_tiled_bitserial(
                **common_kwargs,
                enable_tcs=getattr(module, "tpdcim_enable_tcs", False),
                tcs_thresholds=getattr(module, "tpdcim_tcs_thresholds", None),
            )
            stats["qk_mode"] = "bitserial"
        else:
            raise ValueError(f"Unsupported tpdcim_qk_mode: {qk_mode}")
        module.tpdcim_last_stats = stats

    else:
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        module.tpdcim_last_stats = None

    attention_mask = prepare_tpdcim_attention_mask(
        attention_mask,
        query,
    )

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1)

    attn_weights = F.dropout(
        attn_weights,
        p=dropout,
        training=module.training,
    )

    # A × V는 아직 원본 방식 유지
    attn_output = torch.matmul(attn_weights, value)

    # [B, H, N, D] -> [B, N, H, D]
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
    """
    Hugging Face BertSelfAttention.forward를 instance 단위로 patch하는 함수.

    원본과 같은 흐름:
        hidden_states
        -> Q/K/V projection
        -> attention
        -> attn_output reshape
    """
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


# def enable_qk_tiling(model, tile_n=256):
#     """
#     Hugging Face BertForMaskedLM / BertModel 안의 모든 BertSelfAttention에
#     QK tiling patch를 적용한다.
#     """
#     if hasattr(model, "bert"):
#         layers = model.bert.encoder.layer
#     else:
#         layers = model.encoder.layer

#     for layer in layers:
#         attn = layer.attention.self

#         attn.tpdcim_enable_qk_tiling = True
#         attn.tpdcim_tile_n = tile_n
#         attn.tpdcim_last_stats = None

#         # 이 instance의 forward만 우리가 만든 함수로 교체
#         attn.forward = types.MethodType(patched_bert_self_attention_forward, attn)

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
    """
    모든 BertSelfAttention에 QK tiling patch 적용.
    sparse 옵션도 여기서 같이 설정.
    """
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
        attn.tpdcim_qk_mode = qk_mode # For qk quantization
        attn.tpdcim_last_stats = None

        attn.tpdcim_enable_sparse = enable_sparse
        attn.tpdcim_local_window = local_window
        attn.tpdcim_global_blocks = global_blocks
        attn.tpdcim_num_random_blocks = num_random_blocks

        attn.tpdcim_enable_tcs = enable_tcs
        attn.tpdcim_tcs_thresholds = tcs_thresholds

        # forward patch
        attn.forward = types.MethodType(patched_bert_self_attention_forward, attn)



def disable_qk_tiling(model):
    """
    이미 patch된 forward를 원복하는 함수는 아님.
    단순히 tiling flag만 끈다.
    """
    if hasattr(model, "bert"):
        layers = model.bert.encoder.layer
    else:
        layers = model.encoder.layer

    for layer in layers:
        attn = layer.attention.self
        attn.tpdcim_enable_qk_tiling = False