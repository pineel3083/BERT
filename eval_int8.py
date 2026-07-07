import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForMaskedLM

from tpdcim_bert_patch import enable_qk_tiling


def extend_bert_position_embeddings(model, max_position_embeddings=4096):
    bert = model.bert
    old_pos_emb = bert.embeddings.position_embeddings
    old_max_pos, hidden_size = old_pos_emb.weight.shape

    if max_position_embeddings <= old_max_pos:
        return model

    new_pos_emb = nn.Embedding(max_position_embeddings, hidden_size)

    with torch.no_grad():
        for i in range(max_position_embeddings):
            new_pos_emb.weight[i] = old_pos_emb.weight[i % old_max_pos]

    bert.embeddings.position_embeddings = new_pos_emb
    bert.embeddings.position_ids = torch.arange(max_position_embeddings).expand((1, -1))
    bert.embeddings.token_type_ids = torch.zeros(
        (1, max_position_embeddings),
        dtype=torch.long,
    )

    model.config.max_position_embeddings = max_position_embeddings
    return model


def apply_tpdcim(
    model,
    tile_n=256,
    enable_sparse=False,
    qk_mode="fp32",
    local_window=1,
    global_blocks=(0,),
    num_random_blocks=0,
    enable_tcs=False,
    tcs_thresholds=None,
):
    enable_qk_tiling(
        model,
        tile_n=tile_n,
        enable_sparse=enable_sparse,
        local_window=local_window,
        global_blocks=global_blocks,
        num_random_blocks=num_random_blocks,
        qk_mode=qk_mode,
        enable_tcs=enable_tcs,
        tcs_thresholds=tcs_thresholds,
    )

# def apply_tpdcim(
#     model,
#     tile_n=256,
#     enable_sparse=False,
#     qk_mode="fp32",
#     local_window=1,
#     global_blocks=(0,),
#     num_random_blocks=0,
# ):
#     """
#     enable_qk_tiling에 qk_mode 인자를 넣은 버전/안 넣은 버전 둘 다 대응.
#     """
#     try:
#         enable_qk_tiling(
#             model,
#             tile_n=tile_n,
#             enable_sparse=enable_sparse,
#             local_window=local_window,
#             global_blocks=global_blocks,
#             num_random_blocks=num_random_blocks,
#             qk_mode=qk_mode,
#         )
#     except TypeError:
#         enable_qk_tiling(
#             model,
#             tile_n=tile_n,
#             enable_sparse=enable_sparse,
#             local_window=local_window,
#             global_blocks=global_blocks,
#             num_random_blocks=num_random_blocks,
#         )

#         # enable_qk_tiling이 qk_mode를 아직 안 받는 경우 직접 설정
#         for layer in model.bert.encoder.layer:
#             layer.attention.self.tpdcim_qk_mode = qk_mode


def make_input(tokenizer, device, max_length=2048):
    """
    [MASK]가 중간쯤 오도록 앞뒤 context를 채운다.
    정확도 평가용이 아니라 Original/FP32/INT8/Sparse 차이 확인용.
    """
    prefix = (
        "This is background context about history, geography, language, "
        "science, literature, computers, and everyday facts. "
    ) * 80

    prompt = "The capital of France is [MASK]. "

    suffix = (
        "Additional context is placed here so that the sequence contains "
        "many real tokens across multiple attention blocks. "
    ) * 300

    text = prefix + prompt + suffix

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    ids = inputs["input_ids"][0]
    mask_pos = torch.where(ids == tokenizer.mask_token_id)[0]

    if len(mask_pos) == 0:
        raise RuntimeError("[MASK]가 없음. prefix/suffix 길이 확인 필요.")

    mask_pos = mask_pos[0].item()
    real_tokens = inputs["attention_mask"].sum().item()

    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs, mask_pos, real_tokens


def run_mlm(model, tokenizer, inputs, topk=5):
    model.eval()

    input_ids = inputs["input_ids"]
    mask_pos = torch.where(input_ids[0] == tokenizer.mask_token_id)[0]

    if len(mask_pos) == 0:
        raise RuntimeError("[MASK] 없음.")

    mask_pos = mask_pos[0].item()

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits
    mask_logits = logits[0, mask_pos, :]
    probs = torch.softmax(mask_logits, dim=-1)

    top = torch.topk(mask_logits, k=topk)

    result = []
    for i in range(topk):
        tid = top.indices[i].item()
        result.append(
            {
                "rank": i + 1,
                "token_id": tid,
                "token": tokenizer.decode([tid]),
                "logit": top.values[i].item(),
                "prob": probs[tid].item(),
            }
        )

    return result, mask_logits.detach().cpu()


def print_result(title, result):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for r in result:
        print(
            f"{r['rank']:2d}. "
            f"{r['token']:15s} "
            f"logit={r['logit']:.4f} "
            f"prob={r['prob']:.6f}"
        )


def get_stats(model):
    total_computed = 0
    total_skipped = 0
    total_tiles = 0
    densities = []
    qk_modes = set()

    tcs_rows_total = 0
    tcs_rows_skipped = 0
    bit_ops_total = 0
    bit_ops_skipped_by_tcs = 0
    enable_tcs_set = set()

    for layer in model.bert.encoder.layer:
        stats = getattr(layer.attention.self, "tpdcim_last_stats", None)

        if stats is None:
            continue

        total_computed += stats.get("computed_qk_tiles", 0)
        total_skipped += stats.get("skipped_qk_tiles", 0)
        total_tiles += stats.get("total_qk_tiles", 0)
        densities.append(stats.get("sparse_density", 1.0))
        qk_modes.add(stats.get("qk_mode", "fp32"))

        tcs_rows_total += stats.get("tcs_rows_total", 0)
        tcs_rows_skipped += stats.get("tcs_rows_skipped", 0)
        bit_ops_total += stats.get("bit_ops_total", 0)
        bit_ops_skipped_by_tcs += stats.get("bit_ops_skipped_by_tcs", 0)
        enable_tcs_set.add(stats.get("enable_tcs", False))

    density = sum(densities) / len(densities) if densities else 1.0

    return {
        "qk_mode": ",".join(sorted(qk_modes)),
        "computed": total_computed,
        "skipped": total_skipped,
        "total": total_tiles,
        "density": density,
        "skip_ratio": 1.0 - density,
        "enable_tcs": ",".join(str(x) for x in sorted(enable_tcs_set)),
        "bit_ops_total": bit_ops_total,
        "bit_ops_skipped_by_tcs": bit_ops_skipped_by_tcs,
        "tcs_rows_total": tcs_rows_total,
        "tcs_rows_skipped": tcs_rows_skipped,
        "tcs_row_skip_ratio": (
            0.0 if tcs_rows_total == 0 else tcs_rows_skipped / tcs_rows_total
        ),
    }


def run_case(
    base_model,
    tokenizer,
    inputs,
    name,
    tile_n,
    enable_sparse,
    qk_mode,
    enable_tcs=False,
    tcs_thresholds=None,
):
    model = copy.deepcopy(base_model)
    model.eval()

    apply_tpdcim(
        model,
        tile_n=tile_n,
        enable_sparse=enable_sparse,
        qk_mode=qk_mode,
        local_window=1,
        global_blocks=(0,),
        num_random_blocks=0,
        enable_tcs=enable_tcs,
        tcs_thresholds=tcs_thresholds,
    )

    result, logits = run_mlm(model, tokenizer, inputs)
    stats = get_stats(model)

    return result, logits, stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model_name = "google-bert/bert-base-uncased"
    tile_n = 256
    max_length = 2048

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs, mask_pos, real_tokens = make_input(
        tokenizer,
        device,
        max_length=max_length,
    )

    print("input length:", max_length)
    print("real tokens :", real_tokens)
    print("mask pos    :", mask_pos)
    print("mask block  :", mask_pos // tile_n)
    print("tile_n      :", tile_n)

    print("\nLoading model...")
    base_model = BertForMaskedLM.from_pretrained(model_name)
    base_model = extend_bert_position_embeddings(
        base_model,
        max_position_embeddings=4096,
    )
    base_model.to(device)
    base_model.eval()

    # Original
    original_result, original_logits = run_mlm(
        base_model,
        tokenizer,
        inputs,
    )

    print_result("Original BERT", original_result)

    # cases = [
    #     ("Dense FP32", False, "fp32", False, None),
    #     ("Dense INT8", False, "int8", False, None),
    #     ("Dense BitSerial", False, "bitserial", False, None),

    #     ("Dense BitSerial TCS th0", False, "bitserial", True, [0] * 8),
    #     ("Dense BitSerial TCS th16", False, "bitserial", True, [16] * 8),
    #     ("Dense BitSerial TCS th20", False, "bitserial", True, [20] * 8),
    #     ("Dense BitSerial TCS th24", False, "bitserial", True, [24] * 8),
    #     ("Dense BitSerial TCS th28", False, "bitserial", True, [28] * 8),
    #     ("Dense BitSerial TCS th30", False, "bitserial", True, [30] * 8),
    #     ("Dense BitSerial TCS th32", False, "bitserial", True, [32] * 8),

    #     ("Sparse FP32", True, "fp32", False, None),
    #     ("Sparse INT8", True, "int8", False, None),
    #     ("Sparse BitSerial", True, "bitserial", False, None),

    #     ("Sparse BitSerial TCS th0", True, "bitserial", True, [0] * 8),
    #     ("Sparse BitSerial TCS th16", True, "bitserial", True, [16] * 8),
    #     ("Sparse BitSerial TCS th20", True, "bitserial", True, [20] * 8),
    #     ("Sparse BitSerial TCS th24", True, "bitserial", True, [24] * 8),
    #     ("Sparse BitSerial TCS th28", True, "bitserial", True, [28] * 8),
    #     ("Sparse BitSerial TCS th30", True, "bitserial", True, [30] * 8),
    #     ("Sparse BitSerial TCS th32", True, "bitserial", True, [32] * 8),
    # ]

    cases = [
        ("Dense FP32", False, "fp32", False, None),
        ("Dense INT8", False, "int8", False, None),
        ("Dense BitSerial", False, "bitserial", False, None),

        # uniform TCS baseline
        ("Dense BitSerial TCS uni16", False, "bitserial", True, [16] * 8),
        ("Dense BitSerial TCS uni20", False, "bitserial", True, [20] * 8),
        ("Dense BitSerial TCS uni24", False, "bitserial", True, [24] * 8),

        # bit-wise TCS candidates
        ("Dense BitSerial TCS bw_A", False, "bitserial", True, [20, 20, 18, 18, 16, 16, 14, 12]),
        ("Dense BitSerial TCS bw_B", False, "bitserial", True, [22, 22, 20, 18, 16, 16, 14, 12]),
        ("Dense BitSerial TCS bw_C", False, "bitserial", True, [24, 24, 22, 20, 18, 16, 14, 12]),
        ("Dense BitSerial TCS bw_D", False, "bitserial", True, [24, 22, 20, 18, 16, 14, 12, 10]),
        ("Dense BitSerial TCS bw_E", False, "bitserial", True, [28, 24, 22, 20, 18, 16, 14, 10]),

        ("Sparse FP32", True, "fp32", False, None),
        ("Sparse INT8", True, "int8", False, None),
        ("Sparse BitSerial", True, "bitserial", False, None),

        # uniform TCS baseline
        ("Sparse BitSerial TCS uni16", True, "bitserial", True, [16] * 8),
        ("Sparse BitSerial TCS uni20", True, "bitserial", True, [20] * 8),
        ("Sparse BitSerial TCS uni24", True, "bitserial", True, [24] * 8),

        # bit-wise TCS candidates
        ("Sparse BitSerial TCS bw_A", True, "bitserial", True, [20, 20, 18, 18, 16, 16, 14, 12]),
        ("Sparse BitSerial TCS bw_B", True, "bitserial", True, [22, 22, 20, 18, 16, 16, 14, 12]),
        ("Sparse BitSerial TCS bw_C", True, "bitserial", True, [24, 24, 22, 20, 18, 16, 14, 12]),
        ("Sparse BitSerial TCS bw_D", True, "bitserial", True, [24, 22, 20, 18, 16, 14, 12, 10]),
        ("Sparse BitSerial TCS bw_E", True, "bitserial", True, [28, 24, 22, 20, 18, 16, 14, 10]),
    ]

    summary = []

    for name, sparse_on, qk_mode, enable_tcs, tcs_thresholds in cases:
        result, logits, stats = run_case(
            base_model=base_model,
            tokenizer=tokenizer,
            inputs=inputs,
            name=name,
            tile_n=tile_n,
            enable_sparse=sparse_on,
            qk_mode=qk_mode,
            enable_tcs=enable_tcs,
            tcs_thresholds=tcs_thresholds,
        )

        diff = (original_logits - logits).abs()

        print_result(name, result)

        print("\n--- Compare with Original ---")
        print("top1 same :", result[0]["token_id"] == original_result[0]["token_id"])
        print("diff max  :", diff.max().item())
        print("diff mean :", diff.mean().item())

        print("\n--- TPDCIM Stats ---")
        print(stats)

        summary.append(
            {
                "name": name,
                "top1_same": result[0]["token_id"] == original_result[0]["token_id"],
                "diff_max": diff.max().item(),
                "diff_mean": diff.mean().item(),
                "stats": stats,
            }
        )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for row in summary:
        s = row["stats"]
        print(
            f"{row['name']:12s} | "
            f"top1_same={str(row['top1_same']):5s} | "
            f"diff_mean={row['diff_mean']:.6f} | "
            f"qk_mode={s['qk_mode']:4s} | "
            f"density={s['density']:.4f} | "
            f"skip={s['skip_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()