import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForMaskedLM

from tpdcim_bert_patch import enable_qk_tiling


TCS_CONVENTION = "thresholds are LSB->MSB; active rows satisfy bsum > threshold"


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


def make_input(tokenizer, device, max_length=2048):
    """
    Build a long MLM input with [MASK] near the middle.

    This is a reproducible behavior/regression probe, not an accuracy benchmark.
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
        raise RuntimeError("[MASK] not found. Check prefix/suffix lengths.")

    mask_pos = mask_pos[0].item()
    real_tokens = inputs["attention_mask"].sum().item()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs, mask_pos, real_tokens


def run_mlm(model, tokenizer, inputs, topk=5):
    model.eval()

    input_ids = inputs["input_ids"]
    mask_pos = torch.where(input_ids[0] == tokenizer.mask_token_id)[0]

    if len(mask_pos) == 0:
        raise RuntimeError("[MASK] not found.")

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


def _format_thresholds(thresholds):
    if thresholds in (None, ""):
        return "-"
    if isinstance(thresholds, (list, tuple)):
        return "[" + ",".join(str(x) for x in thresholds) + "]"
    return str(thresholds)


def get_stats(model):
    total_computed = 0
    total_skipped = 0
    total_tiles = 0
    densities = []
    qk_modes = set()
    enable_sparse_set = set()

    tcs_rows_total = 0
    tcs_rows_skipped = 0
    bit_ops_total = 0
    bit_ops_skipped_by_tcs = 0
    enable_tcs_set = set()
    tcs_thresholds_seen = set()
    tcs_orders = set()
    tcs_rules = set()

    for layer in model.bert.encoder.layer:
        stats = getattr(layer.attention.self, "tpdcim_last_stats", None)
        if stats is None:
            continue

        total_computed += stats.get("computed_qk_tiles", 0)
        total_skipped += stats.get("skipped_qk_tiles", 0)
        total_tiles += stats.get("total_qk_tiles", 0)
        densities.append(stats.get("sparse_density", 1.0))
        qk_modes.add(stats.get("qk_mode", "fp32"))
        enable_sparse_set.add(stats.get("enable_sparse", False))

        tcs_rows_total += stats.get("tcs_rows_total", 0)
        tcs_rows_skipped += stats.get("tcs_rows_skipped", 0)
        bit_ops_total += stats.get("bit_ops_total", 0)
        bit_ops_skipped_by_tcs += stats.get("bit_ops_skipped_by_tcs", 0)
        enable_tcs_set.add(stats.get("enable_tcs", False))
        tcs_orders.add(stats.get("tcs_threshold_order", "lsb_to_msb"))
        tcs_rules.add(stats.get("tcs_active_rule", "bsum > threshold"))

        thresholds = stats.get("tcs_thresholds")
        if stats.get("enable_tcs", False) and thresholds is not None:
            tcs_thresholds_seen.add(tuple(thresholds))

    density = sum(densities) / len(densities) if densities else 1.0
    threshold_values = sorted(tcs_thresholds_seen)
    if len(threshold_values) == 1:
        tcs_thresholds = list(threshold_values[0])
    elif len(threshold_values) > 1:
        tcs_thresholds = [list(x) for x in threshold_values]
    else:
        tcs_thresholds = None

    return {
        "qk_mode": ",".join(sorted(qk_modes)),
        "enable_sparse": ",".join(str(x) for x in sorted(enable_sparse_set)),
        "computed": total_computed,
        "skipped": total_skipped,
        "total": total_tiles,
        "density": density,
        "sparse_tile_skip_ratio": 1.0 - density,
        "enable_tcs": ",".join(str(x) for x in sorted(enable_tcs_set)),
        "tcs_thresholds": tcs_thresholds,
        "tcs_threshold_order": ",".join(sorted(tcs_orders)),
        "tcs_active_rule": ",".join(sorted(tcs_rules)),
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


def print_final_summary(summary):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("TCS convention:", TCS_CONVENTION)
    print()

    for row in summary:
        s = row["stats"]
        print(
            f"{row['name']:30s} | "
            f"top1_same={str(row['top1_same']):5s} | "
            f"diff_mean={row['diff_mean']:.6f} | "
            f"qk_mode={s['qk_mode']:9s} | "
            f"tiles={s['computed']:4d}/{s['skipped']:4d}/{s['total']:4d} | "
            f"density={s['density']:.5f} | "
            f"sparse_skip={s['sparse_tile_skip_ratio']:.5f} | "
            f"tcs={s['enable_tcs']:5s} | "
            f"tcs_row_skip={s['tcs_row_skip_ratio']:.5f} | "
            f"bit_ops={s['bit_ops_total']:5d} | "
            f"bit_ops_tcs_skip={s['bit_ops_skipped_by_tcs']:4d}"
        )


def run_sanity_checks(summary, logits_by_name, tile_n, max_length):
    by_name = {row["name"]: row for row in summary}

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    require(by_name["Dense FP32"]["stats"]["qk_mode"] == "fp32", "Dense FP32 qk_mode mismatch")
    require(by_name["Dense INT8"]["stats"]["qk_mode"] == "int8", "Dense INT8 qk_mode mismatch")
    require(
        by_name["Dense BitSerial"]["stats"]["qk_mode"] == "bitserial",
        "Dense BitSerial qk_mode mismatch",
    )

    require(by_name["Dense FP32"]["stats"]["density"] == 1.0, "Dense FP32 density must be 1")
    require(by_name["Dense INT8"]["stats"]["density"] == 1.0, "Dense INT8 density must be 1")
    require(
        by_name["Dense BitSerial"]["stats"]["computed"] == by_name["Dense BitSerial"]["stats"]["total"],
        "Dense BitSerial must compute every tile",
    )

    dense_pair_diff = (logits_by_name["Dense INT8"] - logits_by_name["Dense BitSerial"]).abs()
    sparse_pair_diff = (logits_by_name["Sparse INT8"] - logits_by_name["Sparse BitSerial"]).abs()
    require(dense_pair_diff.max().item() <= 1e-6, "Dense INT8 and Dense BitSerial logits differ")
    require(sparse_pair_diff.max().item() <= 1e-6, "Sparse INT8 and Sparse BitSerial logits differ")

    if max_length == 2048 and tile_n == 256:
        sparse = by_name["Sparse FP32"]["stats"]
        require(sparse["computed"] == 408, "Sparse FP32 expected 408 computed QK tiles")
        require(sparse["skipped"] == 360, "Sparse FP32 expected 360 skipped QK tiles")
        require(sparse["total"] == 768, "Sparse FP32 expected 768 total QK tiles")
        require(abs(sparse["density"] - 0.53125) < 1e-12, "Sparse density expected 0.53125")
        require(
            by_name["Sparse BitSerial"]["stats"]["bit_ops_total"] == 3264,
            "Sparse BitSerial expected 3264 bit ops",
        )
        require(
            by_name["Dense BitSerial"]["stats"]["bit_ops_total"] == 6144,
            "Dense BitSerial expected 6144 bit ops",
        )

    tcs_rows = [
        row["stats"]["tcs_row_skip_ratio"]
        for row in summary
        if "TCS" in row["name"]
    ]
    require(any(x > 0.0 for x in tcs_rows), "TCS cases should report row skips")

    print("\nSANITY CHECKS PASSED")
    print("Dense INT8 == Dense BitSerial max diff:", dense_pair_diff.max().item())
    print("Sparse INT8 == Sparse BitSerial max diff:", sparse_pair_diff.max().item())


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("TCS convention:", TCS_CONVENTION)

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

    original_result, original_logits = run_mlm(
        base_model,
        tokenizer,
        inputs,
    )
    print_result("Original BERT", original_result)

    cases = [
        ("Dense FP32", False, "fp32", False, None),
        ("Dense INT8", False, "int8", False, None),
        ("Dense BitSerial", False, "bitserial", False, None),
        ("Dense BitSerial TCS uni16", False, "bitserial", True, [16] * 8),
        ("Dense BitSerial TCS uni20", False, "bitserial", True, [20] * 8),
        ("Dense BitSerial TCS uni24", False, "bitserial", True, [24] * 8),
        ("Dense BitSerial TCS bw_A", False, "bitserial", True, [20, 20, 18, 18, 16, 16, 14, 12]),
        ("Dense BitSerial TCS bw_B", False, "bitserial", True, [22, 22, 20, 18, 16, 16, 14, 12]),
        ("Dense BitSerial TCS bw_C", False, "bitserial", True, [24, 24, 22, 20, 18, 16, 14, 12]),
        ("Dense BitSerial TCS bw_D", False, "bitserial", True, [24, 22, 20, 18, 16, 14, 12, 10]),
        ("Dense BitSerial TCS bw_E", False, "bitserial", True, [28, 24, 22, 20, 18, 16, 14, 10]),
        ("Sparse FP32", True, "fp32", False, None),
        ("Sparse INT8", True, "int8", False, None),
        ("Sparse BitSerial", True, "bitserial", False, None),
        ("Sparse BitSerial TCS uni16", True, "bitserial", True, [16] * 8),
        ("Sparse BitSerial TCS uni20", True, "bitserial", True, [20] * 8),
        ("Sparse BitSerial TCS uni24", True, "bitserial", True, [24] * 8),
        ("Sparse BitSerial TCS bw_A", True, "bitserial", True, [20, 20, 18, 18, 16, 16, 14, 12]),
        ("Sparse BitSerial TCS bw_B", True, "bitserial", True, [22, 22, 20, 18, 16, 16, 14, 12]),
        ("Sparse BitSerial TCS bw_C", True, "bitserial", True, [24, 24, 22, 20, 18, 16, 14, 12]),
        ("Sparse BitSerial TCS bw_D", True, "bitserial", True, [24, 22, 20, 18, 16, 14, 12, 10]),
        ("Sparse BitSerial TCS bw_E", True, "bitserial", True, [28, 24, 22, 20, 18, 16, 14, 10]),
    ]

    summary = []
    logits_by_name = {}

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
        logits_by_name[name] = logits

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

    print_final_summary(summary)
    run_sanity_checks(summary, logits_by_name, tile_n=tile_n, max_length=max_length)


if __name__ == "__main__":
    main()
