import copy
import csv
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


def find_mask_pos(tokenizer, text, max_length):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    ids = enc["input_ids"][0]
    pos = torch.where(ids == tokenizer.mask_token_id)[0]

    if len(pos) == 0:
        raise ValueError("[MASK]가 truncation 때문에 사라짐.")

    return pos[0].item(), ids.shape[0]


def build_2048_text_with_mask_block(
    tokenizer,
    prompt,
    target_block=4,
    tile_n=256,
    max_length=2048,
):
    prefix_unit = (
        "This is background context about history, geography, language, "
        "science, literature, computers, and everyday facts. "
    )

    suffix_unit = (
        "Additional context is placed here so that the sequence contains "
        "many real tokens across multiple attention blocks. "
    )

    prefix = ""

    target_start = target_block * tile_n
    target_end = (target_block + 1) * tile_n

    for _ in range(1000):
        candidate = prefix + " " + prompt
        mask_pos, _ = find_mask_pos(
            tokenizer,
            candidate,
            max_length=max_length,
        )

        if target_start <= mask_pos < target_end:
            break

        prefix += prefix_unit

        if mask_pos >= target_end:
            raise RuntimeError(
                f"[MASK]가 target block을 지나침. mask_pos={mask_pos}"
            )
    else:
        raise RuntimeError("[MASK] 위치 조정 실패")

    text = prefix + " " + prompt + " " + suffix_unit * 1000

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    ids = inputs["input_ids"][0]
    mask_positions = torch.where(ids == tokenizer.mask_token_id)[0]

    if len(mask_positions) == 0:
        raise ValueError("최종 입력에서 [MASK] 없음.")

    mask_pos = mask_positions[0].item()
    real_tokens = inputs["attention_mask"].sum().item()
    mask_block = mask_pos // tile_n

    return text, mask_pos, mask_block, real_tokens


def run_mlm(model, tokenizer, inputs):
    model.eval()

    input_ids = inputs["input_ids"]
    mask_positions = torch.where(input_ids[0] == tokenizer.mask_token_id)[0]

    if len(mask_positions) == 0:
        raise ValueError("입력에 [MASK] 없음.")

    mask_pos = mask_positions[0].item()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    mask_logits = logits[0, mask_pos, :]
    probs = torch.softmax(mask_logits, dim=-1)

    top1_id = torch.argmax(mask_logits).item()
    top1_token = tokenizer.decode([top1_id])
    top1_logit = mask_logits[top1_id].item()
    top1_prob = probs[top1_id].item()

    top5 = torch.topk(mask_logits, k=5)

    top5_list = []
    for i in range(5):
        tid = top5.indices[i].item()
        top5_list.append(
            {
                "rank": i + 1,
                "token": tokenizer.decode([tid]),
                "logit": top5.values[i].item(),
                "prob": probs[tid].item(),
            }
        )

    return {
        "mask_logits": mask_logits.detach().cpu(),
        "top1_id": top1_id,
        "top1_token": top1_token,
        "top1_logit": top1_logit,
        "top1_prob": top1_prob,
        "top5": top5_list,
    }


def get_tile_summary(model):
    total_computed = 0
    total_skipped = 0
    total_tiles = 0
    densities = []

    for layer in model.bert.encoder.layer:
        stats = getattr(layer.attention.self, "tpdcim_last_stats", None)
        if stats is None:
            continue

        total_computed += stats["computed_qk_tiles"]
        total_skipped += stats["skipped_qk_tiles"]
        total_tiles += stats["total_qk_tiles"]
        densities.append(stats["sparse_density"])

    avg_density = sum(densities) / len(densities)

    return {
        "computed": total_computed,
        "skipped": total_skipped,
        "total": total_tiles,
        "density": avg_density,
        "skip_ratio": 1.0 - avg_density,
    }


def print_top5(title, result):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for r in result["top5"]:
        print(
            f"{r['rank']:2d}. "
            f"{r['token']:15s} "
            f"logit={r['logit']:.4f} "
            f"prob={r['prob']:.6f}"
        )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tile_n = 256
    max_length = 2048
    target_block = 4

    prompt = "The capital of France is [MASK]."

    text, mask_pos, mask_block, real_tokens = build_2048_text_with_mask_block(
        tokenizer=tokenizer,
        prompt=prompt,
        target_block=target_block,
        tile_n=tile_n,
        max_length=max_length,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("prompt      :", prompt)
    print("input length:", max_length)
    print("real tokens :", real_tokens)
    print("mask pos    :", mask_pos)
    print("mask block  :", mask_block)
    print("tile_n      :", tile_n)

    print("\nLoading model...")
    model_ref = BertForMaskedLM.from_pretrained(model_name)
    model_ref = extend_bert_position_embeddings(
        model_ref,
        max_position_embeddings=4096,
    )
    model_ref.to(device)
    model_ref.eval()

    # 같은 weight를 가진 patch 모델 하나만 만들어서 dense/sparse 설정만 바꿔가며 사용
    model_test = copy.deepcopy(model_ref)
    model_test.to(device)
    model_test.eval()

    # Original
    ref = run_mlm(model_ref, tokenizer, inputs)
    print_top5("Original BERT", ref)

    # Dense tiled sanity check
    enable_qk_tiling(
        model_test,
        tile_n=tile_n,
        enable_sparse=False,
    )

    dense = run_mlm(model_test, tokenizer, inputs)
    dense_diff = (ref["mask_logits"] - dense["mask_logits"]).abs()
    dense_tile = get_tile_summary(model_test)

    print_top5("Dense QK Tiling", dense)
    print("\nDense diff max :", dense_diff.max().item())
    print("Dense diff mean:", dense_diff.mean().item())
    print("Dense tile     :", dense_tile)

    # Sparse sweep configs
    configs = [
        # name, local_window, global_blocks, num_random_blocks
        ("lw1_g0_r0", 1, (0,), 0),
        ("lw1_g0_r1", 1, (0,), 1),
        ("lw1_g0_r2", 1, (0,), 2),
        ("lw2_g0_r0", 2, (0,), 0),
        ("lw2_g0_r1", 2, (0,), 1),
        ("lw2_g0_r2", 2, (0,), 2),
        ("lw3_g0_r0", 3, (0,), 0),
        ("lw3_g0_r1", 3, (0,), 1),
        ("lw3_g0_r2", 3, (0,), 2),
        ("lw1_g04_r0", 1, (0, 4), 0),
        ("lw2_g04_r0", 2, (0, 4), 0),
    ]

    rows = []

    print("\n\n" + "#" * 80)
    print("SPARSE SWEEP")
    print("#" * 80)

    for name, local_window, global_blocks, num_random_blocks in configs:
        enable_qk_tiling(
            model_test,
            tile_n=tile_n,
            enable_sparse=True,
            local_window=local_window,
            global_blocks=global_blocks,
            num_random_blocks=num_random_blocks,
        )

        sparse = run_mlm(model_test, tokenizer, inputs)
        sparse_diff = (ref["mask_logits"] - sparse["mask_logits"]).abs()
        tile = get_tile_summary(model_test)

        top1_same = sparse["top1_id"] == ref["top1_id"]

        print("\n" + "-" * 80)
        print(
            f"{name} | "
            f"density={tile['density']:.4f} | "
            f"skip={tile['skip_ratio']:.4f} | "
            f"top1_same={top1_same}"
        )
        print(
            f"ref_top1={ref['top1_token']} "
            f"sparse_top1={sparse['top1_token']} "
            f"diff_max={sparse_diff.max().item():.4f} "
            f"diff_mean={sparse_diff.mean().item():.4f}"
        )

        for r in sparse["top5"]:
            print(
                f"{r['rank']:2d}. "
                f"{r['token']:15s} "
                f"logit={r['logit']:.4f} "
                f"prob={r['prob']:.6f}"
            )

        rows.append(
            {
                "config": name,
                "local_window": local_window,
                "global_blocks": str(global_blocks),
                "num_random_blocks": num_random_blocks,
                "density": tile["density"],
                "skip_ratio": tile["skip_ratio"],
                "computed_tiles": tile["computed"],
                "skipped_tiles": tile["skipped"],
                "total_tiles": tile["total"],
                "top1_same": top1_same,
                "ref_top1": ref["top1_token"],
                "sparse_top1": sparse["top1_token"],
                "sparse_top1_logit": sparse["top1_logit"],
                "sparse_top1_prob": sparse["top1_prob"],
                "diff_max": sparse_diff.max().item(),
                "diff_mean": sparse_diff.mean().item(),
            }
        )

    csv_path = "sparse_sweep_2048.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", csv_path)

    print("\n" + "=" * 80)
    print("SHORT SUMMARY")
    print("=" * 80)

    for row in rows:
        print(
            f"{row['config']:12s} "
            f"density={row['density']:.4f} "
            f"skip={row['skip_ratio']:.4f} "
            f"top1_same={row['top1_same']} "
            f"diff_mean={row['diff_mean']:.4f}"
        )


if __name__ == "__main__":
    main()