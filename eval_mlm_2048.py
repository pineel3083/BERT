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


def make_2048_text(prompt):
    """
    Sanity check용.

    실제 문장은 prompt만 사용한다.
    tokenizer에서 padding='max_length', max_length=2048을 쓰기 때문에
    input tensor shape은 [1, 2048]이 된다.

    즉:
        input length = 2048
        real tokens  = 짧은 prompt 길이
        나머지       = PAD

    이렇게 해야 BERT의 [MASK] prediction이 filler에 오염되지 않는다.
    """
    return prompt


def tokenize_fixed_2048(tokenizer, text, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=2048,
    )

    input_len = inputs["input_ids"].shape[1]
    real_tokens = inputs["attention_mask"].sum().item()

    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs, input_len, real_tokens


def run_one_model(model, tokenizer, inputs, topk=5):
    model.eval()

    input_ids = inputs["input_ids"]
    mask_positions = torch.where(input_ids[0] == tokenizer.mask_token_id)[0]

    if len(mask_positions) == 0:
        raise ValueError("입력에 [MASK]가 없음.")

    mask_pos = mask_positions[0].item()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_logits = logits[0, mask_pos, :]
    mask_probs = torch.softmax(mask_logits, dim=-1)

    topk_result = torch.topk(mask_logits, k=topk, dim=-1)

    results = []

    for rank in range(topk):
        token_id = topk_result.indices[rank].item()
        logit = topk_result.values[rank].item()
        prob = mask_probs[token_id].item()
        token = tokenizer.decode([token_id])

        results.append(
            {
                "rank": rank + 1,
                "token": token,
                "token_id": token_id,
                "logit": logit,
                "prob": prob,
            }
        )

    # full logits를 들고 있으면 메모리 커지니까 mask_logits만 CPU로 반환
    return results, mask_logits.detach().cpu()


def get_sparse_summary(model):
    """
    layer별 sparse stats 평균/합계 요약.
    Dense 모델이면 skipped는 0일 것.
    """
    total_computed = 0
    total_skipped = 0
    total_tiles = 0
    densities = []

    for layer in model.bert.encoder.layer:
        stats = getattr(layer.attention.self, "tpdcim_last_stats", None)

        if stats is None:
            continue

        total_computed += stats.get("computed_qk_tiles", 0)
        total_skipped += stats.get("skipped_qk_tiles", 0)
        total_tiles += stats.get("total_qk_tiles", 0)

        if "sparse_density" in stats:
            densities.append(stats["sparse_density"])

    avg_density = sum(densities) / len(densities) if densities else None

    return {
        "computed_tiles_total": total_computed,
        "skipped_tiles_total": total_skipped,
        "total_tiles_total": total_tiles,
        "avg_sparse_density": avg_density,
    }


def print_topk(title, results):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for r in results:
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

    prompts = [
        "The capital of France is [MASK].",
        "The capital of Germany is [MASK].",
        "The opposite of hot is [MASK].",
        "The largest planet in the solar system is [MASK].",
        "The main language spoken in Japan is [MASK].",
        "The author of Hamlet was [MASK].",
        "The color of the sky is [MASK].",
        "A dog is an [MASK].",
    ]

    expected_answers = {
        "The capital of France is [MASK].": "paris",
        "The capital of Germany is [MASK].": "berlin",
        "The opposite of hot is [MASK].": "cold",
        "The largest planet in the solar system is [MASK].": "jupiter",
        "The main language spoken in Japan is [MASK].": "japanese",
        "The author of Hamlet was [MASK].": "shakespeare",
        "The color of the sky is [MASK].": "blue",
        "A dog is an [MASK].": "animal",
    }

    print("Loading original model...")
    model_ref = BertForMaskedLM.from_pretrained(model_name)
    model_ref = extend_bert_position_embeddings(model_ref, max_position_embeddings=4096)
    model_ref.to(device)
    model_ref.eval()

    print("Building dense tiled model...")
    model_dense = copy.deepcopy(model_ref)
    enable_qk_tiling(
        model_dense,
        tile_n=256,
        enable_sparse=False,
    )
    model_dense.to(device)
    model_dense.eval()

    print("Building sparse tiled model...")
    model_sparse = copy.deepcopy(model_ref)
    enable_qk_tiling(
        model_sparse,
        tile_n=256,
        enable_sparse=True,
        local_window=1,
        global_blocks=(0,),
        num_random_blocks=0,
    )
    model_sparse.to(device)
    model_sparse.eval()

    csv_rows = []

    for idx, prompt in enumerate(prompts):
        print("\n\n" + "#" * 80)
        print(f"CASE {idx}: {prompt}")
        print("#" * 80)

        text = make_2048_text(prompt)
        inputs, input_len, real_tokens = tokenize_fixed_2048(tokenizer, text, device)

        print("input length:", input_len)
        print("real tokens :", real_tokens)

        ref_topk, ref_mask_logits = run_one_model(
            model_ref,
            tokenizer,
            inputs,
            topk=5,
        )

        dense_topk, dense_mask_logits = run_one_model(
            model_dense,
            tokenizer,
            inputs,
            topk=5,
        )

        sparse_topk, sparse_mask_logits = run_one_model(
            model_sparse,
            tokenizer,
            inputs,
            topk=5,
        )

        print_topk("Original BERT", ref_topk)
        print_topk("Dense QK Tiling", dense_topk)
        print_topk("Sparse QK Tiling", sparse_topk)

        dense_diff = (ref_mask_logits - dense_mask_logits).abs()
        sparse_diff = (ref_mask_logits - sparse_mask_logits).abs()

        dense_stats = get_sparse_summary(model_dense)
        sparse_stats = get_sparse_summary(model_sparse)

        print("\n--- Diff Summary ---")
        print("dense max diff :", dense_diff.max().item())
        print("dense mean diff:", dense_diff.mean().item())
        print("sparse max diff :", sparse_diff.max().item())
        print("sparse mean diff:", sparse_diff.mean().item())

        print("\n--- Tile Summary ---")
        print("dense :", dense_stats)
        print("sparse:", sparse_stats)

        csv_rows.append(
            {
                "case_id": idx,
                "prompt": prompt,
                "input_length": input_len,
                "real_tokens": real_tokens,

                "original_top1": ref_topk[0]["token"],
                "original_top1_logit": ref_topk[0]["logit"],
                "original_top1_prob": ref_topk[0]["prob"],

                "dense_top1": dense_topk[0]["token"],
                "dense_top1_logit": dense_topk[0]["logit"],
                "dense_top1_prob": dense_topk[0]["prob"],
                "dense_max_diff": dense_diff.max().item(),
                "dense_mean_diff": dense_diff.mean().item(),

                "sparse_top1": sparse_topk[0]["token"],
                "sparse_top1_logit": sparse_topk[0]["logit"],
                "sparse_top1_prob": sparse_topk[0]["prob"],
                "sparse_max_diff": sparse_diff.max().item(),
                "sparse_mean_diff": sparse_diff.mean().item(),

                "sparse_computed_tiles": sparse_stats["computed_tiles_total"],
                "sparse_skipped_tiles": sparse_stats["skipped_tiles_total"],
                "sparse_total_tiles": sparse_stats["total_tiles_total"],
                "sparse_avg_density": sparse_stats["avg_sparse_density"],
                "expected": expected_answers.get(prompt, ""),
                "original_correct": ref_topk[0]["token"].strip() == expected_answers.get(prompt, ""),
                "dense_correct": dense_topk[0]["token"].strip() == expected_answers.get(prompt, ""),
                "sparse_correct": sparse_topk[0]["token"].strip() == expected_answers.get(prompt, ""),
            }
        )

    csv_path = "mlm_2048_eval_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n\nSaved:", csv_path)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    dense_same = 0
    sparse_same = 0

    for row in csv_rows:
        if row["original_top1"] == row["dense_top1"]:
            dense_same += 1
        if row["original_top1"] == row["sparse_top1"]:
            sparse_same += 1

    original_correct = 0
    dense_correct = 0
    sparse_correct = 0

    for row in csv_rows:
        if row["original_correct"]:
            original_correct += 1
        if row["dense_correct"]:
            dense_correct += 1
        if row["sparse_correct"]:
            sparse_correct += 1

    print(f"Original expected correct: {original_correct}/{len(csv_rows)}")
    print(f"Dense expected correct   : {dense_correct}/{len(csv_rows)}")
    print(f"Sparse expected correct  : {sparse_correct}/{len(csv_rows)}")


    print(f"Dense top1 same as original : {dense_same}/{len(csv_rows)}")
    print(f"Sparse top1 same as original: {sparse_same}/{len(csv_rows)}")

    avg_sparse_density = sum(
        row["sparse_avg_density"] for row in csv_rows
        if row["sparse_avg_density"] is not None
    ) / len(csv_rows)

    print(f"Average sparse density: {avg_sparse_density:.4f}")
    print(f"Average skipped ratio : {1.0 - avg_sparse_density:.4f}")


if __name__ == "__main__":
    main()