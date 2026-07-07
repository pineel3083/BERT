import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForMaskedLM

from tpdcim_bert_patch import enable_qk_tiling


def extend_bert_position_embeddings(model, max_position_embeddings=4096):
    """
    pretrained bert-base는 position embedding이 512까지만 있음.
    실험용으로 512 position embedding을 반복 복사해서 4096까지 확장.
    """

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


def make_long_text():
    text = (
        "The capital of France is [MASK]. "
        + (
            "Paris is the capital city of France. "
            "France is a country in Europe. "
            "The city has museums, monuments, culture, and history. "
        ) * 400
    )

    return text


def tokenize_2048(tokenizer, text, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=2048,
    )

    print("input length:", inputs["input_ids"].shape[1])
    print("real tokens :", inputs["attention_mask"].sum().item())

    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


def print_topk_predictions(model, tokenizer, inputs, title, topk=5):
    model.eval()

    input_ids = inputs["input_ids"]
    mask_pos = torch.where(input_ids[0] == tokenizer.mask_token_id)[0]

    if len(mask_pos) == 0:
        raise ValueError("입력에 [MASK]가 없음.")

    # [MASK]가 여러 개면 첫 번째만 사용
    mask_pos = mask_pos[0].item()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_logits = logits[0, mask_pos, :]
    mask_probs = torch.softmax(mask_logits, dim=-1)

    topk_result = torch.topk(mask_logits, k=topk, dim=-1)

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for rank in range(topk):
        token_id = topk_result.indices[rank].item()
        logit = topk_result.values[rank].item()
        prob = mask_probs[token_id].item()
        token = tokenizer.decode([token_id])

        print(f"{rank + 1:2d}. {token:15s} logit={logit:.4f} prob={prob:.6f}")

    return logits


def compare_logits(logits_a, logits_b, name_a="A", name_b="B"):
    diff = (logits_a - logits_b).abs()

    print("\n" + "-" * 60)
    print(f"Logit difference: {name_a} vs {name_b}")
    print("-" * 60)
    print("max diff :", diff.max().item())
    print("mean diff:", diff.mean().item())


def print_tpdcim_stats(model, title):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)

    for i, layer in enumerate(model.bert.encoder.layer):
        stats = getattr(layer.attention.self, "tpdcim_last_stats", None)
        print(f"layer {i}: {stats}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model_name = "google-bert/bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = make_long_text()

    # 원본 BERT
    model_ref = BertForMaskedLM.from_pretrained(model_name)
    model_ref = extend_bert_position_embeddings(
        model_ref,
        max_position_embeddings=4096,
    )
    model_ref.to(device)
    model_ref.eval()

    inputs = tokenize_2048(tokenizer, text, device)

    # Dense QK tiling BERT
    model_tiled = copy.deepcopy(model_ref)
    enable_qk_tiling(
        model_tiled,
        tile_n=256,
        enable_sparse=False,
    )
    model_tiled.to(device)
    model_tiled.eval()

    # Sparse QK tiling BERT
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

    logits_ref = print_topk_predictions(
        model_ref,
        tokenizer,
        inputs,
        title="Original BERT, N=2048",
        topk=5,
    )

    logits_tiled = print_topk_predictions(
        model_tiled,
        tokenizer,
        inputs,
        title="BERT + Dense QK Tiling, N=2048",
        topk=5,
    )

    logits_sparse = print_topk_predictions(
        model_sparse,
        tokenizer,
        inputs,
        title="BERT + Sparse QK Tiling, N=2048",
        topk=5,
    )

    compare_logits(logits_ref, logits_tiled, "Original", "Dense QK Tiled")
    compare_logits(logits_ref, logits_sparse, "Original", "Sparse QK Tiled")

    print_tpdcim_stats(model_tiled, "Dense QK Tiling Stats")
    print_tpdcim_stats(model_sparse, "Sparse QK Tiling Stats")


if __name__ == "__main__":
    main()