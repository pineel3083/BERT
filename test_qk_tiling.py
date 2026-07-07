import torch
from transformers import BertConfig, BertModel
from tpdcim_bert_patch import enable_qk_tiling


def main():
    torch.manual_seed(0)

    N = 2048
    B = 1

    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=4096,
    )

    model_ref = BertModel(config)
    model_ref.eval()

    model_tiled = BertModel(config)
    model_tiled.load_state_dict(model_ref.state_dict())
    model_tiled.eval()

    # enable_qk_tiling(model_tiled, tile_n=256)

    # For dense
    # enable_qk_tiling(
    # model_tiled,
    # tile_n=256,
    # enable_sparse=False,
    # )

    enable_qk_tiling(
        model_tiled,
        tile_n=256,
        enable_sparse=True,
        local_window=1,
        global_blocks=(0,),
        num_random_blocks=0,
    )


    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(B, N),
    )

    attention_mask = torch.ones(B, N, dtype=torch.long)

    with torch.no_grad():
        out_ref = model_ref(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        out_tiled = model_tiled(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    max_diff = (out_ref - out_tiled).abs().max().item()
    mean_diff = (out_ref - out_tiled).abs().mean().item()

    print("max diff :", max_diff)
    print("mean diff:", mean_diff)

    print("\nLayer stats:")
    for i, layer in enumerate(model_tiled.encoder.layer):
        print(f"layer {i}: {layer.attention.self.tpdcim_last_stats}")


if __name__ == "__main__":
    main()