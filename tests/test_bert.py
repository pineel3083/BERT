import torch

from model.bert import BertModel
from model.config import BertConfig


def tiny_config(**kwargs):
    config = {
        "vocab_size": 101,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 32,
        "dcim_block_size": 3,
    }
    config.update(kwargs)
    return BertConfig(**config)


def test_bert_model_outputs_match_encoder_shapes():
    torch.manual_seed(0)
    model = BertModel(tiny_config())
    input_ids = torch.randint(0, 101, (2, 7))

    outputs = model(input_ids, output_hidden_states=True, output_attentions=True)

    assert outputs["last_hidden_state"].shape == (2, 7, 32)
    assert outputs["pooler_output"].shape == (2, 32)
    assert len(outputs["hidden_states"]) == 3
    assert len(outputs["attentions"]) == 2
    assert outputs["attentions"][0].shape == (2, 4, 7, 7)


def test_attention_mask_blocks_masked_tokens():
    torch.manual_seed(1)
    model = BertModel(tiny_config(num_hidden_layers=1))
    model.eval()
    input_ids = torch.randint(0, 101, (1, 5))
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
    probs_to_masked_tokens = outputs["attentions"][0][..., 3:]

    assert torch.all(probs_to_masked_tokens < 1e-4)


def test_dcim_viha_and_a_stationary_paths_collect_stats():
    torch.manual_seed(2)
    config = tiny_config(
        num_hidden_layers=1,
        dcim_block_size=2,
        dcim_collect_stats=True,
        dcim_qk_mode="viha",
        dcim_av_mode="a_stationary",
    )
    model = BertModel(config)
    outputs = model(torch.randint(0, 101, (1, 5)))

    stats = outputs["dcim_stats"][0]
    assert stats["qk_viha_blocks"] == 3
    assert stats["av_a_stationary_blocks"] == 3


def test_tcs_thresholds_skip_low_activity_bit_planes():
    torch.manual_seed(3)
    config = tiny_config(
        num_hidden_layers=1,
        dcim_collect_stats=True,
        dcim_tcs_thresholds=[64, 64, 64, 64, 64, 64, 64, 64],
    )
    model = BertModel(config)
    outputs = model(torch.randint(0, 101, (1, 4)))

    assert outputs["dcim_stats"][0]["tcs_skipped_bit_planes"] > 0
