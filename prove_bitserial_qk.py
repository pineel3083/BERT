import torch

from tpdcim_bert_patch import (
    quantize_symmetric_int8,
    qkt_tiled_int8,
    qkt_tiled_bitserial,
)


def decompose_q_int8(q_int8):
    """Return uint8 two's-complement view of signed int8 tensor."""
    return torch.bitwise_and(q_int8.to(torch.int16), 255)


def bitserial_q_matmul_k(q_int8, k_int8, disabled_bit=None):
    """Reconstruct q_int8 @ k_int8.T using Q bit-planes."""
    q_uint8 = decompose_q_int8(q_int8)
    k_fp = k_int8.to(torch.float32)

    acc = torch.zeros(
        (*q_int8.shape[:-1], k_int8.shape[-2]),
        device=q_int8.device,
        dtype=torch.float32,
    )

    bit_report = []

    for bit in range(8):
        if bit == disabled_bit:
            bit_report.append((bit, "disabled", 0, 0.0))
            continue

        q_bit = torch.bitwise_and(
            torch.bitwise_right_shift(q_uint8, bit),
            1,
        ).to(torch.float32)

        bit_weight = float(1 << bit) if bit < 7 else -128.0

        partial = torch.matmul(q_bit, k_fp.transpose(2, 3))
        acc = acc + bit_weight * partial

        bit_report.append(
            (
                bit,
                bit_weight,
                int(q_bit.sum().item()),
                float(partial.abs().sum().item()),
            )
        )

    return acc, bit_report


def manual_small_check():
    print("=" * 80)
    print("CHECK 1: manual small tensor bit-serial reconstruction")
    print("=" * 80)

    torch.manual_seed(0)

    B, H, Tq, Tk, D = 1, 1, 4, 4, 8

    q_fp = torch.randn(B, H, Tq, D) * 0.5
    k_fp = torch.randn(B, H, Tk, D) * 0.5

    q_int8, q_scale = quantize_symmetric_int8(q_fp)
    k_int8, k_scale = quantize_symmetric_int8(k_fp)

    direct = torch.matmul(
        q_int8.to(torch.float32),
        k_int8.to(torch.float32).transpose(2, 3),
    )

    bitserial, report = bitserial_q_matmul_k(q_int8, k_int8)

    diff = (direct - bitserial).abs()

    print("q_int8 sample:", q_int8[0, 0, 0, :].tolist())
    print("k_int8 sample:", k_int8[0, 0, 0, :].tolist())
    print()
    print("bit | weight | q_bit_ones | partial_abs_sum")
    for bit, weight, ones, psum in report:
        print(f"{bit:3d} | {str(weight):>6s} | {ones:10d} | {psum:15.4f}")

    print()
    print("direct int8 matmul:")
    print(direct[0, 0])
    print()
    print("bitserial reconstructed matmul:")
    print(bitserial[0, 0])
    print()
    print("max diff :", diff.max().item())
    print("mean diff:", diff.mean().item())

    assert diff.max().item() == 0.0, "manual bitserial != direct int8 matmul"
    print("PASS")


def bit_ablation_check():
    print()
    print("=" * 80)
    print("CHECK 2: bit ablation")
    print("=" * 80)

    torch.manual_seed(1)

    B, H, Tq, Tk, D = 1, 1, 4, 4, 8

    q_fp = torch.randn(B, H, Tq, D) * 0.5
    k_fp = torch.randn(B, H, Tk, D) * 0.5

    q_int8, _ = quantize_symmetric_int8(q_fp)
    k_int8, _ = quantize_symmetric_int8(k_fp)

    direct = torch.matmul(
        q_int8.to(torch.float32),
        k_int8.to(torch.float32).transpose(2, 3),
    )

    normal, _ = bitserial_q_matmul_k(q_int8, k_int8)
    no_bit0, _ = bitserial_q_matmul_k(q_int8, k_int8, disabled_bit=0)
    no_bit7, _ = bitserial_q_matmul_k(q_int8, k_int8, disabled_bit=7)

    normal_diff = (direct - normal).abs()
    no_bit0_diff = (direct - no_bit0).abs()
    no_bit7_diff = (direct - no_bit7).abs()

    print("normal max diff :", normal_diff.max().item())
    print("no bit0 max diff:", no_bit0_diff.max().item())
    print("no bit7 max diff:", no_bit7_diff.max().item())

    assert normal_diff.max().item() == 0.0, "normal bitserial failed"
    assert no_bit0_diff.max().item() > 0.0, "removing bit0 did not change result"
    assert no_bit7_diff.max().item() > 0.0, "removing sign bit did not change result"

    print("PASS")


def tiled_function_check():
    print()
    print("=" * 80)
    print("CHECK 3: qkt_tiled_int8 vs qkt_tiled_bitserial")
    print("=" * 80)

    torch.manual_seed(2)

    B, H, N, D = 1, 2, 16, 8
    tile_n = 4
    scaling = 1.0

    query = torch.randn(B, H, N, D)
    key = torch.randn(B, H, N, D)

    scores_int8, stats_int8 = qkt_tiled_int8(
        query=query,
        key=key,
        scaling=scaling,
        tile_n=tile_n,
        enable_sparse=False,
    )

    scores_bitserial, stats_bitserial = qkt_tiled_bitserial(
        query=query,
        key=key,
        scaling=scaling,
        tile_n=tile_n,
        enable_sparse=False,
    )

    diff = (scores_int8 - scores_bitserial).abs()

    print("int8 stats:", stats_int8)
    print("bitserial stats:", stats_bitserial)
    print("max diff :", diff.max().item())
    print("mean diff:", diff.mean().item())

    expected_tiles = (N // tile_n) * (N // tile_n)
    expected_bit_ops = expected_tiles * 8

    print("expected tiles:", expected_tiles)
    print("expected bit ops:", expected_bit_ops)

    assert diff.max().item() == 0.0, "tiled bitserial != tiled int8"
    assert stats_bitserial.get("computed_qk_tiles") == expected_tiles
    assert stats_bitserial.get("bit_ops_total") == expected_bit_ops

    print("PASS")


def main():
    manual_small_check()
    bit_ablation_check()
    tiled_function_check()
    print()
    print("=" * 80)
    print("ALL CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
