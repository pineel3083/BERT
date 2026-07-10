# MRPC TP-DCIM Evaluation Notes

This note records the MRPC results collected during the TP-DCIM BERT experiments. The selected MRPC TCS operating point is `paper_mrpc_soft`, which follows the direction of the TP-DCIM paper Table 1 threshold pattern for MRPC.

## Setup

- Task: GLUE MRPC, `validation`
- Dataset source: `nyu-mll/glue`
- Dense public checkpoint: `textattack/bert-base-uncased-MRPC`
- Primary metric: F1
- Sequence length: `max_length=128`
- Sparse/TP-DCIM tiling: `tile_n=16`, `num_blocks=8`
- Sparse pattern: local window 1, global block 0, deterministic random block count 1
- TCS threshold convention in code: LSB-to-MSB
- TCS active rule: `bit_sum > threshold`
- Label order: dataset order, use `--checkpoint-label-order dataset`

## Dense Baseline

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8774509804 |
| F1 | 0.9134948097 |
| mean_nonpad_tokens | 53.242647 |
| max_nonpad_tokens | 86 |

The maximum observed non-padding length was 86, so `max_length=128` is sufficient for the current MRPC validation setup.

## Before Sparse-Aware Fine-Tuning

| Stage | Checkpoint | Thresholds LSB-to-MSB | Accuracy | F1 | Delta vs Dense F1 | TCS row skip | Reduction vs Dense | Reduction vs BigBird |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Dense SW baseline | `textattack/bert-base-uncased-MRPC` | - | 0.8774509804 | 0.9134948097 | 0.000000 | 0.000000 | x1.000 | - |
| BigBird proxy before FT | `textattack/bert-base-uncased-MRPC` | - | 0.7352941176 | 0.7822580645 | -0.131237 | 0.000000 | x0.641 | x1.000 |
| TCS medium_p3 before FT | `textattack/bert-base-uncased-MRPC` | `[48,40,34,28,22,16,8,0]` | 0.7303921569 | 0.7791164659 | -0.134378 | 0.364584 | x0.407 | x0.635 |

The sparse attention mask caused a large MRPC F1 drop before sparse-aware fine-tuning. Therefore, threshold tuning alone is not meaningful before recovering the sparse reference.

## Sparse-Aware Fine-Tuning

Sparse-aware fine-tuning was run from the public MRPC checkpoint using sparse FP32 QK during training.

- Output checkpoint: `checkpoints/sparse_mrpc_textattack_len128_tile16_rand1_full_e5_run1`
- Best epoch: 3
- Best validation metrics:
  - Accuracy: `0.8529411765`
  - F1: `0.8986486486`
  - Loss: `0.9183819431`

Epoch-level validation metrics:

| Epoch | Accuracy | F1 | Loss |
| ---: | ---: | ---: | ---: |
| 1 | 0.852941 | 0.895833 | 0.647696 |
| 2 | 0.843137 | 0.892256 | 0.898011 |
| 3 | 0.852941 | 0.898649 | 0.918382 |
| 4 | 0.833333 | 0.887043 | 1.058169 |
| 5 | 0.843137 | 0.892256 | 0.979365 |

Sparse-aware fine-tuning recovered most of the sparse degradation:

- Dense F1: `0.9134948097`
- Sparse before FT F1: `0.7822580645`
- Sparse after FT F1: `0.8986486486`
- Recovery: about `+11.64 percentage points` F1

## Paper-Like TCS Candidate Sweep After Fine-Tuning

The TP-DCIM paper Table 1 uses different threshold patterns for MNLI and MRPC. For MRPC, the paper shows a stronger MSB-to-LSB pattern than MNLI: `[0,16,...,36,40,48]`. Because the omitted values are not fully specified, three paper-like candidates were evaluated.

Reference scores used by `eval_tcs_only.py`:

- `software_score = 0.9134948096885814`
- `sparse_reference_score = 0.8986486486486487`

| Candidate | Thresholds MSB-to-LSB | Thresholds LSB-to-MSB | Accuracy | F1 | Drop vs Sparse F1 | Delta vs Dense F1 | TCS row skip | Reduction vs Dense | Reduction vs BigBird |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `paper_mrpc_soft` | `[0,16,20,24,28,36,40,48]` | `[48,40,36,28,24,20,16,0]` | 0.8504901961 | 0.8960817717 | 0.0025668769 | -0.0174130380 | 0.384853 | x0.394 | x0.615 |
| `paper_mrpc_mid` | `[0,16,24,28,32,36,40,48]` | `[48,40,36,32,28,24,16,0]` | 0.8455882353 | 0.8930390492 | 0.0056095994 | -0.0204557605 | 0.454482 | x0.349 | x0.546 |
| `paper_mrpc_hard` | `[0,16,24,32,36,36,40,48]` | `[48,40,36,36,32,24,16,0]` | 0.8161764706 | 0.8713550600 | 0.0272935886 | -0.0421397497 | 0.539830 | x0.295 | x0.460 |

## Selected Representative Point

Use `paper_mrpc_soft` as the representative MRPC TCS operating point.

- LSB-to-MSB: `[48,40,36,28,24,20,16,0]`
- MSB-to-LSB: `[0,16,20,24,28,36,40,48]`
- Accuracy after sparse-aware FT + TCS: `0.8504901961`
- F1 after sparse-aware FT + TCS: `0.8960817717`
- Additional drop vs sparse-aware BigBird proxy: `0.2567 percentage points` F1
- Total drop vs dense public checkpoint: `1.7413 percentage points` F1
- TCS row skip ratio: `0.384853`
- Reduction vs dense QK activity proxy: `x0.394`
- Reduction vs BigBird proxy: `x0.615`

This point matches the paper-style MRPC trend best: TCS adds a small additional task-score drop over the sparse reference while giving a substantial extra row-skip/activity reduction.

## Interpretation

1. MRPC is much more sensitive to the sparse mask than MNLI before sparse-aware fine-tuning.
2. Sparse-aware fine-tuning is essential for MRPC: before FT, sparse F1 dropped by about `13.12 percentage points`; after FT, the residual dense-to-sparse F1 drop was about `1.48 percentage points`.
3. After sparse-aware fine-tuning, TCS behaves as expected. `paper_mrpc_soft` adds only about `0.26 percentage points` F1 drop relative to the sparse reference.
4. `paper_mrpc_mid` saves more activity but exceeds the paper-like extra-drop target. `paper_mrpc_hard` is too aggressive and should not be used as the representative result.

## Reproduction Commands

Dense baseline:

```bash
CUDA_VISIBLE_DEVICES=7 python eval_glue_dense_baseline.py \
  --task mrpc \
  --checkpoint textattack/bert-base-uncased-MRPC \
  --batch-size 32 \
  --max-length 128
```

Before-FT sparse and TCS medium_p3:

```bash
CUDA_VISIBLE_DEVICES=7 python eval_fixed_tcs_thresholds.py \
  --task mrpc \
  --checkpoint textattack/bert-base-uncased-MRPC \
  --batch-size 4 \
  --max-length 128 \
  --tile-n 16 \
  --num-random-blocks 1 \
  --threshold-candidates-lsb-to-msb 48,40,34,28,22,16,8,0 \
  --threshold-names medium_p3 \
  --output-csv results/before_ft_tcs_medium_p3_mrpc_full_len128_tile16_rand1.csv
```

Sparse-aware fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=7 python finetune_glue_sparse.py \
  --task mrpc \
  --model-name textattack/bert-base-uncased-MRPC \
  --checkpoint-label-order dataset \
  --output-dir checkpoints/sparse_mrpc_textattack_len128_tile16_rand1_full_e5_run1 \
  --epochs 5 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --max-length 128 \
  --tile-n 16 \
  --num-random-blocks 1 \
  --learning-rate 1e-5 \
  --fp16 \
  --log-every 50
```

After-FT paper-like TCS sweep:

```bash
CUDA_VISIBLE_DEVICES=7 python eval_tcs_only.py \
  --task mrpc \
  --checkpoint checkpoints/sparse_mrpc_textattack_len128_tile16_rand1_full_e5_run1 \
  --checkpoint-label-order dataset \
  --reference-score 0.8986486486486487 \
  --software-score 0.9134948096885814 \
  --batch-size 4 \
  --max-length 128 \
  --tile-n 16 \
  --num-random-blocks 1 \
  --threshold-candidates-lsb-to-msb \
    48,40,36,28,24,20,16,0 \
    48,40,36,32,28,24,16,0 \
    48,40,36,36,32,24,16,0 \
  --threshold-names \
    paper_mrpc_soft \
    paper_mrpc_mid \
    paper_mrpc_hard \
  --output-csv results/after_ft_tcs_paper_like_mrpc_full_len128_tile16_rand1.csv
```
