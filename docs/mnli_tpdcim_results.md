# MNLI TP-DCIM Evaluation Notes

This note records the MNLI results collected during the TP-DCIM BERT experiments. The goal is not exact TP-DCIM paper reproduction, but a reproducible task-level trend study with a public Hugging Face BERT checkpoint and the software TP-DCIM attention patch.

## Setup

- Task: GLUE MNLI, `validation_matched`
- Dataset source: `nyu-mll/glue`
- Dense public checkpoint: `textattack/bert-base-uncased-MNLI`
- Primary metric: accuracy
- Sequence length used for the main runs: `max_length=128`
- Sparse/TP-DCIM tiling: `tile_n=16`, `num_blocks=8`
- Sparse pattern: local window 1, global block 0, deterministic random block count 1
- TCS threshold convention: vectors are stored as LSB-to-MSB in code
- TCS active rule: `bit_sum > threshold`
- TextAttack MNLI label order: pass `--checkpoint-label-order textattack`

## Dense Length Sanity Check

The dense public checkpoint was evaluated with both 128 and 512 token padding/truncation. The result difference was negligible, so the later paper-like runs used `max_length=128`.

| max_length | batch_size | accuracy | mean_nonpad_tokens | max_nonpad_tokens |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 0.8457463067 | 39.053693 | 128 |
| 512 | 16 | 0.8458481915 | 39.179929 | 237 |

## Main Full-Validation Results

Dense baseline reference:

- `software_score = 0.845746306673459`

Sparse-aware fine-tuned sparse reference:

- `sparse_reference_score = 0.8407539480387163`

| Stage | Checkpoint | Thresholds LSB-to-MSB | Accuracy | Delta vs Dense | Delta vs Sparse Ref | TCS row skip | Reduction vs Dense | Reduction vs BigBird |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Dense SW baseline | `textattack/bert-base-uncased-MNLI` | - | 0.8457463067 | 0.000000 | - | 0.000000 | x1.000 | - |
| BigBird proxy before FT | `textattack/bert-base-uncased-MNLI` | - | 0.8232297504 | -0.022517 | - | 0.000000 | x0.641 | x1.000 |
| TCS medium before FT | `textattack/bert-base-uncased-MNLI` | `[48,36,24,20,16,12,8,0]` | 0.8211920530 | -0.024554 | -0.002038 | 0.237787 | x0.488 | x0.762 |
| BigBird proxy after sparse-aware FT | `checkpoints/sparse_mnli_textattack_len128_tile16_rand1_full_e1` | - | 0.8407539480 | -0.004992 | 0.000000 | 0.000000 | x0.641 | x1.000 |
| TCS medium after sparse-aware FT | `checkpoints/sparse_mnli_textattack_len128_tile16_rand1_full_e1` | `[48,36,24,20,16,12,8,0]` | 0.8400407539 | -0.005706 | -0.000713 | 0.237826 | x0.488 | x0.762 |
| TCS medium_p3 after sparse-aware FT | `checkpoints/sparse_mnli_textattack_len128_tile16_rand1_full_e1` | `[48,40,34,28,22,16,8,0]` | 0.8372898625 | -0.008456 | -0.003464 | 0.364708 | x0.407 | x0.635 |
| TCS aggressive_p2 after sparse-aware FT | `checkpoints/sparse_mnli_textattack_len128_tile16_rand1_full_e1` | `[56,44,38,30,24,18,10,0]` | 0.8340295466 | -0.011717 | -0.006724 | 0.416836 | x0.374 | x0.583 |

## Selected Representative Point

Use `medium_p3` as the main MNLI TCS operating point:

- LSB-to-MSB: `[48,40,34,28,22,16,8,0]`
- MSB-to-LSB: `[0,8,16,22,28,34,40,48]`
- Accuracy after sparse-aware FT + TCS: `0.8372898625`
- Additional drop vs sparse-aware BigBird proxy: `0.346 percentage points`
- Total drop vs dense public checkpoint: `0.846 percentage points`
- TCS row skip ratio: `0.364708`
- Reduction vs dense QK activity proxy: `x0.407`
- Reduction vs BigBird proxy: `x0.635`

This point is the best current balance: it keeps the additional TCS accuracy loss near the paper-like range while saving substantially more row activity than the original `medium` vector.

## Interpretation

1. The public dense baseline around `84.57%` is consistent with common BERT-base MNLI numbers. It is much lower than the TP-DCIM paper table's reported MNLI baseline, so absolute accuracy should not be compared directly to the paper table.
2. Sparse attention without sparse-aware fine-tuning caused a large MNLI drop: `84.57% -> 82.32%`.
3. One epoch of sparse-aware fine-tuning recovered most of that loss: `82.32% -> 84.08%` sparse accuracy.
4. After sparse-aware fine-tuning, TCS can be applied with controlled additional degradation.
5. `medium_p3` is currently the strongest paper-style tradeoff point for MNLI.

## Reproduction Commands

Dense baseline, length 128:

```bash
CUDA_VISIBLE_DEVICES=7 python eval_glue_dense_baseline.py \
  --task mnli \
  --checkpoint textattack/bert-base-uncased-MNLI \
  --checkpoint-label-order textattack \
  --batch-size 32 \
  --max-length 128
```

Sparse-aware fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=7 python finetune_glue_sparse.py \
  --task mnli \
  --model-name textattack/bert-base-uncased-MNLI \
  --checkpoint-label-order textattack \
  --output-dir checkpoints/sparse_mnli_textattack_len128_tile16_rand1_full_e1 \
  --epochs 1 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --max-length 128 \
  --tile-n 16 \
  --num-random-blocks 1 \
  --learning-rate 1e-5 \
  --fp16 \
  --log-every 1000
```

Before-FT sparse and TCS medium:

```bash
CUDA_VISIBLE_DEVICES=7 python eval_fixed_tcs_thresholds.py \
  --task mnli \
  --checkpoint textattack/bert-base-uncased-MNLI \
  --checkpoint-label-order textattack \
  --batch-size 4 \
  --max-length 128 \
  --tile-n 16 \
  --num-random-blocks 1 \
  --threshold-candidates-lsb-to-msb 48,36,24,20,16,12,8,0 \
  --threshold-names tcs_medium \
  --output-csv results/before_ft_tcs_medium_mnli_full_len128_tile16_rand1.csv
```

After-FT final TCS candidates:

```bash
CUDA_VISIBLE_DEVICES=7 python eval_tcs_only.py \
  --task mnli \
  --checkpoint checkpoints/sparse_mnli_textattack_len128_tile16_rand1_full_e1 \
  --checkpoint-label-order textattack \
  --reference-score 0.8407539480387163 \
  --software-score 0.845746306673459 \
  --batch-size 4 \
  --max-length 128 \
  --tile-n 16 \
  --num-random-blocks 1 \
  --threshold-candidates-lsb-to-msb \
    48,40,34,28,22,16,8,0 \
    56,44,38,30,24,18,10,0 \
  --threshold-names \
    medium_p3 \
    aggressive_p2 \
  --output-csv results/after_ft_tcs_final_candidates_mnli_full_len128_tile16_rand1.csv
```
