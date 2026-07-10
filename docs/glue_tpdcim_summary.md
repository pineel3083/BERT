# GLUE TP-DCIM Final Summary

This document summarizes the final representative MNLI and MRPC task-level results for the software TP-DCIM BERT experiments. The results are trend-oriented and use public Hugging Face checkpoints plus sparse-aware fine-tuned checkpoints. They are not exact TP-DCIM paper reproduction numbers.

Detailed per-task notes:

- [MNLI TP-DCIM evaluation notes](mnli_tpdcim_results.md)
- [MRPC TP-DCIM evaluation notes](mrpc_tpdcim_results.md)

## Common Setup

- Base model family: BERT-base
- Attention patch: QK-only TP-DCIM software patch
- Sparse pattern: BigBird-like block sparsity
- Random sparse blocks: `num_random_blocks=1`
- Local window: `1`
- Sequence length: `max_length=128`
- Tile size: `tile_n=16`
- Number of sequence blocks: `8`
- TCS threshold order in code: LSB-to-MSB
- TCS active rule: `bit_sum > threshold`

## Representative Thresholds

| Task | Selected TCS name | Thresholds MSB-to-LSB | Thresholds LSB-to-MSB |
| --- | --- | --- | --- |
| MNLI | `medium_p3` | `[0,8,16,22,28,34,40,48]` | `[48,40,34,28,22,16,8,0]` |
| MRPC | `paper_mrpc_soft` | `[0,16,20,24,28,36,40,48]` | `[48,40,36,28,24,20,16,0]` |

## Final Task-Level Results

For MNLI, the primary metric is accuracy. For MRPC, the primary metric is F1.

| Task | Metric | Dense baseline | Sparse before FT | Sparse after FT | Sparse + TCS after FT | Sparse recovery from FT | Final drop vs dense | TCS extra drop vs sparse | Reduction vs dense | Reduction vs BigBird |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MNLI | accuracy | 0.845746 | 0.823230 | 0.840754 | 0.837290 | +0.017524 | -0.008456 | -0.003464 | x0.407 | x0.635 |
| MRPC | F1 | 0.913495 | 0.782258 | 0.898649 | 0.896082 | +0.116391 | -0.017413 | -0.002567 | x0.394 | x0.615 |

## Detailed Final Operating Points

| Task | Checkpoint for final sparse/TCS eval | Selected TCS | Score | Accuracy | F1 | TCS row skip | Operation saving vs dense | Operation saving vs BigBird |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNLI | `checkpoints/sparse_mnli_textattack_len128_tile16_rand1_full_e1` | `medium_p3` | 0.837290 | 0.837290 | - | 0.364708 | 0.593016 | 0.364708 |
| MRPC | `checkpoints/sparse_mrpc_textattack_len128_tile16_rand1_full_e5_run1` | `paper_mrpc_soft` | 0.896082 | 0.850490 | 0.896082 | 0.384853 | 0.605922 | 0.384853 |

## Main Interpretation

1. Sparse/BigBird-style block attention causes measurable task degradation before sparse-aware fine-tuning.
2. Sparse-aware fine-tuning recovers most of the sparse degradation for both MNLI and MRPC.
3. TCS can then be applied on top of the sparse-aware checkpoint with a small additional task-score drop.
4. The final representative TCS points reduce the QK activity proxy to about `x0.40` of dense and about `x0.62-0.64` of the BigBird proxy.

## Paper Table 1 Alignment

The TP-DCIM paper Table 1 reports different threshold patterns for MNLI and MRPC. The representative points here follow that direction:

- MNLI uses a softer MNLI-style vector with MSB-to-LSB `[0,8,16,22,28,34,40,48]`.
- MRPC uses a stronger MRPC-style vector with MSB-to-LSB `[0,16,20,24,28,36,40,48]`.

The absolute scores differ from the paper because these experiments use public checkpoints and a software-level attention patch. The useful comparison is the trend: sparse-aware fine-tuning restores the sparse baseline, and TCS adds computation reduction with small incremental task degradation.

## Remaining Work

SQuADv1.1 is not included in the current final summary. Adding SQuAD would require a separate question-answering evaluation path: span preprocessing, start/end logit postprocessing, EM/F1 scoring, and likely sparse-aware QA fine-tuning. It can be added as a future extension if exact Table 1 coverage becomes necessary.
