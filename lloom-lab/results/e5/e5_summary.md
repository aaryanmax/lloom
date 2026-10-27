# E5: No-Free-Lunch (NFL) Theorem Empirical Evaluation

Demonstration of performance collapse when ground truth labels are permuted via derangements (zero fixed points).

## zeroshot_gemini
| $n$ | True F1 (mean ± std) | NFL Permuted F1 (mean ± std) | Performance Collapse |
|---|---|---|---|
| 10 | 0.551 ± 0.074 | 0.018 ± 0.011 | **96.7%** |
| 20 | 0.579 ± 0.071 | 0.031 ± 0.008 | **94.6%** |
| 40 | 0.588 ± 0.043 | 0.029 ± 0.009 | **95.0%** |
| 80 | 0.615 ± 0.016 | 0.034 ± 0.006 | **94.4%** |
| 160 | 0.613 ± 0.024 | 0.032 ± 0.002 | **94.8%** |

## zeroshot_qwen
| $n$ | True F1 (mean ± std) | NFL Permuted F1 (mean ± std) | Performance Collapse |
|---|---|---|---|
| 10 | 0.249 ± 0.077 | 0.014 ± 0.015 | **94.4%** |
| 20 | 0.253 ± 0.028 | 0.032 ± 0.003 | **87.3%** |
| 40 | 0.284 ± 0.060 | 0.027 ± 0.007 | **90.7%** |
| 80 | 0.237 ± 0.034 | 0.029 ± 0.004 | **87.6%** |
| 160 | 0.240 ± 0.012 | 0.029 ± 0.004 | **87.9%** |
