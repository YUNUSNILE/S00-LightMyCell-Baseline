# Val comparison: E1g_smpcnxt-k16-best+mip-ema_20261124-0535, E1b_smpr34-k16_20261121-1317, E1h_unet-k16-best+mip-ema_20261127-1959

Scores at each run's best Val epoch (selected on this same Val set: slightly optimistic for all runs).
Secondary number: LMC-score without Study 20 (reporting only; selection uses all of Val).

## Headline (Val, best checkpoint; organelle columns = study-macro (SSIM+PCC)/2)

| run                   |   params_M |   best_epoch |   LMC (study macro) |   LMC (image mean) |   LMC w/o S20 |   Nucleus |   Mitochondria |   Tubulin |   Actin |   last-epoch LMC |   s/step |   val pass (s) | total time   |   peak GPU GB |
|:----------------------|-----------:|-------------:|--------------------:|-------------------:|--------------:|----------:|---------------:|----------:|--------:|-----------------:|---------:|---------------:|:-------------|--------------:|
| E1g smp_convnext_tiny |      31.95 |           25 |              0.5782 |             0.5395 |        0.6525 |    0.6607 |         0.4799 |    0.5535 |  0.6189 |           0.5588 |     0.73 |            148 | 1h35m        |          2.96 |
| E1b smp_resnet34      |      24.48 |           35 |              0.5328 |             0.4897 |        0.6208 |    0.5782 |         0.4601 |    0.5261 |  0.5669 |           0.5306 |     0.86 |             98 | 1h33m        |          2.11 |
| E1h unet              |       7.77 |           50 |              0.5138 |             0.4741 |        0.5938 |    0.5632 |         0.4242 |    0.4918 |  0.5759 |           0.5138 |     0.68 |            153 | 1h35m        |          2.19 |

## (SSIM+PCC)/2 per organelle x modality (image mean)

| organelle | tl | n | E1b | E1g | E1h |
|---|---|---|---|---|---|
| Actin | BF | 10 | 0.644 | 0.675 | 0.64 |
| Actin | PC | 20 | 0.584 | 0.651 | 0.591 |
| Mitochondria | DIC | 23 | 0.283 | 0.365 | 0.259 |
| Mitochondria | PC | 15 | 0.637 | 0.595 | 0.589 |
| Nucleus | BF | 10 | 0.703 | 0.863 | 0.705 |
| Nucleus | DIC | 28 | 0.28 | 0.354 | 0.278 |
| Nucleus | PC | 20 | 0.628 | 0.692 | 0.607 |
| Tubulin | BF | 10 | 0.684 | 0.688 | 0.57 |
| Tubulin | DIC | 28 | 0.346 | 0.359 | 0.347 |
| Tubulin | PC | 15 | 0.517 | 0.584 | 0.507 |

## SSIM and PCC per organelle x modality

| organelle | tl | pcc_E1b | pcc_E1g | pcc_E1h | ssim_E1b | ssim_E1g | ssim_E1h |
|---|---|---|---|---|---|---|---|
| Actin | BF | 0.628 | 0.734 | 0.607 | 0.659 | 0.616 | 0.674 |
| Actin | PC | 0.599 | 0.661 | 0.603 | 0.57 | 0.642 | 0.58 |
| Mitochondria | DIC | 0.119 | 0.133 | 0.121 | 0.447 | 0.597 | 0.398 |
| Mitochondria | PC | 0.583 | 0.593 | 0.611 | 0.691 | 0.597 | 0.567 |
| Nucleus | BF | 0.625 | 0.914 | 0.595 | 0.781 | 0.813 | 0.814 |
| Nucleus | DIC | 0.177 | 0.244 | 0.187 | 0.384 | 0.464 | 0.369 |
| Nucleus | PC | 0.537 | 0.641 | 0.475 | 0.719 | 0.743 | 0.739 |
| Tubulin | BF | 0.759 | 0.86 | 0.726 | 0.61 | 0.516 | 0.413 |
| Tubulin | DIC | 0.184 | 0.172 | 0.23 | 0.509 | 0.546 | 0.465 |
| Tubulin | PC | 0.589 | 0.652 | 0.597 | 0.446 | 0.517 | 0.417 |

## (SSIM+PCC)/2 per organelle x study

| organelle | study | tl | size | E1b | E1g | E1h |
|---|---|---|---|---|---|---|
| Actin | 72 | PC | 15 | 0.64 | 0.712 | 0.639 |
| Actin | 81 | PC | 5 | 0.417 | 0.47 | 0.449 |
| Actin | 82 | BF | 10 | 0.644 | 0.675 | 0.64 |
| Mitochondria | 20 | DIC | 23 | 0.283 | 0.365 | 0.259 |
| Mitochondria | 72 | PC | 15 | 0.637 | 0.595 | 0.589 |
| Nucleus | 20 | DIC | 23 | 0.196 | 0.275 | 0.208 |
| Nucleus | 72 | PC | 15 | 0.594 | 0.661 | 0.561 |
| Nucleus | 81 | PC | 5 | 0.73 | 0.785 | 0.744 |
| Nucleus | 82 | BF | 10 | 0.703 | 0.863 | 0.705 |
| Nucleus | 86 | DIC | 5 | 0.668 | 0.719 | 0.598 |
| Tubulin | 20 | DIC | 23 | 0.288 | 0.297 | 0.293 |
| Tubulin | 72 | PC | 15 | 0.517 | 0.584 | 0.507 |
| Tubulin | 82 | BF | 10 | 0.684 | 0.688 | 0.57 |
| Tubulin | 86 | DIC | 5 | 0.615 | 0.645 | 0.598 |

## Paired bootstrap of the LMC-score difference (images resampled within study x organelle, 2000 draws; study-level columns resample the Val studies within each organelle instead)

| A | B | LMC(A)-LMC(B) | 95% CI low | 95% CI high | P(A>B) | w/o S20: diff | w/o: CI low | w/o: CI high | study-level CI low | study-level CI high | study-level P(A>B) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1g | E1b | 0.0454 | 0.0322 | 0.0582 | 1.0 | 0.0317 | 0.0163 | 0.0453 | 0.023 | 0.0689 | 1.0 |
| E1g | E1h | 0.0645 | 0.0511 | 0.0769 | 1.0 | 0.0587 | 0.0443 | 0.0721 | 0.0421 | 0.0863 | 1.0 |
| E1b | E1h | 0.019 | 0.0138 | 0.0244 | 1.0 | 0.027 | 0.0209 | 0.0335 | 0.0056 | 0.0339 | 0.999 |

## Task 2 (MIP) on Val, at each run's best Task 1 epoch

| run | task | LMC (study macro) | LMC w/o S20 | Nucleus | Mitochondria | Tubulin | Actin |
|---|---|---|---|---|---|---|---|
| E1g smp_convnext_tiny | mip | 0.5896 | 0.6655 | 0.6604 | 0.4913 | 0.5648 | 0.6421 |
| E1h unet | mip | 0.5405 | 0.6117 | 0.5995 | 0.4493 | 0.5228 | 0.5906 |
