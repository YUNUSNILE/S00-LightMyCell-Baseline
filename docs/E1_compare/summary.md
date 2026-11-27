# Val comparison: E1a_unet-k16_20261121-1041, E1b_smpr34-k16_20261121-1317, E1c_runet-k16_20261121-1512

Scores at each run's best Val epoch (selected on this same Val set: slightly optimistic for all runs).

## Headline (Val, best checkpoint; organelle columns = study-macro (SSIM+PCC)/2)

| run | params_M | best_epoch | LMC (study macro) | LMC (image mean) | Nucleus | Mitochondria | Tubulin | Actin | last-epoch LMC | s/step | val pass (s) | total time | peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1a unet | 7.77 | 45 | 0.5281 | 0.4844 | 0.5578 | 0.4508 | 0.5187 | 0.5852 | 0.5238 | 1.12 | 110 | 1h55m | 2.28 |
| E1b smp_resnet34 | 24.48 | 35 | 0.5328 | 0.4897 | 0.5782 | 0.4601 | 0.5261 | 0.5669 | 0.5306 | 0.86 | 98 | 1h33m | 2.11 |
| E1c runet | 31.01 | 45 | 0.5086 | 0.4787 | 0.5086 | 0.4349 | 0.5193 | 0.5717 | 0.5039 | 1.54 | 107 | 2h31m | 2.3 |

## (SSIM+PCC)/2 per organelle x modality (image mean)

| organelle | tl | n | E1a | E1b | E1c |
|---|---|---|---|---|---|
| Actin | BF | 10 | 0.632 | 0.644 | 0.611 |
| Actin | PC | 20 | 0.61 | 0.584 | 0.603 |
| Mitochondria | DIC | 23 | 0.267 | 0.283 | 0.282 |
| Mitochondria | PC | 15 | 0.635 | 0.637 | 0.588 |
| Nucleus | BF | 10 | 0.667 | 0.703 | 0.598 |
| Nucleus | DIC | 28 | 0.294 | 0.28 | 0.302 |
| Nucleus | PC | 20 | 0.578 | 0.628 | 0.577 |
| Tubulin | BF | 10 | 0.647 | 0.684 | 0.635 |
| Tubulin | DIC | 28 | 0.336 | 0.346 | 0.315 |
| Tubulin | PC | 15 | 0.54 | 0.517 | 0.61 |

## SSIM and PCC per organelle x modality

| organelle | tl | pcc_E1a | pcc_E1b | pcc_E1c | ssim_E1a | ssim_E1b | ssim_E1c |
|---|---|---|---|---|---|---|---|
| Actin | BF | 0.573 | 0.628 | 0.633 | 0.69 | 0.659 | 0.589 |
| Actin | PC | 0.621 | 0.599 | 0.636 | 0.599 | 0.57 | 0.569 |
| Mitochondria | DIC | 0.124 | 0.119 | 0.106 | 0.41 | 0.447 | 0.459 |
| Mitochondria | PC | 0.624 | 0.583 | 0.594 | 0.646 | 0.691 | 0.581 |
| Nucleus | BF | 0.563 | 0.625 | 0.715 | 0.771 | 0.781 | 0.48 |
| Nucleus | DIC | 0.243 | 0.177 | 0.225 | 0.344 | 0.384 | 0.379 |
| Nucleus | PC | 0.491 | 0.537 | 0.53 | 0.665 | 0.719 | 0.625 |
| Tubulin | BF | 0.669 | 0.759 | 0.771 | 0.624 | 0.61 | 0.498 |
| Tubulin | DIC | 0.216 | 0.184 | 0.16 | 0.456 | 0.509 | 0.47 |
| Tubulin | PC | 0.626 | 0.589 | 0.659 | 0.453 | 0.446 | 0.562 |

## (SSIM+PCC)/2 per organelle x study

| organelle | study | tl | size | E1a | E1b | E1c |
|---|---|---|---|---|---|---|
| Actin | 72 | PC | 15 | 0.658 | 0.64 | 0.653 |
| Actin | 81 | PC | 5 | 0.466 | 0.417 | 0.451 |
| Actin | 82 | BF | 10 | 0.632 | 0.644 | 0.611 |
| Mitochondria | 20 | DIC | 23 | 0.267 | 0.283 | 0.282 |
| Mitochondria | 72 | PC | 15 | 0.635 | 0.637 | 0.588 |
| Nucleus | 20 | DIC | 23 | 0.209 | 0.196 | 0.256 |
| Nucleus | 72 | PC | 15 | 0.541 | 0.594 | 0.566 |
| Nucleus | 81 | PC | 5 | 0.689 | 0.73 | 0.61 |
| Nucleus | 82 | BF | 10 | 0.667 | 0.703 | 0.598 |
| Nucleus | 86 | DIC | 5 | 0.683 | 0.668 | 0.513 |
| Tubulin | 20 | DIC | 23 | 0.276 | 0.288 | 0.259 |
| Tubulin | 72 | PC | 15 | 0.54 | 0.517 | 0.61 |
| Tubulin | 82 | BF | 10 | 0.647 | 0.684 | 0.635 |
| Tubulin | 86 | DIC | 5 | 0.612 | 0.615 | 0.573 |

## Paired bootstrap of the LMC-score difference (images resampled within study x organelle, 2000 draws)

| A | B | LMC(A)-LMC(B) | 95% CI low | 95% CI high | P(A>B) |
|---|---|---|---|---|---|
| E1a | E1b | -0.0047 | -0.0111 | 0.002 | 0.08 |
| E1a | E1c | 0.0195 | 0.0084 | 0.0311 | 1.0 |
| E1b | E1c | 0.0242 | 0.0134 | 0.0349 | 1.0 |
