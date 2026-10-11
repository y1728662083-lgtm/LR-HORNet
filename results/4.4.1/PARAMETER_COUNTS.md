# Section 4.4.1 parameter counts

The table reports **total parameters**. Values in parentheses are exact counts.
All available folds were checked and were internally consistent.

| Model | ds003626 | Directional Words–Russian | Directional Words–Spanish | Bimodal Inner Speech |
|---|---:|---:|---:|---:|
| LR-HORNet | 6.726 K (6,726) | 3.592 K (3,592) | 5.192 K (5,192) | 4.938 K (4,938) |
| EEGNet | 5.460 K (5,460) | 2.326 K (2,326) | 3.926 K (3,926) | 3.544 K (3,544) |
| ATCNet | 117.124 K (117,124) | 29.724 K (29,724) | 114.574 K (114,574) | 34.080 K (34,080) |
| EEGMamba | 3.278 M (3,278,047) | 3.278 M (3,278,449) | 3.278 M (3,278,449) | 3.279 M (3,278,851) |
| FAST | 257.380 K (257,380) | 160.966 K (160,966) | 165.094 K (165,094) | 191.784 K (191,784) |
| LaBraM | 5.821 M (5,820,740) | 5.821 M (5,821,142) | 5.821 M (5,821,142) | 5.822 M (5,821,544) |

Notes:

- EEGMamba contains 200 non-trainable parameters in every dataset; all other listed parameters are trainable.
- Counts can vary across datasets because channel count, input length, and output-class dimensions are dataset dependent.
- Official-baseline counts come from `test_metrics.json`; LR-HORNet, EEGNet, and ATCNet are counted from saved state dictionaries with BatchNorm buffers excluded.
