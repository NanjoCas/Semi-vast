| label ratio | method | seeds | accuracy | macro-F1 | AUC | pseudo size |
|---|---|---|---|---|---|---|
| 0.10 | A supervised only | 3 | 0.5190 ± 0.0298 | 0.4480 ± 0.0386 | 0.6537 ± 0.0307 | 0 |
| 0.10 | B confidence threshold | 3 | 0.5796 ± 0.0633 | 0.5662 ± 0.0659 | 0.7398 ± 0.0619 | 1525 |
| 0.10 | L confidence + NLI direction (B+L) | 3 | 0.5758 ± 0.0904 | 0.5348 ± 0.1448 | 0.7370 ± 0.0790 | 773 |
| 0.10 | Q confidence + per-class c quantile (L-q) | 3 | 0.5703 ± 0.0547 | 0.5507 ± 0.0505 | 0.7321 ± 0.0387 | 976 |
| 0.10 | K top-|Q| by confidence (size control) | 3 | 0.5597 ± 0.0695 | 0.5440 ± 0.0639 | 0.7238 ± 0.0577 | 976 |
| 0.10 | O oracle (gold labels) | 3 | 0.6736 ± 0.0153 | 0.6611 ± 0.0138 | 0.8291 ± 0.0087 | 3855 |
