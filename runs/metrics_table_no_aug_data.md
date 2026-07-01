# Evaluation Metrics

Weights: `weights\best.pt`  |  IoU threshold: 0.5  |  Confidence threshold: 0.25

Eval videos: eval, eval_1  |  N eval frames: 47  (sampled at ~1 FPS per video)

Distance assumptions: vehicle length = 4.5 m, horizontal FOV = 90.0 deg

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate | 60.7% | 44.8% |
| Precision | 66.7% | 75.7% |
| False alarms / min | 312.77 | 70.21 |
| Time to first detection (s) | 0.00 | 0.00 |

**mAP@0.5 (both bands combined, bonus metric):** 0.509

## Raw counts

| Band | TP | FP | FN |
|---|---|---|---|
| 0-200 m | 490 | 245 | 317 |
| 200-400 m | 171 | 55 | 211 |

## Time to first detection by video (s)

| Video | 0-200 m | 200-400 m |
|---|---|---|
| eval | 0.00 | N/A |
| eval_1 | 0.00 | 0.00 |
