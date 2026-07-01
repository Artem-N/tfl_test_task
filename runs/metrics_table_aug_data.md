# Evaluation Metrics

Weights: `weights\best.pt`  |  IoU threshold: 0.25  |  Confidence threshold: 0.2

Eval videos: eval, eval_1  |  N eval frames: 47  (sampled at ~1 FPS per video)

Distance assumptions: vehicle length = 4.5 m, horizontal FOV = 90.0 deg

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate | 59.1% | 23.6% |
| Precision | 76.3% | 92.8% |
| False alarms / min | 188.94 | 8.94 |
| Time to first detection (s) | 0.00 | 0.00 |

**mAP@0.5 (both bands combined, bonus metric):** 0.452

## Raw counts

| Band | TP | FP | FN |
|---|---|---|---|
| 0-200 m | 477 | 148 | 330 |
| 200-400 m | 90 | 7 | 292 |

## Time to first detection by video (s)

| Video | 0-200 m | 200-400 m |
|---|---|---|
| eval | 0.00 | N/A |
| eval_1 | 0.00 | 0.00 |
