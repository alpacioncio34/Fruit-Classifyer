# Checkpoint 2026-11-27: why test (45 %) is far below val (67 %)

Stopped on request. Nothing in the existing project files was changed.

## Saved model.torch (ResNet, 64p), diag.py / diag2.py
| Check | Acc |
|---|---|
| val, all | 0.691 |
| val without near-copies of train images (879; 146 removed) | 0.674 |
| grayscale | 0.178 |
| noise | 0.399 |
| hue +0.05 | 0.438 |
| zoom in 2x | 0.439 |
| half size on white | 0.463 |
| blur | 0.502 |

Weak classes without near-copies: mango 0.30, apple 0.36.
The link between this fragility and the 45 % test score is a hypothesis (test images not seen).

## Ablation (exp.py): 200 epochs, seed 0, last-epoch EMA, val without near-copies
"covered" = distortions inside the strong augmentation ranges, "unseen" = never used in augmentation.

| Arch | Aug | Clean | TTA | Covered | Unseen | State |
|---|---|---|---|---|---|---|
| resnet (9978) | base | 0.676 | 0.678 | 0.574 | 0.549 | done |
| resnet (9978) | strong | 0.644 | 0.653 | 0.612 | 0.576 | done |
| mb (9854) | base | 0.729 at epoch 150 | - | - | - | stopped, no weights |
| mb | strong, strongmix | - | - | - | - | not run |
| ds | base, strong, strongmix | - | - | - | - | not run |

## Resume
    docker cp experiments/2026-11-27/exp.py pi1-main:/tmp/exp.py
    docker exec pi1-main sh -c 'cd /tmp && NT=10 python3 exp.py mb strong 200 0'
Args: arch (resnet|mb|ds), aug (base|strong|strongmix), epochs, seed.
CPU time with two runs in parallel: resnet ~11 min, mb ~28 min per 200 epochs.

## Next steps
1. Run mb base and mb strong (then strongmix); compare clean AND unseen.
2. Port the winner into challenge_train.py (Net + augmentation); check params <= 10000 with challenge_test.py.
3. Try longer training for strong augmentation (it was still behind on clean at 200 epochs).
4. Final submission: train on train+val with fixed settings; recompute teacher_logits.pt for the merged set.
5. Check the challenge rules allow the pretrained teacher.
6. Submitted code and weights must match: challenge_test.py loads with strict=False, and a wrong RES loads
   silently (model.torch gives 0.40 val if evaluated at 16p).
