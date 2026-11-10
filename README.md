# Mini-ResNet for 16×16 Fruit Classification

## Overview

The goal of this network is to spend a tiny parameter budget (under 10k parameters) as efficiently as possible:

- keep enough channels from the start to avoid losing information,
- reuse the same width throughout the network,
- use residual connections so the network trains easily and can be regularized with stochastic depth.

## Data flow

```
input image            3 × 16 × 16
   │
 Stem (conv 3→16)      16 × 16 × 16   detect basic patterns: edges, colors
   │
 ResBlock 1            16 × 16 × 16   combine them into textures, simple shapes
 MaxPool               16 × 8 × 8     halve resolution, keep strongest responses
   │
 ResBlock 2            16 × 8 × 8     combine into larger parts of the fruit
 MaxPool               16 × 4 × 4
   │
 Global average pool   16             one number per feature: "how much of it is in the image"
 Dropout
 Linear (16→10)        10             scores for each fruit class
```

## Components

### 1. Stem: one convolution from 3 to 16 channels

It turns raw RGB pixels into 16 feature maps right away. This fixes the main weakness of the original network, which compressed the image into only 2 channels at the start and lost information that no later layer could recover.

### 2. Residual blocks: the core of the network

Each block has two convolutions plus a shortcut:

```
out = x + F(x)
```

The block doesn't have to produce a completely new representation. It only learns a **correction** `F(x)` to add to its input. This brings two advantages:

- **Easier training.** Gradients can flow directly back through the shortcuts, so even early layers receive a strong learning signal.
- **A safe default.** If a block has nothing useful to add, it can learn `F(x) ≈ 0` and simply pass its input through, so extra depth doesn't hurt.

### 3. Pooling between blocks

Each `MaxPool` halves the resolution (16 → 8 → 4). This makes each later layer see a larger area of the original image, so the network moves from local details (edges) to global structure (the shape of a banana vs an apple). It also reduces computation and adds a little robustness to small shifts.

### 4. Constant width (16 channels everywhere)

Networks normally increase the number of channels as resolution decreases, but with 10k parameters there's no room for that. Keeping 16 channels lets every shortcut be a plain identity, with no 1×1 projection that would cost parameters.

### 5. Global average pooling head

Instead of flattening the final 16×4×4 map (256 values), each feature map is averaged into one number. The classifier then asks *"is this feature present anywhere in the image?"*, not *"is it present in this exact position?"*.

- This fits fruit classification well, since the fruit can be anywhere in the image.
- It makes the classifier extremely cheap: 170 parameters instead of thousands.

### 6. Regularization

- **Stochastic depth** randomly skips the residual branch of a block during training, so blocks can't depend too much on each other. It only works because of the shortcuts: when a block is skipped, `x` still passes through.
- **Dropout** before the final layer prevents the classifier from relying on just a few features.
- **BatchNorm** after every convolution keeps the values in a stable range, which speeds up training and adds a small regularizing effect.

## Stochastic depth (`sd_prob`)

`sd_prob` is the probability that a residual block's branch is skipped during training:

```
out = x + F(x)     # block active
out = x            # block dropped: the input passes straight through the shortcut
```

| Technique | What is dropped |
|---|---|
| Dropout | Random neurons |
| Stochastic depth | Random blocks |

- Only applied during **training**. During evaluation all blocks are active, and `StochasticDepth` rescales the outputs so both modes behave consistently.
- The probability increases with depth: block 1 uses `sd_prob / 2`, block 2 uses `sd_prob`. Early layers learn basic features that every later layer depends on, so they are dropped less often.

## Parameter budget

| Part | Parameters | Share |
|---|---|---|
| Stem | 464 | 5% |
| ResBlock 1 | 4,672 | 47% |
| ResBlock 2 | 4,672 | 47% |
| Classifier | 170 | 2% |
| **Total** | **9,978** | |

Almost all the budget goes into the convolutions that learn features, and almost nothing into the classifier. This is the opposite of the original network, where most parameters sat in the last convolution and the flattened linear layer.

## Comparison with the original network

| | Original network | Mini-ResNet |
|---|---|---|
| Channels | 3 → 2 → 4 → 8 → 16 → 32 (funnel) | 3 → 16 → 16 → 16 → 16 → 16 (constant) |
| Skip connections | None | Around each pair of convolutions |
| Layer order | Conv → ReLU → BN | Conv → BN → ReLU |
| Classifier input | Flatten 32×2×2 = 128 values | Global average pool → 16 values |
| Stochastic depth | Not possible (no shortcuts) | Yes |
| Parameters | ~7,650 | 9,978 |


## Summary

A miniature ResNet: a wide-enough stem so no information is lost early, two residual blocks that refine features while pooling builds up a global view, and a cheap, position-invariant classifier, with stochastic depth and dropout to keep such a small model from memorizing the training set.