# DenseNet proposal.
After testing , implementing new features, even distillation learning, we reached the same top, ~66% accuracy, I am suspecting that this might be close to the top performance possible on the RESNet, therefore I am changing 
my strategy, since historically, densenet has had same accuracy as ResNet using fewer parameters, I expect this new implementation to better the performance of the Mini ResNet.

ResBlocks will be substituted by DenseBlocks, in each one, each layer gets the feature maps of all the earlier layers, concateneated as input. Therefore each new layer adds only k new inputs.

We will also need to add transition layers between blocks, that are bascially 1x1 convs + 2x2 average pooling that reduce the size.

The background to starting this approach is that, the net learns too fast, without fitting well to the data, so the tradeoff here is to use other resource, for example DenseNet will need much more memory, because of
the concatenated feature maps.

## Other points to look at

1. Bottleneck block

Uses 1x1 convolutions to reduce channels, then uses 3x3 convolutions to restore to the original channels, the cost is almost the same but gets 4 times wider outputs:
Why would this fail: With 16x16 pixels for each image, wider outputs might not be what we need (if wide refers to more sight of vision, it is worth trying it tho)

2. Depthwise separable convolutions

This feature comes from Efficient CNNs.

Standard convolutions: Each filter mixes space and all channels
Deothwise: One kxk filter per input channel, spatial only

This leads to fewer parameters and its used in MobileNet.

3. Modernization

- Inverted bottleneck
- GELU
- LayerNorm
- Fewer norm and activation layers

Perhaps trying out theese things would help the networks performance.

4. Repeated Augmentation

5. MobileNet

MobileNets are desgined to be efficient under low computational circumstances (embedded systems, mobile phones...), perhaps would be an interesting point to start looking.
