from custom_resnet import _resnet, Bottleneck

net = _resnet('resnet50', Bottleneck, [2, 2, 2, 1], pretrained = False, progress = False)
