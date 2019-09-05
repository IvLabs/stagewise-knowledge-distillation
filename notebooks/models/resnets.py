from fastai.vision import *
import warnings

def res_block_with_depth(n_in, n_out, depth=1, dense:bool=False, norm_type:Optional[NormType]=NormType.Batch, **conv_kwargs):
    "Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`."
    norm2 = norm_type
    if not dense and (norm_type==NormType.Batch): norm2 = NormType.BatchZero
    layer1 = conv_layer(n_in, n_out, norm_type=norm_type, stride=(2, 2), **conv_kwargs)
    layers = []
    for _ in range(depth-1):
        layers.append(conv_layer(n_out, n_out, norm_type=norm2, **conv_kwargs))
    layers.append(MergeLayer(dense))
    if len(layers) == 0: return layer1
    return SequentialEx(layer1, SequentialEx(*layers))

def res_block(nf, depth=1, dense:bool=False, norm_type:Optional[NormType]=NormType.Batch, **conv_kwargs):
    "Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`."
    norm2 = norm_type
    if not dense and (norm_type==NormType.Batch): norm2 = NormType.BatchZero
    layers = []
    for i in range(depth):
        layers.append(conv_layer(nf, nf, norm_type=norm2, **conv_kwargs))
    layers.append(MergeLayer(dense))
    return SequentialEx(*layers)

def get_resnet(input_channels=3, out_classes=10, depth=2, n_blocks=3):
    if depth > 2: warnings.warn("Do not use this function for more than 2 depth, The residual connections will become insufficient")
    net = [conv_layer(3, 64, ks = 7, stride = 2, padding = 3),
            nn.MaxPool2d(3, 2, padding = 1),
            res_block(64, depth=depth)]
    in_c = net[0][0].out_channels
    for i in range(n_blocks):
        net.append(res_block_with_depth(in_c, 2*in_c, depth=depth))
        in_c *= 2
    net.append(create_head(in_c*2, 10, ps=0))
    return SequentialEx(*net)