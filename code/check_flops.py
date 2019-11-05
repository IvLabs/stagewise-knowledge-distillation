
from models import custom_resnet
from thop import profile, clever_format
import torch
from tqdm import tqdm

models = ['resnet10','resnet14', 'resnet20', 'resnet18', 'resnet26', 'resnet34']

model = getattr(custom_resnet, models[0])()

input = torch.randn(1, 3, 224, 224)

flops_dict, params_dict = {}, {}

for md in tqdm(models):
    model = getattr(custom_resnet, md)()

    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    
    flops_dict[md], params_dict[md] = flops, params

print("flops", flops_dict)

print("params", params_dict)

print(f"| models | MACs | Parameters (M) |")
for md in models:
    print(f"| {md} | {flops_dict[md]} | {params_dict[md]} |")
