import torch
from thop import profile
from model.CDNet import opt_sar_CDNet


model = opt_sar_CDNet(Transfer=True, Structure=True, Sementic=True)

# 计算模型计算量（FLOPs）
input_data = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(input_data,input_data))

print(f"模型计算量:")
print(f"  FLOPs: {flops/1e9:.2f} GFLOPs")
print(f"  参数: {params/1e6:.2f} M")