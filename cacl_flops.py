import importlib
import torch
from option import get_option
from thop import profile

opt = get_option()
module = importlib.import_module("model.{}".format(opt.model.lower()))
dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
# 输入
dummy_input = torch.randn(1, 3, 224, 224).to(dev)  # .to(device)
model = module.Net(opt).to(dev)
flops, params = profile(model, inputs=(dummy_input,))

print('FLOPs: ', flops, 'params: ', params)

print('FLOPs: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))