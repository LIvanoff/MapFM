from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
#model_version="radio_v2.5-g" # for RADIOv2.5-g model (ViT-H/14)
# model_version="radio_v2.5-h" # for RADIOv2.5-H model (ViT-H/16)
# model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)
model_version="radio_v2.5-b" # for RADIOv2.5-B model (ViT-B/16)
# model_version="e-radio_v2" # for E-RADIO
model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
# model.eval()
# model = model.cuda()
# if "e-radio_v2" in model_version:
#     model.model.set_optimal_window_size([480, 800])
x = torch.rand(1, 3, 480, 800, device='cpu')
nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
x.div_(255.0)

# summary, spatial_features = model(x, feature_fmt='NCHW')
# x = x.to(torch.bfloat16).cuda()
summary, spatial_features = model(x)
print(model)
print("x.shape:", x.shape)
print("summary.shape:", summary.shape)
print("spatial_features.shape:", spatial_features.shape)

# from transformers import AutoModel
# import torch
# model = AutoModel.from_pretrained("theaiinstitute/theia-base-patch16-224-cdiv", trust_remote_code=True)
# fake_input = torch.zeros((1, 480 ,800, 3), dtype=torch.uint8)

# theia_feature = model(fake_input)
# for key, value in theia_feature.items():
#     print(key, value.shape)
    