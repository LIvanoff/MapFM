import torch
from torch.nn import functional as F

from mmdet.models.builder import BACKBONES
from mmcv.runner.base_module import BaseModule

@BACKBONES.register_module()
class RADIO(BaseModule):
    def __init__(self,
                 model_version="radio_v2.5-b",
                 img_size=[480, 800],
                 freeze=None,
                 chunk_size=None,
                 ):
        super(RADIO, self).__init__()
        self.chunk_size = chunk_size
        self.freeze = freeze
        
        self.radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
        if "e-radio_v2" in model_version:
            self.radio.model.set_optimal_window_size(img_size)
        
        for param in self.radio.parameters():
            param.requires_grad = False
        
        #self._freeze_stages()
        
    def forward(self, x):
        x = x.div(255.0)  # Без in-place — безопаснее для автоград
        B = x.size(0)
        # if self.chunk_size is None: self.chunk_size = B
        
        
        if self.freeze is None:
            self.radio.eval()

        spatial_outputs = []

        if self.chunk_size:
            # assert B % self.chunk_size == 0, "self.chunk_size должно быть кратно 6"
            for i in range(0, B, self.chunk_size):
                x_chunk = x[i:i + self.chunk_size].contiguous()
                
                # nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
                # x_chunk = F.interpolate(x_chunk, nearest_res, mode='bilinear', align_corners=False)
                # x_chunk = x_chunk.to(torch.bfloat16).cuda()
                _, spatial_features = self.radio(x_chunk, feature_fmt='NCHW')
                assert spatial_features.ndim == 4
                spatial_outputs.append(spatial_features)
            spatial_outputs = torch.cat(spatial_outputs, dim=0)
        else:
            _, spatial_features = self.radio(x, feature_fmt='NCHW')
            assert spatial_features.ndim == 4
            spatial_outputs = spatial_features

        return [spatial_outputs]
