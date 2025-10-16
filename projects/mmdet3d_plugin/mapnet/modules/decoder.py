import torch
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer
from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import \
    MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.runner.base_module import BaseModule
import warnings
from mmcv.cnn import xavier_init, constant_init
import torch.nn as nn
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
import copy
import math

def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapNetDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False,
                 query_pos_embedding='none',
                 num_pts_per_vec=20,
                 num_heads=8,
                 **kwargs):
        super(MapNetDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.query_pos_embedding = query_pos_embedding
        self.num_pts_per_vec = num_pts_per_vec
        self.num_heads = num_heads
        if query_pos_embedding == 'instance':
            def _get_clones(module, N):
                return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
            pt_pos_query_proj = nn.Linear(self.embed_dims, self.embed_dims)
            self.pt_pos_query_projs = _get_clones(pt_pos_query_proj, self.num_layers)
            
        # self.seg_attn = nn.MultiheadAttention(self.embed_dims, self.num_heads, dropout=0.1) # Correct place
        # self.norm_seg = nn.LayerNorm(self.embed_dims) # It is very important to add layer norm
        for layer in self.layers:
            layer.seg_attn = nn.MultiheadAttention(self.embed_dims, self.num_heads, dropout=0.1) # Create for layer
            layer.norm_seg = nn.LayerNorm(self.embed_dims) # Add this too

    def init_weights(self):
        if self.query_pos_embedding == 'instance':
            for m in self.pt_pos_query_projs:
                xavier_init(m, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                seg_embed=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        bs = reference_points.shape[0]
        for lid, layer in enumerate(self.layers):
            # import ipdb; ipdb.set_trace()
            if seg_embed is not None: #Проверяем что он существует
                seg_attn_layer = getattr(layer, 'seg_attn', None) #Получаем seg_attn если он есть.
                if seg_attn_layer is not None: #Проверяем что слой внимания есть в данном слое.
                    seg_embed_level = seg_embed[-1] # Shape: (100, 2, 256) from (6, 100, 2, 256)
                    seg_embed_perm = seg_embed_level.permute(0, 1, 2)
                    output = seg_attn_layer(output, seg_embed_perm, seg_embed_perm)[0] # Q,K,V
                    output = layer.norm_seg(output)
            if self.query_pos_embedding == 'instance':
                reference_points_reshape = reference_points.view(bs, -1, self.num_pts_per_vec, 2)
                reference_points_reshape = reference_points_reshape.view(bs, -1, 2)
                query_sine_embed = gen_sineembed_for_position(reference_points_reshape)
                query_sine_embed = query_sine_embed.view(bs, -1, self.num_pts_per_vec, self.embed_dims)
                point_query_pos = self.pt_pos_query_projs[lid](query_sine_embed)
                query_pos_lid = None
                reference_points_input = reference_points_reshape[..., :2].unsqueeze(2)
            else:
                point_query_pos = None
                reference_points_input = reference_points[..., :2].unsqueeze(2)
                query_pos_lid = query_pos
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos_lid,
                pt_query_pos=point_query_pos,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points