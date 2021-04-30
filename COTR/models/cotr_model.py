import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from COTR.utils import debug_utils, constants, utils
from .misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone
from .transformer import build_transformer
from .position_encoding import NerfPositionalEncoding, MLP


class COTR(nn.Module):

    def __init__(self, backbone, transformer, sine_type='lin_sine'):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.corr_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_proj = NerfPositionalEncoding(hidden_dim // 4, sine_type)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor, queries):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        _b, _q, _ = queries.shape
        queries = queries.reshape(-1, 2)
        queries = self.query_proj(queries).reshape(_b, _q, -1)
        queries = queries.permute(1, 0, 2)
        hs = self.transformer(self.input_proj(src), mask, queries, pos[-1])[0]
        outputs_corr = self.corr_embed(hs)
        out = {'pred_corrs': outputs_corr[-1]}
        return out


def build(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = COTR(
        backbone,
        transformer,
        sine_type=args.position_embedding,
    )
    return model
