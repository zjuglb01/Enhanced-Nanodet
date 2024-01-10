import copy

from .gfl_head import GFLHead
from .nanodet_head import NanoDetHead
from .nanodet_plus_head import NanoDetPlusHead
from .simple_conv_head import SimpleConvHead
from .nanodet_plus_plus_head import NanoDetPlusPlusHead
def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "GFLHead":
        return GFLHead(**head_cfg)
    elif name == "NanoDetHead":
        return NanoDetHead(**head_cfg)
    elif name == "NanoDetPlusHead":
        return NanoDetPlusHead(**head_cfg)
    elif name == "SimpleConvHead":
        return SimpleConvHead(**head_cfg)
    elif name == "NanoDetPlusPlusHead":
        return NanoDetPlusPlusHead(**head_cfg)
    else:
        raise NotImplementedError
