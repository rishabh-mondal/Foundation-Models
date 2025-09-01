# models/satmae_vit.py (Updated for modern timm)

from functools import partial
import torch
import torch.nn as nn
import numpy as np # Make sure numpy is imported

# This import will now work because we upgraded timm
import timm.models.vision_transformer

# Import from local models directory
from pos_embed import get_2d_sincos_pos_embed


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    # ==================== THE CRITICAL CHANGE IS HERE ====================
    def __init__(self, patch_size=16, global_pool=False, **kwargs):
        # We now explicitly accept `patch_size` and pass it to super()
        super(VisionTransformer, self).__init__(patch_size=patch_size, **kwargs)
    # =====================================================================

        pos_embed_data = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            if hasattr(self, 'norm'):
                del self.norm  # remove the original norm

    # The rest of the file is the same as before
    def forward_backbone(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x[:, 1:, :]

    def forward_features(self, x):
        x = self.forward_backbone(x)
        if self.global_pool:
            x = x.mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            outcome = x.mean(dim=1)
        return outcome


def vit_base_patch16(**kwargs):
    """Factory function for ViT-Base. `patch_size` is expected in kwargs."""
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    """Factory function for ViT-Large. `patch_size` is expected in kwargs."""
    model = VisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    """Factory function for ViT-Huge. `patch_size` is expected in kwargs."""
    model = VisionTransformer(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model