# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import  PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath


__all__ = [
    'resmlp_12', 'resmlp_24', 'resmlp_36', 'resmlpB_24'
]

class Mlp_pac(nn.Module):
    def __init__(self, hidden_features, ratio):
        super().__init__()
        self.v = nn.Parameter(torch.randn(hidden_features, int(ratio*hidden_features)))
        self.u = nn.Parameter(torch.randn(int(ratio*hidden_features), hidden_features))

        self.b = nn.Parameter(torch.randn(hidden_features))

    def forward(self, x):
        y = x @ self.v @ self.u  + self.b
        return y

class Mlp_correct(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.w = nn.Parameter(torch.eye(features))
        self.b = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        y = x @ self.w  + self.b
        return y

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., fc1_len=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.correct_fc1 = Mlp_correct(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.uv1 = Mlp_pac(hidden_features, 0.125)
        # self.correct_fc1 = Mlp_correct(hidden_features)
        self.act = act_layer()
        self.correct_fc2 = Mlp_correct(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.uv2 = Mlp_pac(out_features, 0.5)
        # self.correct_fc2 = Mlp_correct(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        after_fc1_correct = self.correct_fc1(x)
        after_fc1 = self.fc1(after_fc1_correct)
        after_uv1 = self.uv1(after_fc1)
        after_act = self.act(after_uv1)

        after_fc2_correct = self.correct_fc2(after_act)
        after_fc2 = self.fc2(after_fc2_correct)
        after_uv2 = self.uv2(after_fc2)
        return after_uv2, x, after_fc1_correct, after_act, after_fc2_correct

        # after_fc1 = self.fc1(x)
        # after_uv1 = self.uv1(after_fc1)
        # after_fc1_correct = self.correct_fc1(after_uv1)

        # after_act = self.act(after_fc1_correct)

        # after_fc2 = self.fc2(after_act)
        # after_uv2 = self.uv2(after_fc2)
        # after_fc2_correct = self.correct_fc2(after_uv2)
        # return after_fc2_correct, after_uv1, after_fc1_correct, after_uv2, after_fc2_correct

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta    
    
class layers_scale_mlp_blocks(nn.Module):

    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU,init_values=1e-4,num_patches = 196):
        super().__init__()
        self.norm1 = Affine(dim)
        self.attn = nn.Linear(num_patches, num_patches)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = Affine(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(4.0 * dim), act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x).transpose(1,2)).transpose(1,2))
        y,  before_fc1_correct, after_fc1_correct, before_fc2_correct, after_fc2_correct = self.mlp(self.norm2(x))
        x = x + self.drop_path(self.gamma_2 * y)
        return x , before_fc1_correct, after_fc1_correct, before_fc2_correct, after_fc2_correct


class resmlp_models(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,drop_rate=0.,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                drop_path_rate=0.0,init_scale=1e-4):
        super().__init__()



        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=int(in_chans), embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        dpr = [drop_path_rate for i in range(depth)]

        self.blocks = nn.ModuleList([
            layers_scale_mlp_blocks(
                dim=embed_dim,drop=drop_rate,drop_path=dpr[i],
                act_layer=act_layer,init_values=init_scale,
                num_patches=num_patches)
            for i in range(depth)])


        self.norm = Affine(embed_dim)


        self.correct_feature = Mlp_correct(embed_dim)
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            # nn.init.eye_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        before_fc1_correct_list, after_fc1_correct_list, before_fc2_correct_list, after_fc2_correct_list = [], [], [], []
        for i , blk in enumerate(self.blocks):
            x, before_fc1_correct, after_fc1_correct, before_fc2_correct, after_fc2_correct  = blk(x)
            before_fc1_correct_list.append(before_fc1_correct) 
            after_fc1_correct_list.append(after_fc1_correct) 
            before_fc2_correct_list.append(before_fc2_correct) 
            after_fc2_correct_list.append(after_fc2_correct)
        x = self.norm(x)
        x = x.mean(dim=1).reshape(B,1,-1)

        return x[:, 0], before_fc1_correct_list, after_fc1_correct_list, before_fc2_correct_list, after_fc2_correct_list

    def forward(self, x):
        feature, before_fc1_correct_list, after_fc1_correct_list, before_fc2_correct_list, after_fc2_correct_list  = self.forward_features(x)
        after_correct_feature = self.correct_feature(feature)
        x = self.head(after_correct_feature)
        return x, feature, after_correct_feature ,before_fc1_correct_list, after_fc1_correct_list, before_fc2_correct_list, after_fc2_correct_list

@register_model
def resmlp_12(pretrained=False,dist=False, **kwargs):
    model = resmlp_models(
        patch_size=16, embed_dim=384, depth=12,
        Patch_layer=PatchEmbed,
        init_scale=0.1,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth"
        else:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_12_no_dist.pth"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )
            
        model.load_state_dict(checkpoint)
    return model
  
@register_model
def resmlp_24(pretrained=False,dist=False,dino=False, **kwargs):
    model = resmlp_models(
        patch_size=16, embed_dim=384, depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-5,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth"
        elif dino:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_24_dino.pth"
        else:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_24_no_dist.pth"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )
            
        model.load_state_dict(checkpoint)
    return model
  
@register_model
def resmlp_36(pretrained=False,dist=False, **kwargs):
    model = resmlp_models(
        patch_size=16, embed_dim=384, depth=36,
        Patch_layer=PatchEmbed,
        init_scale=1e-6,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth"
        else:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_36_no_dist.pth"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )
            
        model.load_state_dict(checkpoint)
    return model

@register_model
def resmlpB_24(pretrained=False,dist=False, in_22k = False, **kwargs):
    model = resmlp_models(
        patch_size=8, embed_dim=768, depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-6,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth"
        elif in_22k:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth"
        else:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth"
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )
            
        model.load_state_dict(checkpoint)
    
    return model