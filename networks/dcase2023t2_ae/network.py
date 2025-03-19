import math
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_


class DyT(nn.Module):
    def __init__(self, C, init_a=0.5): 
        super().__init__() 
        self.alpha = nn.Parameter(torch.ones(1) * init_a) 
        self.gamma = nn.Parameter(torch.ones(C)) 
        self.beta = nn.Parameter(torch.zeros(C)) 
        self.scale_func = nn.Tanh()
        
    def forward(self, x): 
        x = self.scale_func(self.alpha * x) 
        return self.gamma * x + self.beta
    
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=DyT,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        o = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        
    def forward(self, x):
        x = self.proj(x)
        return x


class AENetX(nn.Module):
    def __init__(self, input_dim, block_size, embed_dim=192, depth=6,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=DyT,):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.patch_embed = PatchEmbed(input_dim=input_dim, embed_dim=embed_dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.encoder = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,)
            for i in range(depth // 2)])
        
        decoder_blocks = [Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,)
            for i in range(depth // 2, depth)]
        decoder_blocks.append(nn.Linear(self.embed_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_blocks)
        # trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        x = self.patch_embed(x)
        z = self.encoder(x)
        x_re = self.decoder(z)
        return x_re.unsqueeze(1), z.unsqueeze(1)


class AENet(nn.Module):
    def __init__(self,input_dim, block_size, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.embed_dim = 192
        self.neck_dim = 8
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            norm_layer(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            norm_layer(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            norm_layer(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            norm_layer(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.neck_dim),
            norm_layer(8),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.neck_dim, self.embed_dim),
            norm_layer(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            norm_layer(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            norm_layer(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            norm_layer(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.input_dim)
        )

    def forward(self, x):
        z = self.encoder(x.view(-1,self.input_dim))
        return self.decoder(z), z
