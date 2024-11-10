import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
import numpy as np
from einops import rearrange, repeat

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Window_partition(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = pair(window_size)
    def forward(self, x):
        B, C, H, W = x.shape
        kH, kW = self.window_size
        sH, sW = kH // 2, kW // 2
        y = x.unfold(2, kH, kH).unfold(3, kW, kW) # B, C, nH, nW, kH, kW
        windows = y.permute(0, 2, 3, 4, 5, 1).reshape(B, -1, kH*kW*C)

        y_ = x[..., sH:-sH, sW:-sW].unfold(2, kH, kH).unfold(3, kW, kW) # B, C, nH, nW, kH, kW
        windows_ = y_.permute(0, 2, 3, 4, 5, 1).reshape(B, -1, kH*kW*C)

        return torch.cat([windows, windows_], 1)

def window_reverse(windows, window_size, resolution):
    H, W = pair(resolution)
    kH, kW = pair(window_size)
    sH, sW = kH // 2, kW // 2
    num = (H//kH) * (W//kW)
    B = windows.shape[0]
    x = windows[:, :num].view(B, H//kH, W//kW, kH, kW, 3)
    y = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, H, W)

    y_ = y.clone()
    x_ = windows[:, num:].view(B, H//kH-1, W//kW-1, kH, kW, 3)
    y_[..., sH:-sH, sW:-sW] = x_.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, H-kH, W-kW)

    return [y, y_]

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask=None):
        if mask is None:
            return self.fn(self.norm(x))
        else:
            return self.fn(self.norm(x), mask)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., window=16, resolution=512, is_overlap=True):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.window = window
        self.resolution = resolution
        self.is_overlap = is_overlap
        self.pooling = nn.AvgPool2d(2, stride=1)

    def forward(self, x, mask):
        B = x.shape[0]
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        with torch.no_grad():
            updated = (torch.mean(mask[:, 1:], dim=-1, keepdim=True) > 0.) * 1.
            updated = torch.cat([torch.ones_like(updated[:, :1]), updated], 1)
            qkv_m = mask @ torch.abs(self.to_qkv.weight.transpose(1, 0))
            qkv_m = qkv_m.chunk(3, dim = -1)
            q_m, k_m, v_m = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv_m)
            q_m = q_m / (1e-6 + torch.max(q_m, dim=2, keepdim=True)[0])
            k_m = k_m / (1e-6 + torch.max(k_m, dim=2, keepdim=True)[0])
            v_m = v_m / (1e-6 + torch.max(v_m, dim=2, keepdim=True)[0])

        dots = torch.matmul(q * q_m, (k * k_m).transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v_m * v)
        out = rearrange(out, 'b h n d -> b n (h d)') * updated 

        with torch.no_grad():
            m = torch.matmul(attn, v_m)
            m = rearrange(m, 'b h n d -> b n (h d)') * updated

        # bridge update for bridge tokens
        kH, kW = pair(self.window)
        H, W = pair(self.resolution)
        nH, nW = H // kH, W // kW

        if self.is_overlap:
            ## update region for bridge
            out_bridge = out[:, 1:nH*nW+1].reshape(B, nH, nW, -1).unfold(1, 2, 1).unfold(2, 2, 1) # B, nH-1, nW-1, C, 2, 2
            out_bridge = torch.mean(out_bridge, (-2, -1))
            with torch.no_grad():
                m_bridge = m[:, 1:nH*nW+1].reshape(B, nH, nW, -1).unfold(1, 2, 1).unfold(2, 2, 1) # B, nH-1, nW-1, C, 2, 2
                m_bridge = torch.mean(m_bridge, (-2, -1))
                # average & update
                m_bridge = m_bridge.reshape(B, (nH-1) * (nW-1), -1) * (1 - updated[:, -(nH-1)*(nW-1):])

            # average & update 
            out_bridge = out_bridge.reshape(B, (nH-1) * (nW-1), -1) * (1 - updated[:, -(nH-1)*(nW-1):])

            ## update region for origin
            out_origin = F.pad(out[:, -(nH-1)*(nW-1):].reshape(B, nH-1, nW-1, -1), (0, 0, 1, 1, 1, 1), value=0)  
            out_origin = out_origin.unfold(1, 2, 1).unfold(2, 2, 1) # B, nH, nW, C, 2, 2
            out_origin = torch.mean(out_origin, (-2, -1))
            with torch.no_grad():
                m_origin = F.pad(m[:, -(nH-1)*(nW-1):].reshape(B, nH-1, nW-1, -1), (0, 0, 1, 1, 1, 1), value=0)
                m_origin = m_origin.unfold(1, 2, 1).unfold(2, 2, 1) # B, nH, nW, C, 2, 2
                m_origin = torch.mean(m_origin, (-2, -1))

                # average & update
                m_origin = m_origin.reshape(B, nH*nW, -1) * (1 - updated[:, 1:nH*nW+1])

                # final update
                m[:, 1:nH*nW+1] += m_origin
                m[:, -(nH-1)*(nW-1):] += m_bridge

            # average & update
            out_origin = out_origin.reshape(B, nH*nW, -1) * (1 - updated[:, 1:nH*nW+1])

            # final update
            out[:, 1:nH*nW+1] += out_origin
            out[:, -(nH-1)*(nW-1):] += out_bridge

        with torch.no_grad():
            inter = F.interpolate(torch.mean(m[:, 1:], dim=-1, keepdim=True)[:, :16*16].reshape(-1, 1, 16, 16), (256, 256))
            inter_shift = F.interpolate(torch.mean(m[:, 1:], dim=-1, keepdim=True)[:, -15*15:].reshape(-1, 1, 15, 15), (15*16, 15*16))
            inter[..., 8:-8, 8:-8] = (inter[..., 8:-8, 8:-8] + inter_shift) / 2
            m = m @ torch.abs(self.to_out[0].weight.transpose(1, 0))

        return self.to_out(out), m, inter

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., window=16, resolution=512):
        super().__init__()
        self.layers = nn.ModuleList([])
        is_overlap = True
        kH, kW = pair(window)
        H, W = pair(resolution)
        self.num = (H//kH)*(W//kW)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, window=window, resolution=resolution, is_overlap=is_overlap)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, m):
        stack = []
        for i, (attn, ff) in enumerate(self.layers):
            y, m, inter = attn(x, m)
            x = y + x
            x = ff(x) + x
            stack.append(inter)
        return x, stack

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_h_patches = image_height // patch_height
        num_w_patches = image_width // patch_width
        patch_dim = (channels + 1) * patch_height * patch_width
        out_dim = channels * patch_height * patch_width

        self.to_patch = nn.Sequential(
            Window_partition(patch_size),
            nn.Linear(patch_dim, dim),
        )
        num_patches = num_h_patches*num_w_patches + (num_h_patches-1)*(num_w_patches-1)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, patch_size, image_size)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
            nn.Tanh()
        )

        self.window = patch_size
        self.resolution = image_size

        self.mask_to_flat = Window_partition(patch_size)

    def forward(self, img, mask):
        mask = 1 - mask
        x = self.to_patch(torch.cat((img, mask), 1))
        with torch.no_grad():
            m = self.mask_to_flat(mask.repeat(1, 4, 1, 1)) # b, n, 1
            m = torch.cat((torch.ones_like(m[:, :1]), m), dim=1)
            m = m @ torch.abs(self.to_patch[1].weight.transpose(1, 0))

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x, stack = self.transformer(x, m)
        x = x[:, 1:]
        x = self.mlp_head(x) # B, S, 3

        out = window_reverse(x, self.window, self.resolution)

        return out, stack

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

def window_partition_revised(x, window_size):
    B, H, W, C = x.shape
    kH, kW = window_size[0], window_size[1]
    x = x.unfold(1, kH, kH).unfold(2, kW, kW) # B, H, W, C, kH, kW
    windows = x.permute(0, 1, 2, 4, 5, 3).contiguous()
    return windows

def window_reverse_revised(windows, window_size, H, W):
    kH, kW = window_size[0], window_size[1]
    B = int(windows.shape[0] / (H * W / kH / kW))
    x = windows.view(B, H // kH, W // kW, kH, kW, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.dim))
            self.v_bias = nn.Parameter(torch.zeros(self.dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, size=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(attn.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock_revised(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, is_shift=False,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.is_shife = is_shift
        self.kH, self.kW = to_2tuple(window_size)
        self.sH, self.sW = self.kH//2, self.kW//2

        if self.input_resolution[0] <= self.kH:
            self.sH = 0
            self.kH = self.input_resolution[0]
            assert 0 <= self.sH < self.kH, "height shift_size must in 0-window_size"

        if self.input_resolution[1] <= self.kW:
            self.sW = 0
            self.kW = self.input_resolution[1]
            assert 0 <= self.sW < self.kW, "width shift_size must in 0-window-size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.kH, self.kW), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.is_shife:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.kH),
                        slice(-self.kH, -self.sH),
                        slice(-self.sH, None))
            w_slices = (slice(0, -self.kW),
                        slice(-self.kW, -self.sW),
                        slice(-self.sW, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition_revised(img_mask, (self.kH, self.kW))  # nW, H_window_size, W_window_size, 1
            mask_windows = mask_windows.view(-1, self.kH * self.kW)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        if (L != H * W):
            print(L, H, W)
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        if self.is_shife:
            shifted_x = torch.roll(x, shifts=(-self.sH, -self.sW), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition_revised(shifted_x, (self.kH, self.kW))  # nW*B, window_size, window_size, C
        size = x_windows.shape
        x_windows = x_windows.view(-1, self.kH * self.kW, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn(x_windows, mask=self.attn_mask, size=size)  # nW*B, window_size*window_size, C

        attn_windows = attn_windows.view(-1, self.kH, self.kW, C)
        shifted_x = window_reverse_revised(attn_windows, (self.kH, self.kW), H, W)  # B H' W' C

        if self.is_shife:
            x = torch.roll(shifted_x, shifts=(self.sH, self.sW), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={(self.kH, self.kW)}, shift_size={(self.sH, self.sW)}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.kH / self.kW
        flops += nW * self.attn.flops(self.kH * self.kW)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops

class PatchMerging_revised(nn.Module):
    def __init__(self, input_resolution, dim, stride, norm_layer=nn.LayerNorm, reduce_dim=True):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        stride = to_2tuple(stride)
        self.kH, self.kW = stride[0], stride[1]
        self.k = max(self.kH, self.kW)
        if reduce_dim:
            self.reduction = nn.Linear(self.kH * self.kW * dim, (self.k ** 2) * dim // 2, bias=False)
            self.norm = norm_layer((self.k ** 2) * dim // 2)
        else:
            self.reduction = nn.Linear(self.kH * self.kW * dim, dim, bias=False)
            self.norm = norm_layer(dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"
        assert H % self.kH == 0 and W % self.kW == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x = x.unfold(1, self.kH, self.kH).unfold(2, self.kW, self.kW) # B H/kH W/kW C kH kW
        x = x.reshape(B, -1, self.kH * self.kW * C)  # B (H*W)/(kH*kW) kH*kW*C
        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * (self.k ** 2) * self.dim * self.dim // 2
        flops += H * W * (self.k ** 2) * self.dim // (2*self.kH*self.kW)
        return flops

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d)."""
    def __init__(self, embedding_dim, padding_idx, init_size=1024, div_half_dim=False, center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None, div_half_dim=False):
        assert embedding_dim % 2 == 0, ('In this version, we request embedding_dim divisible by 2 but got {embedding_dim}')

        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        assert input.dim() == 2 or input.dim() == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        if center_shift is not None:
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches, 1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches, 1) + h_shift

        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), center_shift)
        return grid.to(x)

class UpSample(nn.Module):
    """ BilinearUpsample Layer. """
    def __init__(self, input_resolution, dim, out_dim, scale):
        super().__init__()
        assert dim % 2 == 0, f"x dim are not even."
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear')
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, out_dim, bias=False)
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.scale = scale
        self.alpha = nn.Parameter(torch.zeros(1))
        self.sin_pos_embed = SinusoidalPositionalEmbedding(embedding_dim=out_dim // 2, padding_idx=0, init_size=out_dim // 2)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size L:{}, H:{}, W{}".format(L, H, W)
        assert C == self.dim, "wrong in PatchMerging"

        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()   # B,C,H,W
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L*(self.scale**2), C)   # B,H,W,C
        x = self.norm(x)
        x = self.reduction(x)

        x = x.reshape(B, H * self.scale, W * self.scale, self.out_dim).permute(0, 3, 1, 2)
        x += self.sin_pos_embed.make_grid2d(H * self.scale, W * self.scale, B) * self.alpha
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * self.scale * W * self.scale, self.out_dim)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (self.scale**2) * H * W * self.dim
        flops += (self.scale**2) * H * W * self.dim * (self.out_dim)
        flops += (self.scale**2) * H * W * 2
        flops += (self.scale**2) * self.input_resolution[0] * self.input_resolution[1] * self.dim * 5
        return flops

class AvgPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return x.mean(1, keepdim=True)

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding """
    def __init__(self, img_size=512, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage. """
    def __init__(self, dim, input_resolution, num_heads, depth=2, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, stride=None, use_checkpoint=False,
                 pretrained_window_size=0, reduce_dim=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock_revised(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 is_shift=i % 2 == 1,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        if stride is not None:
            self.downsample = PatchMerging_revised(input_resolution, dim=dim, stride=stride, norm_layer=norm_layer, reduce_dim=reduce_dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class Refine(nn.Module):
    def __init__(self, in_c):
        super(Refine, self).__init__()
        dim = 32

        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c + 1, dim // 2, kernel_size=7,
                          stride=1, padding=3), nn.GELU()),  # 512, 512
            PatchEmbed(img_size=(512, 512), patch_size=4, in_chans=dim //
                       2, embed_dim=dim, norm_layer=nn.LayerNorm),  # 128, 128, c
            BasicLayer(dim=dim, input_resolution=(128, 128), num_heads=1,
                       depth=2, stride=2, window_size=4),  # 64, 64, c * 2
            BasicLayer(dim=dim * 2, input_resolution=(64, 64), num_heads=2,
                       depth=2, stride=2, window_size=4),  # 32, 32, c * 4
            BasicLayer(dim=dim * 4, input_resolution=(32, 32), num_heads=4,
                       depth=2, stride=2, window_size=4),  # 16, 16, c * 8
            BasicLayer(dim=dim * 8, input_resolution=(16, 16), num_heads=8,
                       depth=4, stride=2, window_size=4),  # 8, 8, c * 16
            nn.Sequential(
                BasicLayer(dim=dim * 16, input_resolution=(8, 8),
                           num_heads=16, depth=2, stride=None, window_size=2),
                AvgPool(dim * 16), nn.Linear(dim * 16, dim * 16), nn.ReLU())])  # 1, 1

        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 16, dim * 16), nn.ReLU(),
                          UpSample((1, 1), dim * 16, dim * 16, 4)),  # 8, 8
            nn.Sequential(
                BasicLayer(dim=dim * 32, input_resolution=(4, 4),
                           num_heads=16, depth=2, stride=None, window_size=2),
                UpSample((4, 4), dim * 32, dim * 16, 2)),  # 16, 16
            nn.Sequential(
                BasicLayer(dim=dim * 16 + dim * 8, input_resolution=(8, 8),
                           num_heads=8, depth=4, stride=None, window_size=4),
                UpSample((8, 8), dim * 16 + dim * 8, dim * 8, 2)),  # 32, 32
            nn.Sequential(
                BasicLayer(dim=dim * 8 + dim * 4, input_resolution=(16, 16),
                           num_heads=4, depth=2, stride=None, window_size=4),
                UpSample((16, 16), dim * 8 + dim * 4, dim * 4, 2)),  # 64, 64
            nn.Sequential(
                BasicLayer(dim=dim * 4 + dim * 2, input_resolution=(32, 32),
                           num_heads=2, depth=2, stride=None, window_size=4),
                UpSample((32, 32), dim * 4 + dim * 2, dim, 2)),  # 128, 128
            nn.Sequential(
                BasicLayer(dim=2 * dim, input_resolution=(64, 64),
                           num_heads=1, depth=2, stride=None, window_size=4),
                UpSample((64, 64), 2 * dim, dim // 2, 4)),  # 512, 512
            nn.Sequential(
                BasicLayer(dim=dim, input_resolution=(256, 256),
                           num_heads=1, depth=2, stride=None, window_size=8),
                UpSample((256, 256), dim, dim, 2),  # 512, 512
                Mlp(dim, out_features=3), nn.Tanh(),),  # 512, 512
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        feats = []
        for f in self.encoder_blocks:
            x = f(x)
            feats.append(x)  # Store the features from the encoder

        feats.pop()  # Remove the last feature which is the output of AvgPool

        for i, f in enumerate(self.decoder_blocks):
            if i == 0:
                x = f(x)
            else:
                feat = feats[-1]
                if len(feat.shape) > 3:
                    feat = feat.view(feat.size(0), feat.size(
                        1), -1).permute(0, 2, 1)

                # Ensure the dimensions match before concatenation
                feat = feat[:, :x.shape[1], :]

                x = f(torch.cat((x, feat), -1))
                feats.pop()
        outputs = x.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        return outputs


class CMT(nn.Module):
    def __init__(self, input_size=512, patch_size=16, depth=15, heads=16):
        super().__init__()
        self.coarse = ViT(input_size, patch_size,
                          (patch_size**2) * 3, depth, heads, 1024)
        self.refine = Refine(6)

    def forward(self, img, mask):
        c_gen, stack = self.coarse(img * (1 - mask), mask)
        c_gen_ = []
        for c_g in c_gen:
            c_gen_.append((c_g * mask) + img * (1 - mask))
        gen = self.refine(torch.cat(c_gen_+[mask], 1))
        gen = (gen * mask) + img * (1 - mask)
        return gen


if __name__ == '__main__':
    model = CMT().cuda()
    img = torch.randn(1, 3, 512, 512).cuda()
    mask = torch.randn(1, 1, 512, 512).cuda()
    gen = model(img, mask)
    print(gen.shape)