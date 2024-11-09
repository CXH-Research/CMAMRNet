import torch
import math
import functools
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DownSample(nn.Module):
    """ sample with convolutional operation
        :param input_nc: input channel
        :param with_conv: use convolution to refine the feature
        :param kernel_size: feature size
        :param return_mask: return mask for the confidential score
    """
    def __init__(self, input_nc, with_conv=False, kernel_size=3, return_mask=False):
        super(DownSample, self).__init__()
        self.with_conv = with_conv
        self.return_mask = return_mask
        if self.with_conv:
            self.conv = PartialConv2d(input_nc, input_nc, kernel_size=kernel_size, stride=2,
                                      padding=int(int(kernel_size-1)/2), return_mask=True)

    def forward(self, x, mask=None):
        if self.with_conv:
            x, mask = self.conv(x, mask)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            mask = F.avg_pool2d(mask, kernel_size=2, stride=2) if mask is not None else mask
        if self.return_mask:
            return x, mask
        else:
            return x
        

class Identity(nn.Module):
    def forward(self, x):
        return x
    

class ResnetBlock(nn.Module):
    def __init__(self, input_nc, output_nc=None, kernel=3, dropout=0.0, activation='gelu', norm='pixel', return_mask=False):
        super(ResnetBlock, self).__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)
        self.return_mask = return_mask

        output_nc = input_nc if output_nc is None else output_nc

        self.norm1 = norm_layer(input_nc)
        self.conv1 = PartialConv2d(input_nc, output_nc, kernel_size=kernel, padding=int((kernel-1)/2), return_mask=True)
        self.norm2 = norm_layer(output_nc)
        self.conv2 = PartialConv2d(output_nc, output_nc, kernel_size=kernel, padding=int((kernel-1)/2), return_mask=True)
        self.dropout = nn.Dropout(dropout)
        self.act = activation_layer

        if input_nc != output_nc:
            self.short = PartialConv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0)
        else:
            self.short = Identity()

    def forward(self, x, mask=None):
        x_short = self.short(x)
        x, mask = self.conv1(self.act(self.norm1(x)), mask)
        x, mask = self.conv2(self.dropout(self.act(self.norm2(x))), mask)
        if self.return_mask:
            return (x + x_short) / math.sqrt(2), mask
        else:
            return (x + x_short) / math.sqrt(2)
        

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask1 = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask1)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask1)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask / self.slide_winsize   # replace the valid value to confident score
        else:
            return output
        

class AttnAware(nn.Module):
    def __init__(self, input_nc, activation='gelu', norm='pixel', num_heads=2):
        super(AttnAware, self).__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)
        head_dim = input_nc // num_heads
        self.num_heads = num_heads
        self.input_nc = input_nc
        self.scale = head_dim ** -0.5

        self.query_conv = nn.Sequential(
            norm_layer(input_nc),
            activation_layer,
            nn.Conv2d(input_nc, input_nc, kernel_size=1)
        )
        self.key_conv = nn.Sequential(
            norm_layer(input_nc),
            activation_layer,
            nn.Conv2d(input_nc, input_nc, kernel_size=1)
        )

        self.weight = nn.Conv2d(self.num_heads*2, 2, kernel_size=1, stride=1)
        self.to_out = ResnetBlock(input_nc * 2, input_nc, 1, 0, activation, norm)

    def forward(self, x, pre=None, mask=None):
        B, C, W, H = x.size()
        q = self.query_conv(x).view(B, -1, W*H)
        k = self.key_conv(x).view(B, -1, W*H)
        v = x.view(B, -1, W*H)

        q = rearrange(q, 'b (h d) n -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b (h d) n -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b (h d) n -> b h n d', h=self.num_heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if pre is not None:
            # attention-aware weight
            B, head, N, N = dots.size()
            mask_n = mask.view(B, -1, 1, W * H).expand_as(dots)
            w_visible = (dots.detach() * mask_n).max(dim=-1, keepdim=True)[0]
            w_invisible = (dots.detach() * (1-mask_n)).max(dim=-1, keepdim=True)[0]
            weight = torch.cat([w_visible.view(B, head, W, H), w_invisible.view(B, head, W, H)], dim=1)
            weight = self.weight(weight)
            weight = F.softmax(weight, dim=1)
            # visible attention score
            pre_v = pre.view(B, -1, W*H)
            pre_v = rearrange(pre_v, 'b (h d) n -> b h n d', h=self.num_heads)
            dots_visible = torch.where(dots > 0, dots * mask_n, dots / (mask_n + 1e-8))
            attn_visible = dots_visible.softmax(dim=-1)
            context_flow = torch.einsum('bhij, bhjd->bhid', attn_visible, pre_v)
            context_flow = rearrange(context_flow, 'b h n d -> b (h d) n').view(B, -1, W, H)
            # invisible attention score
            dots_invisible = torch.where(dots > 0, dots * (1 - mask_n), dots / ((1 - mask_n) + 1e-8))
            attn_invisible = dots_invisible.softmax(dim=-1)
            self_attention = torch.einsum('bhij, bhjd->bhid', attn_invisible, v)
            self_attention = rearrange(self_attention, 'b h n d -> b (h d) n').view(B, -1, W, H)
            # out
            out = weight[:, :1, :, :]*context_flow + weight[:, 1:, :, :]*self_attention
        else:
            attn = dots.softmax(dim=-1)
            out = torch.einsum('bhij, bhjd->bhid', attn, v)

            out = rearrange(out, 'b h n d -> b (h d) n').view(B, -1, W, H)

        out = self.to_out(torch.cat([out, x], dim=1))
        return out
    

class UpSample(nn.Module):
    """ sample with convolutional operation
    :param input_nc: input channel
    :param with_conv: use convolution to refine the feature
    :param kernel_size: feature size
    :param return_mask: return mask for the confidential score
    """
    def __init__(self, input_nc, with_conv=False, kernel_size=3, return_mask=False):
        super(UpSample, self).__init__()
        self.with_conv = with_conv
        self.return_mask = return_mask
        if self.with_conv:
            self.conv = PartialConv2d(input_nc, input_nc, kernel_size=kernel_size, stride=1,
                                      padding=int(int(kernel_size-1)/2), return_mask=True)

    def forward(self, x, mask=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        mask = F.interpolate(mask, scale_factor=2, mode='bilinear', align_corners=True) if mask is not None else mask
        if self.with_conv:
            x, mask = self.conv(x, mask)
        if self.return_mask:
            return x, mask
        else:
            return x
        

class PixelwiseNorm(nn.Module):
    def __init__(self, input_nc):
        super(PixelwiseNorm, self).__init__()
        self.init = False
        self.alpha = nn.Parameter(torch.ones(1, input_nc, 1, 1))

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        # x = x - x.mean(dim=1, keepdim=True)
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).rsqrt()  # [N1HW]
        y = x * y  # normalize the input x volume
        return self.alpha*y
    

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'pixel':
        norm_layer = functools.partial(PixelwiseNorm)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'relu':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'gelu':
        nonlinearity_layer = nn.GELU()
    elif activation_type == 'leakyrelu':
        nonlinearity_layer = nn.LeakyReLU(0.2)
    elif activation_type == 'prelu':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


class ToRGB(nn.Module):
    def __init__(self, input_nc, output_nc, upsample=True, activation='gelu', norm='pixel'):
        super().__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)

        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            input_nc = input_nc + output_nc

        self.conv = nn.Sequential(
            norm_layer(input_nc),
            activation_layer,
            PartialConv2d(input_nc, output_nc, kernel_size=3, padding=1)
        )

    def forward(self, input, skip=None):
        if skip is not None:
            skip = self.upsample(skip)
            input = torch.cat([input, skip], dim=1)

        out = self.conv(input)

        return torch.tanh(out)
    

class TFill(nn.Module):
    def __init__(self, input_nc=3, ngf=64, embed_dim=512, down_layers=3, mid_layers=6, num_res_blocks=1, dropout=0.0,
                 rample_with_conv=True, activation='gelu', norm='pixel'):
        super(TFill, self).__init__()
        self.down_layers = down_layers
        self.mid_layers = mid_layers
        self.num_res_blocks = num_res_blocks
        out_dims = []
        # start
        self.encode = PartialConv2d(
            input_nc, ngf, kernel_size=3, stride=1, padding=1)
        # down
        self.down = nn.ModuleList()
        out_dim = ngf
        for i in range(self.down_layers):
            block = nn.ModuleList()
            down = nn.Module()
            in_dim = out_dim
            out_dims.append(out_dim)
            out_dim = min(int(in_dim * 2), embed_dim)
            down.downsample = DownSample(
                in_dim, rample_with_conv, kernel_size=3)
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(
                    in_dim, out_dim, 3, dropout, activation, norm))
                in_dim = out_dim
            down.block = block
            self.down.append(down)
        # middle
        self.mid = nn.ModuleList()
        for i in range(self.mid_layers):
            self.mid.append(ResnetBlock(
                out_dim, out_dim, 3, dropout, activation, norm))
        # up
        self.up = nn.ModuleList()
        for i in range(self.down_layers):
            block = nn.ModuleList()
            up = nn.Module()
            in_dim = out_dim
            out_dim = max(out_dims[-i-1], ngf)
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(
                    in_dim, out_dim, 3, dropout, activation, norm))
                in_dim = out_dim
            if i == self.down_layers - 3:
                up.attn = AttnAware(out_dim, activation, norm)
            up.block = block
            upsample = True if i != 0 else False
            up.out = ToRGB(
                out_dim, input_nc, upsample, activation, norm)
            up.upsample = UpSample(
                out_dim, rample_with_conv, kernel_size=3)
            self.up.append(up)
        # end
        self.decode = ToRGB(
            out_dim, input_nc, True, activation, norm)

    def forward(self, x, mask=None):
        # start
        x = self.encode(x)
        pre = None
        # down
        for i in range(self.down_layers):
            x = self.down[i].downsample(x)
            if i == 2:
                pre = x
            for i_block in range(self.num_res_blocks):
                x = self.down[i].block[i_block](x)
        # middle
        for i in range(self.mid_layers):
            x = self.mid[i](x)
        # up
        skip = None
        for i in range(self.down_layers):
            for i_block in range(self.num_res_blocks):
                x = self.up[i].block[i_block](x)
            if i == self.down_layers - 3:
                mask = F.interpolate(mask, size=x.size()[
                                     2:], mode='bilinear', align_corners=True) if mask is not None else None
                x = self.up[i].attn(x, pre=pre, mask=mask)
            skip = self.up[i].out(x, skip)
            x = self.up[i].upsample(x)
        # end
        x = self.decode(x, skip)

        return x


if __name__ == '__main__':
    model = TFill()
    x = torch.randn(1, 3, 256, 256)
    mask = torch.randn(1, 1, 256, 256)
    y = model(x, mask)
    print(y.size())