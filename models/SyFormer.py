import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from einops import rearrange
from einops import rearrange
import numpy as np
from torch.autograd import Variable
from torch import Tensor
from typing import Tuple



def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1,  relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size,
                              stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(
            x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        x = self.leakyrelu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)

        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class CPAMEnc(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super(CPAMEnc, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                   norm_layer(in_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                   norm_layer(in_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                   norm_layer(in_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                   norm_layer(in_channels),
                                   nn.ReLU(True))

    def forward(self, x):
        b, c, h, w = x.size()
        feat1 = self.conv1(self.pool1(x)).view(b, c, -1)
        feat2 = self.conv2(self.pool2(x)).view(b, c, -1)
        feat3 = self.conv3(self.pool3(x)).view(b, c, -1)
        feat4 = self.conv4(self.pool4(x)).view(b, c, -1)

        return torch.cat((feat1, feat2, feat3, feat4), 2)


class CPAMDec(nn.Module):
    def __init__(self, in_channels):
        super(CPAMDec, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels//4, kernel_size=1)
        self.conv_key = nn.Linear(in_channels, in_channels//4)
        self.conv_value = nn.Linear(in_channels, in_channels)

    def forward(self, x, y):
        m_batchsize, C, width, height = x.size()
        m_batchsize, K, M = y.size()

        proj_query = self.conv_query(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.conv_key(y).view(m_batchsize, K, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.conv_value(y).permute(0, 2, 1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.scale*out + x
        return out


class CCAMDec(nn.Module):
    def __init__(self):
        super(CCAMDec, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        m_batchsize, C, width, height = x.size()
        x_reshape = x.view(m_batchsize, C, -1)

        B, K, W, H = y.size()
        y_reshape = y.view(B, K, -1)
        proj_query = x_reshape
        proj_key = y_reshape.permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B, K, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, width, height)

        out = x + self.scale*out
        return out


class CPAMDec_Mix(nn.Module):
    """
    CPAM decoding module
    """

    def __init__(self, in_channels):
        super(CPAMDec_Mix, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.scale1 = nn.Parameter(torch.zeros(1))

        # self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        # self.conv_key1 = nn.Linear(in_channels, in_channels//4) # key_conv2
        # self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels//4, kernel_size=1)
        self.conv_key2 = nn.Linear(in_channels, in_channels//4)  # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels)  # value2

    def forward(self, x1, y1, x2, y2):

        m_batchsize, C, width, height = x1.size()
        m_batchsize, K, M = y1.size()

        proj_value1 = self.conv_value2(y1).permute(0, 2, 1)

        proj_query2 = self.conv_query2(x2).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key2 = self.conv_key2(y2).view(
            m_batchsize, K, -1).permute(0, 2, 1)
        proj_value2 = self.conv_value2(y2).permute(0, 2, 1)

        energy2 = torch.bmm(proj_query2, proj_key2)

        energy = torch.abs(energy2)

        attention = self.softmax(energy)

        out1 = torch.bmm(proj_value1, attention.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, width, height)
        out1 = self.scale*out1 + x1

        out2 = torch.bmm(proj_value2, attention.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, width, height)
        out2 = self.scale1*out2 + x2

        return out1, out2


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
        # if self.multi_channel:
        #     self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
        #                                          self.kernel_size[1])
        # else:
        #     self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.weight_maskUpdater = torch.ones(
            1, 1, self.kernel_size[0], self.kernel_size[1])

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
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / \
                    (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view,
                               self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)
        if self.return_mask:
            return output, self.update_mask
        else:
            return output, self.update_mask


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        return q, kv
    

class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(
            qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(
            key)  # per-window pooling -> (n, p^2, c)
        # (n, p^2, p^2)
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(
            attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index
    

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),  # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               # (n, p^2, k, w^2, c_kv)
                               index=r_idx.view(
                                   n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * \
                topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv
    

class Syatt(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, auto_pad=False, dim=64, topK=1):
        super(Syatt, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                           norm_layer(inter_channels),
                                           nn.ReLU())  # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                           norm_layer(inter_channels),
                                           nn.ReLU())  # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer)  # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer)  # en_s

        self.cpam_dec_mix = CPAMDec_Mix(inter_channels)  # de_s

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*3, out_channels, 3, padding=1, bias=False),
                                      norm_layer(out_channels),
                                      nn.ReLU())
        self.topK = topK
        self.dim = dim
        self.auto_pad = auto_pad
        self.n_win = 8
        self.qkv = QKVLinear(self.dim, self.dim)
        self.wo = nn.Identity()
        self.kv_down = nn.Identity()
        self.scale = self.dim ** -0.5

        self.router = TopkRouting(qk_dim=self.dim,
                                  qk_scale=self.scale,
                                  topk=self.topK,
                                  )
        self.kv_gather = KVGather(mul_weight='none')
        self.attn_act = nn.Softmax(dim=-1)
        self.gamma1 = nn.Parameter(-1 * torch.ones((dim)), requires_grad=True)
        self.partial_op = PartialConv2d(
            inter_channels, inter_channels, kernel_size=5, stride=1, padding=3, bias=False, multi_channel=True)
        # PartialConv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=bias, multi_channel=True)
        self.cconv = nn.Conv2d(
            inter_channels, inter_channels, kernel_size=5, padding=1, stride=1,)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = rearrange(
            img_mask, "1 (j h) (i w) 1 -> (1 j i) h w 1", j=self.n_win, i=self.n_win)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, y, mask):

        _, _, H_mask, W_mask = mask.size()

        #

        mask1 = mask
        mask = mask.permute(0, 2, 3, 1)

        cpam_b_x = self.conv_cpam_b_x(x)

        cpam_b_x, mmask = self.partial_op(cpam_b_x, mask1)
        cpam_b_x = self.cconv(cpam_b_x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0, 2, 1)

        x_identity = cpam_b_x.permute(0, 2, 3, 1)
        x = x_identity

        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0

        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c",
                      j=self.n_win, i=self.n_win)
        mask_shape = rearrange(
            mask, "n (j h) (i w) 1 -> n (j i) h w 1", j=self.n_win, i=self.n_win)
        mask_windows = rearrange(
            mask_shape, "n (j i) h w 1 -> (n j i) (h w) 1", j=self.n_win, i=self.n_win)

        q, kv = self.qkv(x)
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)

        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(
            kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.dim].mean([2, 3])

        r_weight, r_idx = self.router(q_win, k_win)
        #print(r_weight.shape) [2, 64, 16][2, 64, 4][2, 64, 1]

        # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)

        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)

        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 c -> (n p2) c (k w2)')
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 c -> (n p2) (k w2) c')
        q_pix = rearrange(q_pix, 'n p2 w2 c -> (n p2) w2 c')
        #print(k_pix_sel.shape)  [128, 64, 1024]

        # ############## mask attention ######################
        # attn_mask = self.calculate_mask((H_mask, W_mask))  # [64, 64, 64]
        # if attn_mask is not None:
        #     nW = attn_mask.shape[0]
        #     attn = attn_weight.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)

        # param-free multihead attention
        # (n*p^2, w^2, c) @ (n*p^2, c, topk*h_kv*w_kv) -> (n*p^2, w^2, topk*w2)
        attn_weight = (q_pix * self.scale) @ k_pix_sel

        if mask_windows is not None:
            _, _, C = attn_weight.shape
            attn_mask_windows = mask_windows.expand(-1, -1, C)
            attn = attn_weight + attn_mask_windows.masked_fill(
                attn_mask_windows == 0, float(-100.0)).masked_fill(attn_mask_windows == 1, float(0.0))
            # with torch.no_grad():
            #     mask_windows = torch.clamp(torch.sum(mask_windows, dim=1, keepdim=True), 0, 1).repeat(1, N, 1)

        attn_weight = self.attn_act(attn)

        # (n*p^2, w^2, topk*h_kv*w_kv) @ (n*p^2, topk*h_kv*w_kv, c) -> (n*p^2, w^2, c)
        out = attn_weight @ v_pix_sel

        out = rearrange(out, '(n j i) (h w) c -> n (j h) (i w) c',
                        j=self.n_win, i=self.n_win, h=H // self.n_win, w=W // self.n_win)
        output1 = x_identity + self.gamma1 * out

        cpam_b_y = self.conv_cpam_b_y(y)

        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0, 2, 1)

        cpam_feat1, cpam_feat2 = self.cpam_dec_mix(
            cpam_b_x, cpam_f_x, cpam_b_y, cpam_f_y)

        feat_sum = self.conv_cat(
            torch.cat([output1.permute(0, 3, 1, 2), cpam_feat1, cpam_feat2], 1))

        return feat_sum, cpam_feat1, cpam_feat2


class FDA(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(FDA, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                         norm_layer(out_channels),
                                         nn.ReLU())  # conv5_s
        self.cpam_enc = CPAMEnc(out_channels, norm_layer)  # en_s
        self.cpam_dec = CPAMDec(out_channels)  # de_s

        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU())
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                      norm_layer(inter_channels//16),
                                      nn.ReLU())
        self.ccam_dec = CCAMDec()

    def forward(self, x):
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b, ccam_f)

        cpam_b = self.conv_cpam_b(ccam_feat)
        cpam_f = self.cpam_enc(cpam_b).permute(0, 2, 1)  # BKD
        cpam_feat = self.cpam_dec(cpam_b, cpam_f)
        return cpam_feat


def np_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a


def th_generate_grid(batch_size, input_height, input_width, dtype, cuda):
    grid = np.meshgrid(
        range(input_height), range(input_width), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)

    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda()
    return Variable(grid, requires_grad=False)


def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def th_batch_map_coordinates(input, coords, order=1):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s)
    coords : tf.Tensor. shape = (b, n_points, 2)
    Returns
    -------
    tf.Tensor. shape = (b, s, s)
    """

    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    n_coords = coords.size(1)

    # coords = torch.clamp(coords, 0, input_size - 1)

    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1),
                        torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

    assert (coords.size(1) == n_coords)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda()

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
        ], 1)
        inds = indices[:, 0] * input.size(1) * input.size(2) + indices[:, 1] * input.size(2) + indices[:, 2]
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0] * (vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0] * (vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 1] * (vals_b - vals_t) + vals_t
    return mapped_vals


def th_batch_map_offsets(input, offsets, grid=None, order=1):
    """Batch map offsets into input
    Parameters
    ---------
    input : torch.Tensor. shape = (b, s, s)
    offsets: torch.Tensor. shape = (b, s, s, 2)
    Returns
    -------
    torch.Tensor. shape = (b, s, s)
    """
    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    offsets = offsets.view(batch_size, -1, 2)
    if grid is None:
        grid = th_generate_grid(batch_size, input_height, input_width, offsets.data.type(), offsets.data.is_cuda)

    coords = offsets + grid

    mapped_vals = th_batch_map_coordinates(input, coords)
    return mapped_vals


class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters,
                                           self.filters * 2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(
            self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(
            x, offsets, grid=self._get_grid(self, x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(
            batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1,
                                int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x
    

class RCNModule(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True):
        super(RCNModule, self).__init__()
        self.num_features = num_features
        self.track_running_stats = track_running_stats
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, mask_t):
        input_m = input * mask_t

        if self.training:
            mask_mean_bn = torch.mean(mask_t, (0, 2, 3), True)
            mean_bn = torch.mean(input_m, (0, 2, 3), True) / mask_mean_bn
            var_bn = torch.mean(((input_m - mean_bn) * mask_t) ** 2, (0, 2, 3), True) / mask_mean_bn

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * mean_bn.data)
            self.running_var.mul_(self.momentum)
            self.running_var.add_((1 - self.momentum) * var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        mask_mean_in = torch.mean(mask_t, (2, 3), True)
        mean_in = torch.mean(input_m, (2, 3), True) / mask_mean_in
        var_in = torch.mean(((input_m - mean_in) * mask_t) ** 2, (2, 3), True) / mask_mean_in

        mask_mean_ln = torch.mean(mask_t, (1, 2, 3), True)
        mean_ln = torch.mean(input_m, (1, 2, 3), True) / mask_mean_ln
        var_ln = torch.mean(((input_m - mean_ln) * mask_t) ** 2, (1, 2, 3), True) / mask_mean_ln

        mean_weight = F.softmax(self.mean_weight)
        var_weight = F.softmax(self.var_weight)

        x_mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
        x_var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn

        x_out = self.weight * (input_m - x_mean) / torch.sqrt(x_var + self.eps) + self.bias
        return x_out * mask_t + input * (1 - mask_t)
    

class DDUM(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, rn=True, sample='none-3', activ='relu',
                 conv_bias=False, defor=True, type='img1'):
        super(DDUM, self).__init__()
        #将所有的可变形卷积都取消
        self.type = type
        if sample == 'down-5':
            if type == 'img1':
                self.conv = nn.Conv2d(in_ch+1, out_ch, 5, 2, 2, bias=conv_bias)
                self.updatemask = nn.MaxPool2d(5, 2, 2)
                if defor:
                    self.offset = ConvOffset2D(in_ch+1)
            elif type == 'img2':
                self.conv = nn.Conv2d(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
                if defor:
                    self.offset = ConvOffset2D(in_ch)

        elif sample == 'down-7':
            if type == 'img1':
                self.conv = nn.Conv2d(in_ch+1, out_ch, 7, 2, 3, bias=conv_bias)
                self.updatemask = nn.MaxPool2d(7, 2, 3)
                if defor:
                    self.offset = ConvOffset2D(in_ch+1)
            elif type == 'img2':
                self.conv = nn.Conv2d(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
                if defor:
                    self.offset = ConvOffset2D(in_ch)

        elif sample == 'down-3':
            if type == 'img1':
                self.conv = nn.Conv2d(in_ch+1, out_ch, 3, 2, 1, bias=conv_bias)
                self.updatemask = nn.MaxPool2d(3, 2, 1)
                if defor:
                    self.offset = ConvOffset2D(in_ch+1)
            elif type == 'img2':
                self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
                if defor:
                    self.offset = ConvOffset2D(in_ch)

        else:
            self.conv = nn.Conv2d(in_ch+2, out_ch, 3, 1, 1, bias=conv_bias)
            self.updatemask = nn.MaxPool2d(3, 1, 1)
            if defor:
                self.offset0 = ConvOffset2D(in_ch-out_ch+1)
                self.offset1 = ConvOffset2D(out_ch+1)
        self.in_ch = in_ch
        self.out_ch = out_ch

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if rn:
            # Regional Composite Normalization
            self.rn = RCNModule(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input, input_mask):
        if hasattr(self, 'offset'):
            if self.type == 'img1':
                input = torch.cat([input, input_mask[:, :1, :, :]], dim=1)
                h = self.offset(input)
                h = input * input_mask[:, :1, :, :] + \
                    (1 - input_mask[:, :1, :, :]) * h
                h = self.conv(h)
                h_mask = self.updatemask(input_mask[:, :1, :, :])
                h = h * h_mask
            else:
                h = self.offset(input)
                h = self.conv(h)

                h_mask = input_mask
        elif hasattr(self, 'offset0'):

            h1_in = torch.cat(
                [input[:, self.in_ch - self.out_ch:, :, :], input_mask[:, 1:, :, :]], dim=1)
            m1_in = input_mask[:, 1:, :, :]
            h0 = torch.cat([input[:, :self.in_ch - self.out_ch,
                           :, :], input_mask[:, :1, :, :]], dim=1)
            h1 = self.offset1(h1_in)
            h1 = m1_in * h1_in + (1 - m1_in) * h1
            h = self.conv(torch.cat([h0, h1], dim=1))
            h_mask = F.interpolate(
                input_mask[:, :1, :, :], scale_factor=2, mode='nearest')

        else:
            h = self.conv(torch.cat([input, input_mask[:, :, :, :]], dim=1))
            h_mask = self.updatemask(input_mask[:, :1, :, :])
            h = h * h_mask

        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class SyFormer(nn.Module):
    def __init__(self, num_classes=3, normal_init=True):
        super(SyFormer, self).__init__()

        self.enc_1 = DDUM(3, 16, sample='down-7', defor=True)
        self.enc_2 = DDUM(16, 32, sample='down-5', defor=True)
        self.enc_3 = DDUM(32, 64, sample='down-5', defor=True)
        self.enc_4 = DDUM(64, 128, sample='down-3', defor=True)
        self.enc_5 = DDUM(128, 256, sample='down-3', defor=True)

        self.enc_11 = DDUM(3, 16, sample='down-7', defor=True, type='img2')
        self.enc_22 = DDUM(16, 32, sample='down-5', defor=True, type='img2')
        self.enc_33 = DDUM(32, 64, sample='down-5', defor=True, type='img2')
        self.enc_44 = DDUM(64, 128, sample='down-3', defor=True, type='img2')
        self.enc_55 = DDUM(128, 256, sample='down-3', defor=True, type='img2')

        self.dec_3 = DDUM(256 + 128, 128, activ='leaky')
        self.dec_2 = DDUM(128 + 64, 64, activ='leaky')

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsamplex22 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.consrative2 = Syatt(256, 128, dim=128, topK=1)
        self.consrative3 = Syatt(128, 64, dim=64, topK=4)
        self.consrative4 = Syatt(64, 32, dim=32, topK=16)

        self.Translayer2_1 = BasicConv2d(128, 64, 1)
        self.fam32_1 = FDA(128, 64)
        self.Translayer3_1 = BasicConv2d(64, 32, 1)
        self.fam43_1 = FDA(64, 32)

        self.Translayer2_2 = BasicConv2d(128, 64, 1)
        self.fam32_2 = FDA(128, 64)
        self.Translayer3_2 = BasicConv2d(64, 32, 1)
        self.fam43_2 = FDA(64, 32)

        self.final1 = nn.Sequential(
            Conv(32, 32, 3,  relu=True),
            Conv(32, num_classes, 3, relu=False)
        )
        self.final2 = nn.Sequential(
            Conv(32, 32, 3, relu=True),
            Conv(32, num_classes, 3, relu=False)
        )
        self.final_middle_1 = nn.Sequential(
            Conv(64, 32, 3, relu=True),
            Conv(32, num_classes, 3, relu=False)
        )
        self.final_middle_2 = nn.Sequential(
            Conv(64, 32, 3, relu=True),
            Conv(32, num_classes, 3, relu=False)
        )
        self.out = nn.Sequential(
            Conv(12, 12, 3, relu=True),

        )
        self.out1 = Conv(12, 3, 3, relu=False)

        if normal_init:
            self.init_weights()

    def forward(self, imgs1, mask):

        c00, mask00 = self.enc_1(imgs1, mask)
        c00_img2, mask_00 = self.enc_11(imgs1, mask)

        c0, mask0 = self.enc_2(c00, mask00)
        c0_img2, mask_0 = self.enc_22(c00_img2, mask_00)

        c1, mask1 = self.enc_3(c0, mask0)
        c1_img2, mask_1 = self.enc_33(c0_img2, mask_0)

        c2, mask2 = self.enc_4(c1, mask1)
        c2_img2, mask_2 = self.enc_44(c1_img2, mask_1)

        c3, mask3 = self.enc_5(c2, mask2)
        c3_img2, mask_3 = self.enc_55(c2_img2, mask_2)

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2, mask1)
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2, mask2)
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2, mask3)

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(
            torch.cat([cross_result3, self.upsamplex2(out2)], 1))
        out4 = self.fam43_1(
            torch.cat([cross_result4, self.upsamplex2(self.Translayer3_1(out3))], 1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2+cur2_2))
        out3_2 = self.fam32_2(
            torch.cat([torch.abs(cur1_3+cur2_3), self.upsamplex2(out2_2)], 1))
        out4_2 = self.fam43_2(torch.cat(
            [torch.abs(cur1_4+cur2_4), self.upsamplex2(self.Translayer3_2(out3_2))], 1))

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))
        output = self.out(
            torch.cat((out_middle_1, out_middle_2, out_1, out_2), dim=1))
        output1 = self.out1(self.upsamplex22(output))

        return output1

    def init_weights(self):
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)
        self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)


def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


if __name__ == '__main__':
    model = SyFormer().cuda()
    imgs1 = torch.randn(2, 3, 512, 512).cuda()
    mask = torch.randn(2, 1, 512, 512).cuda()
    out = model(imgs1, mask)
    print(out.size())