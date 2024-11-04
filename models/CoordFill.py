import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from math import pi
from torch.nn import init


class BilinearDownsample(nn.Module):
    def __init__(self, stride, channels, use_gpu):
        super().__init__()
        self.stride = stride
        self.channels = channels

        # create tent kernel
        kernel = np.arange(1,2*stride+1,2) # ramp up
        kernel = np.concatenate((kernel,kernel[::-1])) # reflect it and concatenate
        if use_gpu:
            kernel = torch.Tensor(kernel/np.sum(kernel)).to(device='cuda') # normalize
        else:
            kernel = torch.Tensor(kernel / np.sum(kernel))
        self.register_buffer('kernel_horz', kernel[None,None,None,:].repeat((self.channels,1,1,1)))
        self.register_buffer('kernel_vert', kernel[None,None,:,None].repeat((self.channels,1,1,1)))

        self.refl = nn.ReflectionPad2d(int(stride/2))#nn.ReflectionPad2d(int(stride/2))

    def forward(self, input):
        return F.conv2d(F.conv2d(self.refl(input), self.kernel_horz, stride=(1,self.stride), groups=self.channels),
                    self.kernel_vert, stride=(self.stride,1), groups=self.channels)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params = num_params + param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            '''
            for name, param in m.named_parameters():
                if (name == "lowres_stream.params_pred.weight"):
                    print("%s_init" % name)
                    init.zeros_(param.data[0:13])
                    init.normal_(param.data[13:13 + 64 * 64], 0.0, 0.02)
                    for i in range(1,6):
                        init.zeros_(param.data[13+i*64*64+(i-1)*64:13+64*64+i*64])
                        init.normal_(param.data[13+i*64*64+i*64:13+i*64+(i+1)*64*64], 0.0, 0.02)
                    init.zeros_(param.data[13 + i * 64 * 64 + (i - 1) * 64:13 + 64 * 64 + i * 64 + 3])
                    init.normal_(param.data[13 + i * 64 * 64 + i * 64 + 3 :13 + i * 64 + i * 64 * 64 +64*3], 0.0, 0.02)
                if (name == "lowres_stream.params_pred.bias"):
                    print("%s_init" % name)
                    init.zeros_(param.data)
            '''


        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class CoordFillGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(lr_instance=True)
        parser.set_defaults(no_instance_dist=True)
        parser.set_defaults(hr_coor="cosine")
        return parser

    def __init__(self, opt, hr_stream=None, lr_stream=None, fast=False):
        super(CoordFillGenerator, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if (opt.no_instance_edge & opt.no_instance_dist) else 1)
        self.lr_instance = opt.lr_instance
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.downsampling = opt.crop_size // opt.ds_scale

        self.highres_stream = PixelQueryNet(self.downsampling, num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth,
                                               no_one_hot=opt.no_one_hot, lr_instance=opt.lr_instance,
                                               **hr_stream)

        num_params = self.highres_stream.num_params
        self.lowres_stream = ParaGenNet(num_params, scale_injection=opt.scale_injection)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def get_lowres(self, im):
        """Creates a lowres version of the input."""
        device = self.use_gpu()
        if(self.learned_ds_factor != self.downsampling):
            myds = BilinearDownsample(int(self.downsampling//self.learned_ds_factor), self.num_inputs,device)
            return myds(im)
        else:
            return im

    def forward(self, highres):
        lowres = self.get_lowres(highres)
        lr_features = self.lowres_stream(lowres)
        output = self.highres_stream(highres, lr_features)
        return output, lr_features#, lowres


def _get_coords(bs, h, w, device, ds):
    """Creates the position encoding for the pixel-wise MLPs"""
    x = torch.arange(0, w).float()
    y = torch.arange(0, h).float()
    scale = 7 / 8
    x_cos = torch.remainder(x, ds).float() / ds
    x_sin = torch.remainder(x, ds).float() / ds
    y_cos = torch.remainder(y, ds).float() / ds
    y_sin = torch.remainder(y, ds).float() / ds
    x_cos = x_cos / (max(x_cos) / scale)
    x_sin = x_sin / (max(x_sin) / scale)
    y_cos = x_cos / (max(y_cos) / scale)
    y_sin = x_cos / (max(y_sin) / scale)
    xcos = torch.cos((2 * pi * x_cos).float())
    xsin = torch.sin((2 * pi * x_sin).float())
    ycos = torch.cos((2 * pi * y_cos).float())
    ysin = torch.sin((2 * pi * y_sin).float())
    xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
    xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
    ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
    ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
    coords = torch.cat([xcos, xsin, ycos, ysin], 1).to(device)

    return coords.to(device)



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class ParaGenNet(torch.nn.Module):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, num_out, scale_injection=False):
        super(ParaGenNet, self).__init__()

        self.num_out = num_out
        self.scale_injection = scale_injection

        ngf = 64
        if self.scale_injection:
            self.out_para = nn.Sequential(
                torch.nn.Linear(ngf * 8 + 1, self.num_out)
            )
        else:
            self.out_para = nn.Sequential(
                torch.nn.Linear(ngf * 8, self.num_out)
            )

    def forward(self, model, x, x_hr):
        structure = model(x)
        if self.scale_injection:
            scale = (torch.ones(x_hr.size(0), 1, 1, 1) * (structure.size(3) / x_hr.size(3))) \
                .to(structure.device)
            scale = scale.repeat(1, structure.size(2), structure.size(3), 1)
            structure = torch.cat([structure.permute(0, 2, 3, 1), scale], dim=-1)
            para = self.out_para(structure).permute(0, 3, 1, 2)
        else:
            para = self.out_para(structure.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return para

    def mask_predict(self, model, x, x_hr, mask):
        structure = model(x)

        if self.scale_injection:
            scale = (torch.ones(x_hr.size(0), 1, 1, 1) * (structure.size(3) / x_hr.size(3))) \
                .to(structure.device)
            scale = scale.repeat(1, structure.size(2), structure.size(3), 1)
            structure = torch.cat([structure.permute(0, 2, 3, 1), scale], dim=-1)
        else:
            structure = structure.permute(0, 2, 3, 1)

        bs, h, w, c = structure.size()
        k = mask.size(2) // h
        mask = mask.unfold(2, k, k).unfold(3, k, k)
        mask = mask.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h, w, int(k * k))
        lr_mask = torch.mean(mask, dim=-1).view(h * w)
        structure = structure.view(bs, h * w, c)
        index = torch.nonzero(1 - lr_mask).squeeze(1)
        structure = structure[:, index, :]
        para = self.out_para(structure).permute(0, 2, 1)
        return para, mask


class PixelQueryNet(torch.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling,
                 num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine",
                 no_one_hot=False, lr_instance=False):
        super(PixelQueryNet, self).__init__()

        self.lr_instance = lr_instance
        self.downsampling = downsampling
        self.num_inputs = num_inputs - (1 if self.lr_instance else 0)
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.no_one_hot = no_one_hot
        self.channels = []
        self._set_channels()

        self.num_params = 0
        self.splits = {}
        self._set_num_params()

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling

    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        in_ch = in_ch + int(4)
        self.channels = [in_ch]
        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)

    def _set_num_params(self):
        nparams = 0
        self.splits = {
            "biases": [],
            "weights": [],
        }

        # # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1]
            nparams = nparams + nco  # FC biases
            self.splits["biases"].append((idx, idx + nco))
            idx = idx + nco

            nparams = nparams + nci * nco  # FC weights
            self.splits["weights"].append((idx, idx + nco * nci))
            idx = idx + nco * nci

        self.num_params = nparams

    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]

    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"

        if self.lr_instance:
            highres = highres[:, :-1, :, :]

        # Fetch sizes
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape
        k = h // h_lr

        self.xy_coords = _get_coords(1, h, w, highres.device, h // h_lr)

        highres = torch.repeat_interleave(self.xy_coords, repeats=bs, dim=0)

        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]

        tiles = highres.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)
        out = tiles
        num_layers = len(self.channels) - 1

        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)

            out = torch.matmul(out, w_) + b_

            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            # out = torch.nn.functional.leaky_relu(out, 0.01)
            if idx < num_layers - 1:
                out = torch.nn.functional.leaky_relu(out, 0.01)
            else:
                out = torch.tanh(out)
        #
        # reorder the tiles in their correct position, and put channels first
        out = out.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out

    def mask_predict(self, highres, lr_params, hr_mask, lr_mask):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"

        if self.lr_instance:
            highres = highres[:, :-1, :, :]

        bs, _, h, w = highres.shape
        bs, h_lr, w_lr, _ = lr_mask.shape
        k = h // h_lr

        self.xy_coords = _get_coords(1, h, w, highres.device, h // h_lr)
        pe = torch.repeat_interleave(self.xy_coords, repeats=bs, dim=0)
        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = pe.shape[1]
        # bs, 5 rgbxy, h//k=h_lr, w//k=w_lr, k, k
        tiles = pe.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)

        mask = torch.mean(lr_mask, dim=-1).view(h_lr * w_lr)
        index = torch.nonzero(1 - mask).squeeze(1)
        out = tiles
        num_layers = len(self.channels) - 1

        out = out.view(bs, h_lr * w_lr, int(k * k), nci)[:, index, :, :]
        num = out.size(1)

        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 1).view(bs, num, nci, nco)
            b_ = b_.permute(0, 2, 1).view(bs, num, 1, nco)

            out = torch.matmul(out, w_) + b_

            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = torch.nn.functional.leaky_relu(out, 0.01)
            else:
                out = torch.tanh(out)

        highres = highres.unfold(2, k, k).unfold(3, k, k)
        highres = highres.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), 3).view(bs, h_lr * w_lr, int(k * k), 3)

        highres[:, index, :, :] = out
        out = highres.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out


class AttFFC(nn.Module):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, ngf):
        super(AttFFC, self).__init__()
        self.add = FFC_BN_ACT(ngf, ngf, kernel_size=3, stride=1, padding=1,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                           **{"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False})
        self.minus = FFC_BN_ACT(ngf+1, ngf, kernel_size=3, stride=1, padding=1,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                           **{"ratio_gin": 0, "ratio_gout": 0.75, "enable_lfu": False})
        self.mask = FFC_BN_ACT(ngf, 1, kernel_size=3, stride=1, padding=1,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.Sigmoid,
                           **{"ratio_gin": 0.75, "ratio_gout": 0, "enable_lfu": False})

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)

        mask, _ = self.mask((x_l, x_g))

        minus_l, minus_g = self.minus(torch.cat([x_l, x_g, mask], 1))

        add_l, add_g = self.add((x_l - minus_l, x_g - minus_g))

        x_l, x_g = x_l - minus_l + add_l, x_g - minus_g + add_g

        return x_l, x_g


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU()

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        # ffted = self.relu(self.bn(ffted))
        ffted = self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            # nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output
    

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg

class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact()
        self.act_g = gact()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        # x_l = self.act_l(self.bn_l(x_l))
        # x_g = self.act_g(self.bn_g(x_g))
        x_l = self.act_l(x_l)
        x_g = self.act_g(x_g)
        return x_l, x_g
    

class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)
    

class AttFFCResNetGenerator(nn.Module):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, ngf):
        super(AttFFCResNetGenerator, self).__init__()

        self.dowm = nn.Sequential(
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(4, 64, kernel_size=7, padding=0, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                           **{"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}),
            FFC_BN_ACT(64, 128, kernel_size=4, stride=2, padding=1,
                       norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                       **{"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}),
            FFC_BN_ACT(128, 256, kernel_size=4, stride=2, padding=1,
                       norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                       **{"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}),
            FFC_BN_ACT(256, 512, kernel_size=4, stride=2, padding=1,
                       norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                       **{"ratio_gin": 0, "ratio_gout": 0.75, "enable_lfu": False}),
        )
        self.block1 = AttFFC(ngf)
        self.block2 = AttFFC(ngf)
        self.block3 = AttFFC(ngf)
        self.block4 = AttFFC(ngf)
        self.block5 = AttFFC(ngf)
        self.block6 = AttFFC(ngf)
        self.c = ConcatTupleLayer()

    def forward(self, x):
        x = self.dowm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.c(x)

        return x
    

class CoordFill(nn.Module):
    def __init__(self, mask_prediction=False, attffc=False,
                 scale_injection=False):
        super(CoordFill, self).__init__()
        self.n_channels = 3
        self.n_classes = 3
        self.in_size = 256
        self.mask_prediction = mask_prediction
        self.attffc = attffc
        self.scale_injection = scale_injection

        self.opt = self.get_opt()
        self.asap = CoordFillGenerator(self.opt)

        self.refine = AttFFCResNetGenerator(512)

    def get_opt(self):
        from yacs.config import CfgNode as CN
        opt = CN()
        opt.label_nc = 0
        # opt.label_nc = 1
        opt.lr_instance = False
        opt.crop_size = 512
        opt.ds_scale = 32
        opt.aspect_ratio = 1.0
        opt.contain_dontcare_label = False
        opt.no_instance_edge = True
        opt.no_instance_dist = True
        opt.gpu_ids = 0
        opt.output_nc = 3
        opt.hr_width = 64
        opt.hr_depth = 5
        opt.scale_injection = self.scale_injection

        opt.no_one_hot = False
        opt.lr_instance = False
        opt.norm_G = 'batch'

        opt.lr_width = 256
        opt.lr_max_width = 256
        opt.lr_depth = 5
        opt.learned_ds_factor = 1
        opt.reflection_pad = False

        return opt

    def forward(self, img, mask):
        hr_hole = img * mask

        lr_img = F.interpolate(img, size=(self.in_size, self.in_size), mode='bilinear')
        lr_mask = F.interpolate(mask, size=(self.in_size, self.in_size), mode='nearest')
        lr_hole = lr_img * lr_mask

        lr_features = self.asap.lowres_stream(self.refine, torch.cat([lr_hole, lr_mask], dim=1), hr_hole)

        output = self.asap.highres_stream(hr_hole, lr_features)

        if self.mask_prediction:
            output = output * (1 - mask) + hr_hole

        return output

    def mask_predict(self, inp):
        img, mask = inp
        hr_hole = img * mask

        lr_img = F.interpolate(img, size=(self.in_size, self.in_size), mode='bilinear')
        lr_mask = F.interpolate(mask, size=(self.in_size, self.in_size), mode='nearest')
        lr_hole = lr_img * lr_mask

        lr_features, temp_mask = self.asap.lowres_stream.mask_predict(self.refine, torch.cat([lr_hole, lr_mask], dim=1), hr_hole, mask)

        output = self.asap.highres_stream.mask_predict(hr_hole, lr_features, mask, temp_mask)
        output = output * (1 - mask) + hr_hole

        return output
    

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    mask = torch.randn(1, 1, 256, 256)
    model = CoordFill()
    res = model(x, mask)
    print(res.shape)