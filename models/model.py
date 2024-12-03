import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Callable, Tuple
from timm.layers import SqueezeExcite, SeparableConvNormAct, drop_path, trunc_normal_, Mlp, DropPath, to_2tuple


def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


class MBConv(nn.Module):
    """ MBConv block as described in: https://arxiv.org/pdf/2204.01697.pdf.

        Without downsampling:
        x ← x + Proj(SE(DWConv(Conv(Norm(x)))))

        With downsampling:
        x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).

        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.

        Note: This implementation differs slightly from the original MobileNet implementation!

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true downscale by a factor of two is performed. Default: False
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            drop_path: float = 0.,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MBConv, self).__init__()
        # Save parameter
        self.drop_path_rate: float = drop_path
        # Check parameters for downscaling
        if not downscale:
            assert in_channels == out_channels, "If downscaling is utilized input and output channels must be equal."
        # Ignore inplace parameter if GELU is used
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels, kernel_size=(1, 1)),
            SeparableConvNormAct(in_channels=in_channels, out_channels=out_channels, stride=2 if downscale else 1,
                                 act_layer=act_layer, norm_layer='layernorm2d'),
            SqueezeExcite(channels=out_channels, rd_ratio=0.25),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=(1, 1))
        )
        # Make skip path
        self.skip_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=(1, 1))
        ) if downscale else nn.Identity()

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output


def window_partition(
        input: torch.Tensor,
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Window partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)

    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    windows = input.view(
        B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous(
    ).view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(
        windows: torch.Tensor,
        original_size: Tuple[int, int],
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.

    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output


def grid_partition(
        input: torch.Tensor,
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Grid partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        grid_size (Tuple[int, int], optional): Grid size to be applied. Default (7, 7)

    Returns:
        grid (torch.Tensor): Unfolded input tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    grid = input.view(
        B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], window_size[1], C]
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous(
    ).view(-1, grid_size[0], grid_size[1], C)
    return grid


def grid_reverse(
        grid: torch.Tensor,
        original_size: Tuple[int, int],
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the grid partition.

    Args:
        Grid (torch.Tensor): Grid tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        grid_size (Tuple[int, int], optional): Grid size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height, width, and channels
    (H, W), C = original_size, grid.shape[-1]
    # Compute original batch size
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    # Fold grid tensor
    output = grid.view(
        B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.

    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.

    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid(
        [torch.arange(win_h), torch.arange(win_w)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class RelativeSelfAttention(nn.Module):
    """ Relative Self-Attention similar to Swin V1. Implementation inspired by Timms Swin V1 implementation.

    Args:
        in_channels (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
            self,
            in_channels: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(RelativeSelfAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.scale: float = num_heads ** -0.5
        self.attn_area: int = grid_window_size[0] * grid_window_size[1]
        # Init layers
        self.qkv_mapping = nn.Linear(
            in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=in_channels,
                              out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(grid_window_size[0],
                                                                                    grid_window_size[1]))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.

        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B_, N, C].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B_, N, C].
        """
        # Get shape of input
        B_, N, C = input.shape
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(
            B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Scale query
        q = q * self.scale
        # Compute attention maps
        attn = self.softmax(q @ k.transpose(-2, -1) +
                            self._get_relative_positional_bias())
        # Map value with attention maps
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class MaxViTTransformerBlock(nn.Module):
    """ MaxViT Transformer block.

        With block partition:
        x ← x + Unblock(RelAttention(Block(LN(x))))
        x ← x + MLP(LN(x))

        With grid partition:
        x ← x + Ungrid(RelAttention(Grid(LN(x))))
        x ← x + MLP(LN(x))

        Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
        Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.

    Args:
        in_channels (int): Number of input channels.
        partition_function (Callable): Partition function to be utilized (grid or window partition).
        reverse_function (Callable): Reverse function to be utilized  (grid or window reverse).
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
    """

    def __init__(
            self,
            in_channels: int,
            partition_function: Callable,
            reverse_function: Callable,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """ Constructor method """
        super(MaxViTTransformerBlock, self).__init__()
        # Save parameters
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        self.grid_window_size: Tuple[int, int] = grid_window_size
        # Init layers
        self.norm_1 = norm_layer(in_channels)
        self.attention = RelativeSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=int(mlp_ratio * in_channels),
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)].
        """
        # Save original shape
        B, C, H, W = input.shape
        # Perform partition
        input_partitioned = self.partition_function(
            input, self.grid_window_size)
        input_partitioned = input_partitioned.view(
            -1, self.grid_window_size[0] * self.grid_window_size[1], C)
        # Perform normalization, attention, and dropout
        output = input_partitioned + \
            self.drop_path(self.attention(self.norm_1(input_partitioned)))
        # Perform normalization, MLP, and dropout
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        # Reverse partition
        output = self.reverse_function(output, (H, W), self.grid_window_size)
        return output


class MaxViTBlock(nn.Module):
    """ MaxViT block composed of MBConv block, Block Attention, and Grid Attention.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true spatial downscaling is performed. Default: False
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            num_heads: int = 3,
            grid_window_size: Tuple[int, int] = (8, 8),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViTBlock, self).__init__()
        # Init MBConv block
        self.mb_conv = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            downscale=downscale,
            act_layer=act_layer,
            drop_path=drop_path
        )
        # Init Block and Grid Transformer
        self.block_transformer = MaxViTTransformerBlock(
            in_channels=out_channels,
            partition_function=window_partition,
            reverse_function=window_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer
        )
        self.grid_transformer = MaxViTTransformerBlock(
            in_channels=out_channels,
            partition_function=grid_partition,
            reverse_function=grid_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2] (downscaling is optional)
        """
        output = self.grid_transformer(
            self.block_transformer(self.mb_conv(input)))
        return output 


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.ln = LayerNorm2d(out_channels * 2) 
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        batch, c, h, w = x.size()
        # (batch, c, h, w/2+1) 复数
        ffted = torch.fft.rfftn(x, s=(h, w), dim=(2, 3), norm='ortho')
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.gelu(self.ln(ffted))

        ffted = torch.tensor_split(ffted, 2, dim=1)
        ffted = torch.complex(ffted[0], ffted[1])
        output = torch.fft.irfftn(ffted, s=(h, w), dim=(2, 3), norm='ortho')
        return output


class CALayer(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.fu = FourierUnit(in_planes, in_planes)

    def forward(self, x):
        avg_out = self.fc2(self.gelu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.gelu(self.fc1(self.max_pool(x))))

        fu_out = self.fu(x)

        out = avg_out + max_out + fu_out
        return torch.sigmoid(out)


class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.fu = FourierUnit(2, 1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        fu_out = self.fu(x)

        x = self.conv1(x + fu_out)
        return self.sigmoid(x)


class FCBAM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=7):
        super(FCBAM, self).__init__()
        self.ca = CALayer(in_planes, ratio)
        self.sa = SALayer(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out
    

class Aggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Aggregation, self).__init__()
        self.body = [FCBAM(in_planes=in_channels)]
        self.body = nn.Sequential(*self.body)
        self.tail = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.body(x)
        out = self.tail(out)
        return out


class HBlock(nn.Module):
    def __init__(self, channels=3):
        super(HBlock, self).__init__()
        self.block1 = MaxViTBlock(in_channels=channels, out_channels=channels)
        self.block2 = MaxViTBlock(in_channels=channels, out_channels=channels)

        self.agg_rgb = Aggregation(in_channels = channels * 3, out_channels=channels)
        self.agg_mas = Aggregation(in_channels = channels * 3, out_channels=channels)
    

    def forward(self, x):
        out_1 = self.block1(x)
        out_2 = self.block2(out_1)
        
        agg_rgb = self.agg_rgb(torch.cat([out_1, out_2, x], dim=1))
        agg_mas = self.agg_mas(torch.cat([out_1, out_2, x], dim=1))
        
        output = agg_rgb.mul(torch.sigmoid(agg_mas))
        
        return output + x


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# Multi-axis Partial Queried Learning Block (MPQLB)
class MPQLB(nn.Module):
    def __init__(self, dim, x=8, y=8, bias=False):
        super(MPQLB, self).__init__()

        partial_dim = int(dim // 4)

        self.hw = nn.Parameter(torch.ones(1, partial_dim, x, y), requires_grad=True)
        self.conv_hw = nn.Conv2d(partial_dim, partial_dim, kernel_size=to_2tuple(3), padding=1, groups=partial_dim, bias=bias)

        self.ch = nn.Parameter(torch.ones(1, 1, partial_dim, x), requires_grad=True)
        self.conv_ch = nn.Conv1d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.cw = nn.Parameter(torch.ones(1, 1, partial_dim, y), requires_grad=True)
        self.conv_cw = nn.Conv1d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.conv_4 = nn.Conv2d(partial_dim, partial_dim, kernel_size=to_2tuple(1), bias=bias)

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias),
        )

    def forward(self, x):
        input_ = x
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # hw
        x1 = x1 * self.conv_hw(F.interpolate(self.hw, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ch
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2 * self.conv_ch(
            F.interpolate(self.ch, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # cw
        x3 = x3.permute(0, 2, 1, 3)
        x3 = x3 * self.conv_cw(
            F.interpolate(self.cw, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)

        x4 = self.conv_4(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        x = self.mlp(x) + input_

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super(BasicLayer, self).__init__()
        self.blocks = nn.ModuleList([MPQLB(dim, dim) for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=to_2tuple(3), padding=1, bias=bias),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=to_2tuple(3), padding=1, bias=bias),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# Supervised Cross-scale Transposed Attention Module (SCTAM)
class SCTAM(nn.Module):
    def __init__(self, dim, up_scale=2, bias=False):
        super(SCTAM, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

        self.up = nn.PixelShuffle(up_scale)

        self.qk_pre = nn.Conv2d(int(dim // (up_scale ** 2)), 3, kernel_size=to_2tuple(1), bias=bias)
        self.qk_post = nn.Sequential(LayerNorm2d(3),
                                     nn.Conv2d(3, int(dim * 2), kernel_size=to_2tuple(1), bias=bias))

        self.v = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias)
        )

        self.conv = nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias)

        self.norm =LayerNorm2d(dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qk = self.qk_pre(self.up(x))
        fake_image = qk
        qk = self.qk_post(qk).reshape(b, 2, c, -1).transpose(0, 1)
        q, k = qk[0], qk[1]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        v = self.v(x)
        v_ = v.reshape(b, c, h*w)

        attn = (q @ k.transpose(-1, -2)) * self.alpha
        attn = attn.softmax(dim=-1)
        x = (attn @ v_).reshape(b, c, h, w) + self.conv(v)

        x = self.norm(x)
        x = self.proj(x)

        return x, fake_image


class PIFM(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(PIFM, self).__init__()

        hidden_features = int(channel // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.a = nn.Sequential(
            nn.Conv2d(channel, hidden_features, kernel_size=to_2tuple(1), bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features, channel * 2, kernel_size=to_2tuple(1), bias=bias),
            nn.Softmax(dim=1)
        )
        self.t = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=to_2tuple(3), padding=1, groups=channel, bias=bias),
            nn.Conv2d(channel, hidden_features, kernel_size=to_2tuple(1), bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features, channel * 2, kernel_size=to_2tuple(1), bias=bias),
            nn.Sigmoid()
        )

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats_ = in_feats.view(B, 2, C, H, W)
        x = torch.sum(in_feats_, dim=1)

        a = self.a(self.avg_pool(x))
        t = self.t(x)
        j = torch.mul((1 - t), a) + torch.mul(t, in_feats)

        j = j.view(B, 2, C, H, W)
        j = torch.sum(j, dim=1)
        return j


class LBlock(nn.Module):
    def __init__(self, in_channel=3, out_channel=4, dim=24, depths=(4, 4, 4, 2, 2)):
        super(LBlock, self).__init__()

        self.patch_embed = nn.Conv2d(in_channel, dim, kernel_size=to_2tuple(3), padding=1)
        self.skip1 = BasicLayer(dim, depths[0])

        self.down1 = Downsample(dim)
        self.skip2 = BasicLayer(int(dim * 2 ** 1), depths[1])

        self.down2 = Downsample(int(dim * 2 ** 1))
        self.latent = BasicLayer(int(dim * 2 ** 2), depths[2])

        self.up1 = Upsample(int(dim * 2 ** 2))
        self.sctam1 = SCTAM(int(dim * 2 ** 1), up_scale=2)
        self.pifm1 = PIFM(int(dim * 2 ** 1))
        self.layer4 = BasicLayer(int(dim * 2 ** 1), depths[3])

        self.up2 = Upsample(int(dim * 2 ** 1))
        self.sctam2 = SCTAM(dim, up_scale=1)
        self.pifm2 = PIFM(dim)
        self.layer5 = BasicLayer(dim, depths[4])

        self.patch_unembed = nn.Conv2d(dim, out_channel, kernel_size=to_2tuple(3), padding=1, bias=False)

        self.agg = Aggregation(in_channel * 4, in_channel)


    def forward_features(self, x):

        x = self.patch_embed(x)
        skip1 = x

        x = self.down1(x)
        skip2 = x

        x = self.down2(x)
        x = self.latent(x)
        x = self.up1(x)
        x, fake_image_x4 = self.sctam1(x)

        x = self.pifm1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.up2(x)
        x, fake_image_x2 = self.sctam2(x)

        x = self.pifm2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)

        return x, fake_image_x4, fake_image_x2


    def forward(self, x):
        input_ = x
        _, _, h, w = input_.shape

        x, fake_image_x4, fake_image_x2 = self.forward_features(x)
        K, B = torch.split(x, [1, 3], dim=1)

        x = K * input_ - B + input_
        x = x[:, :, :h, :w]

        x = torch.cat([input_, x, fake_image_x4, fake_image_x2], dim=1)

        x = self.agg(x)

        return x
        

class Model(nn.Module):
    def __init__(self, depth=2):
        super(Model, self).__init__()
        self.depth = depth
        self.LowFrequency = LBlock()
        self.HighFrequency = nn.ModuleList([HBlock() for _ in range(self.depth)])

    def laplacian_pyramid_decomposition(self, img):
        current = img
        pyramid = []
        for i in range(self.depth):
            blurred = F.interpolate(current, scale_factor=0.5, mode='bicubic', align_corners=True)
            expanded = F.interpolate(blurred, current.shape[2:], mode='bicubic', align_corners=True)
            residual = current - expanded
            pyramid.append(residual)
            current = blurred
        pyramid.append(current)
        return pyramid

    def laplacian_pyramid_reconstruction(self, pyramid):
        current = pyramid[-1]
        for i in reversed(range(self.depth)):
            expanded = F.interpolate(current, pyramid[i].shape[2:], mode='bicubic', align_corners=True)
            current = expanded + pyramid[i]
            current = self.HighFrequency[i](current)
        return current

    def forward(self, inp):
        inps = self.laplacian_pyramid_decomposition(inp)
        inps[-1] = self.LowFrequency(inps[-1])
        res = self.laplacian_pyramid_reconstruction(inps)
        return res


# Sample usage with a random image
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = torch.randn(1, 3, 256, 256).to(device)

    mask = torch.randn(1, 1, 256, 256).to(device)

    levels = 3

    lp_model = Model(levels).to(device)

    reconstructed_img = lp_model(img)

    print(reconstructed_img.shape)
