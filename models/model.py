import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch import einsum
from typing import Type, Callable, Tuple
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath


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
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
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
            norm_layer(in_channels),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels, kernel_size=(1, 1)),
            DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=2 if downscale else 1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            SqueezeExcite(in_chs=out_channels, rd_ratio=0.25),
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
            num_heads: int = 10,
            grid_window_size: Tuple[int, int] = (8, 8),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
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
            norm_layer=norm_layer,
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
    

class ChannelFocus(nn.Module):
    def __init__(self, channels):
        super(ChannelFocus, self).__init__()
        
        # AvgPool和MaxPool的输出都是 Cx1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP with hidden size = channels
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: BxCxHxW
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # 将avg_out和max_out相加后经过sigmoid
        out = self.sigmoid(avg_out + max_out)
        
        return out


class SpatialFocus(nn.Module):
    def __init__(self, channels):
        super(SpatialFocus, self).__init__()
        
        # 多尺度卷积分支
        self.conv3 = nn.Conv2d(2, channels, kernel_size=3, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(2, channels, kernel_size=5, padding=2, dilation=1)
        self.conv7 = nn.Conv2d(2, channels, kernel_size=7, padding=3, dilation=1)
        
        # 特征融合
        self.conv_merge = nn.Sequential(
            nn.Conv2d(channels * 6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x1, x2):
        # 通道统计信息
        avg_out_x1 = torch.mean(x1, dim=1, keepdim=True)
        max_out_x1, _ = torch.max(x1, dim=1, keepdim=True)
        spatial_info_x1 = torch.cat([avg_out_x1, max_out_x1], dim=1)
        
        avg_out_x2 = torch.mean(x2, dim=1, keepdim=True)
        max_out_x2, _ = torch.max(x2, dim=1, keepdim=True)
        spatial_info_x2 = torch.cat([avg_out_x2, max_out_x2], dim=1)

        feat3_x1 = self.relu(self.bn(self.conv3(spatial_info_x1)))
        feat5_x1 = self.relu(self.bn(self.conv5(spatial_info_x1)))
        feat7_x1 = self.relu(self.bn(self.conv7(spatial_info_x1)))
        
        feat3_x2 = self.relu(self.bn(self.conv3(spatial_info_x2)))
        feat5_x2 = self.relu(self.bn(self.conv5(spatial_info_x2)))
        feat7_x2 = self.relu(self.bn(self.conv7(spatial_info_x2)))

        feat_all = torch.cat([
            feat3_x1, feat5_x1, feat7_x1,
            feat3_x2, feat5_x2, feat7_x2, 
        ], dim=1)
        
        spatial_attention = self.conv_merge(feat_all)
        spatial_attention = self.sigmoid(spatial_attention)
        return spatial_attention
    

class Inception(nn.Module):
    def __init__(self, in_channels, filters):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1,
                      padding=(1 - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(3, 3), stride=(1, 1), dilation=1,
                      padding=(3 - 1) // 2),
            nn.LeakyReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1,
                      padding=(1 - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(5, 5), stride=(1, 1), dilation=1,
                      padding=(5 - 1) // 2),
            nn.LeakyReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1),
            nn.LeakyReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1),
            nn.LeakyReLU(),
        )

    def forward(self, input):
        o1 = self.branch1(input)
        o2 = self.branch2(input)
        o3 = self.branch3(input)
        o4 = self.branch4(input)
        return torch.cat([o1, o2, o3, o4], dim=1)


class EdgeDFN(nn.Module):
    def __init__(self, channels=3):
        super(EdgeDFN, self).__init__()
        # Inception edge enhancement module
        self.edge_enhance = nn.Sequential(
            Inception(in_channels=channels, filters=channels),  # outputs channels*4
            nn.ReLU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1),    # reduce to channels
            nn.ReLU()
        )
        
        # Edge feature fusion
        self.conv_redu = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.edge_fuse = nn.Conv2d(channels * 3 + 1, channels, kernel_size=1, bias=False)
        
        self.cn_att = ChannelFocus(channels=channels*2)
        self.sp_att = SpatialFocus(channels=channels**2)

        self.max = MaxViTBlock(in_channels=channels * 3 + 1, out_channels=channels * 3 + 1)
    

    def rgb_to_grayscale(self, x):
        # x: [B, C, H, W]
        if x.shape[1] == 3:
            # Using standard weights for RGB to grayscale conversion
            r = x[:, 0:1, :, :]
            g = x[:, 1:2, :, :]
            b = x[:, 2:3, :, :]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            # For non-RGB images, take the mean across channels
            gray = x.mean(dim=1, keepdim=True)
        return gray


    def get_gaussian_kernel(self, kernel_size=5, sigma=1.0):
        # Create a Gaussian kernel for blurring
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid([ax, ax], indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel.to(next(self.parameters()).device)
    

    def apply_gaussian_blur(self, img, kernel_size=5, sigma=1.0):
        # Apply Gaussian blur to the image
        kernel = self.get_gaussian_kernel(kernel_size, sigma)
        img_blur = F.conv2d(img, kernel, padding=kernel_size//2)
        return img_blur


    def get_sobel_kernels(self):
        # Define Sobel kernels for edge detection
        sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]])
        sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                       [0., 0., 0.],
                                       [1., 2., 1.]])
        sobel_kernel_x = sobel_kernel_x.view(1, 1, 3, 3)
        sobel_kernel_y = sobel_kernel_y.view(1, 1, 3, 3)
        return sobel_kernel_x.to(next(self.parameters()).device), sobel_kernel_y.to(next(self.parameters()).device)


    def compute_gradients(self, img):
        # Compute gradients using Sobel kernels
        sobel_kernel_x, sobel_kernel_y = self.get_sobel_kernels()
        grad_x = F.conv2d(img, sobel_kernel_x, padding=1)
        grad_y = F.conv2d(img, sobel_kernel_y, padding=1)
        return grad_x, grad_y


    def canny_edge(self, x):
        # x: [B, C, H, W]
        # Convert to grayscale
        gray = self.rgb_to_grayscale(x)
        
        # Apply Gaussian blur
        blurred = self.apply_gaussian_blur(gray, kernel_size=5, sigma=1.0)
        
        # Compute gradients
        grad_x, grad_y = self.compute_gradients(blurred)
        
        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Normalize gradient magnitude to [0,1]
        grad_magnitude = grad_magnitude / (grad_magnitude.max() + 1e-6)
        
        # Apply threshold to get edges
        edges = (grad_magnitude > 0.1).float()
        
        # Repeat edges to match the number of channels
        edges = edges.repeat(1, x.shape[1], 1, 1)
        
        # Edge enhancement using the Inception module
        edge = self.edge_enhance(edges)
        
        return edge

    def forward(self, x, mask):
        # Extract enhanced edges
        edge_x = self.canny_edge(x)
        
        # Adaptive feature fusion
        x_edge = x + edge_x
        
        output = torch.cat([x, x_edge], dim=1)

        # Channel attention
        cn_weight = self.cn_att(output)
        f_cn = cn_weight * output
        cn_output = self.conv_redu(f_cn)

        # Spatial attention
        sp_weight = self.sp_att(x_edge, x)
        output = cn_output * sp_weight

        output = torch.cat([output, mask, x, x_edge], dim=1)
        
        output = self.edge_fuse(self.max(output))
        
        return output + x


def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class OCAB(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.mask_process = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, mask):
        b, c, h, w = x.shape
        
        f_mask = self.mask_process(mask)
        masked_x = x * f_mask

        qkv_1 = self.qkv(masked_x)
        qs_1, ks_1, vs_1 = qkv_1.chunk(3, dim=1)

        qkv_2 = self.qkv(x)
        qs_2, ks_2, vs_2 = qkv_2.chunk(3, dim=1)

        qs_1 = rearrange(qs_1, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks_2, vs_2 = map(lambda t: self.unfold(t), (ks_2,vs_2))
        ks_2, vs_2 = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks_2,vs_2))

        qs_2 = rearrange(qs_2, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks_1, vs_1 = map(lambda t: self.unfold(t), (ks_1,vs_1))
        ks_1, vs_1 = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks_1,vs_1))


        qs_1, ks_2, vs_2 = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', 
                                              head=self.num_spatial_heads), (qs_1, ks_2, vs_2))
        qs_2, ks_1, vs_1 = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', 
                                              head=self.num_spatial_heads), (qs_2, ks_1, vs_1))
        qs_1 = qs_1 * self.scale
        attn_1 = (qs_1 @ ks_2.transpose(-2, -1))
        attn_1 += self.rel_pos_emb(qs_1)
        attn_1 = attn_1.softmax(dim=-1)
        out_1 = (attn_1 @ vs_2)
    
        qs_2 = qs_2 * self.scale
        attn_2 = (qs_2 @ ks_1.transpose(-2, -1))
        attn_2 += self.rel_pos_emb(qs_2)
        attn_2 = attn_2.softmax(dim=-1)
        out_2 = (attn_2 @ vs_1)
        # 重排回原始维度
        out_1 = rearrange(out_1, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', 
                      head=self.num_spatial_heads,
                      h=h//self.window_size,
                      w=w//self.window_size,
                      p1=self.window_size,
                      p2=self.window_size)

        out_2 = rearrange(out_2, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', 
                      head=self.num_spatial_heads,
                      h=h//self.window_size,
                      w=w//self.window_size,
                      p1=self.window_size,
                      p2=self.window_size)

        out = torch.sigmoid(self.alpha) * out_1 + (1 - torch.sigmoid(self.alpha)) * out_2
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_channel_heads, num_spatial_heads, spatial_dim_head, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()


        self.spatial_attn = OCAB(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = ChannelAttention(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.norm5 = LayerNorm(1, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x_masked, mask):
        x = x_masked + self.spatial_attn(self.norm3(x_masked), self.norm5(mask))
        x = x + self.spatial_ffn(self.norm4(x))
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        return x
    

class EncoderSequential(nn.Sequential):
    def forward(self, x, mask):
        for module in self:
            x = module(x, mask)
        return x
    

class Trans_low(nn.Module):
    def __init__(self, num_residual_blocks):
        super(Trans_low, self).__init__()

        start = [nn.Conv2d(3, 16, 3, padding=1),
                 nn.InstanceNorm2d(16),
                 nn.LeakyReLU(),
                 nn.Conv2d(16, 64, 3, padding=1),
                 nn.LeakyReLU()]

        final = [nn.Conv2d(64, 16, 3, padding=1),
                  nn.LeakyReLU(),
                  nn.Conv2d(16, 3, 3, padding=1)]

        self.start = nn.Sequential(*start)
        self.refine = EncoderSequential(*[TransformerBlock(dim=64, window_size=8, overlap_ratio=0.5, num_channel_heads=8, num_spatial_heads=4,
                                        spatial_dim_head=16, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for _ in range(num_residual_blocks)])
        self.final = nn.Sequential(*final)

    def forward(self, x, mask):
        out = self.start(x)
        out = self.refine(out, mask)
        out = self.final(out)
        out = torch.tanh(out)
        return out


class MuralRestormer(nn.Module):
    def __init__(self, levels):
        super(MuralRestormer, self).__init__()
        self.levels = levels
        self.bottom_model = Trans_low(num_residual_blocks=4)
        self.reconstruction_models = nn.ModuleList(
            [EdgeDFN() for _ in range(levels)])

    def gaussian_kernel(self, size, sigma, channels, device):
        grid = torch.arange(size, device=device)
        mean = (size - 1) / 2.
        variance = sigma ** 2.

        kernel_1d = torch.exp(-((grid - mean) ** 2) / (2 * variance))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()

        kernel = kernel_2d.view(1, 1, size, size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel

    def apply_gaussian_blur(self, img, kernel_size=5, sigma=1.0):
        device = img.device
        channels = img.shape[1]
        kernel = self.gaussian_kernel(
            kernel_size, sigma, channels, device).type_as(img)
        padding = kernel_size // 2
        img = F.pad(img, (padding, padding, padding, padding), mode='reflect')
        blurred = F.conv2d(img, kernel, groups=channels)
        return blurred

    def downsample(self, img):
        B, C, H, W = img.shape
        new_H = (H + 1) // 2
        new_W = (W + 1) // 2
        return F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=False)

    def upsample(self, img, size):
        return F.interpolate(img, size=size, mode='bilinear', align_corners=False)

    def forward(self, img, mask):
        # Decompose
        current = img
        pyramid = []
        for _ in range(self.levels):
            blurred = self.apply_gaussian_blur(
                current, kernel_size=5, sigma=1.0)
            down = self.downsample(blurred)
            up = self.upsample(down, size=current.shape[2:])
            laplacian = current - up
            pyramid.append(laplacian)
            current = down

        current_mask = mask
        mask_pyramid = []
        for _ in range(self.levels):
            blurred = self.apply_gaussian_blur(
                current_mask, kernel_size=5, sigma=1.0)
            down = self.downsample(blurred)
            up = self.upsample(down, size=current_mask.shape[2:])
            mask_pyramid.append(current_mask - up)
            current_mask = down

        # Apply bottom model
        refined_bottom = self.bottom_model(current, current_mask)
        pyramid.append(refined_bottom)
        mask_pyramid.append(current_mask)

        # Reconstruct
        image = pyramid[-1]
        mas = mask_pyramid[-1]
        for i, lev in enumerate(zip(reversed(pyramid[:-1]), reversed(mask_pyramid[:-1]))):
            image = self.upsample(image, size=lev[0].shape[2:])
            mas = self.upsample(mas, size=lev[1].shape[2:])
            image = self.reconstruction_models[i](image, mas) + lev[0]
        return image


# Sample usage with a random image
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = torch.randn(1, 3, 256, 256).to(device)

    mask = torch.randn(1, 1, 256, 256).to(device)

    levels = 5

    lp_model = MuralRestormer(levels).to(device)

    reconstructed_img = lp_model(img, mask)

    print(reconstructed_img.shape)
