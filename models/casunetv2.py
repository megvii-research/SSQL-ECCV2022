import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_state_dict


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        has_bn=True,
        has_relu=False,
    ):
        super().__init__()

        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                groups=groups,
            ),
        )
        if has_bn:
            self.conv.add_module(
                "bn", nn.BatchNorm2d(num_features=out_channels, affine=True)
            )
        if has_relu:
            self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Deconv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        has_bn=True,
        has_relu=False,
    ):
        super().__init__()

        self.deconv = nn.Sequential()
        self.deconv.add_module(
            "deconv",
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                groups=groups,
            ),
        )
        if has_bn:
            self.deconv.add_module(
                "bn", nn.BatchNorm2d(num_features=out_channels, affine=True)
            )
        if has_relu:
            self.deconv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.deconv(x)
        return x


class ResNextBlock(nn.Module):
    def __init__(self, stride, in_channels, out_channels, groups, has_proj=False):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.has_proj = has_proj
        self.bottleneck = out_channels // 4

        if self.has_proj:
            if self.stride == 2:
                self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.conv = Conv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                has_bn=True,
                has_relu=False,
            )

        self.conv_group = nn.Sequential(
            Conv(
                in_channels=self.in_channels,
                out_channels=self.bottleneck,
                kernel_size=1,
                has_bn=True,
                has_relu=True,
            ),
            Conv(
                in_channels=self.bottleneck,
                out_channels=self.bottleneck,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=self.groups,
                has_bn=True,
                has_relu=True,
            ),
            Conv(
                in_channels=self.bottleneck,
                out_channels=self.out_channels,
                kernel_size=1,
                has_bn=True,
                has_relu=False,
            ),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        proj = x

        if self.has_proj:
            if self.stride:
                proj = self.pool(proj)
            proj = self.conv(proj)

        x = self.conv_group(x)
        x = x + proj
        x = self.relu(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ks):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = ks

        self.conv0 = nn.Sequential(
            Conv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, self.ks),
                padding=(0, self.ks // 2),
                has_bn=True,
                has_relu=True,
            ),
            Conv(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(self.ks, 1),
                padding=(self.ks // 2, 0),
                has_bn=True,
                has_relu=False,
            ),
        )
        self.conv1 = nn.Sequential(
            Conv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.ks, 1),
                padding=(self.ks // 2, 0),
                has_bn=True,
                has_relu=True,
            ),
            Conv(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(1, self.ks),
                padding=(0, self.ks // 2),
                has_bn=True,
                has_relu=False,
            ),
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x = x0 + x1

        return x


class RefineBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ks):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = ks

        self.conv_group = nn.Sequential(
            Conv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.ks,
                padding=self.ks // 2,
                has_bn=True,
                has_relu=True,
            ),
            Conv(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.ks,
                padding=self.ks // 2,
                has_bn=True,
                has_relu=False,
            ),
        )

    def forward(self, x):
        proj = x
        x = self.conv_group(x)
        x = x + proj

        return x


class DownConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        nr_convs,
        pooling=True,
        kernel_size=16,
        stride=16,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nr_convs = nr_convs
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.stride = stride

        self.down_conv = nn.Sequential()
        for i in range(nr_convs):
            if i == 0:
                self.down_conv.add_module(
                    "conv0",
                    ResNextBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        groups=2,
                        stride=2,
                        has_proj=True,
                    ),
                )
            else:
                self.down_conv.add_module(
                    "conv%s" % i,
                    ResNextBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        groups=2,
                        stride=1,
                        has_proj=False,
                    ),
                )

        if self.pooling:
            self.pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        x = self.down_conv(x)
        if self.pooling:
            after_pool = self.pool(x)
            return x, after_pool

        return x, None


class UpConv(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_out_channels,
        out_channels,
        is_conv_block,
        has_from_up=True,
        deconv_has_bn=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_out_channels = mid_out_channels
        self.out_channels = out_channels
        self.is_conv_block = is_conv_block
        self.has_from_up = has_from_up
        self.deconv_has_bn = deconv_has_bn

        if self.is_conv_block:
            self.conv = ConvBlock(
                in_channels=self.in_channels, out_channels=self.mid_out_channels, ks=3
            )
        else:
            self.conv = Conv(
                in_channels=self.in_channels,
                out_channels=self.mid_out_channels,
                kernel_size=3,
                padding=1,
            )

        self.refine = RefineBlock(
            in_channels=self.mid_out_channels, out_channels=self.mid_out_channels, ks=3
        )
        self.deconv = Deconv(
            in_channels=self.mid_out_channels,
            out_channels=self.out_channels,
            kernel_size=2,
            stride=2,
            has_bn=self.deconv_has_bn,
            has_relu=False,
        )

    def forward(self, from_down, from_up=None):
        x = self.conv(from_down)
        if self.has_from_up:
            x = x + from_up
        x = self.refine(x)
        x = self.deconv(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, max_disp, base_ch=32, depth=5):  # base_ch=32/32/24
        super().__init__()

        self.in_channels = in_channels
        self.max_disp = max_disp
        self.base_ch = base_ch
        self.down_convs = []
        self.up_convs = []

        nr_convs = [3, 4, 7, 7, 7]
        kernel_size = [16, 8, 4, 2, None]  # for pooling
        stride = [16, 8, 4, 2, None]  # for pooling
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.base_ch * (
                2 ** (i + 1)
            )  # base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16, base_ch * 32
            pooling = True if i < depth - 1 else False
            self.down_convs.append(
                DownConv(
                    in_channels=ins,
                    out_channels=outs,
                    nr_convs=nr_convs[i],
                    pooling=pooling,
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                )
            )

        in_channels = [_ * self.base_ch for _ in [62, 16, 8, 4, 2]]
        mid_out_channels = [_ * 4 for _ in [16, 16, 8, 2, 1]]
        out_channels = [_ * 4 for _ in [16, 8, 2, 1]] + [1]  # final output channel is 1
        is_conv_block = [True, True, True, False, False]
        for i in range(depth):
            has_from_up = True if i > 0 else False
            deconv_has_bn = True if i < depth - 1 else False
            self.up_convs.append(
                UpConv(
                    in_channels=in_channels[i],
                    mid_out_channels=mid_out_channels[i],
                    out_channels=out_channels[i],
                    is_conv_block=is_conv_block[i],
                    has_from_up=has_from_up,
                    deconv_has_bn=deconv_has_bn,
                )
            )

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        from_downs = []
        encoder_pools = []
        last_fm = None

        for module in self.down_convs:
            x, after_pool = module(x)
            from_downs.append(x)
            if after_pool is not None:
                encoder_pools.append(after_pool)

        encoder_pools.append(from_downs[-1])
        from_downs[-1] = torch.cat(encoder_pools, dim=1)

        for i, module in enumerate(self.up_convs):
            if i == 0:
                from_up = module(from_downs[-1 - i])
            else:
                from_up = module(from_downs[-1 - i], from_up)

            if i == len(from_downs) - 2:
                last_fm = from_up

        pred = self.sigmoid(from_up) * self.max_disp

        return pred, last_fm


class CascadedUnets(nn.Module):
    def __init__(self, max_disp, channel_base=1) -> None:
        super().__init__()

        self.unet1 = UNet(
            in_channels=6, max_disp=max_disp // 4, base_ch=32 * channel_base
        )
        self.unet2 = UNet(
            in_channels=10, max_disp=max_disp // 2, base_ch=32 * channel_base
        )
        self.unet3 = UNet(in_channels=10, max_disp=max_disp, base_ch=24 * channel_base)

    def forward(self, data):

        data_dw4 = F.interpolate(input=data, scale_factor=0.25, mode="bilinear")
        data_dw2 = F.interpolate(input=data, scale_factor=0.5, mode="bilinear")

        pred1_dw4, fm1_dw8 = self.unet1(data_dw4)

        fm1_dw2 = F.interpolate(input=fm1_dw8, scale_factor=4, mode="bilinear")
        data_dw2 = torch.cat([data_dw2, fm1_dw2], dim=1)
        pred2_dw2, fm2_dw4 = self.unet2(data_dw2)

        fm2 = F.interpolate(fm2_dw4, scale_factor=4, mode="bilinear")
        data = torch.cat([data, fm2], dim=1)
        pred3, _ = self.unet3(data)

        pred_pyramid = [pred1_dw4, pred2_dw2, pred3]

        return pred_pyramid


def casunetv2(ckpt_path=None):
    model = CascadedUnets(max_disp=192)
    if ckpt_path is not None:
        state_dict = get_state_dict(ckpt_path)
        model.load_state_dict(state_dict)
    return model


def casunetv2_double(ckpt_path=None):
    model = CascadedUnets(max_disp=192, channel_base=2)
    if ckpt_path is not None:
        state_dict = get_state_dict(ckpt_path)
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    model = CascadedUnets(max_disp=192)
    print(model)
