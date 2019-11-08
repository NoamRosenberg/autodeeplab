import torch
import torch.nn as nn
import platform
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

# TODO: NOW I DONT KNOW HOW TO USE ABN ON WINDOWS SYSTEM

if platform.system() == 'Windows':

    class ABN(nn.Module):
        def __init__(self, C_out, affine=False):
            super(ABN, self).__init__()
            self.op = nn.Sequential(
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False)
            )

        def forward(self, x):
            return self.op(x)
else:
    from modeling.modules import InPlaceABNSync as ABN

OPS = {
    'none': lambda C, stride, affine, use_ABN: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, use_ABN: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine, use_ABN: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine, use_ABN: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, use_ABN: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine, use_ABN: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine, use_ABN: DilConv(C, C, 3, stride, 2, 2, affine=affine, use_ABN=use_ABN),
    'dil_conv_5x5': lambda C, stride, affine, use_ABN: DilConv(C, C, 5, stride, 4, 2, affine=affine, use_ABN=use_ABN),
}


class NaiveBN(nn.Module):
    def __init__(self, C_out, momentum=0.1, affine=True):
        super(NaiveBN, self).__init__()
        self.op = nn.Sequential(
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU()
        )
        self._initialize_weights()



    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, use_ABN=False):
        super(ReLUConvBN, self).__init__()
        if use_ABN:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                ABN(C_out)
            )

        else:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine)
            )
        self._initialize_weights()



    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, seperate=True,
                 use_ABN=False):
        super(DilConv, self).__init__()
        if use_ABN:
            if seperate:
                self.op = nn.Sequential(
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation,
                              groups=C_in, bias=False),
                    nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                    ABN(C_out, affine=affine),
                )
            else:
                self.op = nn.Sequential(
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False),
                    nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                    ABN(C_out, affine=affine),
                )

        else:
            if seperate:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation,
                              groups=C_in, bias=False),
                    nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine),
                )
            else:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False),
                    nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(C_out, affine=affine),
                )
        self._initialize_weights()



    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, use_ABN=False):
        super(SepConv, self).__init__()
        if use_ABN:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                          bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                ABN(C_in, affine=affine),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                ABN(C_out, affine=affine)
            )

        else:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                          bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_in, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )
        self._initialize_weights()



    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()
        self._initialize_weights()



    def forward(self, x):
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self._initialize_weights()



    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()



class FactorizedReduce(nn.Module):
    # TODO: why conv1 and conv2 in two parts ?
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self._initialize_weights()



    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


class DoubleFactorizedReduce(nn.Module):
    # TODO: why conv1 and conv2 in two parts ?
    def __init__(self, C_in, C_out, affine=True):
        super(DoubleFactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self._initialize_weights()


    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

class FactorizedIncrease(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ReLU(inplace=False),
            nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )
        self._initialize_weights()


    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


class DoubleFactorizedIncrease(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleFactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.ReLU(inplace=False),
            nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )
        self._initialize_weights()


    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, paddings, dilations, momentum=0.0003):
        super(ASPP, self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False, ),
                                    nn.BatchNorm2d(in_channels))
        self.conv33 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3,
                                              padding=paddings, dilation=dilations, bias=False, ),
                                    nn.BatchNorm2d(in_channels))
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False, ),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU())

        self.concate_conv = nn.Conv2d(in_channels * 3, in_channels, 1, bias=False, stride=1, padding=0)
        self.concate_bn = nn.BatchNorm2d(in_channels, momentum)
        self.final_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False, stride=1, padding=0)
        self._initialize_weights()

    def forward(self, x):
        conv11 = self.conv11(x)
        conv33 = self.conv33(x)

        # image pool and upsample
        image_pool = nn.AvgPool2d(kernel_size=x.size()[2:])
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        image_pool = image_pool(x)
        conv_image_pool = self.conv_p(image_pool)
        upsample = upsample(conv_image_pool)

        # concate
        concate = torch.cat([conv11, conv33, upsample], dim=1)
        concate = self.concate_conv(concate)
        concate = self.concate_bn(concate)

        return self.final_conv(concate)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
