import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm2d(C_out),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )


  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
#TODO: why conv1 and conv2 in two parts ?
  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

class DoubleFactorizedReduce(nn.Module):
#TODO: why conv1 and conv2 in two parts ?
  def __init__(self, C_in, C_out, affine=True):
    super(DoubleFactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

class FactorizedIncrease (nn.Module) :
    def __init__ (self, in_channel, out_channel) :
        super(FactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential (
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ReLU(inplace = False),
            nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )
    def forward (self, x) :
        return self.op (x)

class DoubleFactorizedIncrease (nn.Module) :
    def __init__ (self, in_channel, out_channel) :
        super(DoubleFactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential (
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.ReLU(inplace = False),
            nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )
    def forward (self, x) :
        return self.op (x)

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

        self.concate_conv = nn.Conv2d(in_channels * 3, in_channels, 1, bias=False,  stride=1, padding=0)
        self.concate_bn = nn.BatchNorm2d(in_channels, momentum)
        self.final_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False,  stride=1, padding=0)

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