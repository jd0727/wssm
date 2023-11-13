from models.base.modules import *


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, act=ACT.RELU):
        super(Residual, self).__init__()
        self.conv1 = Ck3BA(in_channels=in_channels, out_channels=out_channels,
                           stride=stride, dilation=dilation, bn=True, act=act)
        self.conv2 = Ck3BA(in_channels=out_channels, out_channels=out_channels,
                           stride=1, dilation=dilation, bn=True, act=None)

        self.act = ACT.build(act)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Ck1BA(in_channels=in_channels, out_channels=out_channels,
                                  stride=stride, bn=True, act=None)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out + residual)
        return out


class ResidualX(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, act=ACT.RELU):
        super(ResidualX, self).__init__()
        self.conv1 = Ck3BA(in_channels=in_channels, out_channels=out_channels * 2,
                           stride=stride, dilation=dilation, bn=True, act=act)
        self.conv2 = Ck3BA(in_channels=out_channels * 2, out_channels=out_channels,
                           stride=1, dilation=dilation, groups=32, bn=True, act=None)

        self.act = ACT.build(act)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Ck1BA(in_channels=in_channels, out_channels=out_channels,
                                  stride=stride, bn=True, act=None)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out + residual)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, act=ACT.RELU):
        super(Bottleneck, self).__init__()
        inner_channels = out_channels // 4
        self.conv1 = Ck1s1BA(in_channels=in_channels, out_channels=inner_channels,
                             bn=True, act=act)
        self.conv2 = Ck3BA(in_channels=inner_channels, out_channels=inner_channels,
                           stride=stride, dilation=dilation, bn=True, act=act)
        self.conv3 = Ck1s1BA(in_channels=inner_channels, out_channels=out_channels,
                             bn=True, act=None)

        self.act = ACT.build(act)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Ck1BA(in_channels=in_channels, out_channels=out_channels,
                                  stride=stride, bn=True, act=None)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(out + residual)
        return out


class BottleneckX(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, act=ACT.RELU):
        super(BottleneckX, self).__init__()
        inner_channels = out_channels // 4
        self.conv1 = Ck1s1BA(in_channels=in_channels, out_channels=inner_channels * 2,
                             bn=True, act=act)
        self.conv2 = Ck3BA(in_channels=inner_channels * 2, out_channels=inner_channels * 2,
                           stride=stride, dilation=dilation, groups=32, bn=True, act=act)
        self.conv3 = Ck1s1BA(in_channels=inner_channels * 2, out_channels=out_channels,
                             bn=True, act=None)

        self.act = ACT.build(act)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Ck1BA(in_channels=in_channels, out_channels=out_channels,
                                  stride=stride, bn=True, act=None)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(out + residual)
        return out


class BottleneckV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, ratio=26 / 64, dilations=(1, 1, 1), act=ACT.RELU):
        super(BottleneckV2, self).__init__()
        inner_channels = int(math.floor(out_channels / 4 * ratio))
        self.num_scale = len(dilations)
        self.inner_channels = inner_channels
        self.stride = stride

        self.conv1 = Ck1s1BA(in_channels=in_channels, out_channels=inner_channels * (self.num_scale + 1), act=act)
        self.conv2s = nn.ModuleList()
        for i in range(self.num_scale):
            self.conv2s.append(Ck3BA(in_channels=inner_channels, out_channels=inner_channels, stride=1,
                                     dilation=dilations[i], act=act))
        self.conv3 = Ck1s1B(in_channels=inner_channels * (self.num_scale + 1), out_channels=out_channels)

        self.act = ACT.build(act)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Ck1(in_channels=in_channels, out_channels=out_channels, stride=stride)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        x = x if self.stride == 1 else F.max_pool2d(x, kernel_size=self.stride, stride=self.stride)
        x = self.conv1(x)
        xs = torch.split(x, self.inner_channels, dim=1)
        buff = None
        outs = [xs[0]]
        for i in range(self.num_scale):
            if i == 0:
                buff = self.conv2s[i](xs[i + 1])
            else:
                buff = self.conv2s[i](xs[i + 1] + buff)
            outs.append(buff)
        out = torch.cat(outs, dim=1)
        out = self.conv3(out)
        out = self.act(out + residual)
        return out


class ResNetBkbn(nn.Module):
    def __init__(self, Module, repeat_nums, channels=64, act=ACT.RELU):
        super(ResNetBkbn, self).__init__()
        self.pre = CpaBA(in_channels=3, out_channels=64, kernel_size=7, stride=2, bn=True, act=act)
        self.stage1 = ResNetBkbn.ModuleRepeat(Module, in_channels=64, out_channels=channels, stride=1,
                                              repeat_num=repeat_nums[0], act=act, with_pool=True)
        self.stage2 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels, out_channels=channels * 2, stride=2,
                                              repeat_num=repeat_nums[1], act=act, with_pool=False)
        self.stage3 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels * 2, out_channels=channels * 4, stride=2,
                                              repeat_num=repeat_nums[2], act=act, with_pool=False)
        self.stage4 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels * 4, out_channels=channels * 8, stride=2,
                                              repeat_num=repeat_nums[3], act=act, with_pool=False)

    @staticmethod
    def ModuleRepeat(Module, in_channels, out_channels, repeat_num=1, stride=1, dilation=1, act=ACT.RELU,
                     with_pool=False):
        backbone = [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)] if with_pool else []
        last_channels = in_channels
        for i in range(repeat_num):
            backbone.append(
                Module(in_channels=last_channels, out_channels=out_channels, dilation=dilation, stride=stride, act=act))
            last_channels = out_channels
            stride = 1
        backbone = nn.Sequential(*backbone)
        return backbone

    def forward(self, imgs):
        feat0 = self.pre(imgs)
        feat1 = self.stage1(feat0)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        return feat4

    R18_PARA = dict(Module=Residual, repeat_nums=(2, 2, 2, 2), channels=64)
    R34_PARA = dict(Module=Residual, repeat_nums=(3, 4, 6, 3), channels=64)
    R50_PARA = dict(Module=Bottleneck, repeat_nums=(3, 4, 6, 3), channels=256)
    R101_PARA = dict(Module=Bottleneck, repeat_nums=(3, 4, 23, 3), channels=256)
    R152_PARA = dict(Module=Bottleneck, repeat_nums=(3, 8, 36, 3), channels=256)

    V2R50_PARA = dict(Module=BottleneckV2, repeat_nums=(3, 4, 6, 3), channels=256)
    V2R101_PARA = dict(Module=BottleneckV2, repeat_nums=(3, 4, 23, 3), channels=256)

    X18_PARA = dict(Module=ResidualX, repeat_nums=(2, 2, 2, 2), channels=64)
    X34_PARA = dict(Module=ResidualX, repeat_nums=(3, 4, 6, 3), channels=64)
    X50_PARA = dict(Module=BottleneckX, repeat_nums=(3, 4, 6, 3), channels=256)
    X101_PARA = dict(Module=BottleneckX, repeat_nums=(3, 4, 23, 3), channels=256)
    X152_PARA = dict(Module=BottleneckX, repeat_nums=(3, 8, 36, 3), channels=256)

    @staticmethod
    def R18(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.R18_PARA, act=act)

    @staticmethod
    def R34(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.R34_PARA, act=act)

    @staticmethod
    def R50(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.R50_PARA, act=act)

    @staticmethod
    def R101(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.R101_PARA, act=act)

    @staticmethod
    def V2R50(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.V2R50_PARA, act=act)

    @staticmethod
    def V2R101(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.V2R101_PARA, act=act)

    @staticmethod
    def R152(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.R152_PARA, act=act)

    @staticmethod
    def X18(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.X18_PARA, act=act)

    @staticmethod
    def X34(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.X34_PARA, act=act)

    @staticmethod
    def X50(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.X50_PARA, act=act)

    @staticmethod
    def X101(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.X101_PARA, act=act)

    @staticmethod
    def X152(act=ACT.RELU):
        return ResNetBkbn(**ResNetBkbn.X152_PARA, act=act)


class ResNetMain(ResNetBkbn):
    def __init__(self, Module, repeat_nums, act=ACT.RELU, channels=64, num_cls=0):
        super(ResNetMain, self).__init__(Module=Module, repeat_nums=repeat_nums, channels=channels, act=act)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=channels * 8, out_features=num_cls)
        self.num_cls = num_cls

    def forward(self, imgs):
        feat = super(ResNetMain, self).forward(imgs)
        feat = self.pool(feat)
        feat = feat.squeeze(dim=-1).squeeze(dim=-1)
        feat = self.linear(feat)
        return feat

    @staticmethod
    def R18(act=ACT.RELU, num_cls=10):
        return ResNetMain(**ResNetBkbn.R18_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def R34(act=ACT.RELU, num_cls=10):
        return ResNetMain(**ResNetBkbn.R34_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def R50(act=ACT.RELU, num_cls=10):
        return ResNetMain(**ResNetBkbn.R50_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def R101(act=ACT.RELU, num_cls=10):
        return ResNetMain(**ResNetBkbn.R101_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def R152(act=ACT.RELU, num_cls=10):
        return ResNetMain(**ResNetBkbn.R152_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def X18(act=ACT.RELU, num_cls=10):
        return ResNetMain(**ResNetBkbn.X18_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def X34(act=ACT.RELU, num_cls=10):
        return ResNetMain(**ResNetBkbn.X34_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def X50(act=ACT.RELU, num_cls=10):
        return ResNetMain(**ResNetBkbn.X50_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def X101(act=ACT.RELU, num_cls=10):
        return ResNetMain(**ResNetBkbn.X101_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def X152(act=ACT.RELU, num_cls=10):
        return ResNetMain(**ResNetBkbn.X152_PARA, act=act, num_cls=num_cls)


class ResNetCBkbn(nn.Module):
    def __init__(self, Module, repeat_nums, act=ACT.RELU):
        super(ResNetCBkbn, self).__init__()
        self.pre = CpaBA(in_channels=3, out_channels=16, kernel_size=7, stride=2, bn=True, act=act)
        self.stage1 = ResNetBkbn.ModuleRepeat(Module, in_channels=16, out_channels=16, stride=1,
                                              repeat_num=repeat_nums[0], act=act, with_pool=True)
        self.stage2 = ResNetBkbn.ModuleRepeat(Module, in_channels=16, out_channels=32, stride=2,
                                              repeat_num=repeat_nums[1], act=act, with_pool=False)
        self.stage3 = ResNetBkbn.ModuleRepeat(Module, in_channels=32, out_channels=64, stride=2,
                                              repeat_num=repeat_nums[2], act=act, with_pool=False)

    def forward(self, imgs):
        feat0 = self.pre(imgs)
        feat1 = self.stage1(feat0)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        return feat3

    R20_PARA = dict(Module=Residual, repeat_nums=(3, 3, 3))
    R32_PARA = dict(Module=Residual, repeat_nums=(5, 5, 5))
    R44_PARA = dict(Module=Residual, repeat_nums=(7, 7, 7))
    R56_PARA = dict(Module=Residual, repeat_nums=(9, 9, 9))
    R110_PARA = dict(Module=Residual, repeat_nums=(18, 18, 18))

    @staticmethod
    def R20(act=ACT.RELU):
        return ResNetCBkbn(**ResNetCBkbn.R20_PARA, act=act)

    @staticmethod
    def R32(act=ACT.RELU):
        return ResNetCBkbn(**ResNetCBkbn.R32_PARA, act=act)

    @staticmethod
    def R44(act=ACT.RELU):
        return ResNetCBkbn(**ResNetCBkbn.R44_PARA, act=act)

    @staticmethod
    def R56(act=ACT.RELU):
        return ResNetCBkbn(**ResNetCBkbn.R56_PARA, act=act)

    @staticmethod
    def R110(act=ACT.RELU):
        return ResNetCBkbn(**ResNetCBkbn.R110_PARA, act=act)


class ResNetCMain(ResNetCBkbn):
    def __init__(self, Module, repeat_nums, act=ACT.RELU, num_cls=0):
        super(ResNetCMain, self).__init__(Module=Module, repeat_nums=repeat_nums, act=act)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=64, out_features=num_cls)
        self.num_cls = num_cls

    def forward(self, imgs):
        feat = super(ResNetCMain, self).forward(imgs)
        feat = self.pool(feat)
        feat = feat.squeeze(dim=-1).squeeze(dim=-1)
        feat = self.linear(feat)
        return feat

    @staticmethod
    def R20(act=ACT.RELU, num_cls=0):
        return ResNetCMain(**ResNetCBkbn.R20_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def R32(act=ACT.RELU, num_cls=0):
        return ResNetCMain(**ResNetCBkbn.R32_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def R44(act=ACT.RELU, num_cls=0):
        return ResNetCMain(**ResNetCBkbn.R44_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def R56(act=ACT.RELU, num_cls=0):
        return ResNetCMain(**ResNetCBkbn.R56_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def R110(act=ACT.RELU, num_cls=0):
        return ResNetCMain(**ResNetCBkbn.R110_PARA, act=act, num_cls=num_cls)


class ResNet(OneStageClassifier):
    def __init__(self, backbone, device=None, pack=None, img_size=(512, 512)):
        num_cls = backbone.num_cls
        super(ResNet, self).__init__(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def R18(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetMain.R18(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def X18(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetMain.X18(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def R20(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetCMain.R20(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def R32(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetCMain.R32(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def R34(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetMain.R34(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def X34(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetMain.X34(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def R44(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetCMain.R32(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def R50(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetMain.R50(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def X50(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetMain.X50(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def R56(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetCMain.R56(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def R101(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetMain.R101(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def X101(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetMain.X101(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def R110(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetCMain.R110(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def R152(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetMain.R152(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def X152(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512)):
        backbone = ResNetMain.X152(act=ACT.RELU, num_cls=num_cls)
        return ResNet(backbone=backbone, device=device, pack=pack, img_size=img_size)


if __name__ == '__main__':
    # model = ResNetBkbn.V2R50(img_size=(512, 512), num_cls=20, device='cpu')
    model = ResNetBkbn.V2R50()
    # imgs = torch.zeros(5, 3, 512, 512)
    # cindsN = model.imgs2clses(imgs)
    model2onnx(model, './x18cus', 224)
    # imgs = torch.zeros(1, 3, 224, 224)
    # y = model(imgs)
