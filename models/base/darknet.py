from models.base.modules import *


# CBA+CBA+Res
class DarkNetResidual(nn.Module):
    def __init__(self, channels, inner_channels, act=ACT.LK):
        super(DarkNetResidual, self).__init__()
        self.backbone = nn.Sequential(
            Ck1s1BA(in_channels=channels, out_channels=inner_channels, bn=True, act=act),
            Ck3s1BA(in_channels=inner_channels, out_channels=channels, bn=True, act=act),
        )

    def forward(self, x):
        x = x + self.backbone(x)
        return x


class DarkNetTinyBkbn(nn.Module):
    def __init__(self, channels, act=ACT.LK, in_channels=3):
        super(DarkNetTinyBkbn, self).__init__()
        self.stage1 = nn.Sequential(
            Ck3s1BA(in_channels=in_channels, out_channels=channels, act=act),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Ck3s1BA(in_channels=channels, out_channels=channels * 2, act=act),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DarkNetTinyBkbn.C3C1Repeat(in_channels=channels * 2, out_channels=channels * 4, repeat_num=3, act=act),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DarkNetTinyBkbn.C3C1Repeat(in_channels=channels * 4, out_channels=channels * 8, repeat_num=3, act=act),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DarkNetTinyBkbn.C3C1Repeat(in_channels=channels * 8, out_channels=channels * 16, repeat_num=5, act=act),
        )
        self.stage2 = DarkNetTinyBkbn.C3C1Repeat(
            in_channels=channels * 16, out_channels=channels * 32, repeat_num=5, act=act)
        self.stage3 = nn.Sequential(
            Ck1BA(in_channels=channels * 32, out_channels=channels * 32, act=act),
            Ck1BA(in_channels=channels * 32, out_channels=channels * 32, act=act)
        )

    @staticmethod
    def C3C1Repeat(in_channels, out_channels, repeat_num, act=ACT.LK):
        inner_channels = out_channels // 2
        convs = [Ck3s1BA(in_channels, out_channels, act=act)]
        for i in range((repeat_num - 1) // 2):
            convs.append(Ck1s1BA(out_channels, inner_channels, act=act))
            convs.append(Ck3s1BA(inner_channels, out_channels, act=act))
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feats1 = self.stage1(imgs)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        return feats1, feats2, feats3


class DarkNetBkbn(nn.Module):
    def __init__(self, channels, repeat_nums, act=ACT.LK, in_channels=3):
        super(DarkNetBkbn, self).__init__()
        self.pre = Ck3s1BA(in_channels=in_channels, out_channels=channels, act=act)
        self.stage1 = DarkNetBkbn.ResidualRepeat(in_channels=channels, out_channels=channels * 2,
                                                 repeat_num=repeat_nums[0], stride=2, act=act)
        self.stage2 = DarkNetBkbn.ResidualRepeat(in_channels=channels * 2, out_channels=channels * 4,
                                                 repeat_num=repeat_nums[1], stride=2, act=act)
        self.stage3 = DarkNetBkbn.ResidualRepeat(in_channels=channels * 4, out_channels=channels * 8,
                                                 repeat_num=repeat_nums[2], stride=2, act=act)
        self.stage4 = DarkNetBkbn.ResidualRepeat(in_channels=channels * 8, out_channels=channels * 16,
                                                 repeat_num=repeat_nums[3], stride=2, act=act)
        self.stage5 = DarkNetBkbn.ResidualRepeat(in_channels=channels * 16, out_channels=channels * 32,
                                                 repeat_num=repeat_nums[4], stride=2, act=act)

    @staticmethod
    def ResidualRepeat(in_channels, out_channels, repeat_num=1, stride=2, act=ACT.LK):
        convs = [Ck3BA(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act)]
        for i in range(repeat_num):
            convs.append(DarkNetResidual(channels=out_channels, inner_channels=out_channels // 2, act=act))
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        return feats5

    PARA_R53 = dict(channels=32, repeat_nums=(1, 2, 8, 8, 4))

    @staticmethod
    def R53(act=ACT.RELU):
        return DarkNetBkbn(**DarkNetBkbn.PARA_R53, act=act)


# ConvResidualRepeat+CBA+Res
class CSPBlockV4(nn.Module):
    def __init__(self, channels, shortcut_channels, backbone_channels, backbone_inner_channels,
                 repeat_num, act=ACT.LK):
        super(CSPBlockV4, self).__init__()
        self.shortcut = Ck1BA(in_channels=channels, out_channels=shortcut_channels, act=act)
        backbone = [Ck1s1BA(in_channels=channels, out_channels=backbone_channels, act=act)]
        for i in range(repeat_num):
            backbone.append(DarkNetResidual(
                channels=backbone_channels, inner_channels=backbone_inner_channels, act=act))
        backbone.append(Ck1s1BA(in_channels=backbone_channels, out_channels=backbone_channels, act=act))
        self.backbone = nn.Sequential(*backbone)
        self.concater = Ck1s1BA(in_channels=shortcut_channels + backbone_channels, out_channels=channels, act=act)

    def forward(self, x):
        xc = torch.cat([self.backbone(x), self.shortcut(x)], dim=1)
        xc = self.concater(xc)
        return xc


# ConvResidualRepeat+CBA+Res
class CSPBlockV5(nn.Module):
    def __init__(self, channels, shortcut_channels, backbone_channels, backbone_inner_channels,
                 repeat_num, act=ACT.LK):
        super(CSPBlockV5, self).__init__()
        self.shortcut = Ck1s1BA(in_channels=channels, out_channels=shortcut_channels, act=act)

        backbone = [Ck1s1BA(in_channels=channels, out_channels=backbone_channels, act=act)]
        for i in range(repeat_num):
            backbone.append(DarkNetResidual(
                channels=backbone_channels, inner_channels=backbone_inner_channels, act=act))
        self.backbone = nn.Sequential(*backbone)
        self.concater = Ck1s1BA(in_channels=shortcut_channels + backbone_channels, out_channels=channels, act=act)

    def forward(self, x):
        xc = torch.cat([self.backbone(x), self.shortcut(x)], dim=1)
        xc = self.concater(xc)
        return xc


class CSPDarkNetV4Bkbn(nn.Module):
    def __init__(self, channels, repeat_nums, act=ACT.MISH, in_channels=3):
        super(CSPDarkNetV4Bkbn, self).__init__()
        self.pre = Ck3s1BA(in_channels=in_channels, out_channels=channels, act=act)
        self.stage1 = CSPDarkNetV4Bkbn.ResidualRepeat(in_channels=channels, out_channels=channels * 2,
                                                      repeat_num=repeat_nums[0], stride=2, act=act)
        self.stage2 = CSPDarkNetV4Bkbn.ResidualRepeat(in_channels=channels * 2, out_channels=channels * 4,
                                                      repeat_num=repeat_nums[1], stride=2, act=act)
        self.stage3 = CSPDarkNetV4Bkbn.ResidualRepeat(in_channels=channels * 4, out_channels=channels * 8,
                                                      repeat_num=repeat_nums[2], stride=2, act=act)
        self.stage4 = CSPDarkNetV4Bkbn.ResidualRepeat(in_channels=channels * 8, out_channels=channels * 16,
                                                      repeat_num=repeat_nums[3], stride=2, act=act)
        self.stage5 = CSPDarkNetV4Bkbn.ResidualRepeat(in_channels=channels * 16, out_channels=channels * 32,
                                                      repeat_num=repeat_nums[4], stride=2, act=act)

    @staticmethod
    def ResidualRepeat(in_channels, out_channels, repeat_num=1, stride=2, act=ACT.LK):
        convs = nn.Sequential(
            Ck3BA(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act),
            CSPBlockV4(channels=out_channels, shortcut_channels=out_channels // 2,
                       backbone_channels=out_channels // 2, backbone_inner_channels=out_channels // 2,
                       repeat_num=repeat_num, act=act))
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        return feats5

    @staticmethod
    def R53(act=ACT.RELU):
        return CSPDarkNetV4Bkbn(**DarkNetBkbn.PARA_R53, act=act)


class CSPDarkNetV5Bkbn(nn.Module):
    def __init__(self, channels, repeat_nums, act=ACT.SILU, in_channels=3):
        super(CSPDarkNetV5Bkbn, self).__init__()
        self.pre = nn.Sequential(
            Focus(),
            Ck3s1BA(in_channels=in_channels * 4, out_channels=channels, act=act))
        self.stage1 = CSPDarkNetV5Bkbn.ResidualRepeat(in_channels=channels, out_channels=channels * 2,
                                                      repeat_num=repeat_nums[0], stride=2, act=act)
        self.stage2 = CSPDarkNetV5Bkbn.ResidualRepeat(in_channels=channels * 2, out_channels=channels * 4,
                                                      repeat_num=repeat_nums[1], stride=2, act=act)
        self.stage3 = CSPDarkNetV5Bkbn.ResidualRepeat(in_channels=channels * 4, out_channels=channels * 8,
                                                      repeat_num=repeat_nums[2], stride=2, act=act)
        self.stage4 = CSPDarkNetV5Bkbn.ResidualRepeat(in_channels=channels * 8, out_channels=channels * 16,
                                                      repeat_num=repeat_nums[3], stride=2, act=act)

    @staticmethod
    def ResidualRepeat(in_channels, out_channels, repeat_num=1, stride=2, act=ACT.LK):
        convs = nn.Sequential(
            Ck3BA(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act),
            CSPBlockV5(channels=out_channels, shortcut_channels=out_channels // 2,
                       backbone_channels=out_channels // 2, backbone_inner_channels=out_channels // 2,
                       repeat_num=repeat_num, act=act))
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        return feats4

    PARA_NANO = dict(channels=16, repeat_nums=(1, 2, 3, 1))
    PARA_SMALL = dict(channels=32, repeat_nums=(1, 2, 3, 1))
    PARA_MEDIUM = dict(channels=48, repeat_nums=(2, 4, 6, 2))
    PARA_LARGE = dict(channels=64, repeat_nums=(3, 6, 9, 3))
    PARA_XLARGE = dict(channels=80, repeat_nums=(4, 8, 12, 4))

    @staticmethod
    def NANO(act=ACT.RELU):
        return CSPDarkNetV5Bkbn(**CSPDarkNetV5Bkbn.PARA_NANO, act=act)

    @staticmethod
    def SMALL(act=ACT.RELU):
        return CSPDarkNetV5Bkbn(**CSPDarkNetV5Bkbn.PARA_SMALL, act=act)

    @staticmethod
    def MEDIUM(act=ACT.RELU):
        return CSPDarkNetV5Bkbn(**CSPDarkNetV5Bkbn.PARA_MEDIUM, act=act)

    @staticmethod
    def LARGE(act=ACT.RELU):
        return CSPDarkNetV5Bkbn(**CSPDarkNetV5Bkbn.PARA_LARGE, act=act)

    @staticmethod
    def XLARGE(act=ACT.RELU):
        return CSPDarkNetV5Bkbn(**CSPDarkNetV5Bkbn.PARA_XLARGE, act=act)


class CSPDarkNetV5ExtBkbn(CSPDarkNetV5Bkbn):
    def __init__(self, channels, repeat_nums, act=ACT.SILU, in_channels=3):
        super(CSPDarkNetV5ExtBkbn, self).__init__(channels, repeat_nums, act=act, in_channels=in_channels)
        self.stage5 = CSPDarkNetV5Bkbn.ResidualRepeat(in_channels=channels * 16, out_channels=channels * 16,
                                                      repeat_num=repeat_nums[3], stride=2, act=act)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        return feats5

    @staticmethod
    def NANO(act=ACT.RELU):
        return CSPDarkNetV5ExtBkbn(**CSPDarkNetV5Bkbn.PARA_NANO, act=act)

    @staticmethod
    def SMALL(act=ACT.RELU):
        return CSPDarkNetV5ExtBkbn(**CSPDarkNetV5Bkbn.PARA_SMALL, act=act)

    @staticmethod
    def MEDIUM(act=ACT.RELU):
        return CSPDarkNetV5ExtBkbn(**CSPDarkNetV5Bkbn.PARA_MEDIUM, act=act)

    @staticmethod
    def LARGE(act=ACT.RELU):
        return CSPDarkNetV5ExtBkbn(**CSPDarkNetV5Bkbn.PARA_LARGE, act=act)

    @staticmethod
    def XLARGE(act=ACT.RELU):
        return CSPDarkNetV5ExtBkbn(**CSPDarkNetV5Bkbn.PARA_XLARGE, act=act)


class CSPBlockV4Tiny(nn.Module):
    def __init__(self, channels, act=ACT.LK):
        super(CSPBlockV4Tiny, self).__init__()
        self.inner_channels = channels // 2
        self.conv1 = Ck3s1BA(in_channels=channels, out_channels=channels, bn=True, act=act)
        self.conv2 = Ck3s1BA(in_channels=self.inner_channels, out_channels=self.inner_channels, bn=True, act=act)

        self.conv3 = Ck3s1BA(in_channels=self.inner_channels, out_channels=self.inner_channels, bn=True, act=act)
        self.conv4 = Ck1s1BA(in_channels=channels, out_channels=channels, bn=True, act=act)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_p = x1[:, self.inner_channels:, :, :]
        x2 = self.conv2(x1_p)
        x3 = self.conv3(x2)
        x32 = torch.cat([x3, x2], dim=1)
        x4 = self.conv4(x32)
        x14 = torch.cat([x1, x4], dim=1)
        return x14


class CSPDarkNetV4TinyBkbn(nn.Module):
    def __init__(self, channels, act=ACT.LK, in_channels=3):
        super(CSPDarkNetV4TinyBkbn, self).__init__()

        self.stage1 = nn.Sequential(
            Ck3BA(in_channels=in_channels, out_channels=channels, stride=2, act=act),
            Ck3BA(in_channels=channels, out_channels=channels * 2, stride=2, act=act),
            CSPBlockV4Tiny(channels=channels * 2, act=act),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            CSPBlockV4Tiny(channels=channels * 4, act=act),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            CSPBlockV4Tiny(channels=channels * 8, act=act),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Ck3s1BA(in_channels=channels * 16, out_channels=channels * 16, act=act),
            Ck1s1BA(in_channels=channels * 16, out_channels=channels * 8, act=act)

        )
        self.c1_2 = nn.Sequential(
            Ck1s1BA(in_channels=channels * 8, out_channels=channels * 4, act=act),
            nn.UpsamplingNearest2d(scale_factor=2),

        )
        self.c2 = Ck3s1BA(in_channels=channels * 8, out_channels=channels * 16, act=act)
        self.c1 = Ck3s1BA(in_channels=channels * 20, out_channels=channels * 8, act=act)

    def forward(self, imgs):
        feat1 = self.stage1(imgs)
        feat2 = self.stage2(feat1)
        c1 = torch.cat([feat1, self.c1_2(feat2)], dim=1)
        c1 = self.c1(c1)
        c2 = self.c2(feat2)
        return c1, c2


if __name__ == '__main__':
    # model = CSPDarkNetV5Bkbn.NANO()
    model = CSPDarkNetV4Bkbn.R53()
    imgs = torch.rand(2, 3, 416, 416)
    y = model(imgs)
    print(y.size())
