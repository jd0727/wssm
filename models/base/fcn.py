from utils.frame import *
from models.modules import *
from models.template import *
from models.base.vgg import VGGBkbn


class FlatHead(nn.Module):
    def __init__(self, in_channels, out_channels, head_channel=4096, act=ACT.RELU):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv1 = Ck1A(in_channels=in_channels, out_channels=head_channel, act=act)
        self.dp1 = nn.Dropout2d()
        self.conv2 = Ck1A(in_channels=head_channel, out_channels=head_channel, act=act)
        self.dp2 = nn.Dropout2d()
        self.conv3 = Ck1A(in_channels=head_channel, out_channels=out_channels, act=act)

    def forward(self, feats):
        feats = self.pool(feats)
        feats = self.conv1(feats)
        feats = self.dp1(feats)
        feats = self.conv2(feats)
        feats = self.dp2(feats)
        feats = self.conv3(feats)
        return feats


class FCN32VGGMain(VGGBkbn):
    def __init__(self, repeat_nums=(1, 1, 2, 2, 2), act=ACT.RELU, num_cls=20, head_channel=4096, img_size=(224, 224)):
        super(FCN32VGGMain, self).__init__(repeat_nums=repeat_nums, act=act)
        self.stage6 = FlatHead(in_channels=512, head_channel=head_channel, out_channels=num_cls + 1, act=act)
        self.upsamper32 = CTpa(in_channels=num_cls + 1, out_channels=num_cls + 1, kernel_size=64, stride=32)
        self.img_size = img_size
        self.num_cls = num_cls

    def forward(self, imgs):
        feats5 = super(FCN32VGGMain, self).forward(imgs)
        feats6 = self.stage6(feats5)
        feats6 = self.upsamper32(feats6)
        return feats6

    @staticmethod
    def A(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN32VGGMain(**VGGBkbn.A_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=4096)

    @staticmethod
    def B(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN32VGGMain(**VGGBkbn.B_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=4096)

    @staticmethod
    def D(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN32VGGMain(**VGGBkbn.D_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=4096)

    @staticmethod
    def E(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN32VGGMain(**VGGBkbn.E_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=4096)

    @staticmethod
    def AC(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN32VGGMain(**VGGBkbn.A_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=512)

    @staticmethod
    def BC(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN32VGGMain(**VGGBkbn.B_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=512)

    @staticmethod
    def DC(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN32VGGMain(**VGGBkbn.D_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=512)

    @staticmethod
    def EC(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN32VGGMain(**VGGBkbn.E_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=512)


class FCN16VGGMain(VGGBkbn):
    def __init__(self, repeat_nums=(1, 1, 2, 2, 2), act=ACT.RELU, num_cls=20, head_channel=4096, img_size=(224, 224)):
        super(FCN16VGGMain, self).__init__(repeat_nums=repeat_nums, act=act)
        self.stage6 = FlatHead(in_channels=512, head_channel=head_channel, out_channels=num_cls + 1, act=act)
        self.upsamper2 = CTpa(in_channels=num_cls + 1, out_channels=num_cls + 1, kernel_size=4, stride=2)
        self.upsamper16 = CTpa(in_channels=num_cls + 1, out_channels=num_cls + 1, kernel_size=32, stride=16)
        self.conv5 = Ck1s1(in_channels=512, out_channels=num_cls + 1)
        self.img_size = img_size
        self.num_cls = num_cls

    def forward(self, imgs):
        feats1 = self.stage1(imgs)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        feats6 = self.stage6(feats5)

        feats5 = self.conv5(feats5)
        feats6 = self.upsamper2(feats6)

        feats = self.upsamper16(feats5 + feats6)
        return feats

    @staticmethod
    def A(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN16VGGMain(**VGGBkbn.A_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=4096)

    @staticmethod
    def AC(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN16VGGMain(**VGGBkbn.A_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=512)


class FCN8VGGMain(VGGBkbn):
    def __init__(self, repeat_nums=(1, 1, 2, 2, 2), act=ACT.RELU, num_cls=20, head_channel=4096, img_size=(224, 224)):
        super(FCN8VGGMain, self).__init__(repeat_nums=repeat_nums, act=act)
        self.stage6 = FlatHead(in_channels=512, head_channel=head_channel, out_channels=num_cls + 1, act=act)
        self.upsamper2 = CTpa(in_channels=num_cls + 1, out_channels=num_cls + 1, kernel_size=4, stride=2)
        self.upsamper2x = CTpa(in_channels=num_cls + 1, out_channels=num_cls + 1, kernel_size=4, stride=2)
        self.upsamper8 = CTpa(in_channels=num_cls + 1, out_channels=num_cls + 1, kernel_size=16, stride=8)
        self.conv5 = Ck1s1BA(in_channels=512, out_channels=num_cls + 1, act=None)
        self.conv4 = Ck1s1BA(in_channels=512, out_channels=num_cls + 1, act=None)
        self.img_size = img_size
        self.num_cls = num_cls

    def forward(self, imgs):
        feats1 = self.stage1(imgs)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        feats6 = self.stage6(feats5)

        feats5 = self.conv5(feats5)
        feats4 = self.conv4(feats4)

        feats = self.upsamper2x(feats5 + self.upsamper2(feats6))
        feats = self.upsamper8(feats4 + feats)
        return feats

    @staticmethod
    def A(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN8VGGMain(**VGGBkbn.A_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=4096)

    @staticmethod
    def AC(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return FCN8VGGMain(**VGGBkbn.A_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=512)


class FCNConstMain(nn.Module):
    def __init__(self, batch_size=1, num_cls=20, img_size=(224, 224)):
        super().__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.featmaps = nn.Parameter(torch.zeros(batch_size, num_cls + 1, img_size[1], img_size[0]))
        # init_sig(self.featmaps, prior_prob=0.01)

    def forward(self, imgs):
        return self.featmaps


class FCN(OneStageSegmentor):
    def __init__(self, backbone, device=None, pack=None):
        super().__init__(backbone, device=device, pack=pack)

    @staticmethod
    def V32A(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = FCN32VGGMain.A(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return FCN(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def V16A(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = FCN16VGGMain.A(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return FCN(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def V8A(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = FCN8VGGMain.A(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return FCN(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, batch_size=1, num_cls=20, img_size=(224, 224)):
        backbone = FCNConstMain(batch_size=batch_size, num_cls=num_cls, img_size=img_size)
        return FCN(backbone=backbone, device=device, pack=PACK.NONE)


if __name__ == '__main__':
    model = FCN32VGGMain.AC(num_cls=20, img_size=(224, 224))
    imgs = torch.zeros(size=(2, 3, 224, 224))
    y = model(imgs)
    print(y.size())
