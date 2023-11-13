from models.base.resnet import BottleneckV2
from models.base.resunet import ResUNetMain, ResUNetEnc, ResUNetDec
from models.modules import *
from utils import *


# <editor-fold desc='分类层'>
class UInstLayerClsMLP(nn.Module):
    def __init__(self, in_channels, inner_channels, num_cls=1, img_size=(224, 224), feat_size=(8, 8),
                 act=ACT.RELU, inner_features=1024):
        super().__init__()
        self._img_size = img_size
        self.num_cls = num_cls
        self.cvtor = Ck1s1BA(in_channels=in_channels, out_channels=inner_channels, act=act)
        in_features = inner_channels * feat_size[0] * feat_size[1]
        self.stem = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=inner_features),
            ACT.build(act),
            nn.Linear(in_features=inner_features, out_features=num_cls)
        )

    @property
    def img_size(self):
        return self._img_size

    def forward(self, feat):
        feat = self.cvtor(feat)
        feat = feat.view(feat.size(0), -1)
        chot = self.stem(feat)
        return chot


class UInstLayerClsPool(nn.Module):
    def __init__(self, in_channels, num_cls=1, img_size=(224, 224), act=ACT.RELU, inner_features=1024):
        super().__init__()
        self._img_size = img_size
        self.num_cls = num_cls
        self.stem = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=inner_features),
            ACT.build(act),
            nn.Linear(in_features=inner_features, out_features=num_cls)
        )

    @property
    def img_size(self):
        return self._img_size

    def forward(self, feat):
        feat = F.adaptive_avg_pool2d(feat, output_size=1)
        feat = feat.view(feat.size(0), -1)
        chot = self.stem(feat)
        return chot


class UInstLayerClsBdrMLP(nn.Module):
    def __init__(self, in_channels, inner_channels, num_cls=1, img_size=(224, 224), feat_size=(8, 8),
                 act=ACT.RELU, inner_features=1024):
        super().__init__()
        self._img_size = img_size
        self.num_cls = num_cls
        self.cvtor = Ck1s1BA(in_channels=in_channels, out_channels=inner_channels, act=act)
        in_features = inner_channels * feat_size[0] * feat_size[1]
        self.stem = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=inner_features),
            ACT.build(act),
        )
        self.head_border = nn.Sequential(
            nn.Linear(in_features=inner_features, out_features=inner_features),
            ACT.build(act),
            nn.Linear(in_features=inner_features, out_features=5)
        )
        self.head_cls = nn.Sequential(
            nn.Linear(in_features=inner_features, out_features=inner_features),
            ACT.build(act),
            nn.Linear(in_features=inner_features, out_features=num_cls)
        )

    @property
    def img_size(self):
        return self._img_size

    # @staticmethod
    # def decode(feat_xywhuv, img_size):
    #     img_size = torch.as_tensor(img_size).to(feat_xywhuv.device)
    #
    #     xy = torch.sigmoid(feat_xywhuv[..., :2].clamp(min=-4, max=4)) * img_size
    #     wh = torch.exp(feat_xywhuv[..., 2:4].clamp(min=-4, max=4)) * img_size / 2 + 1e-7
    #     uv = feat_xywhuv[..., 4:6]
    #     norm = torch.norm(uv, dim=1, keepdim=True)
    #     uv = torch.where(norm < 1e-7, torch.as_tensor([[1.0, 0.0]]).to(feat_xywhuv.device).expand_as(uv), uv / norm)
    #
    #     xywhuv = torch.cat([xy, wh, uv], dim=-1)
    #     return xywhuv

    @staticmethod
    def decode(feat_xywha, img_size):
        img_size = torch.as_tensor(img_size).to(feat_xywha.device)
        xy = torch.sigmoid(feat_xywha[..., :2].clamp(min=-4, max=4)) * img_size
        wh = torch.exp(feat_xywha[..., 2:4].clamp(min=-4, max=4)) * img_size / 2 + 1e-7
        xywha = torch.cat([xy, wh, feat_xywha[..., 4:5]], dim=-1)
        return xywha

    def forward(self, feat):
        feat = self.cvtor(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.stem(feat)
        border = self.head_border(feat)
        chot = self.head_cls(feat)
        border = UInstLayerClsBdrMLP.decode(border, self._img_size)
        return chot, border


class UInstLayerClsBdrPool(nn.Module):
    def __init__(self, in_channels, num_cls=1, img_size=(224, 224),
                 act=ACT.RELU, inner_features=1024):
        super().__init__()
        self._img_size = img_size
        self.num_cls = num_cls
        self.stem = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=inner_features),
            ACT.build(act),
        )
        self.head_border = nn.Sequential(
            nn.Linear(in_features=inner_features, out_features=inner_features),
            ACT.build(act),
            nn.Linear(in_features=inner_features, out_features=5)
        )
        self.head_cls = nn.Sequential(
            nn.Linear(in_features=inner_features, out_features=inner_features),
            ACT.build(act),
            nn.Linear(in_features=inner_features, out_features=num_cls)
        )

    @property
    def img_size(self):
        return self._img_size

    def forward(self, feat):
        feat = F.adaptive_avg_pool2d(feat, output_size=1)
        feat = feat.view(feat.size(0), -1)
        feat = self.stem(feat)
        border = self.head_border(feat)
        chot = self.head_cls(feat)
        border = UInstLayerClsBdrMLP.decode(border, self._img_size)
        return chot, border


# </editor-fold>

# <editor-fold desc='网络主干'>
class BottleneckV3(BottleneckV2):
    def __init__(self, in_channels, out_channels, stride=1, ratio=26 / 64, dilations=(1, 3, 5), act=ACT.RELU):
        super(BottleneckV3, self).__init__(
            in_channels, out_channels, stride=stride, ratio=ratio, dilations=dilations, act=act)


class UInstResUNetMain(nn.Module):

    def __init__(self, Module, repeat_nums_enc, repeat_nums_dec, pre_channels,
                 channelss, strides, num_cls=20, img_size=(224, 224), act=ACT.RELU, in_channels=3):
        super(UInstResUNetMain, self).__init__()
        self.img_size = img_size
        self.num_cls = num_cls
        self.strides = strides
        self.pre = CpaBA(in_channels=in_channels, out_channels=pre_channels, kernel_size=7, stride=1, bn=True,
                         act=act)
        self.enc = ResUNetEnc(Module, repeat_nums_enc, in_channels=pre_channels, out_channelss=channelss,
                              strides=strides, act=act)
        self.dec = ResUNetDec(Module, repeat_nums_dec, in_channelss=channelss, out_channelss=channelss,
                              strides=strides, act=act)
        self.out = Ck1s1(in_channels=channelss[0] + pre_channels, out_channels=num_cls)

    def forward(self, imgs, attn=None):
        feat = self.pre(imgs)
        feats = self.enc(feat)
        if attn is not None:
            attns = _gen_attns(attn, self.strides)
            for i in range(len(feats)):
                feats[i] = feats[i] * attns[i]

        feats_rec = self.dec(feats)
        feat_rec = feats_rec[0]
        if not self.strides[0] == 1:
            feat_rec = F.interpolate(feat_rec, scale_factor=self.strides[0])
        imgs_rec = self.out(torch.cat([feat, feat_rec], dim=1))
        # show_arrs(imgs_rec[:, 0])
        # plt.pause(1e5)
        return imgs_rec

    PARA_V2R50 = dict(Module=BottleneckV2, repeat_nums_enc=(2, 3, 4, 6, 3), repeat_nums_dec=[1, 2, 2, 3, 2],
                      channelss=(128, 256, 512, 1024, 2048), strides=(2, 2, 2, 2, 2), pre_channels=64)
    PARA_V3R50 = dict(Module=BottleneckV3, repeat_nums_enc=(2, 3, 4, 6, 3), repeat_nums_dec=[1, 2, 2, 3, 2],
                      channelss=(128, 256, 512, 1024, 2048), strides=(2, 2, 2, 2, 2), pre_channels=64)

    @staticmethod
    def R34(img_size=(224, 224), num_cls=2, act=ACT.RELU, in_channels=3, **kwargs):
        return UInstResUNetMain(**ResUNetMain.PARA_R34, img_size=img_size, act=act, num_cls=num_cls,
                                in_channels=in_channels)

    @staticmethod
    def R50(img_size=(224, 224), num_cls=2, act=ACT.RELU, in_channels=3, **kwargs):
        return UInstResUNetMain(**ResUNetMain.PARA_R50, img_size=img_size, act=act, num_cls=num_cls,
                                in_channels=in_channels)

    @staticmethod
    def V2R50(img_size=(224, 224), num_cls=2, act=ACT.RELU, in_channels=3, **kwargs):
        return UInstResUNetMain(**UInstResUNetMain.PARA_V2R50, img_size=img_size, act=act, num_cls=num_cls,
                                in_channels=in_channels)

    @staticmethod
    def V3R50(img_size=(224, 224), num_cls=2, act=ACT.RELU, in_channels=3, **kwargs):
        return UInstResUNetMain(**UInstResUNetMain.PARA_V3R50, img_size=img_size, act=act, num_cls=num_cls,
                                in_channels=in_channels)


class UInstConstMain(nn.Module):
    def __init__(self, batch_size, num_cls=1, img_size=(224, 224)):
        super().__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.featmaps = nn.Parameter(torch.zeros(batch_size, num_cls, img_size[1], img_size[0]))

    def forward(self, imgs, masks_attn=None, *args):
        return self.featmaps


class UInstConstClsMain(nn.Module):
    def __init__(self, batch_size, num_cls=1, img_size=(224, 224)):
        super().__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.masks = nn.Parameter(torch.zeros(batch_size, 2, img_size[1], img_size[0]))
        self.chots = nn.Parameter(torch.zeros(batch_size, num_cls))

    def forward(self, imgs, attns_hbb=None, *args):
        return self.masks, self.chots


class UInstConstClsBdrMain(nn.Module):
    def __init__(self, batch_size, num_cls=1, img_size=(224, 224)):
        super().__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.masks = nn.Parameter(torch.zeros(batch_size, 2, img_size[1], img_size[0]))
        self.chots = nn.Parameter(torch.zeros(batch_size, num_cls))
        self.xywhuv = nn.Parameter(torch.rand(batch_size, 6))

    def forward(self, imgs, attns_hbb=None, *args):
        xywhuv = UInstLayerClsBdrMLP.decode(self.xywhuv, self.img_size)
        return self.masks, self.chots, xywhuv


class UInstResUNetClsMLPMain(ResUNetMain):

    def __init__(self, Module, repeat_nums_enc, repeat_nums_dec, pre_channels, channelss, strides, img_size=(224, 224),
                 act=ACT.RELU,
                 in_channels=4, num_cls=3, inner_channels=512, inner_features=512):
        super(UInstResUNetClsMLPMain, self).__init__(Module, repeat_nums_enc, repeat_nums_dec, pre_channels, channelss,
                                                     strides, num_cls=2, img_size=img_size, act=act,
                                                     in_channels=in_channels, )

        self.layer = UInstLayerClsMLP(in_channels=channelss[-1], inner_channels=inner_channels,
                                      num_cls=num_cls, img_size=img_size,
                                      feat_size=(img_size[0] // self.stride, img_size[1] // self.stride), act=act,
                                      inner_features=inner_features)

    def forward(self, imgs, attns_hbb=None):
        feat = self.pre(imgs)
        feats = self.enc(feat)
        chot = self.layer(feats[-1])
        feats_rec = self.dec(feats)
        feat_rec = feats_rec[0]
        if not self.strides[0] == 1:
            feat_rec = F.interpolate(feat_rec, scale_factor=self.strides[0])
        masks = self.out(torch.cat([feat, feat_rec], dim=1))
        return masks, chot

    @staticmethod
    def R50(img_size=(224, 224), num_cls=3, act=ACT.RELU, **kwargs):
        return UInstResUNetClsMLPMain(**ResUNetMain.PARA_R50, img_size=img_size, act=act, num_cls=num_cls,
                                      in_channels=3, inner_channels=512, inner_features=1024)

    @staticmethod
    def R34(img_size=(224, 224), num_cls=3, act=ACT.RELU, **kwargs):
        return UInstResUNetClsMLPMain(**ResUNetMain.PARA_R34, img_size=img_size, act=act, num_cls=num_cls,
                                      in_channels=3, inner_channels=512, inner_features=1024)


class UInstResUNetClsPoolMain(ResUNetMain):

    def __init__(self, Module, repeat_nums_enc, repeat_nums_dec, pre_channels, channelss, strides, img_size=(224, 224),
                 act=ACT.RELU,
                 in_channels=4, num_cls=3, inner_features=512):
        super(UInstResUNetClsPoolMain, self).__init__(Module, repeat_nums_enc, repeat_nums_dec, pre_channels, channelss,
                                                      strides,
                                                      num_cls=2, img_size=img_size, act=act, in_channels=in_channels, )
        self.layer = UInstLayerClsPool(in_channels=channelss[-1], num_cls=num_cls, img_size=img_size,
                                       act=act, inner_features=inner_features)

    def forward(self, imgs, attns_hbb=None):
        feat = self.pre(imgs)
        feats = self.enc(feat)
        chot = self.layer(feats[-1])
        feats_rec = self.dec(feats)
        feat_rec = feats_rec[0]
        if not self.strides[0] == 1:
            feat_rec = F.interpolate(feat_rec, scale_factor=self.strides[0])
        masks = self.out(torch.cat([feat, feat_rec], dim=1))
        return masks, chot

    @staticmethod
    def R50(img_size=(224, 224), num_cls=3, act=ACT.RELU, **kwargs):
        return UInstResUNetClsPoolMain(**ResUNetMain.PARA_R50, img_size=img_size, act=act, num_cls=num_cls,
                                       in_channels=3, inner_features=1024)

    @staticmethod
    def V2R50(img_size=(224, 224), num_cls=3, act=ACT.RELU, **kwargs):
        return UInstResUNetClsPoolMain(**UInstResUNetMain.PARA_V2R50, img_size=img_size, act=act, num_cls=num_cls,
                                       in_channels=3, inner_features=1024)

    @staticmethod
    def V3R50(img_size=(224, 224), num_cls=3, act=ACT.RELU, **kwargs):
        return UInstResUNetClsPoolMain(**UInstResUNetMain.PARA_V3R50, img_size=img_size, act=act, num_cls=num_cls,
                                       in_channels=3, inner_features=1024)

    @staticmethod
    def R34(img_size=(224, 224), num_cls=3, act=ACT.RELU, **kwargs):
        return UInstResUNetClsPoolMain(**ResUNetMain.PARA_R34, img_size=img_size, act=act, num_cls=num_cls,
                                       in_channels=3, inner_features=1024)


class UInstResUNetClsBdrPoolMain(ResUNetMain):

    def __init__(self, Module, repeat_nums_enc, repeat_nums_dec, pre_channels, channelss, strides, img_size=(224, 224),
                 act=ACT.RELU, in_channels=4, num_cls=3, inner_features=512):
        super(UInstResUNetClsBdrPoolMain, self).__init__(Module, repeat_nums_enc, repeat_nums_dec, pre_channels,
                                                         channelss, strides,
                                                         num_cls=2, img_size=img_size, act=act,
                                                         in_channels=in_channels, )
        self.layer = UInstLayerClsBdrPool(in_channels=channelss[-1], num_cls=num_cls, img_size=img_size,
                                          act=act, inner_features=inner_features)

    def forward(self, imgs, attns_hbb=None):
        feat = self.pre(imgs)
        feats = self.enc(feat)
        chot, bdr = self.layer(feats[-1])
        feats_rec = self.dec(feats)
        feat_rec = feats_rec[0]
        if not self.strides[0] == 1:
            feat_rec = F.interpolate(feat_rec, scale_factor=self.strides[0])
        masks = self.out(torch.cat([feat, feat_rec], dim=1))
        return masks, chot, bdr

    @staticmethod
    def R50(img_size=(224, 224), num_cls=3, act=ACT.RELU, **kwargs):
        return UInstResUNetClsBdrPoolMain(**ResUNetMain.PARA_R50, img_size=img_size, act=act, num_cls=num_cls,
                                          in_channels=3, inner_features=1024)

    @staticmethod
    def V2R50(img_size=(224, 224), num_cls=3, act=ACT.RELU, **kwargs):
        return UInstResUNetClsBdrPoolMain(**UInstResUNetMain.PARA_V2R50, img_size=img_size, act=act, num_cls=num_cls,
                                          in_channels=3, inner_features=1024)

    @staticmethod
    def V3R50(img_size=(224, 224), num_cls=3, act=ACT.RELU, **kwargs):
        return UInstResUNetClsBdrPoolMain(**UInstResUNetMain.PARA_V3R50, img_size=img_size, act=act, num_cls=num_cls,
                                          in_channels=3, inner_features=1024)

    @staticmethod
    def R34(img_size=(224, 224), num_cls=3, act=ACT.RELU, **kwargs):
        return UInstResUNetClsBdrPoolMain(**ResUNetMain.PARA_R34, img_size=img_size, act=act, num_cls=num_cls,
                                          in_channels=3, inner_features=1024)


# </editor-fold>


# <editor-fold desc='导引层'>

def _gen_attns(attn, strides):
    attns = []
    for stride in strides:
        attn = F.avg_pool2d(attn, stride=stride, kernel_size=stride)
        attns.append(attn)
    return attns


class GuideLayerV0(nn.Module):
    def __init__(self, strides, img_size):
        super(GuideLayerV0, self).__init__()
        self.strides = strides
        self.img_size = img_size

    def forward(self, feats, bdr):
        attn = xywhasT2masksT_guss(bdr, self.img_size)[:, None, :, :]
        feats_rec = [None] * len(feats)
        attns = _gen_attns(attn, self.strides)
        for i in range(len(feats) - 1, -1, -1):
            feats_rec[i] = feats[i] * attns[i]
        return feats_rec


class GuideLayerV3(nn.Module):
    def __init__(self, channelss, strides, img_size):
        super(GuideLayerV3, self).__init__()
        self.strides = strides
        self.img_size = img_size
        self.cvtors = nn.ModuleList()
        for i, channels in enumerate(channelss):
            self.cvtors.append(Ck1s1(in_channels=channels + 1, out_channels=channels))

    def forward(self, feats, bdr):
        attn = xywhasT2masksT_guss(bdr, self.img_size)[:, None, :, :]
        feats_rec = [None] * len(feats)
        attns = _gen_attns(attn, self.strides)
        for i in range(len(feats) - 1, -1, -1):
            feat = torch.cat([feats[i], attns[i]], dim=1)
            feats_rec[i] = self.cvtors[i](feat)
        return feats_rec


class GuideLayerV1(nn.Module):
    def __init__(self, channelss, strides, img_size, reduce_ratio=16):
        super(GuideLayerV1, self).__init__()
        self.squzes = nn.ModuleList()
        self.cvtors = nn.ModuleList()
        self.unsquzes = nn.ModuleList()
        self.strides = strides
        self.img_size = img_size
        for i, channels in enumerate(channelss):
            inner_channels = channels // reduce_ratio
            self.squzes.append(Ck1s1(in_channels=channels, out_channels=inner_channels))
            self.cvtors.append(
                Cpa(in_channels=inner_channels + 1, out_channels=inner_channels, kernel_size=5, stride=1))
            self.unsquzes.append(Ck1s1(in_channels=inner_channels, out_channels=channels))

    def forward(self, feats, bdr):
        attn = xywhasT2masksT_guss(bdr, self.img_size)[:, None, :, :]
        attns = _gen_attns(attn, self.strides)
        feats_rec = [None] * len(feats)
        for i in range(len(feats) - 1, -1, -1):
            feat_sqzd = self.squzes[i](feats[i])
            feat_sqzd = torch.cat([attns[i], feat_sqzd], dim=1)
            feat_sqzd = self.cvtors[i](feat_sqzd)
            attn_mixd_i = self.unsquzes[i](feat_sqzd)
            feats_rec[i] = torch.sigmoid(attn_mixd_i) * feats[i]
        return feats_rec


class GuideLayerV2(nn.Module):
    def __init__(self, channelss, strides, img_size, reduce_ratio=16):
        super(GuideLayerV2, self).__init__()
        self.squzes = nn.ModuleList()
        self.cvtors = nn.ModuleList()
        self.unsquzes = nn.ModuleList()
        self.strides = strides
        self.img_size = img_size
        for i, channels in enumerate(channelss):
            inner_channels = channels // reduce_ratio
            concat_channels = inner_channels + 1 + (0 if i == len(channelss) - 1 else channelss[i + 1] // reduce_ratio)
            self.squzes.append(Ck1s1(in_channels=channels, out_channels=inner_channels))
            self.cvtors.append(Cpa(in_channels=concat_channels, out_channels=inner_channels, kernel_size=5, stride=1))
            self.unsquzes.append(Ck1s1(in_channels=inner_channels, out_channels=channels))

    def forward(self, feats, bdr):
        attn = xywhasT2masksT_guss(bdr, self.img_size)[:, None, :, :]
        attns = _gen_attns(attn, self.strides)
        feats_rec = [None] * len(feats)
        feat_sqzd_last = None
        for i in range(len(feats) - 1, -1, -1):
            feat_sqzd = self.squzes[i](feats[i])
            if i < len(feats) - 1:
                feat_sqzd_sld = F.interpolate(feat_sqzd_last, scale_factor=self.strides[i + 1])
                feat_sqzd = torch.cat([feat_sqzd_sld, attns[i], feat_sqzd], dim=1)
            else:
                feat_sqzd = torch.cat([attns[i], feat_sqzd], dim=1)
            feat_sqzd = self.cvtors[i](feat_sqzd)
            attn_mixd_i = self.unsquzes[i](feat_sqzd)
            feats_rec[i] = torch.sigmoid(attn_mixd_i) * feats[i]
            feat_sqzd_last = feat_sqzd
            # show_arrs(feats_rec[i][0, :64])
            # plt.pause(1e5)
        return feats_rec


# </editor-fold>


# <editor-fold desc='ds xml标签转化'>
# </editor-fold>

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = UInstResUNetMain.V3R50(img_size=(128, 128), num_cls=1)
    model.to(device)
    imgs = torch.zeros(2, 3, 128, 128).to(device)
    attn = torch.zeros(2, 1, 128, 128).to(device)
    # y = model(imgs)
    # torch.save(model.state_dict(), './test.pth')
    torch.onnx.export(model, imgs, f='./test.onnx', opset_version=11)
