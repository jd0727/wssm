from models.base.modules import *
from models.base.resnet import ResNetBkbn
from models.base.vgg import VGGBkbn


class ResNetBkbnMutiOut3(nn.Module):
    def __init__(self, Module, repeat_nums, channels=64, act=ACT.RELU):
        super().__init__()
        self.pre = CpaBA(in_channels=3, out_channels=64, kernel_size=7, stride=2, act=act)
        self.stage1 = ResNetBkbn.ModuleRepeat(Module, in_channels=64, out_channels=channels, stride=1,
                                              repeat_num=repeat_nums[0], act=act, with_pool=True)
        self.stage2 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels, out_channels=channels * 2, stride=2,
                                              repeat_num=repeat_nums[1], act=act, with_pool=False)
        self.stage3 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels * 2, out_channels=channels * 4, stride=2,
                                              repeat_num=repeat_nums[2], act=act, with_pool=False)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        return feats1, feats2, feats3


class ResNetBkbnMutiOut4(nn.Module):
    def __init__(self, Module, repeat_nums, channels=64, act=ACT.RELU):
        super(ResNetBkbnMutiOut4, self).__init__()
        self.pre = CpaBA(in_channels=3, out_channels=64, kernel_size=7, stride=2, act=act)
        self.stage1 = ResNetBkbn.ModuleRepeat(Module, in_channels=64, out_channels=channels, stride=1,
                                              repeat_num=repeat_nums[0], act=act, with_pool=True)
        self.stage2 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels, out_channels=channels * 2, stride=2,
                                              repeat_num=repeat_nums[1], act=act, with_pool=False)
        self.stage3 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels * 2, out_channels=channels * 4, stride=2,
                                              repeat_num=repeat_nums[2], act=act, with_pool=False)
        self.stage4 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels * 4, out_channels=channels * 8, stride=2,
                                              repeat_num=repeat_nums[3], act=act, with_pool=False)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        return feats1, feats2, feats3, feats4


class FPNDownStreamConcat(nn.Module):
    def __init__(self, in_channelss, out_channelss, act=ACT.RELU):
        super(FPNDownStreamConcat, self).__init__()
        assert len(in_channelss) == len(out_channelss), 'len err'
        self.cvts = nn.ModuleList()
        for i in range(len(in_channelss)):
            last_channels = in_channelss[i] if i == len(in_channelss) - 1 else in_channelss[i] + out_channelss[i + 1]
            self.cvts.append(Ck1s1(in_channels=last_channels, out_channels=out_channelss[i]))

    def forward(self, feats):
        assert len(feats) == len(self.cvts), 'len err'
        feat_buff = None
        feats_out = []
        for i in range(len(feats) - 1, -1, -1):
            if feat_buff is None:
                feat_buff = self.cvts[i](feats[i])
            else:
                feat_buff = self.cvts[i](torch.cat([feats[i], F.upsample(feat_buff, scale_factor=2)], dim=1))
            feats_out.append(feat_buff)
        feats_out = list(reversed(feats_out))
        return feats_out


class FPNDownStreamAdd(nn.Module):
    def __init__(self, in_channelss, out_channelss, act=ACT.RELU):
        super(FPNDownStreamAdd, self).__init__()
        self.cvts = nn.ModuleList()
        self.adprs = nn.ModuleList()
        for i in range(len(in_channelss)):
            self.cvts.append(Ck1s1BA(in_channels=in_channelss[i], out_channels=out_channelss[i], act=act))
            if i < len(in_channelss) - 1:
                adpr = nn.Identity() if out_channelss[i + 1] == out_channelss[i] else \
                    Ck1s1(in_channels=out_channelss[i + 1], out_channels=out_channelss[i])
                self.adprs.append(adpr)

    def forward(self, feats):
        assert len(feats) == len(self.cvts), 'len err'
        feat_buff = None
        feats_out = []
        for i in range(len(feats) - 1, -1, -1):
            if i == len(feats) - 1:
                feat_buff = self.cvts[i](feats[i])
            else:
                feat_buff = self.adprs[i](feat_buff)
                feat_buff = self.cvts[i](feats[i]) + F.upsample(feat_buff, scale_factor=2)
            feats_out.append(feat_buff)
        feats_out = list(reversed(feats_out))
        return feats_out


class FPNUpStreamAdd(nn.Module):
    def __init__(self, in_channelss, out_channelss, act=ACT.RELU):
        super(FPNUpStreamAdd, self).__init__()
        self.cvts = nn.ModuleList()
        self.mixor = nn.ModuleList()
        for i in range(len(in_channelss)):
            self.cvts.append(nn.Identity() if i == 0 else
                             Ck3(in_channels=out_channelss[i - 1], out_channels=in_channelss[i], stride=2))
            self.mixor.append(Ck3s1BA(in_channels=in_channelss[i], out_channels=out_channelss[i], act=act))

    def forward(self, feats):
        assert len(feats) == len(self.cvts), 'len err'
        feat_buff = None
        feats_out = []
        for i in range(len(feats)):
            if i == 0:
                feat_buff = self.mixor[i](feats[i])
            else:
                feat_buff = self.mixor[i](feats[i] + self.cvts[i](feat_buff))
            feats_out.append(feat_buff)
        return feats_out




