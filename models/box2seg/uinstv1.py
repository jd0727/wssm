from models.base.deeplab import DeepLabV3ResNetMain
from models.base.unet import UNetV1Main
from models.box2seg.modules import *
from models.funcational import *
from models.template import OneStageBoxSupervisedSegmentor


class UInstV1(OneStageBoxSupervisedSegmentor):
    def __init__(self, backbone, device=None, pack=None, num_cls=10, img_size=(128, 128), dilate_size=1, **kwargs):
        super(UInstV1, self).__init__(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)
        self.dilate_size = dilate_size

    @torch.no_grad()
    def imgs_labels2labels(self, imgs, labels, cind2name=None, conf_thres=0.4, only_inner=True, sxy=40, srgb=10,
                           num_infer=0, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        labels = labels_rescale(labels, imgsT, ratios)
        imgsT = imgsT.to(self.device)
        Nb, _, H, W = imgsT.size()
        masksT = self.pkd_modules['backbone'](imgsT)
        masksT = torch.softmax(masksT, dim=1)
        labels_pd = []
        for i, (maskT, label, imgT) in enumerate(zip(masksT, labels, imgsT)):
            if num_infer > 0:
                maskT = masksT_crf(imgT=imgT, masksT=maskT, sxy=sxy, srgb=srgb, num_infer=num_infer)

            insts = InstsLabel.from_boxes_masksT_abs(
                boxes=label, masksT=maskT, conf_thres=conf_thres, cind=0, only_inner=only_inner)
            labels_pd.append(insts)
        return labels_rescale(labels_pd, imgs, 1 / ratios)

    def labels2tars(self, labels, **kwargs):
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        xywhas_tg = [np.zeros(shape=(0, 5))]
        masks_tg = [np.zeros(shape=(0, self.img_size[1], self.img_size[0]), dtype=np.int32)]

        for i, label in enumerate(labels):
            assert isinstance(label, BorderExportable) and isinstance(label, CategoryExportable)
            inds_b_pos.append(np.full(fill_value=i, shape=len(label)))
            if len(label) > 0:
                masksN = label.export_masksN_enc(img_size=self.img_size, num_cls=self.num_cls)
                masksN = (masksN < self.num_cls).astype(np.float32)
                if self.dilate_size > 1:
                    kernel = np.ones(shape=(self.dilate_size, self.dilate_size))
                    masksN = cv2.dilate(masksN, kernel)
                masks_tg.append(masksN[None].astype(np.int32))
            xywhas = label.export_xywhasN()
            xywhas_tg.append(xywhas)

        masks_tg = np.concatenate(masks_tg, axis=0, dtype=np.int32)
        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        xywhas_tg = np.concatenate(xywhas_tg, axis=0)
        targets = (inds_b_pos, masks_tg, xywhas_tg)
        return targets

    def imgs_tars2loss(self, imgs, targets, with_frgd=True, **kwargs):
        inds_b_pos, masks_tg, xywhas_tg = targets
        masks_tg = torch.as_tensor(masks_tg).to(self.device, non_blocking=True).long()
        inds_b_pos = torch.as_tensor(inds_b_pos).to(self.device, non_blocking=True).long()
        xywhas_tg = torch.as_tensor(xywhas_tg).to(self.device, non_blocking=True).float()

        imgs = imgs.to(self.device)
        masks_pd = self.pkd_modules['backbone'](imgs)
        mask_pd_align = masks_pd[inds_b_pos]
        masks_pd_sft = torch.softmax(mask_pd_align, dim=1)

        proj_loss = xywhasT_mil(masks_pd_sft[:, 0], xywhas_tg, background=0, with_log=True,
                                spatial_scale=1.0)

        masks_fltr_expd = xyxysT2masksTb(xywhasT2xyxysT(xywhas_tg), self.img_size)
        masks_fltr = xywhasT2masksTb(xywhas_tg, self.img_size).long()
        smth_loss = crf_loss_with_img3(imgs[inds_b_pos], mask_pd_align, masks_fltr_expd, kernel_size=7, dilation=1, )
        # smth_loss = nei_loss_with_img2(imgs[inds_b_pos], mask_pd_align, fliter_in, kernel_size=3, dilation=1, **kwargs)

        mask_loss = F.cross_entropy(mask_pd_align, 1 - masks_tg * masks_fltr, reduction='mean')
        loss_dict = dict(
            proj=proj_loss,
            smth=smth_loss * 3,
            mask=mask_loss
        )
        return loss_dict

    @staticmethod
    def ResUNetR34(img_size=(224, 224), num_cls=1, device=None, pack=PACK.AUTO, **kwargs):
        backbone = UInstResUNetMain.R34(img_size=img_size, num_cls=2, act=ACT.RELU)
        return UInstV1(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size, **kwargs)

    @staticmethod
    def ResUNetR50(img_size=(224, 224), num_cls=1, device=None, pack=PACK.AUTO, **kwargs):
        backbone = UInstResUNetMain.R50(img_size=img_size, num_cls=2, act=ACT.RELU)
        return UInstV1(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size, **kwargs)

    @staticmethod
    def ResUNetV2R50(img_size=(224, 224), num_cls=1, device=None, pack=PACK.AUTO, **kwargs):
        backbone = UInstResUNetMain.V2R50(img_size=img_size, num_cls=2, act=ACT.RELU)
        return UInstV1(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size, **kwargs)

    @staticmethod
    def ResUNetV3R50(img_size=(224, 224), num_cls=1, device=None, pack=PACK.AUTO, **kwargs):
        backbone = UInstResUNetMain.V3R50(img_size=img_size, num_cls=2, act=ACT.RELU)
        return UInstV1(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size, **kwargs)

    @staticmethod
    def DeepLabV3R50(img_size=(224, 224), num_cls=1, device=None, pack=PACK.AUTO, **kwargs):
        backbone = DeepLabV3ResNetMain.R50(img_size=img_size, num_cls=2, act=ACT.RELU)
        return UInstV1(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size, **kwargs)

    @staticmethod
    def DeepLabV3R34(img_size=(224, 224), num_cls=1, device=None, pack=PACK.AUTO, **kwargs):
        backbone = DeepLabV3ResNetMain.R34(img_size=img_size, num_cls=2, act=ACT.RELU)
        return UInstV1(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size, **kwargs)

    @staticmethod
    def UNet(img_size=(224, 224), num_cls=1, device=None, pack=PACK.AUTO, **kwargs):
        backbone = UNetV1Main.Std(act=ACT.RELU, num_cls=2, img_size=img_size)
        return UInstV1(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size, **kwargs)

    @staticmethod
    def Const(img_size=(224, 224), num_cls=1, device=None, batch_size=4, **kwargs):
        backbone = UInstConstMain(img_size=img_size, num_cls=num_cls, batch_size=batch_size)
        return UInstV1(backbone=backbone, device=device, pack=PACK.NONE, num_cls=num_cls, img_size=img_size, **kwargs)


if __name__ == '__main__':
    imgs = torch.zeros(2, 3, 128, 128)
    attn = torch.zeros(2, 1, 128, 128)
    model = UInstResUNetMain.R34(img_size=(128, 128), num_cls=5)
    torch.onnx.export(model, args=(imgs, attn), f='./test.onnx', opset_version=11, )
