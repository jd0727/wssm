from models.box2seg.modules import *
from models.funcational import *
from models.template import OneStageBoxSupervisedSegmentor


class UInstV6(OneStageBoxSupervisedSegmentor):
    def __init__(self, backbone, device=None, pack=None, num_cls=10, img_size=(128, 128), dilate_size=1, **kwargs):
        super(UInstV6, self).__init__(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)
        self.dilate_size = dilate_size

    @torch.no_grad()
    def imgs_labels2labels(self, imgs, labels, cind2name=None, conf_thres=0.4, only_inner=False, sxy=40, srgb=10,
                           num_infer=0, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        labels = labels_rescale(labels, imgsT, ratios)
        imgsT = imgsT.to(self.device)
        Nb, _, H, W = imgsT.size()
        masksT = self.pkd_modules['backbone'](imgsT)
        masksT = torch.sigmoid(masksT)
        labels_pd = []
        for i, (maskT, label, imgT) in enumerate(zip(masksT, labels, imgsT)):
            if num_infer > 0:
                maskT = masksT_crf(imgT=imgT, masksT=maskT, sxy=sxy, srgb=srgb, num_infer=num_infer)
            insts = InstsLabel.from_boxes_masksT_abs(
                boxes=label, masksT=maskT, conf_thres=conf_thres, cind=None, only_inner=only_inner)
            labels_pd.append(insts)
        return labels_rescale(labels_pd, imgs, 1 / ratios)

    def labels2tars(self, labels, **kwargs):
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        xywhas_tg = [np.zeros(shape=(0, 5))]
        masks_tg = [np.zeros(shape=(0, self.img_size[1], self.img_size[0]), dtype=np.float32)]
        cinds_tg = [np.zeros(shape=(0,), dtype=np.int32)]
        for i, label in enumerate(labels):
            assert isinstance(label, BorderExportable) and isinstance(label, CategoryExportable)
            inds_b_pos.append(np.full(fill_value=i, shape=len(label)))
            cinds = label.export_cindsN()
            if len(label) > 0:
                masksN = label.export_masksN_enc(img_size=self.img_size, num_cls=self.num_cls)
                masksN = (masksN < self.num_cls).astype(np.float32)
                if self.dilate_size > 1:
                    kernel = np.ones(shape=(self.dilate_size, self.dilate_size))
                    masksN = cv2.dilate(masksN, kernel)
                masks_tg.append(masksN[None])
                cinds_tg.append(cinds)
            xywhas = label.export_xywhasN()
            xywhas_tg.append(xywhas)
        cinds_tg = np.concatenate(cinds_tg, axis=0)
        masks_tg = np.concatenate(masks_tg, axis=0)
        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        xywhas_tg = np.concatenate(xywhas_tg, axis=0)
        targets = (inds_b_pos, cinds_tg, masks_tg, xywhas_tg)
        return targets

    def imgs_tars2loss(self, imgs, targets, with_frgd=True, dyam_mask=None, **kwargs):
        inds_b_pos, cinds_tg, masks_tg, xywhas_tg = targets
        masks_tg = torch.as_tensor(masks_tg).to(self.device, non_blocking=True).float()
        cinds_tg = torch.as_tensor(cinds_tg).to(self.device, non_blocking=True).long()
        inds_b_pos = torch.as_tensor(inds_b_pos).to(self.device, non_blocking=True).long()
        xywhas_tg = torch.as_tensor(xywhas_tg).to(self.device, non_blocking=True).float()

        imgs = imgs.to(self.device)
        # xywhas_tg_expd = copy.deepcopy(xywhas_tg)
        # xywhas_tg_expd[:, 2:4] *= 1.1
        masks_fltr_expd = xyxysT2masksTb(xywhasT2xyxysT(xywhas_tg), self.img_size)
        masks_fltr = xywhasT2masksTb(xywhas_tg, self.img_size)
        masks_pd = self.pkd_modules['backbone'](imgs)
        mask_pd_align = masks_pd[inds_b_pos, cinds_tg]
        masks_pd_sft = torch.sigmoid(mask_pd_align)

        proj_loss = xywhasT_mil(masks_pd_sft, xywhas_tg, background=0, with_log=True,
                                spatial_scale=1.0)

        # smth_loss = nei_loss_with_img2(imgs[inds_b_pos], mask_pd_align[:, None], masks_fltr,
        #                                kernel_size=3, dilation=1, reduction='sum', **kwargs)
        smth_loss = crf_loss_with_img3(imgs[inds_b_pos], mask_pd_align[:, None], masks_fltr_expd, kernel_size=7,
                                       dilation=1, srgb=1, )
        loss_dict = dict(
            proj=proj_loss,
            smth=smth_loss * 3,
        )
        masks_tg = masks_tg * masks_fltr
        if with_frgd:
            mask_loss = F.binary_cross_entropy_with_logits(mask_pd_align, masks_tg, reduction='mean', )
            loss_dict['mask'] = mask_loss
        else:
            if dyam_mask is not None:
                dyam_mask = torch.Tensor(dyam_mask).to(self.device).float()
                dyam_mask = dyam_mask[cinds_tg]
                weight_bkgd = 1 - masks_fltr.float() + masks_fltr.float() * dyam_mask[:, None, None]
            else:
                weight_bkgd = 1 - masks_fltr.float()
            bkgd_loss = F.binary_cross_entropy_with_logits(
                mask_pd_align, masks_tg, reduction='mean', weight=weight_bkgd)
            loss_dict['bkgd'] = bkgd_loss
        return loss_dict

    @staticmethod
    def ResUNetR50(img_size=(224, 224), num_cls=1, device=None, pack=PACK.AUTO, **kwargs):
        backbone = UInstResUNetMain.R50(img_size=img_size, num_cls=num_cls, act=ACT.RELU)
        return UInstV6(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size, **kwargs)

    @staticmethod
    def ResUNetV2R50(img_size=(224, 224), num_cls=1, device=None, pack=PACK.AUTO, **kwargs):
        backbone = UInstResUNetMain.V2R50(img_size=img_size, num_cls=num_cls, act=ACT.RELU)
        return UInstV6(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size, **kwargs)

    @staticmethod
    def ResUNetV3R50(img_size=(224, 224), num_cls=1, device=None, pack=PACK.AUTO, **kwargs):
        backbone = UInstResUNetMain.V3R50(img_size=img_size, num_cls=num_cls, act=ACT.RELU)
        return UInstV6(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size, **kwargs)

    @staticmethod
    def Const(img_size=(224, 224), num_cls=1, device=None, batch_size=4, **kwargs):
        backbone = UInstConstMain(img_size=img_size, num_cls=num_cls, batch_size=batch_size)
        return UInstV6(backbone=backbone, device=device, pack=PACK.NONE, num_cls=num_cls, img_size=img_size, **kwargs)
