from utils import *


# <editor-fold desc='分类'>
class OneStageClassifier(OneStageTorchModel, IndependentInferable):
    def __init__(self, backbone, device=None, pack=None, img_size=(224, 224), num_cls=10):
        super(OneStageClassifier, self).__init__(backbone=backbone, device=device, pack=pack)
        self.img_size = img_size
        self._num_cls = num_cls
        self.forward = self.imgs2labels

    @property
    def num_cls(self):
        return self._num_cls

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    def imgs2labels(self, imgs, cind2name=None, **kwargs):
        imgs, _ = imgs2imgsT(imgs, img_size=self.img_size)
        _, _, H, W = imgs.size()
        chotsT = self.pkd_modules['backbone'](imgs.to(self.device))
        chotsT = torch.softmax(chotsT, dim=-1)
        cates = chotsT2cates(chotsT, img_size=(W, H), cind2name=cind2name)
        return cates

    def labels2tars(self, labels, **kwargs):
        targets = []
        for label in labels:
            assert isinstance(label, CategoryLabel), 'class err ' + label.__class__.__name__
            targets.append(OneHotCategory.convert(label.category).chotN)
        targets = np.array(targets)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        imgs = imgs.to(self.device)
        pred = self.pkd_modules['backbone'](imgs)
        target = torch.as_tensor(targets).to(pred.device, non_blocking=True)
        loss = F.cross_entropy(pred, target, reduction='mean')
        return loss

    def grad_cam(self, imgs, cindsN=None):
        imgs, _ = imgs2imgsT(imgs, img_size=self.img_size)
        hmaps = grad_cam(self.backbone, modules=(self.backbone.pool,), imgs=imgs.to(self.device), cindsN=cindsN)
        return hmaps

    def grad_cam_visual(self, imgs, cts, cls2name=None):
        clses = self.imgs2clses(imgs, cls2name=cls2name)
        hmap = self.grad_cam(imgs, clses)
        imgs = torch.cat([imgs, hmap], dim=1)
        mclses = []
        for cls, ct in zip(clses, cts):
            mcls = copy.deepcopy(cls)
            mcls_2 = copy.deepcopy(ct)
            mcls_2['tcls'] = mcls_2['chots_tg']
            del mcls_2['chots_tg']
            del mcls_2['name']
            mcls.update(mcls_2)
            mclses.append(mcls)
        return imgs, mclses


def fwd_hook(module, data_input, data_output, dist, ind):
    fwd_buffer = data_input[0].detach()
    dist[ind] = fwd_buffer
    return None


def bkwd_hook(module, grad_input, grad_output, dist, ind):
    bkwd_buffer = grad_input[0]
    dist[ind] = bkwd_buffer
    return None


def grad_cam(model, modules, imgs, cindsN=None):
    model.eval()
    model.zero_grad()
    # 确定层
    fwd_buffer = {}
    bkwd_buffer = {}

    fwd_handlers = []
    bkwd_handlers = []
    for ind, module in enumerate(modules):
        fwd_handler = module.register_forward_hook(partial(fwd_hook, dist=fwd_buffer, ind=ind))
        bkwd_handler = module.register_backward_hook(partial(bkwd_hook, dist=bkwd_buffer, ind=ind))
        fwd_handlers.append(fwd_handler)
        bkwd_handlers.append(bkwd_handler)

    # 传播
    imgs = imgs.to(next(iter(model.parameters())).device)
    chotsT = model(imgs)
    cindsT = torch.softmax(chotsT, dim=-1)

    mask = torch.zeros_like(cindsT).to(cindsT.device)
    if cindsN is None:
        ids = torch.argmax(cindsT, dim=-1)
        mask[torch.arange(cindsT.size(0)), ids] = 1
    else:
        mask[torch.arange(cindsT.size(0)), cindsN] = 1
    torch.sum(cindsT * mask).backward()

    # 画图
    hmaps = torch.zeros(imgs.size(0), 1, imgs.size(2), imgs.size(3))
    for ind in range(len(modules)):
        fwd_data = fwd_buffer[ind]
        bkwd_data = bkwd_buffer[ind]
        pows = bkwd_data.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        hmap_ind = torch.clamp(torch.sum(pows * fwd_data, dim=1, keepdim=True), min=0)
        hmap_ind = F.interpolate(input=hmap_ind, size=(imgs.size(2), imgs.size(3)), mode='bicubic',
                                 align_corners=True).detach().cpu()
        hmaps += hmap_ind

    # 移除hook
    for fwd_handler in fwd_handlers:
        fwd_handler.remove()
    for bkwd_handler in bkwd_handlers:
        bkwd_handler.remove()
    return hmaps


# </editor-fold>

# <editor-fold desc='语义分割'>
class OneStageSegmentor(OneStageTorchModel, IndependentInferable):
    def __init__(self, backbone, device=None, pack=None, num_cls=10, img_size=(128, 128)):
        super(OneStageSegmentor, self).__init__(backbone=backbone, device=device, pack=pack)
        self.forward = self.imgs2labels
        self._num_cls = num_cls
        self._img_size = img_size

    @property
    def num_cls(self):
        return self._num_cls

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    @torch.no_grad()
    def imgs2labels(self, imgs, cind2name=None, conf_thres=0.4, sxy=40, srgb=10, num_infer=0, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        Nb, _, H, W = imgsT.size()
        maskssT = self.pkd_modules['backbone'](imgsT.to(self.device))
        maskssT = torch.softmax(maskssT, dim=1)
        labels = []
        for i, (masksT, imgT) in enumerate(zip(maskssT, imgsT)):
            if num_infer > 0:
                masksT = masksT_crf(imgT=imgT, masksT=masksT, sxy=sxy, srgb=srgb, num_infer=num_infer)
            segs = SegsLabel.from_masksT(masksT, num_cls=self.num_cls, conf_thres=conf_thres, cind2name=cind2name)
            labels.append(segs)
        return labels_rescale(labels, imgs, 1 / ratios)

    def labels2tars(self, labels, **kwargs):
        masks_tg = [np.zeros(shape=(0, self._img_size[1], self._img_size[0]), dtype=np.int32)]
        # time1=time.time()
        for label in labels:
            assert isinstance(label, RegionExportable), 'class err ' + label.__class__.__name__
            masksN = label.export_masksN_enc(img_size=self.img_size, num_cls=self.num_cls)
            masks_tg.append(masksN[None, ...])
        masks_tg = np.concatenate(masks_tg, axis=0, dtype=np.int32)
        # time2 = time.time()
        # print(time2-time1)
        return masks_tg

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        imgs = imgs.to(self.device)
        preds = self.pkd_modules['backbone'](imgs)

        masks_tg = torch.as_tensor(targets).to(preds.device, non_blocking=True).long()

        masks_tg_chot = torch.zeros(
            size=(masks_tg.size(0), self.num_cls + 1, masks_tg.size(1), masks_tg.size(2)),
            device=masks_tg.device)
        filler = torch.ones_like(masks_tg, device=masks_tg.device, dtype=torch.float32)
        masks_tg_chot.scatter_(dim=1, index=masks_tg[:, None, :, :], src=filler[:, None, :, :])
        loss = F.cross_entropy(preds, masks_tg_chot, reduction='mean')
        return loss


class OneStageBoxSupervisedSegmentor(OneStageTorchModel, SurpervisedInferable):
    def __init__(self, backbone, device=None, pack=None, num_cls=10, img_size=(128, 128)):
        self._num_cls = num_cls
        self._img_size = img_size
        super(OneStageBoxSupervisedSegmentor, self).__init__(backbone=backbone, device=device, pack=pack)
        self.forward = self.imgs_labels2labels

    @property
    def num_cls(self):
        return self._num_cls

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    def labels2masks_attn(self, labels):
        masks_attn = [np.zeros(shape=(0, self.img_size[1], self.img_size[0]))]
        for label in labels:
            msk = label.export_border_masksN_enc(img_size=self.img_size, num_cls=self.num_cls)
            masks_attn.append(msk[None])
        masks_attn = np.concatenate(masks_attn, axis=0)
        masks_attn = (masks_attn < self.num_cls).astype(np.float32)[:, None, :, :]
        return masks_attn

    @torch.no_grad()
    def imgs_labels2labels(self, imgs, labels, cind2name=None, conf_thres=0.4, sxy=80, srgb=10, num_infer=0,
                           only_inner=True, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        labels = labels_rescale(labels, imgsT, ratios)
        imgsT = imgsT.to(self.device)
        masks_attn = self.labels2masks_attn(labels)
        masks_attnT = torch.from_numpy(masks_attn).to(self.device).float()
        Nb, _, H, W = imgsT.size()
        maskssT = self.pkd_modules['backbone'](imgsT, masks_attnT)
        maskssT = torch.softmax(maskssT, dim=1)
        labels_pd = []
        for i, (masksT, label, imgT) in enumerate(zip(maskssT, labels, imgsT)):
            if num_infer > 0:
                masksT = masksT_crf(imgT=imgT, masksT=masksT, sxy=sxy, srgb=srgb, num_infer=num_infer)
            insts = InstsLabel.from_boxes_masksT_abs(
                boxes=label, masksT=masksT, conf_thres=conf_thres, cind=0, only_inner=only_inner)
            labels_pd.append(insts)
        return labels_rescale(labels_pd, imgs, 1 / ratios)

    def loader2labels_with_surp(self, loader, **kwargs):
        print('< Annotate >' + '  Length %d' % len(loader) + '  Batch %d' % loader.batch_size
              + '  ImgSize ' + str(loader.img_size))
        labels_anno = []
        for i, (imgs, labels_ds) in MEnumerate(loader, prefix='Annotating ', broadcast=print):
            labels_md = self.imgs_labels2labels(imgs=imgs, labels=labels_ds, **kwargs)
            for j, (label_md, label_ds) in enumerate(zip(labels_md, labels_ds)):
                label_md.meta = label_ds.meta
                label_md.kwargs = label_ds.kwargs
                label_md.ctx_from(label_ds)
                labels_anno.append(label_md)
        return labels_anno

    def labels2tars(self, labels, **kwargs):
        masks_attn = self.labels2masks_attn(labels)
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        xyxys_tg = [np.zeros(shape=(0, 4))]
        cinds_tg = [np.zeros(shape=(0,), dtype=np.int32)]
        for i, label in enumerate(labels):
            assert isinstance(label, BorderExportable) and isinstance(label, CategoryExportable)
            cinds = label.export_cindsN()
            xyxys = label.export_xyxysN()
            inds_b_pos.append(np.full(fill_value=i, shape=len(cinds)))
            xyxys_tg.append(xyxys)
            cinds_tg.append(cinds)
        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        xyxys_tg = np.concatenate(xyxys_tg, axis=0)
        cinds_tg = np.concatenate(cinds_tg, axis=0)
        targets = (inds_b_pos, xyxys_tg, cinds_tg, masks_attn)
        return targets


# </editor-fold>

# <editor-fold desc='级联模型'>
class CascadeDetector(IndependentInferable):

    @property
    def img_size(self):
        return self.regionprop.img_size

    @property
    def num_cls(self):
        return self.classifier.num_cls

    def __init__(self, regionprop, classifier, img_size_rpn, img_size_cls):
        self.regionprop = regionprop
        self.classifier = classifier
        self.img_size_rpn = img_size_rpn
        self.img_size_cls = img_size_cls

    def imgs2labels(self, imgs, conf_thres=0.3, iou_thres=0.45, with_classifier=False, expend_ratio=1.2, cind2name=None,
                    **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size_rpn)
        labels_rpn = self.regionprop.imgs2labels(imgs=imgsT, conf_thres=conf_thres, iou_thres=iou_thres)
        labels_rescale(labels_rpn, imgs=imgs, ratios=1 / ratios)
        if not with_classifier:
            return labels_rpn
        for img, label_rpn in zip(imgs, labels_rpn):
            if len(label_rpn) == 0:
                continue
            imgP = img2imgP(img)
            patchs = []
            for box in label_rpn:
                border_ext = copy.deepcopy(XYXYBorder.convert(box.border))
                border_ext.expend(ratio=expend_ratio).clip(xyxyN_rgn=np.array([0, 0, imgP.size[0], imgP.size[1]]))
                patchs.append(imgP.crop(border_ext.xyxyN.astype(np.int32)))
            patchsT, _ = imgs2imgsT(patchs, img_size=self.img_size_cls)
            cates = self.classifier.imgs2labels(imgs=patchsT, cind2name=cind2name)
            for cate, box in zip(cates, label_rpn):
                cate.category.conf_scale(box.category.conf)
                box.category = cate.category
                box.update(cate)
        return labels_rpn


class CascadeSegmentor(IndependentInferable):

    @property
    def img_size(self):
        return self.regionprop.img_size

    @property
    def num_cls(self):
        return self.segmentor.num_cls

    def __init__(self, regionprop, segmentor, img_size_rpn, img_size_cls):
        self.regionprop = regionprop
        self.segmentor = segmentor
        self.img_size_rpn = img_size_rpn
        self.img_size_cls = img_size_cls

    def imgs2labels(self, imgs, conf_thres=0.3, iou_thres=0.45, with_segmentor=False, expend_ratio=1.2, cind2name=None,
                    **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size_rpn)
        labels_rpn = self.regionprop.imgs2labels(imgs=imgsT, conf_thres=conf_thres, iou_thres=iou_thres)
        for img, label_rpn, ratio in zip(imgs, labels_rpn, ratios):
            label_rpn.linear(scale=[1 / ratio, 1 / ratio], img_size=img2size(img))
        if not with_segmentor:
            return labels_rpn
        labels = []
        for img, label_rpn in zip(imgs, labels_rpn):
            imgP = img2imgP(img)
            xyxys = []
            patchs = []
            for box in label_rpn:
                border_ext = copy.deepcopy(XYXYBorder.convert(box.border))
                border_ext.expend(ratio=expend_ratio).clip(xyxyN_rgn=np.array([0, 0, imgP.size[0], imgP.size[1]]))
                xyxy = border_ext.xyxyN.astype(np.int32)
                xyxys.append(xyxy)
                patchs.append(imgP.crop(xyxy))
            patchsT, ratios_seg = imgs2imgsT(patchs, img_size=self.img_size_cls)
            labels_seg = self.segmentor.imgs2labels(imgs=patchsT, cind2name=cind2name)
            label = SegsLabel(img_size=self.img_size)
            for segs, xyxy, ratio_seg in zip(labels_seg, xyxys, ratios_seg):
                for seg in segs:
                    seg.linear(scale=[1 / ratio_seg, 1 / ratio_seg], bias=xyxy[:2], img_size=img2size(img))
                    label.append(seg)
                    labels.append(label)
        return labels
# </editor-fold>
