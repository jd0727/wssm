# import imgaug as ia
# import imgaug.augmenters as iaa

import albumentations as A
from scipy.ndimage import gaussian_filter

from .ropr import *


# random.seed(10)

# <editor-fold desc='基础标签变换'>

# 混合类别标签
def blend_cates(cate0, cate1, mix_rate=0.5):
    oc0 = OneHotCategory.convert(cate0.category)
    oc1 = OneHotCategory.convert(cate1.category)
    chot = (1 - mix_rate) * oc0.chotN + mix_rate * oc1.chotN
    cate = copy.deepcopy(cate0)
    cate.category = OneHotCategory(chotN=chot)
    return cate


# 混合检测类标签
def blend_cate_contss(cate_conts0, cate_conts1, mix_rate=0.5):
    cate_conts0 = copy.deepcopy(cate_conts0)
    cate_conts1 = copy.deepcopy(cate_conts1)
    for cate_cont in cate_conts0:
        cate_cont.category.conf_scale(1 - mix_rate)
    for cate_cont in cate_conts1:
        cate_cont.category.conf_scale(mix_rate)
        cate_conts0.append(cate_cont)
    return cate_conts0


# </editor-fold>


# <editor-fold desc='基本功能'>

# 默认图像输出为pil
class ITransform(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, imgs, labels):
        pass


class AugSeq(ITransform):
    def __init__(self, img_size, **kwargs):
        self.kwargs = kwargs
        self._img_size = img_size
        self.transform = self._build_transform(img_size, **self.kwargs)

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        if not self._img_size == img_size:
            self._img_size = img_size
            self.transform = self._build_transform(img_size, **self.kwargs)

    def __call__(self, imgs, labels):
        imgs, labels = self.transform(imgs, labels)
        return imgs, labels

    @abstractmethod
    def _build_transform(self, img_size, **kwargs):
        pass


# 标准tnsor输出
class ToTensor(ITransform):
    def __init__(self, concat=True):
        self.concat = concat

    def __call__(self, imgs, labels):
        for i in range(len(imgs)):
            imgs[i] = img2imgT(imgs[i])
        if self.concat:
            imgs = torch.cat(imgs, dim=0)
        return imgs, labels


# Albumentations增广接口
class AlbuCompose(ITransform):
    def __init__(self, *children, thres=10):
        self.transform = A.Compose(
            children, keypoint_params=dict(format='xy', remove_invisible=False, ))
        self.thres = thres

    def __call__(self, imgs, labels, force_apply: bool = False, **kwargs):
        imgs_aug, labels_aug = [], []
        for img, label in zip(imgs, labels):
            assert isinstance(label, ImageLabel), 'err fmt ' + label.__class__.__name__
            imgN = img2imgN(img)
            xlyl = label.extract_xlylN()
            kwargs = dict(image=imgN, keypoints=xlyl)  # A包图片节点定义不一致，进行简易坐标变换

            masksN = []
            if label.num_bool_chan > 0:
                maskN_enc = label.extract_maskNb_enc(index=1)
                masksN.append(maskN_enc)
            if label.num_chan > 0:
                maskN_val = label.extract_maskN()
                masksN.append(maskN_val)
            if len(masksN) > 0:
                masksN = np.concatenate(masksN, axis=2)
                kwargs.update(dict(mask=masksN))

            transformed = self.transform(**kwargs)
            img_aug, xlyl_aug = transformed['image'], transformed['keypoints']
            img_size = (img_aug.shape[1], img_aug.shape[0])
            label.refrom_xlylN(np.array(xlyl_aug), img_size)
            offset = 0
            if label.num_bool_chan > 0:
                maskN_enc_aug = transformed['mask'][..., 0:1]
                label.refrom_maskNb_enc(maskN_enc_aug, index=1)
                offset = 1
            if label.num_chan > 0:
                maskN_val_aug = transformed['mask'][..., offset:]
                label.refrom_maskN(maskN_val_aug)

            if isinstance(label, ImageItemsLabel):
                label.flit(xyxyN_rgn=np.array([0, 0, img_aug.shape[1], img_aug.shape[0]]), thres=self.thres)
            imgs_aug.append(img_aug)
            labels_aug.append(label)
        return imgs_aug, labels_aug


#  <editor-fold desc='Albumentations偏移修改'>

class AlbumentationsKeyPointPatch():
    def apply_to_keypoint(self, keypoint, **params):
        keypoint_ofst = (keypoint[0] - 0.5, keypoint[1] - 0.5, keypoint[2], keypoint[3])
        keypoint_trd = super(AlbumentationsKeyPointPatch, self).apply_to_keypoint(
            keypoint_ofst, **params)
        return (keypoint_trd[0] + 0.5, keypoint_trd[1] + 0.5, keypoint_trd[2], keypoint_trd[3])


class A_Flip(AlbumentationsKeyPointPatch, A.Flip):
    pass


class A_HorizontalFlip(AlbumentationsKeyPointPatch, A.HorizontalFlip):
    pass


class A_VerticalFlip(AlbumentationsKeyPointPatch, A.VerticalFlip):
    pass


class A_Affine(AlbumentationsKeyPointPatch, A.Affine):
    pass


class A_RandomRotate90(AlbumentationsKeyPointPatch, A.RandomRotate90):
    pass


# </editor-fold>


# class A_Affine(A.Affine):
#     def apply_to_keypoint(self, keypoint, matrix, scale, **params):
#         keypoint_ofst = (keypoint[0] - 0.5, keypoint[1] - 0.5, keypoint[2], keypoint[3])
#         keypoint_trd = super(A_Affine, self).apply_to_keypoint(keypoint_ofst, matrix, scale, **params)
#         return (keypoint_trd[0] + 0.5, keypoint_trd[1] + 0.5, keypoint_trd[2], keypoint_trd[3])


# torch增广接口
class TorchCompose(ITransform):
    def __init__(self, *args):
        super().__init__(args)

    def __call__(self, imgs, labels=None, **kwargs):
        imgs_aug = []
        for img, label in zip(imgs, labels):
            if isinstance(label, CategoryLabel):
                img_aug = super(TorchCompose, self).__call__(img2imgP(img))
            else:
                raise Exception('err fmt ' + label.__class__.__name__)
            imgs_aug.append(img_aug)
        return imgs_aug, labels


# 缩放最大边
class LargestMaxSize(ITransform):
    def __init__(self, max_size=(256, 256), resample=Image.ANTIALIAS, thres=10):
        self.max_size = max_size
        self.resample = resample
        self.thres = thres

    def __call__(self, imgs, labels):

        for i in range(len(imgs)):
            if img2size(imgs[i]) == self.max_size:
                continue
            imgP, label = img2imgP(imgs[i]), labels[i]
            assert isinstance(label, ImageLabel), 'fmt err ' + label.__class__.__name__
            imgP_scld, ratio = imgP_lmtsize(imgP, max_size=self.max_size, resample=self.resample)
            imgs[i] = imgP_scld
            if not ratio == 1:
                label.linear(scale=np.array([ratio, ratio]), size=imgP_scld.size)
                if isinstance(label, ImageItemsLabel):
                    label.flit(thres=self.thres)
        return imgs, labels


class LargestMaxSizeWithPadding(ITransform):
    def __init__(self, max_size=(256, 256), resample=Image.BILINEAR, thres=10):
        self.max_size = max_size
        self.resample = resample
        self.thres = thres

    def __call__(self, imgs, labels):
        for i in range(len(imgs)):
            imgP, label = img2imgP(imgs[i]), labels[i]
            assert isinstance(label, ImageLabel), 'fmt err ' + label.__class__.__name__
            imgs[i], ratio = imgP_lmtsize_pad(imgP, max_size=self.max_size, pad_val=PAD_CVAL, resample=self.resample)
            label.linear(scale=np.array([ratio, ratio]), size=imgs[i].size)
            if isinstance(label, BoxesLabel):
                label.flit(thres=self.thres)
        return imgs, labels


# 缩放最大边
class CenterRescale(ITransform):
    def __init__(self, size=(256, 256), expand_ratio=1.0, resample=cv2.INTER_LANCZOS4, thres=10):
        self.size = size
        self.resample = resample
        self.thres = thres
        self.expand_ratio = expand_ratio

    def __call__(self, imgs, labels):
        for i in range(len(imgs)):
            imgN = img2imgN(imgs[i])
            img_size = np.array((imgN.shape[1], imgN.shape[0]))
            size = np.array(self.size)
            ratio = min(size / img_size) * self.expand_ratio
            bias = size[0] / 2 - ratio * img_size / 2
            A = np.array([[ratio, 0, bias[0]], [0, ratio, bias[1]]]).astype(np.float32)
            imgN = cv2.warpAffine(imgN.astype(np.float32), A, size, flags=self.resample)
            imgs[i] = np.clip(imgN, a_min=0, a_max=255).astype(np.uint8)

            label = labels[i]
            label.linear(scale=[ratio, ratio], bias=bias, size=tuple(size))
            if isinstance(label, ImageItemsLabel):
                label.flit(thres=self.thres)
        return imgs, labels


# 组合增广
class Sequential(list, ITransform):
    def __init__(self, *item):
        super().__init__(item)

    def __call__(self, imgs, labels, **kwargs):
        for seq in self:
            imgs, labels = seq.__call__(imgs, labels)
        return imgs, labels


class ADD_TYPE:
    APPEND = 'append'
    REPLACE = 'replace'
    COVER = 'cover'
    COVER_SRC = 'cover_src'
    COVER_FIRST = 'cover_first'


# 图像混合基类
class MutiMixer(ITransform):
    def __init__(self, repeat=3.0, inupt_num=0, add_type=ADD_TYPE.APPEND):
        super().__init__()
        self.inupt_num = inupt_num
        self.repeat = repeat
        self.add_type = add_type

    def __call__(self, imgs, labels):
        if len(imgs) < self.inupt_num:
            return imgs, labels
        repeat = self.repeat
        if isinstance(repeat, float):
            repeat = int(np.ceil(repeat * len(imgs)))
        imgs_procd = []
        labels_procd = []
        inds_sampd = []
        for n in range(repeat):
            inds = np.random.choice(a=len(imgs), replace=False, size=self.inupt_num)
            imgs_c = [imgs[int(ind)] for ind in inds]
            labels_c = [labels[int(ind)] for ind in inds]
            # 执行混合
            img_procd, label_procd = self.forward(imgs_c, labels_c)
            imgs_procd.append(img_procd)
            labels_procd.append(label_procd)
            inds_sampd.append(inds)
            # 添加
        if self.add_type == ADD_TYPE.REPLACE:
            return imgs_procd, labels_procd
        elif self.add_type == ADD_TYPE.APPEND:
            imgs += imgs_procd
            labels += labels_procd
            return imgs, labels
        elif self.add_type == ADD_TYPE.COVER:
            cover_num = min(len(imgs), repeat)
            inds_dist = np.random.choice(a=len(imgs), replace=False, size=cover_num)
            inds_src = np.random.choice(a=repeat, replace=False, size=cover_num)
            for ind_dist, ind_src in zip(inds_dist, inds_src):
                imgs[ind_dist] = imgs_procd[ind_src]
                labels[ind_dist] = labels_procd[ind_src]
            return imgs, labels
        elif self.add_type == ADD_TYPE.COVER_SRC:
            for i, inds in enumerate(inds_sampd):
                ind_dist = np.random.choice(inds)
                imgs[ind_dist] = imgs_procd[i]
                labels[ind_dist] = labels_procd[i]
            return imgs, labels
        elif self.add_type == ADD_TYPE.COVER_FIRST:
            for i, inds in enumerate(inds_sampd):
                imgs[inds[0]] = imgs_procd[i]
                labels[inds[0]] = labels_procd[i]
            return imgs, labels
        else:
            raise Exception('err add type')

    @abstractmethod
    def forward(self, imgs, boxess):
        pass


# 按透明度混合
class MixAlpha(MutiMixer):
    def __init__(self, repeat=3.0, mix_rate=0.5, add_type=ADD_TYPE.COVER):
        super().__init__(repeat=repeat, inupt_num=2, add_type=add_type)
        self.mix_rate = mix_rate

    def forward(self, imgs, labels):
        img0 = img2imgP(imgs[0])
        img1 = img2imgP(imgs[1])
        img = Image.blend(img0, img1, self.mix_rate)
        if isinstance(labels[0], CategoryLabel):
            label = blend_cates(labels[0], labels[1], mix_rate=self.mix_rate)
        elif isinstance(labels[0], BoxesLabel) or \
                isinstance(labels[0], PointsLabel) or \
                isinstance(labels[0], SegsLabel):
            label = blend_cate_contss(labels[0], labels[1], mix_rate=self.mix_rate)
        else:
            raise Exception('err fmt ' + labels[0].__class__.__name__)
        return img, label


# 马赛克增广
class Mosaic(MutiMixer):
    def __init__(self, repeat=3.0, img_size=(416, 416), add_type=ADD_TYPE.COVER, pad_val=(0, 0, 0)):
        super().__init__(repeat=repeat, inupt_num=4, add_type=add_type)
        self.img_size = img_size
        self.pad_val = pad_val

    def forward(self, imgs, labels):
        imgs = [img2imgP(img) for img in imgs]
        w, h = self.img_size
        labels = copy.deepcopy(labels)  # 隔离
        wp = int((np.random.rand() * 0.5 + 0.25) * w)
        hp = int((np.random.rand() * 0.5 + 0.25) * h)
        # 定义box偏移量
        xyxys_rgn = np.array([
            [0, 0, wp, hp],
            [0, hp, wp, h],
            [wp, 0, w, hp],
            [wp, hp, w, h]
        ])
        # 定义img偏移量
        whs = np.array([img.size for img in imgs], dtype=np.float32)
        scales = np.random.uniform(0.6, 1, size=4) * np.sqrt(w * h / whs[:, 0] / whs[:, 1])
        whs *= scales[:, None]
        xyxys_img = np.array([
            [wp - whs[0, 0], hp - whs[0, 1], wp, hp],
            [hp - whs[1, 1], hp, wp, wp + whs[1, 0]],
            [wp, hp - whs[2, 1], wp + whs[2, 0], hp],
            [wp, hp, wp + whs[3, 0], hp + whs[3, 1]]
        ])
        # 整合
        img_sum = Image.new(mode='RGB', size=(w, h), color=self.pad_val)
        for img, label, scale, bias, xyxy_rgn, wh in zip(imgs, labels, scales, xyxys_img[:, :2], xyxys_rgn, whs):
            scale = np.array([scale, scale])
            if isinstance(label, BoxesLabel):
                label.linear(scale=scale, bias=bias, size=(w, h))
                label.flit(xyxyN_rgn=xyxy_rgn, thres=10)
            elif isinstance(label, PointsLabel):
                label.linear(scale=scale, bias=bias, size=(w, h))
                label.flit(xyxyN_rgn=xyxy_rgn)
            elif isinstance(label, SegsLabel):
                label.linear(scale=scale, bias=bias, size=(w, h))
                label.flit(xyxyN_rgn=xyxy_rgn)
            else:
                Exception('err fmt ' + label.__class__.__name__)
            img = img.resize(size=wh.astype(np.int32))
            img_sum.paste(img, box=tuple(bias.astype(np.int32)))

        label_sum = labels[0]
        for i in range(1, len(labels)):
            label_sum.extend(labels[i])
        label_sum.ctx_size = self.img_size
        return img_sum, label_sum


# 目标区域的裁剪混合
class CutMix(MutiMixer):
    def __init__(self, repeat=3.0, num_patch=2.0, add_type=ADD_TYPE.COVER):
        super().__init__(repeat=repeat, inupt_num=2, add_type=add_type)
        self.num_patch = num_patch

    # 随机生成区域
    def rand_patch(self, xyxy_rgn, thres=10):
        xyxy_rgn = np.array(xyxy_rgn)
        wh = xyxy_rgn[2:4] - xyxy_rgn[0:2]
        wh_patch = np.maximum((wh * np.random.uniform(low=0.5, high=1, size=2)), thres)
        wh_patch = np.minimum(wh_patch, wh).astype(np.int32)

        x1 = np.random.uniform(low=xyxy_rgn[0], high=xyxy_rgn[2] - wh_patch[0])
        y1 = np.random.uniform(low=xyxy_rgn[1], high=xyxy_rgn[3] - wh_patch[1])
        xyxy_patch = np.array([x1, y1, x1 + wh_patch[0], y1 + wh_patch[1]]).astype(np.int32)
        return xyxy_patch

        # 随机抠图并遮挡

    def cutmix_cates(self, imgs, cates):
        imgs = [img2imgP(img) for img in imgs]
        xyxy_patch = self.rand_patch((0, 0, imgs[1].size[0], imgs[1].size[1]))
        patch = imgs[1].crop(xyxy_patch)
        x = np.random.uniform(low=0, high=imgs[0].size[0] - patch[0])
        y = np.random.uniform(low=0, high=imgs[0].size[1] - patch[1])
        img = copy.deepcopy(imgs[0])
        img.paste(patch, box=(x, y))
        mix_rate = patch.size[0] * patch.size[1] / (imgs[0].size[0] * imgs[0].size[1])
        cate = blend_cates(cates[0], cates[1], mix_rate=mix_rate)
        return img, cate

    def cutmix_boxess(self, imgs, boxess):
        num_src = len(boxess[1])
        num_patch = int(np.ceil(self.num_patch * num_src)) \
            if isinstance(self.num_patch, float) else self.num_patch
        if num_patch == 0:
            return imgs[0], boxess[0]
        # 开始混合
        imgs = [img2imgP(img) for img in imgs]
        img_mixed = copy.deepcopy(imgs[0])
        boxes_mixed = copy.deepcopy(boxess[0])
        # 提取patch
        patches = []
        xyxys_patch = []
        boxes_patch = []
        # print(min(num_src, num_patch), num_patch)
        np.random.choice(size=min(num_src, num_patch), a=num_patch, replace=False)
        for i in range(num_patch):
            ind = np.random.choice(a=len(boxess[1]))
            xyxy = XYXYBorder.convert(boxess[1][ind].border).xyxyN
            patches.append(imgs[1].crop(xyxy))
            boxes_patch.append(copy.deepcopy(boxess[1][ind]))
            xyxys_patch.append(xyxy)
        xyxys_patch = np.array(xyxys_patch, dtype=np.int32)
        xyxys_dist = boxess[0].extract_xyxysN()
        irate = ropr_mat_xyxysN(xyxys_patch, xyxys_dist, opr_type=OPR_TYPE.IRATE)
        # 放置patch
        for i in range(num_patch):
            if np.max(irate[i, :]) > 0.2 or np.any(np.array(patches[i].size) < 10):  # 防止新粘贴的图像影响原有目标
                continue
            # 放置patch
            img_mixed.paste(patches[i], box=tuple(xyxys_patch[i, :2]))
            boxes_mixed.append(boxes_patch[i])
        return img_mixed, boxes_mixed

    def forward(self, imgs, labels):
        if isinstance(labels[0], CategoryLabel):
            img, label = self.cutmix_cates(imgs, labels)
        elif isinstance(labels[0], BoxesLabel):
            img, label = self.cutmix_boxess(imgs, labels)
        else:
            raise Exception('err fmt ' + labels[0].__class__.__name__)
        return img, label


# </editor-fold>

# <editor-fold desc='lean扩展'>
class ExpendAlpha(ITransform):
    def __init__(self):
        super().__init__()

    def __call__(self, imgs, labels):
        for img, boxes in zip(imgs, labels):
            assert isinstance(boxes, ImageItemsLabel), 'fmt err ' + boxes.__class__.__name__
            for j, box in enumerate(boxes):
                if isinstance(box, BoxItem) or isinstance(box, InstItem):
                    box.border = XYWHABorder.convert(box.border)
        return imgs, labels


class RandMaskBox(ITransform):
    def __init__(self, ratio=0.2):
        super().__init__()
        self.ratio = ratio

    def __call__(self, imgs, labels):
        imgs_aug = []
        labels_aug = []
        for img, boxes in zip(imgs, labels):
            assert isinstance(boxes, BoxesLabel), 'fmt err ' + boxes.__class__.__name__
            imgP = img2imgP(img)
            boxes_aug = boxes.empty()
            for box in boxes:
                if np.random.rand() < self.ratio:
                    xywha = XYWHABorder.convert(box.border).xywhaN
                    imgP = imgP_fill_xywhaN(imgP, xywha, color=PAD_CVALS)
                else:
                    boxes_aug.append(box)
            labels_aug.append(boxes_aug)
            imgs_aug.append(imgP)
        return imgs_aug, labels_aug


class FlipBox(ITransform):
    def __init__(self, wflip=0.5, hflip=0.5):
        super(FlipBox, self).__init__()
        self.wflip = wflip
        self.hflip = hflip

    def _flipbox_pnt(self, xy0, a0, xy, a, wflip=False, hflip=False):
        if not wflip and not hflip:
            return xy, a
        elif wflip and hflip:
            return 2 * xy0 - xy, a + math.pi
        ar0 = a0 if wflip else a0 + math.pi / 2
        v = np.array([-np.cos(ar0), np.sin(ar0)])
        scale = np.dot(xy - xy0, v) * 2
        return xy - scale * v, 2 * ar0 - a

    def _flipbox_boxes(self, imgP, boxes):
        xywhas = boxes.export_xywhasN(aname_bdr='border_ref')
        for j, box in enumerate(boxes):
            assert isinstance(box, InstRefItem), 'fmt err ' + box.__class__.__name__
            wflip = np.random.rand() < self.wflip
            hflip = np.random.rand() < self.hflip
            if not wflip and not hflip:
                continue
            xywha = box.border_ref.xywhaN
            xywhas_other = np.concatenate([xywhas[:j], xywhas[(j + 1):]], axis=0)
            iareas = ropr_arr_xywhasN(
                np.broadcast_to(xywha, shape=xywhas_other.shape), xywhas_other, opr_type=OPR_TYPE.IAREA)
            if np.any(iareas > 0):
                continue
            imgP = imgP_rflip_paste_xywhaN(imgP, xywha, vflip=wflip, flip=hflip)
            box.border.xywhaN[:2], box.border.xywhaN[4] = self._flipbox_pnt(
                xy0=xywha[:2], a0=xywha[4], xy=box.border.xywhaN[:2], a=box.border.xywhaN[4], wflip=wflip, hflip=hflip)

        return imgP

    def _flipbox_insts(self, imgP, insts):
        xywhas = insts.export_xywhasN(aname_bdr='border_ref')
        for j, inst in enumerate(insts):
            assert isinstance(inst, InstRefItem), 'fmt err ' + inst.__class__.__name__
            wflip = np.random.rand() < self.wflip
            hflip = np.random.rand() < self.hflip
            if not wflip and not hflip:
                continue
            xywha = inst.border_ref.xywhaN
            xywhas_other = np.concatenate([xywhas[:j], xywhas[(j + 1):]], axis=0)
            iareas = ropr_arr_xywhasN(
                np.broadcast_to(xywha, shape=xywhas_other.shape), xywhas_other, opr_type=OPR_TYPE.IAREA)
            if np.any(iareas > 0):
                continue
            imgP = imgP_rflip_paste_xywhaN(imgP, xywha, vflip=wflip, flip=hflip)
            maskP = inst.rgn.maskP
            maskP = imgP_rflip_paste_xywhaN(maskP, xywha, vflip=wflip, flip=hflip)
            inst.rgn = AbsBoolRegion(maskP)
            inst.border.xywhaN[:2], inst.border.xywhaN[4] = self._flipbox_pnt(
                xy0=xywha[:2], a0=xywha[4], xy=inst.border.xywhaN[:2], a=inst.border.xywhaN[4], wflip=wflip,
                hflip=hflip)
        return imgP

    def __call__(self, imgs, labels):
        for i, (img, label) in enumerate(zip(imgs, labels)):
            imgP = img2imgP(img)
            if isinstance(label, BoxesLabel):
                imgP = self._flipbox_boxes(imgP, label)
            elif isinstance(label, InstsLabel):
                imgP = self._flipbox_insts(imgP, label)
            else:
                raise Exception('fmt err ' + label.__class__.__name__)
            imgs[i] = imgP
        return imgs, labels


class CutMixRotate(ITransform):
    def __init__(self, reapet_num=1, samp_ratio=0.3):
        super(CutMixRotate, self).__init__()
        self.reapet_num = reapet_num
        self.samp_ratio = samp_ratio

    def __call__(self, imgs, labels):
        # 取patch
        patches = []
        boxes_ptch = []
        for i, (imgP, boxes) in enumerate(zip(imgs, labels)):
            imgP = img2imgP(imgP)
            for box in boxes:
                assert isinstance(box, BoxRefItem), 'fmt err ' + box.__class__.__name__
                if np.random.rand() < self.samp_ratio and box['long']:
                    box_ptch = copy.deepcopy(box)
                    xywha = box_ptch.border.xywhaN
                    box_ptch.border_ref.xywhaN = xywha
                    w_scale = np.random.uniform(0.2, 1)
                    xywha[2] *= w_scale
                    assert xywha[2] * xywha[3] > 0, 'err'
                    patch = imgP_crop_xywhaN(imgP, xywha)
                    patches.append(patch)
                    boxes_ptch.append(box_ptch)

        # 随机复制
        num_ptch = len(patches)
        for i in range(int((self.reapet_num - 1) * num_ptch)):
            ptr = np.random.randint(low=0, high=num_ptch)
            patches.append(copy.deepcopy(patches[ptr]))
            boxes_ptch.append(copy.deepcopy(boxes_ptch[ptr]))

        # 随机增广
        for i, (box_ptch, patch) in enumerate(zip(boxes_ptch, patches)):
            theta = np.random.uniform(-0.5, 0.5) * np.pi
            patch = patch.convert('RGBA')
            patch = patch.rotate(-theta / np.pi * 180, expand=True)
            pw, ph = patch.size
            scale = np.random.uniform(0.5, 1.5)
            pw_new, ph_new = int(pw * scale), int(ph * scale)
            patch = patch.resize((pw_new, ph_new))

            xywha = box_ptch.border.xywhaN
            xywha[4] = (xywha[4] + theta) % np.pi
            xywha[2] = xywha[2] * scale
            xywha[3] = xywha[3] * scale
            xywha[0] = pw_new / 2
            xywha[1] = ph_new / 2
            patches[i] = patch
        # return patches,[[p] for p in boxes_ptch]

        # 随机放置
        for i, (box_ptch, patch) in enumerate(zip(boxes_ptch, patches)):
            ptr = np.random.randint(low=0, high=len(imgs))
            imgP = img2imgP(imgs[ptr])
            xywhas = labels[ptr].export_xywhasN()

            w, h = imgP.size
            pw, ph = patch.size
            px1 = int(np.random.uniform(0, w - pw))
            py1 = int(np.random.uniform(0, h - ph))
            pxc = px1 + pw // 2
            pyc = py1 + ph // 2
            xywha = box_ptch.border.xywhaN
            xywha[0] = pxc
            xywha[1] = pyc
            iareas = ropr_arr_xywhasN(np.broadcast_to(xywha, shape=xywhas.shape), xywhas,
                                      opr_type=OPR_TYPE.IAREA)
            if not np.any(iareas > 0):
                r, g, b, a = patch.split()
                imgP.paste(patch, box=(px1, py1), mask=a)
                imgs[ptr] = imgP
                labels[ptr].append(box_ptch)
        return imgs, labels


#
#
# </editor-fold>

#  <editor-fold desc='增广序列'>
PAD_CVAL = 127
PAD_CVALS = (PAD_CVAL, PAD_CVAL, PAD_CVAL)


class MODE:
    CONSTANT = 'constant'
    REFLECT = 'reflect'

    MODE_MAPPER_CV2 = {
        CONSTANT: cv2.BORDER_CONSTANT,
        REFLECT: cv2.BORDER_REFLECT,
    }

    @staticmethod
    def mode2cv2(mode):
        return MODE.MODE_MAPPER_CV2[mode]


class AugToTensor(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugToTensor, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, **kwargs):
        return ToTensor()


class AugLimtSize(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugLimtSize, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, **kwargs):
        return LargestMaxSize(max_size=img_size)


class AugFlip(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugFlip, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, to_tensor=True, **kwargs):
        trans = [AlbuCompose(A.Flip(always_apply=True), )]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugFlipRot(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugFlipRot, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, to_tensor=True, **kwargs):
        trans = [AlbuCompose(A.Flip(always_apply=True), A.RandomRotate90(always_apply=True))]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugRigid(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugRigid, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, to_tensor=True, uni_size=True, percent=(-0.33, 0), p=0.5, thres=1,
                         mode=MODE.CONSTANT, **kwargs):
        W, H = img_size
        mode_cv2 = MODE.mode2cv2(mode)
        trans = [LargestMaxSize(max_size=img_size)] if uni_size else []
        a_trans = [A.PadIfNeeded(min_height=H, min_width=W, value=PAD_CVALS, border_mode=mode_cv2, always_apply=True)] \
            if uni_size else []
        a_trans += [A.Flip(always_apply=True),
                    A.RandomRotate90(always_apply=True),
                    A.RandomResizedCrop(H, W, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1,
                                        always_apply=False, p=1.0),
                    # A.CropAndPad(percent=percent, keep_size=True, pad_mode=mode_cv2, p=p, sample_independently=False,
                    #              interpolation=cv2.INTER_CUBIC)
                    ]
        trans.append(AlbuCompose(*a_trans, thres=thres))
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugUInstV0(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugUInstV0, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, to_tensor=True, thres=1, **kwargs):
        W, H = img_size
        a_trans = [A.Flip(always_apply=True),
                   A.RandomResizedCrop(H, W, scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333),
                                       interpolation=cv2.INTER_CUBIC, always_apply=True),
                   ]
        trans = [AlbuCompose(*a_trans, thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class ExchangeMix(MutiMixer):
    def __init__(self, repeat=0.3, add_type=ADD_TYPE.COVER_SRC):
        super().__init__(repeat=repeat, inupt_num=2, add_type=add_type)

    def forward(self, imgs, labels):
        img0 = img2imgN(imgs[0])
        mask0 = labels[0][0].rgn.maskNb[..., None]

        img1 = img2imgN(imgs[1])
        img1 = cv2.resize(img1, labels[0].img_size)

        mask1 = labels[1][0].rgn.maskNb.astype(np.float32)
        mask1 = cv2.resize(mask1, labels[0].img_size)

        mask_join = (mask0 + mask1[..., None] > 0.5).astype(np.float32)
        mask_fltd = gaussian_filter(mask_join, sigma=1)
        img0_ed = img0 * mask_fltd + img1 * (1 - mask_fltd)
        img0_ed = img0_ed.astype(np.uint8)
        return img0_ed, labels[0]


class AugUInstV4(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugUInstV4, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, to_tensor=True, percent=(-0.33, 0), p=0.5, thres=1,
                         mode=MODE.REFLECT, with_exchange=True, prob=0.3, **kwargs):
        W, H = img_size
        mode_cv2 = MODE.mode2cv2(mode)
        trans = []
        a_trans = [
            A_Flip(always_apply=True),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=p),
            # A.RandomRotate90(always_apply=True),
            A_Affine(rotate=(0, 360), cval=PAD_CVALS, mode=mode_cv2, interpolation=cv2.INTER_LANCZOS4, p=p, ),
            A.RandomResizedCrop(H, W, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333),
                                interpolation=cv2.INTER_LANCZOS4,
                                always_apply=True),
            # A.Resize(height=H, width=W, interpolation=cv2.INTER_LANCZOS4)
        ]
        trans.append(AlbuCompose(*a_trans, thres=thres))
        if with_exchange:
            trans.append(ExchangeMix(repeat=prob))
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugUInstAnno(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugUInstAnno, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, to_tensor=True, thres=1, p=0.5, mode=MODE.CONSTANT, **kwargs):
        W, H = img_size
        trans = [
            AlbuCompose(
                A_Flip(always_apply=True),
                # A_Affine(rotate=(-90, 90), cval=PAD_CVALS, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_LANCZOS4,
                #          p=p, ),
                A_RandomRotate90(always_apply=True),
                # A.Resize(H, W, interpolation=cv2.INTER_LANCZOS4, always_apply=True),
                A.RandomResizedCrop(H, W, scale=(0.85, 0.85), ratio=(1, 1), interpolation=cv2.INTER_LANCZOS4,
                                    always_apply=True),
                thres=thres),
        ]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugUInstNorm(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugUInstNorm, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, thres=1, to_tensor=True, mode=MODE.CONSTANT, **kwargs):
        W, H = img_size
        # trans = [
        #     CenterRescale(size=img_size, expand_ratio=1.1, thres=thres), ]
        trans = [
            AlbuCompose(
                A.RandomResizedCrop(H, W, scale=(0.85, 0.85), ratio=(1, 1), interpolation=cv2.INTER_LANCZOS4,
                                    always_apply=True),
                thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugV2(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugV2, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, thres=1, to_tensor=True, p=1, mode=MODE.CONSTANT, **kwargs):
        W, H = img_size
        mode_cv2 = MODE.mode2cv2(mode)
        trans = [LargestMaxSize(max_size=img_size),
                 AlbuCompose(
                     A.PadIfNeeded(min_height=H, min_width=W, value=PAD_CVALS, border_mode=mode_cv2,
                                   always_apply=True),
                     A.Affine(rotate=(-10, 10), shear=(-5, 5), scale=(0.8, 1.1),
                              translate_percent=(-0.1, 0.1), cval=PAD_CVALS, p=p, mode=mode_cv2),
                     A.HorizontalFlip(p=p),
                     thres=thres), ]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


def imgP_affine(imgP, scale=1.0, angle=0.0, shear=0.0, resample=Image.BICUBIC):
    img_size = np.array(imgP.size)
    img_size_scled = (img_size * scale).astype(np.int32)
    A = np.array([[np.cos(angle + shear), np.sin(angle + shear)],
                  [-np.sin(angle - shear), np.cos(angle - shear)]]) * scale
    Ai = np.linalg.inv(A)
    bi = img_size / 2 - Ai @ img_size_scled / 2
    data = [Ai[0, 0], Ai[0, 1], bi[0], Ai[1, 0], Ai[1, 1], bi[1]]
    imgP = imgP.transform(size=tuple(img_size_scled), data=data,
                          method=Image.AFFINE, resample=resample, )
    return imgP


class MaskCutMix(MutiMixer):
    def __init__(self, repeat=3.0, num_patch=2, add_type=ADD_TYPE.COVER_FIRST,
                 scale=(0.5, 2), rotate=(0, np.pi), with_frgd=True, thres=32, samp_dct=None, ):
        super().__init__(repeat=repeat, inupt_num=2, add_type=add_type)

        self.scale = scale
        self.rotate = rotate
        self.thres = thres
        self.with_frgd = with_frgd
        self.num_patch = num_patch
        self.samp_dct = samp_dct

    def forward(self, imgs, labels):
        imgs = [img2imgP(img) for img in imgs]
        # 生成采样
        pieces = []
        masks = []
        insts = []
        for i, (img, label) in enumerate(zip(imgs, labels)):
            for inst in label:
                name = inst['name']
                difficult = inst['difficult']
                measure = inst.border.measure()
                if difficult:
                    continue
                if np.random.rand() > self.samp_dct[name]:
                    continue
                if self.thres > 0 and measure < self.thres:
                    continue
                xywh = XYWHBorder.convert(inst.border).xywhN
                xywh[2:4] = max(xywh[2:4]) * 1.2
                xyxy = xywhN2xyxyN(xywh).astype(np.int32)
                piece = img.crop(tuple(xyxy))
                rgn = copy.deepcopy(inst.rgn if self.with_frgd else inst.border)
                rgn.linear(bias=-xyxy[:2], size=xyxy[2:4] - xyxy[:2])
                mask = imgN2imgP(rgn.maskNb.astype(np.float32) * 255)
                pieces.append(piece)
                masks.append(mask)
                insts.append(inst)
        num_patch_crop = len(pieces)
        img, label = imgs[0], copy.deepcopy(labels[0])
        if num_patch_crop == 0:
            return img, label

        xyxys = label.export_xyxysN()
        inds_src = np.random.choice(num_patch_crop, size=min(num_patch_crop, self.num_patch))
        for ind_src in inds_src:
            piece, mask, inst = pieces[ind_src], masks[ind_src], copy.deepcopy(insts[ind_src])
            ratio = np.random.uniform(low=self.scale[0], high=self.scale[1])
            angle = np.random.uniform(low=self.rotate[0], high=self.rotate[1]) if self.with_frgd else 0
            piece = imgP_affine(piece, scale=ratio, angle=angle)
            mask = imgP_affine(mask, scale=ratio, angle=angle)

            xyxy_samp = RefValRegion._maskNb2xyxyN(np.array(mask) > 0)
            wh = xyxy_samp[2:4] - xyxy_samp[:2]
            measure = np.sqrt(np.prod(wh))
            if np.any(wh >= np.array(img.size)) or measure < self.thres:
                continue
            xy = [np.random.randint(low=0, high=img.size[0] - wh[0]),
                  np.random.randint(low=0, high=img.size[1] - wh[1])]
            xy_paste = xy - xyxy_samp[:2]
            xyxy_samp = xyxy_samp + np.array([xy_paste[0], xy_paste[1], xy_paste[0], xy_paste[1]])
            iareas = ropr_arr_xyxysN(np.repeat(xyxy_samp[None, :], axis=0, repeats=xyxys.shape[0]), xyxys,
                                     opr_type=OPR_TYPE.IAREA)
            if np.any(iareas > 0): continue
            xyxys = np.concatenate([xyxys, xyxy_samp[None, :]], axis=0)
            kernel = np.ones(shape=(int(measure / 10), int(measure / 10)))
            mask = cv2.dilate(np.array(mask), kernel)
            mask = gaussian_filter(np.array(mask), sigma=1)
            mask = imgN2imgP(mask)
            img.paste(piece, tuple(xy_paste), mask=mask)
            # print('paste', inst)
            inst.rgn = RefValRegion(xyN=xy_paste, maskN_ref=np.array(mask), size=label.img_size)
            inst.border = XYXYBorder(xyxy_samp, size=label.img_size)
            label.append(inst)
        # print('output', len(label))
        return img, label


class AugNorm(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugNorm, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, thres=1, to_tensor=True, mode=MODE.CONSTANT, **kwargs):
        W, H = img_size
        mode_cv2 = MODE.mode2cv2(mode)
        trans = [LargestMaxSize(max_size=img_size),
                 AlbuCompose(A.PadIfNeeded(min_height=H, min_width=W, value=PAD_CVALS, border_mode=mode_cv2,
                                           always_apply=True), thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugV1(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugV1, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, thres=1, to_tensor=True, p=1, mode=MODE.CONSTANT, **kwargs):
        W, H = img_size
        mode_cv2 = MODE.mode2cv2(mode)
        trans = [LargestMaxSize(max_size=img_size),
                 AlbuCompose(
                     A.PadIfNeeded(min_height=H, min_width=W, value=PAD_CVALS, border_mode=mode_cv2,
                                   always_apply=True),
                     A.Affine(rotate=(-10, 10), scale=(0.8, 1), cval=PAD_CVALS, p=p, mode=mode_cv2),
                     A.HorizontalFlip(p=p),
                     thres=thres), ]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugV1R(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugV1R, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, thres=1, to_tensor=True, p=1, mode=MODE.REFLECT, **kwargs):
        W, H = img_size
        mode_cv2 = MODE.mode2cv2(mode)
        trans = [LargestMaxSize(max_size=img_size),
                 AlbuCompose(
                     A.PadIfNeeded(min_height=H, min_width=W, value=PAD_CVALS, border_mode=mode_cv2,
                                   always_apply=True),
                     A.Affine(rotate=(-90, 90), scale=(0.9, 1.1), cval=PAD_CVALS, p=p, mode=mode_cv2),
                     A.HorizontalFlip(p=p),
                     thres=thres), ]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugV3(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugV3, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, thres=1, p=1, to_tensor=True, mode=MODE.CONSTANT, **kwargs):
        W, H = img_size
        mode_cv2 = MODE.mode2cv2(mode)
        trans = [LargestMaxSize(max_size=img_size),
                 AlbuCompose(
                     A.PadIfNeeded(min_height=H, min_width=W, value=PAD_CVALS, border_mode=mode_cv2,
                                   always_apply=True),
                     A.Affine(rotate=(-20, 20), shear=(-5, 5), scale=(0.8, 1.2), p=p,
                              translate_percent=(-0.2, 0.2), cval=PAD_CVALS, mode=mode_cv2),
                     A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=p),
                     A.HorizontalFlip(p=p),
                     thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugV3R(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugV3R, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, thres=1, p=1, to_tensor=True, mode=MODE.REFLECT, **kwargs):
        W, H = img_size
        mode_cv2 = MODE.mode2cv2(mode)
        trans = [LargestMaxSize(max_size=img_size),
                 AlbuCompose(
                     A.PadIfNeeded(min_height=H, min_width=W, value=PAD_CVALS, border_mode=mode_cv2,
                                   always_apply=True),
                     A.Affine(rotate=(-90, 90), shear=(-5, 5), scale=(0.8, 1.2), p=p,
                              translate_percent=(-0.1, 0.1), cval=PAD_CVALS, mode=mode_cv2),
                     A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=p),
                     A.Flip(p=p),
                     thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugV4R(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugV4R, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, thres=1, p=1, to_tensor=True, with_mix=True, with_frgd=True,
                         repeat=0.1, samp_dct=None, rotate=(-10, 10), scale=(0.8, 1.2), **kwargs):
        W, H = img_size
        trans = [
            Mosaic(repeat=0.2, img_size=img_size, add_type=ADD_TYPE.COVER, pad_val=PAD_CVALS),
            LargestMaxSize(max_size=img_size),
            AlbuCompose(
                A.PadIfNeeded(min_height=H, min_width=W, value=PAD_CVALS, border_mode=cv2.BORDER_CONSTANT),
                A.Affine(rotate=rotate, shear=(-5, 5), scale=scale,
                         translate_percent=(-0.1, 0.1), cval=PAD_CVALS, always_apply=True),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, ),
                A.HorizontalFlip(p=0.5),
            )]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)


class AugV4(AugSeq):
    def __init__(self, img_size=(416, 352), **kwargs):
        super(AugV4, self).__init__(img_size, **kwargs)

    def _build_transform(self, img_size, thres=1, p=1, to_tensor=True, with_mix=True, with_frgd=True,
                         repeat=0.1, samp_dct=None, rotate=(-10, 10), scale=(0.8, 1.2), **kwargs):
        W, H = img_size
        trans = [
            Mosaic(repeat=0.2, img_size=img_size, add_type=ADD_TYPE.COVER, pad_val=PAD_CVALS),
            LargestMaxSize(max_size=img_size),
            AlbuCompose(
                A.PadIfNeeded(min_height=H, min_width=W, value=PAD_CVALS, border_mode=cv2.BORDER_CONSTANT),
                A.Affine(rotate=rotate, shear=(-5, 5), scale=scale,
                         translate_percent=(-0.1, 0.1), cval=PAD_CVALS),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, ),
                A.HorizontalFlip(p=0.5),
            )]
        # if with_mix:
        #     trans.append(MaskCutMix(repeat=repeat, samp_dct=samp_dct,
        #                             thres=32, scale=(1 / 3, 3), rotate=(-np.pi / 4, np.pi / 4), with_frgd=with_frgd,
        #                             num_patch=2))
        if to_tensor:
            trans.append(ToTensor(concat=True))
        return Sequential(*trans)

# # </editor-fold>
