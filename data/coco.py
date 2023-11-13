from collections import Counter

import pycocotools
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO

from data.processing import *
from utils import *


# <editor-fold desc='数据集转化'>

def _ensure_folders(root, folders):
    dirs = []
    for folder in folders:
        root_dir = os.path.join(root, folder)
        ensure_folder_pth(root_dir)
        dirs.append(root_dir)
    return dirs


# 逐检测框地生成分割数据集
def datasetI2cocoI_perbox(dataset, root, json_name, img_folder='images', json_folder='annotation',
                          name2cind_remapper=None, expend_ratio=1.2, with_clip=False, as_square=False,
                          avoid_overlap=True, cls_names=None, only_one=True, thres=5,
                          ratio_thres=0.2, ignore_empty=True):
    img_dir, json_dir = _ensure_folders(root, [img_folder, json_folder])
    json_pth = ensure_extend(os.path.join(json_dir, json_name), 'json')
    print('Crop Instance to ' + root + ' < ' + img_folder + ' , ' + os.path.join(json_folder, json_name) + ' >')
    plabels_all = []
    for i, (img, insts) in MEnumerate(dataset, prefix='Writing', broadcast=print, with_eta=True):
        pieces, plabels = img2piece_perbox(
            img, insts, expend_ratio=expend_ratio, with_clip=with_clip, as_square=as_square,
            avoid_overlap=avoid_overlap, cls_names=cls_names, only_one=only_one, thres=thres,
            ratio_thres=ratio_thres, ignore_empty=ignore_empty)
        for piece, plabel in zip(pieces, plabels):
            patch_pth = os.path.join(img_dir, plabel.meta + '.jpg')
            piece.save(patch_pth)
            plabels_all.append(plabel)

    CoCoDataSet.create_json(plabels_all, json_pth=json_pth, name2cind_remapper=name2cind_remapper, img_extend='jpg')
    print('Crop completed with %d patches' % len(plabels_all))
    return plabels_all


# 按尺寸分割数据集
def datasetI2cocoI_persize(dataset, root, json_name, img_folder='images', json_folder='annotation',
                           name2cind_remapper=None, ignore_empty=True, with_clip=False, piece_size=(640, 640),
                           thres=5, over_lap=(100, 100), cls_names=None, img_extend='jpg'):
    img_dir, json_dir = _ensure_folders(root, [img_folder, json_folder])
    json_pth = ensure_extend(os.path.join(json_dir, json_name), 'json')
    print('Crop Instance to ' + root + ' < ' + img_folder + ' , ' + os.path.join(json_folder, json_name) + ' >')
    plabels_all = []
    for i, (img, items) in MEnumerate(dataset, prefix='Converting ', with_eta=True):
        if ignore_empty and len(items) == 0:
            continue
        pieces, plabels = img2piece_persize(
            img, items, piece_size=piece_size, thres=thres, over_lap=over_lap, ignore_empty=ignore_empty,
            with_clip=with_clip, cls_names=cls_names)
        del img, items
        for piece, plabel in zip(pieces, plabels):
            patch_pth = os.path.join(img_dir, ensure_extend(plabel.meta, img_extend))
            piece.save(patch_pth)
            plabels_all.append(plabel)
        del pieces, plabels
    CoCoDataSet.create_json(labels=plabels_all, json_pth=json_pth, name2cind_remapper=name2cind_remapper,
                            img_extend=img_extend)
    print('Crop completed with %d patches' % len(plabels_all))
    return plabels_all


# 按提取子类
def datasetI2cocoI(dataset, root, json_name, img_folder='images', json_folder='annotation',
                   name2cind_remapper=None, ignore_empty=True, thres=5, cls_names=None, img_extend='jpg'):
    img_dir, json_dir = _ensure_folders(root, [img_folder, json_folder])
    json_pth = ensure_extend(os.path.join(json_dir, json_name), 'json')
    print('Extract Instance to ' + root + ' < ' + img_folder + ' , ' + os.path.join(json_folder, json_name) + ' >')
    labels_all = []
    for i, (img, items) in MEnumerate(dataset, prefix='Extracting ', with_eta=True):
        img, items = img2img_flit(img, items, thres=thres, cls_names=cls_names)
        if ignore_empty and len(items) == 0:
            continue
        img_pth = os.path.join(img_dir, ensure_extend(items.meta, img_extend))
        img.save(img_pth)
        labels_all.append(items)

    CoCoDataSet.create_json(labels=labels_all, json_pth=json_pth, name2cind_remapper=name2cind_remapper,
                            img_extend=img_extend)
    print('Extract completed with %d images' % len(labels_all))
    return labels_all


# </editor-fold>


class CoCoDataSet(NameMapper, PreLoader, DataSet):

    def __init__(self, root, json_name, img_folder='images', json_folder='annotation',
                 cind2name_remapper=None, cls_names=None, **kwargs):
        json_name = ensure_extend(json_name, 'json')
        json_pth = os.path.join(root, json_folder, json_name)
        img_dir = os.path.join(root, img_folder)
        self.root = root
        self.img_folder = img_folder
        self.json_folder = json_folder
        self.json_name = json_name
        if os.path.exists(json_pth):
            coco_tool = COCO(json_pth)
            if cls_names is None:
                cind2name_remapper = lambda cind: coco_tool.cats[cind]['name']
                cls_names = list(sorted([cate_dict['name'] for cate_dict in coco_tool.cats.values()]))
            self.cind2name_remapper = cind2name_remapper
            metas = list(sorted(coco_tool.imgs.keys()))
            self.img_annos = [coco_tool.imgs[meta] for meta in metas]
            self.obj_annoss = [coco_tool.imgToAnns[meta] for meta in metas]
            self.img_pths = [os.path.join(img_dir, img_anno['file_name']) for img_anno in self.img_annos]
            self.metas = [img_anno['file_name'].split('.')[0] for img_anno in self.img_annos]
        else:
            self.img_pths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]
            self.obj_annoss = [[]] * len(self.img_pths)
            self.img_annos = [{}] * len(self.img_pths)
            self.cind2name_remapper = lambda cind: 'box'
            self.metas = [img_name.split('.')[0] for img_name in os.listdir(img_dir)]

        super(CoCoDataSet, self).__init__(cls_names=cls_names)
        super(NameMapper, self).__init__(**kwargs)

    # <editor-fold desc='coco工具集'>
    @staticmethod
    def collect_names(obj_annoss, cind2name_remapper):
        names = []
        for obj_annos in obj_annoss:
            for obj_anno in obj_annos:
                name = obj_anno['category_name'] if cind2name_remapper is None \
                    else cind2name_remapper(int(obj_anno['category_id']))
                names.append(name)
        return np.array(names)

    @staticmethod
    def collect_img_sizes(img_annos):
        img_sizes = []
        for img_anno in img_annos:
            img_sizes.append((img_anno['width'], img_anno['height']))
        return np.array(img_sizes)

    @staticmethod
    def collect_sizes(obj_annoss):
        sizes = []
        for obj_annos in obj_annoss:
            for obj_anno in obj_annos:
                xyxy = np.array(obj_anno['bbox'])
                sizes.append(xyxy[2:4])
        return np.array(sizes)

    @staticmethod
    def binary_mask2rle(maskN, as_list=True):
        if not as_list:
            rle = mask_utils.encode(np.array(maskN, order='F', dtype=np.uint8))
            rle['counts'] = rle['counts'].decode('utf-8')
        else:
            binary_arr = maskN.ravel(order='F')
            binary_diff = np.diff(binary_arr, axis=0, prepend=0, append=0)
            binary_diff[-1] = 1
            counts_abs = np.nonzero(binary_diff)[0]
            counts = np.diff(counts_abs, axis=0, prepend=0).astype(int)
            rle = {'counts': counts.tolist(), 'size': list(maskN.shape)}
        return rle

    # @staticmethod
    # def binary_mask2rle(binary_mask):
    #     rle = {'counts': [], 'size': list(binary_mask.shape)}
    #     counts = rle.get('counts')
    #     for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
    #         if i == 0 and value == 1:
    #             counts.append(0)
    #         counts.append(len(list(elements)))
    #     return rle

    @staticmethod
    def create_json(labels, json_pth='instances.json', name2cind_remapper=None, img_extend='jpg'):
        json_dict = CoCoDataSet.labels2json_dct(labels, name2cind_remapper=name2cind_remapper, img_extend=img_extend)
        ensure_file_dir(json_pth)
        json_pth = ensure_extend(json_pth, 'json')
        print('Write json annotation ' + json_pth)
        save_json(json_pth, json_dict)
        return json_dict

    @staticmethod
    def json_dct2coco_obj(json_dct):
        coco_obj = pycocotools.coco.COCO()
        coco_obj.dataset = json_dct
        coco_obj.createIndex()
        return coco_obj

    @staticmethod
    def labels2coco_obj(labels, name2cind_remapper=None, img_extend='jpg', with_score=False, with_rgn=True,
                        as_list=False):
        json_dct = CoCoDataSet.labels2json_dct(labels, name2cind_remapper=name2cind_remapper, img_extend=img_extend,
                                               with_score=with_score, with_rgn=with_rgn, as_list=as_list)
        return CoCoDataSet.json_dct2coco_obj(json_dct)

    @staticmethod
    def labels2json_lst(labels, name2cind_remapper=None, with_score=False, with_rgn=True, img_id_mapper=None,
                        as_list=False):
        annotations = []
        for i, label in MEnumerate(labels, prefix='Generating ', with_eta=True):
            img_id = i if img_id_mapper is None else img_id_mapper(label.meta)
            for item in label:
                assert isinstance(item, InstItem)
                cind = int(IndexCategory.convert(item.category).cindN) \
                    if name2cind_remapper is None else name2cind_remapper(item['name'])
                xyxy = XYXYBorder.convert(item.border).xyxyN.astype(float)
                xydwdh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                ann_dict = {'image_id': img_id, 'category_id': cind, 'bbox': xydwdh, }
                if with_score:
                    ann_dict['score'] = item.category.conf
                if with_rgn:
                    if isinstance(item.rgn, XLYLBorder):
                        segment = [list(item.rgn.xlylN.reshape(-1).astype(float))]
                    else:
                        segment = CoCoDataSet.binary_mask2rle(item.rgn.maskNb, as_list=as_list)
                    ann_dict['segmentation'] = segment
                annotations.append(ann_dict)
        return annotations

    @staticmethod
    def labels2json_dct(labels, name2cind_remapper=None, img_extend='jpg', with_score=False, with_rgn=True,
                        as_list=True):
        img_infos = []
        cate_dict = {}
        annotations = []
        img_id = 0
        ann_id = 0
        print('Generate json annotation')
        for i, label in MEnumerate(labels, prefix='Generating ', with_eta=True):
            assert isinstance(label, ImageItemsLabel)
            w, h = label.img_size
            img_dict = {
                'id': img_id,
                'width': int(w), 'height': int(h),
                'file_name': ensure_extend(label.meta, img_extend),
            }
            img_infos.append(img_dict)
            for item in label:
                name = item['name']
                cind = int(IndexCategory.convert(item.category).cindN) \
                    if name2cind_remapper is None else name2cind_remapper(name)
                cate_dict[name] = cind

                iscrowd = 0
                xyxy = XYXYBorder.convert(item.border).xyxyN.astype(float)
                xydwdh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                ann_dict = {'id': ann_id, 'image_id': img_id, 'category_name': name, 'category_id': cind,
                            'bbox': xydwdh, 'iscrowd': iscrowd, 'area': xydwdh[2] * xydwdh[3]}
                if with_score:
                    ann_dict['score'] = item.category.conf
                if with_rgn and isinstance(item, InstItem):
                    if isinstance(item.rgn, XLYLBorder):
                        segment = [list(item.rgn.xlylN.reshape(-1).astype(float))]
                    else:
                        segment = CoCoDataSet.binary_mask2rle(item.rgn.maskNb, as_list=as_list)
                    area = float(item.rgn.area())
                    ann_dict['segmentation'] = segment
                    ann_dict['area'] = area

                annotations.append(ann_dict)
                ann_id += 1
            img_id += 1
            del label
        categories = [{'id': id, 'name': name} for name, id in cate_dict.items()]
        json_dict = {
            'images': img_infos,
            'categories': categories,
            'annotations': annotations,
        }
        return json_dict

    # </editor-fold>
    @property
    def sizes(self):
        return CoCoDataSet.collect_sizes(self.obj_annoss)

    @property
    def names(self):
        return CoCoDataSet.collect_names(self.obj_annoss, self.cind2name_remapper)

    def __len__(self):
        return len(self.img_pths)

    def _getitem_protype(self, item):
        return None

    def rename(self, rename_dict, json_name='instances2'):
        json_pth = ensure_extend(os.path.join(self.root, self.json_folder, self.json_name), 'json')
        json_pth_new = ensure_extend(os.path.join(self.root, self.json_folder, json_name), 'json')
        if not os.path.exists(json_pth):
            return self
        json_dct = load_json(json_pth)
        categories = json_dct['categories']
        annotations = json_dct['annotations']
        for cate in categories:
            if cate['name'] in rename_dict.keys():
                cate['name'] = rename_dict[cate['name']]
        for anno in annotations:
            if anno['category_name'] in rename_dict.keys():
                anno['category_name'] = rename_dict[anno['category_name']]
        save_json(json_pth_new, json_dct)
        return self

    def rename_cind(self, name2cind, json_name='instances2'):
        json_pth = ensure_extend(os.path.join(self.root, self.json_folder, self.json_name), 'json')
        json_pth_new = ensure_extend(os.path.join(self.root, self.json_folder, json_name), 'json')
        if not os.path.exists(json_pth):
            return self
        json_dct = load_json(json_pth)
        categories = json_dct['categories']
        annotations = json_dct['annotations']
        for cate in categories:
            cate['id'] = name2cind(cate['name'])
        for anno in annotations:
            anno['category_id'] = name2cind(anno['category_name'])
        save_json(json_pth_new, json_dct)
        return self

    # 使用函数对标签形式进行变换
    def label_apply(self, func, json_name='instances2', img_extend='jpg'):
        json_pth_new = ensure_extend(os.path.join(self.root, self.json_folder, json_name), 'json')
        print('Convert Instance to ' + self.root + ' < ' + os.path.join(self.json_folder, json_name) + ' >')
        labels = []
        for i, img_ann in MEnumerate(self.img_annos, prefix='Converting', with_eta=True):
            obj_annos = self.obj_annoss[i]
            insts = CoCoIDataSet.prase_anns(
                img_ann=img_ann, obj_anns=obj_annos, num_cls=self.num_cls, cind2name_remapper=self.cind2name_remapper,
                name2cind=self.name2cind)
            insts = func(insts)
            labels.append(insts)
        CoCoDataSet.create_json(labels, json_pth_new, name2cind_remapper=None, img_extend=img_extend)
        return self

    # 部分标注形式没有对应图像尺寸
    # 添加图像尺寸标注
    def add_img_size_anno(self):
        json_name = ensure_extend(self.json_name, 'json')
        json_pth = os.path.join(self.root, self.json_folder, json_name)
        json_dct = load_json(json_pth)
        img_annos = json_dct['images']
        print('Add image size anno ', json_pth)
        for i, img_ann in MEnumerate(img_annos, prefix='Reading', with_eta=True):
            img_pth = os.path.join(self.root, self.img_folder, img_ann['file_name'])
            img = Image.open(img_pth)
            width, height = img.size
            img_ann['width'] = width
            img_ann['height'] = height
        save_json(json_pth, json_dct)
        return None

    def __repr__(self):
        num_dict = Counter(self.names)
        msg = '\n'.join(['%-30s ' % name + ' %-6d' % num for name, num in num_dict.items()])
        return msg


class CoCoDDataSet(CoCoDataSet):
    def __init__(self, root, json_name, img_folder='images', json_folder='annotation',
                 cind2name_remapper=None, cls_names=None, **kwargs):
        super().__init__(root, json_name=json_name, img_folder=img_folder, json_folder=json_folder,
                         cind2name_remapper=cind2name_remapper, cls_names=cls_names, **kwargs)

    @staticmethod
    def ann2cate_name(obj_ann, cind2name_remapper, name2cind, num_cls):
        name = obj_ann['category_name'] if cind2name_remapper is None \
            else cind2name_remapper(int(obj_ann['category_id']))
        cind = name2cind(name)
        category = IndexCategory(cindN=cind, num_cls=num_cls, conf=1)
        return category, name

    @staticmethod
    def ann2border(obj_ann, img_size, bdr_type=None):
        xyxy = np.array(obj_ann['bbox'])
        xyxy[2:4] += xyxy[:2]
        border = XYXYBorder(xyxyN=xyxy, size=img_size)
        if bdr_type is not None and not isinstance(border, bdr_type):
            border = bdr_type.convert(border)
        return border

    @staticmethod
    def prase_anns(img_ann, obj_anns, num_cls, cind2name_remapper, name2cind, bdr_type=None):
        meta = img_ann['file_name'].split('.')[0]
        img_size = (img_ann.get('width', 0), img_ann.get('height', 0))
        boxes = BoxesLabel(meta=meta, img_size=img_size)
        for obj_ann in obj_anns:
            border = CoCoDDataSet.ann2border(obj_ann, img_size=img_size, bdr_type=bdr_type)
            category, name = CoCoDDataSet.ann2cate_name(obj_ann, cind2name_remapper, name2cind, num_cls)
            boxes.append(BoxItem(border=border, category=category, name=name))
        return boxes

    def _getitem_protype(self, index):
        img_pth = self.img_pths[index]
        img = self.load_img(img_pth)
        img_ann = self.img_annos[index]
        img_ann.update(dict(file_name=os.path.basename(img_pth), width=img.size[0], height=img.size[1]))
        boxes = CoCoDDataSet.prase_anns(img_ann=img_ann, obj_anns=self.obj_annoss[index],
                                        num_cls=self.num_cls, cind2name_remapper=self.cind2name_remapper,
                                        name2cind=self.name2cind, bdr_type=self.bdr_type)
        return img, boxes

    @property
    def labels(self):
        labels = []
        for img_ann, obj_anns in zip(self.img_annos, self.obj_annoss):
            boxes = CoCoDDataSet.prase_anns(
                img_ann=img_ann, obj_anns=obj_anns, num_cls=self.num_cls, cind2name_remapper=self.cind2name_remapper,
                name2cind=self.name2cind, bdr_type=self.bdr_type)
            labels.append(boxes)
        return labels


class CoCoSDataSet(CoCoDataSet):
    def __init__(self, root, json_name, img_folder='images', json_folder='annotation',
                 cind2name_remapper=None, cls_names=None, **kwargs):
        super().__init__(root, json_name=json_name, img_folder=img_folder, json_folder=json_folder,
                         cind2name_remapper=cind2name_remapper, cls_names=cls_names, **kwargs)

    @staticmethod
    def ann2rgn(obj_ann, img_size, rgn_type=None):
        segmentation = obj_ann['segmentation']
        xyxy = np.array(obj_ann['bbox'])
        xyxy[2:4] += xyxy[:2]
        if isinstance(segmentation, list):
            if len(segmentation) == 1:
                xlylN = np.array(segmentation[0]).reshape(-1, 2)
                rgn = XLYLBorder(xlylN, img_size)
            else:
                xlylNs = [np.array(xlylN).reshape(-1, 2) for xlylN in segmentation]
                maskNb = xlylNs2maskNb(xlylNs, size=img_size)
                rgn = RefValRegion.from_maskNb_xyxyN(maskNb, xyxy)
        else:
            sp0, sp1 = segmentation['size']
            rle = mask_utils.frPyObjects(segmentation, sp0, sp1)
            maskN = np.array(mask_utils.decode(rle), dtype=bool)
            rgn = RefValRegion.from_maskNb_xyxyN(maskN, xyxy)
        if rgn_type is not None and not isinstance(rgn, rgn_type):
            rgn = rgn_type.convert(rgn)
        return rgn

    @staticmethod
    def prase_anns(img_ann, obj_anns, num_cls, cind2name_remapper, name2cind):
        meta = img_ann['file_name'].split('.')[0]
        img_size = (img_ann['width'], img_ann['height'])
        segs = SegsLabel(meta=meta, img_size=img_size)
        for obj_ann in obj_anns:
            category, name = CoCoDDataSet.ann2cate_name(obj_ann, cind2name_remapper, name2cind, num_cls)
            rgn = CoCoSDataSet.ann2rgn(obj_ann, img_size)
            inserted = False
            for seg in segs:
                if seg.category.cindN == category.cindN:
                    seg.rgn = AbsBoolRegion(maskNb_abs=seg.rgn.maskNb + rgn.maskNb)
                    inserted = True
                    break
            if not inserted:
                segs.append(SegItem(rgn=rgn, category=category, name=name))
        return segs

    def _getitem_protype(self, index):
        img_pth = self.img_pths[index]
        img = self.load_img(img_pth)
        img_ann = self.img_annos[index]
        img_ann.update(dict(file_name=os.path.basename(img_pth), width=img.size[0], height=img.size[1]))
        boxes = CoCoSDataSet.prase_anns(img_ann=img_ann, obj_anns=self.obj_annoss[index],
                                        num_cls=self.num_cls, cind2name_remapper=self.cind2name_remapper,
                                        name2cind=self.name2cind)
        return img, boxes


class CoCoIDataSet(CoCoDataSet):
    def __init__(self, root, json_name, img_folder='images', json_folder='annotation',
                 cind2name_remapper=None, cls_names=None, rgn_type=None, bdr_type=None, **kwargs):
        self.rgn_type = rgn_type
        self.bdr_type = bdr_type
        super().__init__(root, json_name=json_name, img_folder=img_folder, json_folder=json_folder,
                         cind2name_remapper=cind2name_remapper, cls_names=cls_names, **kwargs)

    @staticmethod
    def prase_anns(img_ann, obj_anns, num_cls, cind2name_remapper, name2cind, rgn_type=None, bdr_type=None):
        meta = img_ann['file_name'].split('.')[0]
        img_size = (img_ann.get('width', 0), img_ann.get('height', 0))
        insts = InstsLabel(meta=meta, img_size=img_size)
        for obj_ann in obj_anns:
            border = CoCoDDataSet.ann2border(obj_ann, img_size=img_size, bdr_type=bdr_type)
            category, name = CoCoDDataSet.ann2cate_name(obj_ann, cind2name_remapper, name2cind, num_cls)
            rgn = CoCoSDataSet.ann2rgn(obj_ann, img_size, rgn_type=rgn_type)
            inst = InstItem(border=border, rgn=rgn, category=category, name=name)
            insts.append(inst)
        return insts

    def _getitem_protype(self, index):
        img_pth = self.img_pths[index]
        img = self.load_img(img_pth)
        img_ann = self.img_annos[index]
        img_ann.update(dict(file_name=os.path.basename(img_pth), width=img.size[0], height=img.size[1]))
        insts = CoCoIDataSet.prase_anns(img_ann=img_ann, obj_anns=self.obj_annoss[index],
                                        num_cls=self.num_cls, cind2name_remapper=self.cind2name_remapper,
                                        name2cind=self.name2cind, rgn_type=self.rgn_type, bdr_type=self.bdr_type)

        return img, insts

    def labels(self):
        labels = []
        for img_ann, obj_anns in zip(self.img_annos, self.obj_annoss):
            insts = CoCoIDataSet.prase_anns(
                img_ann=img_ann, obj_anns=obj_anns, num_cls=self.num_cls, cind2name_remapper=self.cind2name_remapper,
                name2cind=self.name2cind, rgn_type=self.rgn_type, bdr_type=self.bdr_type)
            labels.append(insts)
        return labels


class CoCo(DataSource):
    CIND2NAME_REMAPPER_DICT = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
        77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
        82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
        88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }
    CIND2NAME_REMAPPER = lambda cind: CoCo.CIND2NAME_REMAPPER_DICT[cind]
    NEME2CIND_REMAPPER_DICT = dict([val, key] for key, val in CIND2NAME_REMAPPER_DICT.items())
    CLS_NAMES = tuple(CIND2NAME_REMAPPER_DICT.values())
    IMG_PREFIX = 'images_'
    JSON_PREFIX = 'instances_'
    JSON_FOLDER = 'annotation'
    ROOT_SEV_NEW = '/home/data-storage/COCO'
    ROOT_SEV_OLD = '//home//exspace//dataset//COCO'
    ROOT_DES = 'D://Datasets//COCO//'
    IMG_EXTEND = 'jpg'

    BUILDER_MAPPER = {
        TASK_TYPE.DETECTION: CoCoDDataSet,
        TASK_TYPE.SEGMENTATION: CoCoSDataSet,
        TASK_TYPE.INSTANCE: CoCoIDataSet,
    }

    @staticmethod
    def SEV_NEW(**kwargs):
        return CoCo(root=CoCo.ROOT_SEV_NEW, **kwargs)

    @staticmethod
    def SEV_OLD(**kwargs):
        return CoCo(root=CoCo.ROOT_SEV_OLD, **kwargs)

    @staticmethod
    def DES(**kwargs):
        return CoCo(root=CoCo.ROOT_DES, **kwargs)

    def __init__(self, root, img_prefix=IMG_PREFIX, json_prefix=JSON_PREFIX, json_folder=JSON_FOLDER,
                 cind2name_remapper=CIND2NAME_REMAPPER, task_type=TASK_TYPE.DETECTION,
                 cls_names=CLS_NAMES, set_names=None, **kwargs):
        if set_names is None:
            json_dir = os.path.join(root, json_folder)
            set_names = [file_name.split('.')[0] for file_name in listdir_extend(json_dir, 'json')] \
                if os.path.exists(json_dir) else tuple()

        super().__init__(root=root, set_names=set_names, task_type=task_type)
        self.root = root
        self.img_prefix = img_prefix
        self.json_folder = json_folder
        self.json_prefix = json_prefix
        self.cind2name_remapper = cind2name_remapper
        self.kwargs = kwargs
        self.cls_names = cls_names

    def dataset(self, set_name, task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = CoCo.BUILDER_MAPPER[task_type]
        kwargs_update = dict(root=self.root, json_folder=self.json_folder,
                             cind2name_remapper=self.cind2name_remapper, cls_names=self.cls_names)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        if 'img_folder' not in kwargs_update.keys():
            kwargs_update['img_folder'] = self.img_prefix + set_name
        if 'json_name' not in kwargs_update.keys():
            kwargs_update['json_name'] = self.json_prefix + set_name
        dataset = builder(fmt=set_name + '_%d', **kwargs_update)
        return dataset


# if __name__ == '__main__':
#     ds = CoCo.SEV_NEW()
#     loader = ds.loader(set_name='val', batch_size=4, shuffle=False, num_workers=0, aug_seq=None)
#     imgs, labels = next(iter(loader))

# if __name__ == '__main__':
#     ds = CoCo.SEV_NEW(task_type=TASK_TYPE.INSTANCE)
#     dataset = ds.dataset('train')
#     for img, label in dataset:
#         print(label.meta)
#         pass


if __name__ == '__main__':
    # maskN = np.zeros((10, 10))
    maskN = np.random.rand(1000, 1000)
    maskN = np.where(maskN > 0.7, np.ones_like(maskN), np.zeros_like(maskN))
    time1 = time.time()
    for i in range(100):
        rle = CoCoDataSet.binary_mask2rle(maskN)
        # rle2 = binary_mask2rle(maskN)

        # print(rle['counts'])
        # print(rle3['counts'])
        # print(np.sum(np.abs(np.array(rle['counts']) - np.array(rle3['counts']))))
    time2 = time.time()
    print(time2 - time1)
